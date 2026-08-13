"""SimpleFusion baseline.

A concatenation-based fusion baseline (``[15, 30, 32, 38, 43, 46]`` in the
paper), used to illustrate the *highly coupled feature fusion* problem discussed
in Sec. 1.  Missing modalities are zeroed according to the missing prompt of
Eq. 2, which mirrors the protocol of Table 1.
"""

import torch
import torch.nn as nn

from common.abstract import AbstractModel
from models.modules.cka import ensure_sequence
from utils.utils import xavier_normal_initialization


class SimpleFusion(AbstractModel):
    def __init__(self, config, debunk_data=None):
        super(SimpleFusion, self).__init__()
        self.config = config
        self.dim = self.config['fea_dim']
        self.dropout = self.config['dropout']

        self.linear_txt = nn.Sequential(
            nn.Linear(self.config['text_dim'], self.dim), nn.ReLU(), nn.Dropout(p=self.dropout)
        )
        self.linear_img = nn.Sequential(
            nn.Linear(self.config['image_dim'], self.dim), nn.ReLU(), nn.Dropout(p=self.dropout)
        )
        self.linear_audio = nn.Sequential(
            nn.Linear(self.config['audio_dim'], self.dim), nn.ReLU(), nn.Dropout(p=self.dropout)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.apply(xavier_normal_initialization)
        self.classifier = nn.Linear(self.dim * 3, 2)

    @staticmethod
    def _read_prompt(batch):
        prompt = batch.get('prompt', batch.get('masker'))
        if prompt is None:
            raise KeyError("batch must provide either 'prompt' or 'masker'")
        return prompt.float()

    def predict(self, batch):
        return self.forward(batch)

    def forward(self, batch):
        prompt = self._read_prompt(batch)
        keep_video = (1.0 - prompt[:, 1]).view(-1, 1, 1)
        keep_text = (1.0 - prompt[:, 2]).view(-1, 1, 1)
        keep_audio = (1.0 - prompt[:, 3]).view(-1, 1, 1)

        fea_text = self.linear_txt(ensure_sequence(batch['text']) * keep_text).mean(dim=1)
        fea_img = self.linear_img(ensure_sequence(batch['frames']) * keep_video).mean(dim=1)
        fea_audio = self.linear_audio(
            ensure_sequence(batch['audioframes']) * keep_audio
        ).mean(dim=1)

        fea_mm = torch.cat((fea_text, fea_img, fea_audio), dim=-1)
        return self.classifier(fea_mm)

    def calculate_loss(self, batch):
        logits = self.forward(batch)
        return self.criterion(logits, batch['label'])
