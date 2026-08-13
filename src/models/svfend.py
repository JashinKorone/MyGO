# coding: utf-8
r"""SVFEND baseline (Qi et al., AAAI 2023).

Kept as a reference implementation for the comparison of Sec. 4.5 and as the
transformer-based host of the plug-and-play PMA study of Sec. 4.8.  Unlike the
original release, missing modalities are honoured through the missing prompt of
Eq. 2 instead of being hard-coded, so that the model can be evaluated on the
modality-incomplete FakeSV+ dataset.
"""

import torch
import torch.nn as nn

from common.abstract import AbstractModel
from common.function import co_attention
from models.modules import PromptAssistedModalityAligning
from models.modules.cka import ensure_sequence
from utils.utils import xavier_normal_initialization


class SVFEND(AbstractModel):
    def __init__(self, config, debunk_data=None):
        super(SVFEND, self).__init__()
        self.config = config
        self.dim = self.config['fea_dim']
        self.dropout = self.config['dropout']
        self.num_heads = self.config['num_heads']

        self.co_attention_ta = co_attention(
            d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
            d_model=self.dim, visual_len=512, sen_len=512, fea_v=self.dim, fea_s=self.dim,
            pos=False,
        )
        self.co_attention_tv = co_attention(
            d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
            d_model=self.dim, visual_len=512, sen_len=512, fea_v=self.dim, fea_s=self.dim,
            pos=False,
        )
        self.trm = nn.TransformerEncoderLayer(
            d_model=self.dim, nhead=self.num_heads, batch_first=True
        )
        self.linear_txt = nn.Sequential(
            nn.Linear(self.config['text_dim'], self.dim), nn.ReLU(), nn.Dropout(p=self.dropout)
        )
        self.linear_img = nn.Sequential(
            nn.Linear(self.config['image_dim'], self.dim), nn.ReLU(), nn.Dropout(p=self.dropout)
        )
        self.linear_audio = nn.Sequential(
            nn.Linear(self.config['audio_dim'], self.dim), nn.ReLU(), nn.Dropout(p=self.dropout)
        )

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.apply(xavier_normal_initialization)
        self.classifier = nn.Linear(self.dim, 2)

        # Sec. 4.8: PMA is plug-and-play, it only rescales the per-sample loss.
        self.pma = PromptAssistedModalityAligning(config)

    # --------------------------------------------------------------- lifecycle
    def post_epoch_processing(self):
        self.pma.step_epoch()
        return self.pma.summary()

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
        # The last three digits mark missing video / text / audio modalities.
        keep_video = (1.0 - prompt[:, 1]).view(-1, 1, 1)
        keep_text = (1.0 - prompt[:, 2]).view(-1, 1, 1)
        keep_audio = (1.0 - prompt[:, 3]).view(-1, 1, 1)

        fea_text = self.linear_txt(ensure_sequence(batch['text']) * keep_text)
        fea_audio = self.linear_audio(ensure_sequence(batch['audioframes']) * keep_audio)
        fea_audio, fea_text = self.co_attention_ta(
            v=fea_audio, s=fea_text, v_len=fea_audio.shape[1], s_len=fea_text.shape[1]
        )
        fea_audio = torch.mean(fea_audio, -2)

        fea_img = self.linear_img(ensure_sequence(batch['frames']) * keep_video)
        fea_img, fea_text = self.co_attention_tv(
            v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1]
        )
        fea_img = torch.mean(fea_img, -2)
        fea_text = torch.mean(fea_text, -2)

        fea_mm = torch.stack((fea_text, fea_img, fea_audio), dim=1)
        fea_mm = torch.mean(self.trm(fea_mm), -2)
        return self.classifier(fea_mm)

    def calculate_loss(self, batch):
        logits = self.forward(batch)
        loss_per_sample = self.criterion(logits, batch['label'])
        loss, _ = self.pma(self._read_prompt(batch), loss_per_sample)
        return loss
