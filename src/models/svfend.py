# coding: utf-8
r"""

################################################

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import xavier_normal_initialization
from common.abstract import AbstractModel
from common.function import *


class SVFEND(AbstractModel):
    def __init__(self, config, debunk_data):
        super(SVFEND, self).__init__()
        # Initialize hyperparameters
        self.config = config
        self.dim = self.config['fea_dim']
        self.dropout = self.config['dropout']
        self.num_heads = self.config['num_heads']
        self.batch_size = self.config['batch_size']

        # Initialize network
        self.co_attention_ta = co_attention(d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=self.dim,
                                            visual_len=512, sen_len=512, fea_v=self.dim,
                                            fea_s=self.dim, pos=False)
        self.co_attention_tv = co_attention(d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=self.dim,
                                            visual_len=512, sen_len=512, fea_v=self.dim, fea_s=self.dim,
                                            pos=False)
        self.trm = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.num_heads, batch_first=True)
        self.linear_txt = nn.Sequential(torch.nn.Linear(self.config['text_dim'], self.dim), torch.nn.ReLU(),
                                         nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.config['image_dim'], self.dim), torch.nn.ReLU(),
                                        nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(self.config['audio_dim'], self.dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))


        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Initialize
        self.apply(xavier_normal_initialization)
        self.classifier = nn.Linear(self.dim, 2)

    def predict(self, batch):
        predict = self.forward(batch)
        return predict

    def forward(self, batch):
        text = 1
        img = 1
        audio = 1

        # Title
        fea_text = batch['text'].squeeze(dim=1)
        zero_text = torch.zeros_like(fea_text)
        if text == 0:
            fea_text = zero_text
        fea_text = self.linear_txt(fea_text)

        # Audio Frames
        fea_audio = batch['audioframes'].squeeze(dim=1)
        zero_audio = torch.zeros_like(fea_audio)
        if audio == 0:
            fea_audio = zero_audio
        fea_audio = self.linear_audio(fea_audio)
        fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1],
                                                   s_len=fea_text.shape[1])
        fea_audio = torch.mean(fea_audio, -2)

        # Image Frames
        fea_img = batch['frames'].squeeze(dim=1)
        zero_img = torch.zeros_like(fea_img)
        if img == 0:
            fea_img = zero_img
        fea_img = self.linear_img(fea_img)
        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1],
                                                 s_len=fea_text.shape[1])

        fea_img = torch.mean(fea_img, -2)
        fea_text = torch.mean(fea_text, -2)

        fea_text = fea_text.unsqueeze(1)
        fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)

        empty_fea = torch.zeros_like(fea_text)
        fea_mm = torch.cat((fea_text, fea_img, fea_audio), 1)  # (bs, 6, 128)
        fea_mm = self.trm(fea_mm)
        fea_mm = torch.mean(fea_mm, -2)

        output = self.classifier(fea_mm)
        return output

    def calculate_loss(self, batch):
        label = batch['label']
        predict = self.forward(batch)
        loss = self.criterion(predict, label)
        return loss


