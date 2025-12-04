import torch
import torch.nn as nn

from utils.utils import xavier_normal_initialization
from common.abstract import AbstractModel


class SimpleFusion(AbstractModel):
    def __init__(self, config, debunk_data):
        super(SimpleFusion, self).__init__()
        self.config = config
        self.dim = self.config['fea_dim']
        self.dropout = self.config['dropout']
        self.batch_size = self.config['batch_size']

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
        self.classifier = nn.Linear(self.dim * 3, 2)

    def predict(self, batch):
        predict = self.forward(batch)
        return predict

    def forward(self, batch):
        text = 1
        img = 1
        audio = 0

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

        # Image Frames
        fea_img = batch['frames'].squeeze(dim=1)
        zero_img = torch.zeros_like(fea_img)
        if img == 0:
            fea_img = zero_img
        fea_img = self.linear_img(fea_img)

        fea_mm = torch.cat((fea_text, fea_img, fea_audio), -1)  # (bs, 6, 128)

        output = self.classifier(fea_mm)
        return output

    def calculate_loss(self, batch):
        label = batch['label']
        predict = self.forward(batch)
        loss = self.criterion(predict, label)
        return loss