import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

from utils.utils import xavier_normal_initialization
from common.abstract import AbstractModel
from common.function import *


class MCL(nn.Module):
    # Consistency-aware Fusion Layer
    def __init__(self, config):
        super(MCL, self).__init__()
        self.config = config
        self.dim = self.config['fea_dim']
        self.dropout = self.config['dropout']
        self.num_heads = self.config['num_heads']
        self.batch_size = self.config['batch_size']
        self.cl_temp = config['cl_temp']
        self.cl_block = CLBlock(config)

        self.co_attention_ta = co_attention(d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=self.dim,
                                            visual_len=50, sen_len=50, fea_v=self.dim,
                                            fea_s=self.dim, pos=False)
        self.co_attention_tv = co_attention(d_k=self.dim, d_v=self.dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=self.dim,
                                            visual_len=50, sen_len=50, fea_v=self.dim,
                                            fea_s=self.dim, pos=False)
        self.trm_fusion = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.num_heads, batch_first=True)
        self.linear_txt = nn.Sequential(nn.Linear(self.config['text_dim'], self.dim),
                                        nn.ReLU(),
                                        nn.Linear(self.dim, self.dim - 1),
                                        nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(nn.Linear(self.config['image_dim'], self.dim),
                                        nn.ReLU(),
                                        nn.Linear(self.dim, self.dim - 1),
                                        nn.Dropout(p=self.dropout))
        self.linear_au = nn.Sequential(nn.Linear(self.config['audio_dim'], self.dim),
                                       nn.ReLU(),
                                       nn.Linear(self.dim, self.dim - 1),
                                       nn.Dropout(p=self.dropout))

    def forward(self, batch):
        fea_masker = batch['masker']

        fea_text = batch['text'].squeeze(dim=1)
        fea_text = self.linear_txt(fea_text)
        fea_text_masker = fea_masker[:, -1].unsqueeze(dim=1)
        fea_text = torch.cat((fea_text, fea_text_masker), dim=1)

        # Image Frames
        fea_img = batch['frames'].squeeze(dim=1)
        fea_img = self.linear_img(fea_img)
        fea_img_masker = fea_masker[:, -3].unsqueeze(dim=1)
        fea_img = torch.cat((fea_img, fea_img_masker), dim=1)
        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1],
                                                 s_len=fea_text.shape[1])
        fea_img = torch.mean(fea_img, -2)

        # Audio Frames
        fea_audio = batch['audioframes'].squeeze(dim=1)
        fea_audio = self.linear_au(fea_audio)
        fea_audio_masker = fea_masker[:, -2].unsqueeze(dim=1)
        fea_audio = torch.cat((fea_audio, fea_audio_masker), dim=1)
        fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1],
                                                   s_len=fea_text.shape[1])
        fea_audio = torch.mean(fea_audio, -2)
        fea_text = torch.mean(fea_text, -2)

        fea_text = fea_text.unsqueeze(1)
        fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)

        fea_shared = torch.cat((fea_img, fea_text, fea_audio), 1)  # (bs, 3, 128)
        fea_shared = self.trm_fusion(fea_shared)
        fea_shared = torch.mean(fea_shared, -2)

        event_label = batch['event']
        fea_masker = batch['masker']
        label = batch['label']
        event_id_list = event_label.tolist()
        event_id_len = len(event_id_list)

        cl_loss = self.cl_block(features=fea_shared, labels=event_label)

        return fea_shared, fea_text, fea_img, fea_audio, fea_masker, cl_loss


class CLBlock(nn.Module):
    """
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy
        reduction: Reduction method applied to the output. Values must be one of ['none', 'sum', 'mean'].
        negative_mode: Determines how the (optional) negative_keys are handled. Values must be one of ['unpaired', 'paired'].
            If paired, then each query sample is paired with a number of negative keys. Comparable to a triplet loss,
            but with multiple negatives per sample.
            If unpaired, then the set of negative keys are all unrelated to any positive keys.

    Input Shape:
        query: (N, D) Tensor with query items (e.g. embedding of the input)
        key_matrix: (N, N) Tensor to indicate the positive and negative samples of target query items.

    Returns:
        Value of the InfoNCE Loss.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config['cl_temp']
        self.base_temperature = 1
        self.device = config['device']
        self.contrast_mode = 'one'
        # self.positive_sample_encoder = nn.MultiheadAttention(self.dim, self.num_heads)

    def forward(self, features, labels):
        cl_loss = self.contrast(features, labels)
        return cl_loss

    def contrast(self, features, labels):
        bsz = features.shape[0]
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        mask = torch.ones_like(similarity_matrix) * (labels.expand(bsz, bsz).eq(labels.expand(bsz, bsz).t()))
        mask_no_sim = torch.ones_like(mask) - mask
        mask_eye_0 = (torch.ones(bsz, bsz) - torch.eye(bsz, bsz)).to(self.device)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        similarity_matrix = similarity_matrix * mask_eye_0
        sim = mask*similarity_matrix
        no_sim = similarity_matrix - sim
        no_sim_sum = torch.sum(no_sim, dim=1)
        no_sim_sum_expand = no_sim_sum.repeat(bsz, 1).T
        sim_sum = sim + no_sim_sum_expand
        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(bsz, bsz).to(self.device)
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1))/(2*bsz)
        return loss

    def simple_contrast(self, features, labels, mask):
        if len(features.shape) < 3:
            features = features.unsqueeze(-1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and masks')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels dose not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            raise ValueError('Unknown mode')

        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def building_cl_samples(self, query, key_matrix, fea_masker, fea_shared, fea_visual, fea_audio, fea_text):
        # Check if the dimensions are correct.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if (key_matrix.dim() != 2) | (key_matrix.shape[0] != key_matrix.shape[1]):
            raise ValueError('<key_matrix> must have 2 dimensions, and the two dimensions should be the same.')

        # Constructing negative samples
        M = self.negative_sample_num
        N = query.shape[0]
        if self.negative_mode == 'paired':
            # If 'paired', then each query item is paired with a certain number of negative keys
            # The output negative_keys is a (N, M, D) Tensor.
            negative_sampled_list1 = []
            for i in range(N):
                keys_slice = key_matrix[:, i]
                negative_total = torch.sum(keys_slice == 0)
                assert negative_total >= M
                # Make sure the negative sample number is smaller than negative total number

                zeros_indices = torch.nonzero(keys_slice == 0)[:, 0]
                indices = random.sample(zeros_indices.tolist(), k=M)

                sampled_slice_list2 = []
                for idx in indices:
                    if idx == 0:
                        sampled_slice_list2.append(query[:1, :])
                    else:
                        sampled_slice_list2.append(query[idx:idx+1, :])
                negative_samples = torch.cat(sampled_slice_list2, dim=0).unsqueeze(dim=0)
                negative_sampled_list1.append(negative_samples)
            negative_samples = torch.cat(negative_sampled_list1, dim=0)
        elif self.negative_mode == 'unpaired':
            # if 'unpaired', then the set of negative keys is defined as any unpaired keys
            # The output negative_keys is a (M, D) Tensor.
            sum_vector = torch.sum(key_matrix, dim=1, keepdim=True)
            indices = torch.nonzero(sum_vector == 1)[:, 0]
            sampled_indices = random.sample(indices.tolist(), k=M)

            sampled_tensor_list = []
            for idx in sampled_indices:
                sampled_tensor_list.append(query[idx:idx+1, :])
            negative_samples = torch.cat(sampled_tensor_list, dim=0)
        else:
            raise ValueError('Unsupported negative sampling modes')

        # Constructing positive samples
        sum_vector = torch.sum(key_matrix, dim=0, keepdim=True).squeeze()
        no_matching_list = torch.nonzero(sum_vector == 1)[:, 0].tolist()
        matching_list = torch.nonzero(sum_vector != 1)[:, 0].tolist()

        positive_samples_dict = {}
        for i in range(N):
            if i in no_matching_list:
                # If there is no same event in a batch, the module will generate a positive sample by data augment
                # masker[1] == Visual, masker[2] == Audio, masker[3] == Text
                masker = fea_masker[i, :]
                masker = [masker[1].int(), masker[2].int(), masker[3].int()]
                masker_indices = [i for i, value in enumerate(masker) if value == 0]
                positive_sample_list = []
                if 0 in masker_indices:
                    positive_sample_list.append(fea_visual[i, :].unsqueeze(dim=0))
                if 1 in masker_indices:
                    positive_sample_list.append(fea_audio[i, :].unsqueeze(dim=0))
                if 2 in masker_indices:
                    positive_sample_list.append(fea_text[i, :].unsqueeze(dim=0))
                positive_samples = torch.cat(positive_sample_list, dim=0)
                # positive_sample, _ = self.positive_sample_encoder(query=query, key=key, value=key)
                positive_samples_dict.update({i: positive_samples})
            elif i in matching_list:
                # If there exists same event in a batch, the module will query the posts belonging to the same event
                key_slice = key_matrix[:, i]
                assert torch.sum(key_slice) > 1
                # Make sure all posts are not unique in the batch
                key_indices = [i for i, value in enumerate(key_slice) if value == 1]
                fea_shared_selected = fea_shared[key_indices, :]
                positive_samples_dict.update({i: fea_shared_selected})
            else:
                raise ValueError('The inner process reports error, please check the code')

        return positive_samples_dict, negative_samples


class MIL(nn.Module):
    def __init__(self, config):
        super(MIL, self).__init__()
        self.config = config
        self.dim = config['fea_dim']
        self.dropout = config['dropout']
        self.batch_size = config['batch_size']
        self.num_heads = config['num_heads']
        self.lstm_num_layer = config['lstm_num_layer']

        # Initialize the co-attention structure
        self.coherence_t2m = nn.Sequential(nn.Linear(self.dim * 2, self.dim),
                                                  nn.ReLU(),
                                                  nn.Linear(self.dim, self.dim),
                                                  nn.Dropout(self.dropout))
        self.coherence_v2m = nn.Sequential(nn.Linear(self.dim * 2, self.dim),
                                                  nn.ReLU(),
                                                  nn.Linear(self.dim, self.dim),
                                                  nn.Dropout(self.dropout))
        self.coherence_a2m = nn.Sequential(nn.Linear(self.dim * 2, self.dim),
                                                  nn.ReLU(),
                                                  nn.Linear(self.dim, self.dim),
                                                  nn.Dropout(self.dropout))

        self.conflict_detector = nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.num_heads,
                                                            dropout=self.dropout, batch_first=True)
        self.shared_encoder = nn.LSTM(input_size=self.dim, hidden_size=self.dim, num_layers=self.lstm_num_layer,
                                      batch_first=True, bidirectional=True)

    def forward(self, fea_shared, fea_text, fea_visual, fea_audio, fea_masker):
        if (fea_text.dim() > 2) | (fea_visual.dim() > 2) | (fea_audio.dim() > 2):
            fea_text = fea_text.squeeze()
            fea_visual = fea_visual.squeeze()
            fea_audio = fea_audio.squeeze()

        # The coherence components of Text-Multimodal, Visual-Multimodal, and Audio-Multimodal
        # Linear -> ReLU -> Linear
        cohere_t2m = self.coherence_t2m(torch.cat((fea_shared, fea_text), dim=-1)).unsqueeze(dim=1)
        cohere_v2m = self.coherence_v2m(torch.cat((fea_shared, fea_visual), dim=-1)).unsqueeze(dim=1)
        cohere_a2m = self.coherence_a2m(torch.cat((fea_shared, fea_audio), dim=-1)).unsqueeze(dim=1)

        # Introducing the masker
        text_masker = fea_masker[:, -1].unsqueeze(dim=1).bool()
        visual_masker = fea_masker[:, -3].unsqueeze(dim=1).bool()
        audio_masker = fea_masker[:, -2].unsqueeze(dim=1).bool()
        masker = torch.cat((text_masker, visual_masker, audio_masker), dim=1)

        cohere = torch.cat((cohere_t2m, cohere_v2m, cohere_a2m), dim=1)
        conflict = self.conflict_detector(src=cohere, src_key_padding_mask=masker)
        conflict = torch.mean(conflict, -2)

        fea_shared, _ = self.shared_encoder(fea_shared)
        a, b = fea_shared.split(self.dim, dim=-1)
        h_shared = (a + b) / 2

        return h_shared


class Classifier(nn.Module):
    def __init__(self, config):
        super(Classifier, self).__init__()
        self.config = config
        self.dim = config['fea_dim']

        self.fc = nn.Linear(self.dim, 2)

    def forward(self, fea):
        output = self.fc(fea)
        return output


class DIF(AbstractModel):
    def __init__(self, config, debunk_data):
        super(DIF, self).__init__()
        # Initialize hyperparameters
        self.config = config
        self.dim = config['fea_dim']
        self.cl_loss_wgt = config['cl_loss_wgt']

        # Initialize network
        self.mcl = MCL(self.config)
        self.mil = MIL(self.config)
        self.classifier = Classifier(self.config)

        # Loss
        # self.criterion = FocalLoss(gamma=2, weight=None)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize
        self.apply(xavier_normal_initialization)

    def predict(self, batch):
        predict, _ = self.forward(batch)
        return predict

    def forward(self, batch):
        fea_shared, fea_text, fea_visual, fea_audio, fea_masker, cl_loss = self.mcl(batch)
        fea = self.mil(fea_shared, fea_text, fea_visual, fea_audio, fea_masker)

        output = self.classifier(fea)
        return output, cl_loss

    def calculate_loss(self, batch):
        label = batch['label']
        predict, cl_loss = self.forward(batch)
        loss = self.criterion(predict, label)

        return loss + self.cl_loss_wgt * cl_loss


