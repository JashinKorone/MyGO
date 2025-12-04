import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from einops import rearrange


def pad_evidence(seq_len, evidence, mode='text'):
    ori_len = evidence.shape[0]
    if mode == 'text':
        if ori_len >= seq_len:
            evidence = evidence[:seq_len, :]
        else:
            evidence = torch.cat((evidence, torch.zeros([seq_len - ori_len, evidence.shape[1]], dtype=torch.float)),
                                 dim=0)
    elif mode == 'visual':
        if ori_len >= seq_len:
            evidence = evidence[:seq_len, :, :]
        else:
            evidence = torch.cat(
                (evidence, torch.zeros([seq_len - ori_len, evidence.shape[1], evidence.shape[2]], dtype=torch.float)),
                dim=0)
    else:
        raise NotImplementedError
    return evidence


def pad_sequence(seq_len, lst, emb):
    result = []
    for video in lst:
        if isinstance(video, list):
            video = torch.stack(video)
        ori_len = video.shape[0]
        if ori_len == 0:
            video = torch.zeros([seq_len, emb], dtype=torch.long)
        elif ori_len >= seq_len:
            if emb == 200:
                video = torch.FloatTensor(video[:seq_len])
            else:
                video = torch.LongTensor(video[:seq_len])
        else:
            video = torch.cat([video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.long)], dim=0)
            if emb == 200:
                video = torch.FloatTensor(video)
            else:
                video = torch.LongTensor(video)
        result.append(video)
    return torch.stack(result)


def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones(seq_len)
        else:
            video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def pad_frame_sequence_dataset(seq_len, video):
    attention_masks = []
    result = []
    video = torch.FloatTensor(video)
    ori_len = video.shape[0]
    if ori_len >= seq_len:
        gap = ori_len // seq_len
        video = video[::gap][:seq_len]
        mask = np.ones(seq_len)
    else:
        video = torch.cat((video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)), dim=0)
        mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
    mask = torch.IntTensor(mask)
    return video, mask


class Attention(nn.Module):
    def __init__(self, dim, heads=2, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        # q: [b_size x n_heads x len_q x d_k]
        # k: [b_size x n_heads x len_k x d_k]
        # v: [b_size x n_heads x len_v x d_v] note: (len_k == len_v)

        # attn: [b_size x n_heads x len_q x len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))

        # outputs: [b_size x n_heads x len_q x d_v]
        context = torch.matmul(attn, v)

        return context, attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True, )
        std = z.std(dim=-1, keepdim=True, )
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
             for pos in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pad_row = np.zeros([1, d_word_vec])
        pos_enc = np.concatenate([pad_row, pos_enc]).astype(np.float32)

        # additional single row for PAD idx
        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        # fix positional encoding: exclude weight from grad computation
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)
        self.max_len = int(max_seq_len / 10)

    def forward(self, input_len):
        max_len = self.max_len  # torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        input_pos = tensor([list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.pos_enc(input_pos)


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = Linear(d_model, d_k * n_heads)
        self.w_k = Linear(d_model, d_k * n_heads)
        self.w_v = Linear(d_model, d_v * n_heads)

    def forward(self, q, k, v):
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_k x d_model]
        b_size = q.size(0)

        # q_s: [b_size x n_heads x len_q x d_k]
        # k_s: [b_size x n_heads x len_k x d_k]
        # v_s: [b_size x n_heads x len_k x d_v]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        return q_s, k_s, v_s


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)

    def forward(self, inputs):
        # inputs: [b_size x len_q x d_model]
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))

        # outputs: [b_size x len_q x d_model]
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)

        return self.layer_norm(residual + output)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.multihead_attn_v = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.multihead_attn_s = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.pos_emb_v = PosEncoding(visual_len * 10, d_model)
        self.pos_emb_s = PosEncoding(sen_len * 10, d_model)
        self.linear_v = nn.Linear(in_features=fea_v, out_features=d_model)
        self.linear_s = nn.Linear(in_features=fea_s, out_features=d_model)
        self.proj_v = Linear(n_heads * d_v, d_model)
        self.proj_s = Linear(n_heads * d_v, d_model)
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_v = LayerNormalization(d_model)
        self.layer_norm_s = LayerNormalization(d_model)
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.pos = pos

    def forward(self, v, s, v_len, s_len):
        b_size = v.size(0)
        # q: [b_size x len_q x d_model]
        # k: [b_size x len_k x d_model]
        # v: [b_size x len_v x d_model] note (len_k == len_v)
        v, s = self.linear_v(v), self.linear_s(s)
        if self.pos:
            pos_v, pos_s = self.pos_emb_v(v_len), self.pos_emb_s(s_len)
            residual_v, residual_s = v + pos_v, s + pos_s
        else:
            residual_v, residual_s = v, s
        # context: a tensor of shape [b_size x len_q x n_heads * d_v]
        q_v, k_v, v_v = self.multihead_attn_v(v, v, v)
        q_s, k_s, v_s = self.multihead_attn_s(s, s, s)
        context_v, attn_v = self.attention(q_v, k_s, v_s)
        context_s, attn_s = self.attention(q_s, k_v, v_v)
        context_v = context_v.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        context_s = context_s.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        # project back to the residual size, outputs: [b_size x len_q x d_model]
        output_v = self.dropout(self.proj_v(context_v))
        output_s = self.dropout(self.proj_s(context_s))
        return self.layer_norm_v(residual_v + output_v), self.layer_norm_s(residual_s + output_s)


class co_attention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(co_attention, self).__init__()
        self.multi_head = MultiHeadAttention(d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
                                             visual_len=visual_len, sen_len=sen_len, fea_v=fea_v, fea_s=fea_s, pos=pos)
        self.PoswiseFeedForwardNet_v = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)
        self.PoswiseFeedForwardNet_s = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)

    def forward(self, v, s, v_len, s_len):
        v, s = self.multi_head(v, s, v_len, s_len)
        v = self.PoswiseFeedForwardNet_v(v)
        s = self.PoswiseFeedForwardNet_s(s)
        return v, s


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss



