"""Attention primitives shared by the baselines.

Only the two-stream co-attention stack of TikTec / SVFEND is kept here; the
MyGO modules live in :mod:`models.modules`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        if bias:
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
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask, -1e9)
        attn = self.dropout(self.softmax(scores))
        return torch.matmul(attn, v), attn


class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True)
        std = z.std(dim=-1, keepdim=True)
        return self.gamma * ((z - mean) / (std + self.eps)) + self.beta


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_word_vec):
        super(PosEncoding, self).__init__()
        pos_enc = np.array(
            [[pos / np.power(10000, 2.0 * (j // 2) / d_word_vec) for j in range(d_word_vec)]
             for pos in range(max_seq_len)]
        )
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        pos_enc = np.concatenate([np.zeros([1, d_word_vec]), pos_enc]).astype(np.float32)

        self.pos_enc = nn.Embedding(max_seq_len + 1, d_word_vec)
        self.pos_enc.weight = nn.Parameter(torch.from_numpy(pos_enc), requires_grad=False)
        self.max_len = int(max_seq_len / 10)

    def forward(self, input_len):
        max_len = self.max_len
        input_pos = torch.tensor(
            [list(range(1, length + 1)) + [0] * (max_len - length) for length in input_len],
            device=self.pos_enc.weight.device, dtype=torch.long,
        )
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
        b_size = q.size(0)
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
        residual = inputs
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.dropout(self.conv2(output).transpose(1, 2))
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
        v, s = self.linear_v(v), self.linear_s(s)
        if self.pos:
            residual_v = v + self.pos_emb_v(v_len)
            residual_s = s + self.pos_emb_s(s_len)
        else:
            residual_v, residual_s = v, s
        q_v, k_v, v_v = self.multihead_attn_v(v, v, v)
        q_s, k_s, v_s = self.multihead_attn_s(s, s, s)
        context_v, _ = self.attention(q_v, k_s, v_s)
        context_s, _ = self.attention(q_s, k_v, v_v)
        context_v = context_v.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        context_s = context_s.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)
        output_v = self.dropout(self.proj_v(context_v))
        output_s = self.dropout(self.proj_s(context_s))
        return self.layer_norm_v(residual_v + output_v), self.layer_norm_s(residual_s + output_s)


class co_attention(nn.Module):
    """Two-stream co-attention used by the TikTec / SVFEND style baselines."""

    def __init__(self, d_k, d_v, n_heads, dropout, d_model, visual_len, sen_len, fea_v, fea_s, pos):
        super(co_attention, self).__init__()
        self.multi_head = MultiHeadAttention(
            d_k=d_k, d_v=d_v, n_heads=n_heads, dropout=dropout, d_model=d_model,
            visual_len=visual_len, sen_len=sen_len, fea_v=fea_v, fea_s=fea_s, pos=pos,
        )
        self.PoswiseFeedForwardNet_v = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)
        self.PoswiseFeedForwardNet_s = PoswiseFeedForwardNet(d_model=d_model, d_ff=128, dropout=dropout)

    def forward(self, v, s, v_len, s_len):
        v, s = self.multi_head(v, s, v_len, s_len)
        return self.PoswiseFeedForwardNet_v(v), self.PoswiseFeedForwardNet_s(s)
