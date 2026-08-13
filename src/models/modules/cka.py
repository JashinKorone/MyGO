"""Caption-guided Keyframe Attention (CKA).

Implements Section 3.2 of the MyGO paper.

Given the frame-level raw features of the three modalities
(:math:`\\mathcal{V}_i, \\mathcal{T}_i, \\mathcal{A}_i`) and the missing prompt
:math:`\\mathcal{P}_i`, CKA

1. derives a *caption weight* :math:`\\mathcal{U}_i` from the embedded caption
   (textual) frames and re-weights the visual / acoustic frames with it
   (Eq. 3).  When the text modality is missing the weight degenerates into a
   uniform distribution determined by the learnable bias :math:`b_1`, which
   keeps the module robust;
2. runs a two-stream co-attention in which the missing prompt is appended as a
   *feature suffix* of the query stream (Eq. 4), so that the attention focuses
   on the available modalities;
3. returns the refined modality representations
   :math:`\\mathbf{F}_v, \\mathbf{F}_t, \\mathbf{F}_a`, where
   :math:`\\mathbf{F}_t` is the average of ``CKA(T, V)`` and ``CKA(T, A)``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ensure_sequence(x):
    """Return ``x`` with an explicit frame axis, i.e. ``(B, K, D)``."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 4:  # (B, 1, K, D) -> (B, K, D)
        return x.squeeze(1)
    return x


class PromptEncoder(nn.Module):
    """Encode the 4-digit missing prompt into ``num_tokens`` suffix tokens.

    The prompt template follows Sec. 3.1: the first digit marks whether *any*
    modality is missing, the remaining three digits mark the missing state of
    video / text / audio respectively (``1`` denotes missing).  Both a
    continuous projection of the digits and a look-up embedding of the
    (at most 16) modality combinations are used, so that every combination owns
    an explicit, learnable identity.
    """

    def __init__(self, dim, num_tokens=1, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.digit_proj = nn.Linear(4, dim * num_tokens)
        self.combination_emb = nn.Embedding(16, dim * num_tokens)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def to_code(prompt):
        """Map a ``(B, 4)`` binary prompt to its integer combination id."""
        weight = torch.tensor([8, 4, 2, 1], device=prompt.device, dtype=torch.long)
        return (prompt.long() * weight).sum(dim=-1)

    def forward(self, prompt):
        prompt = prompt.float()
        code = self.to_code(prompt)
        token = self.digit_proj(prompt) + self.combination_emb(code)
        token = token.view(prompt.size(0), self.num_tokens, self.dim)
        return self.dropout(self.norm(token))


class PromptedCoAttentionBlock(nn.Module):
    """One direction of the two-stream co-attention of Eq. 4.

    ``query`` is concatenated with the prompt suffix before attending to
    ``context``; the prompt positions are dropped from the output so that the
    frame axis stays unchanged.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1, ffn_ratio=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, query, context, prompt_token=None):
        residual = query
        if prompt_token is not None:
            query = torch.cat((query, prompt_token), dim=1)
        attended, _ = self.attn(query, context, context, need_weights=False)
        attended = attended[:, : residual.size(1), :]
        hidden = self.norm_attn(residual + self.dropout(attended))
        return self.norm_ffn(hidden + self.dropout(self.ffn(hidden)))


class CaptionGuidedKeyframeAttention(nn.Module):
    """Caption-guided Keyframe Attention (Sec. 3.2)."""

    def __init__(self, config):
        super().__init__()
        self.dim = config['fea_dim']
        self.dropout = config['dropout']
        self.num_heads = config['num_heads']
        self.prompt_tokens = config['prompt_tokens'] or 1
        self.use_prompt = config['use_prompt'] if config['use_prompt'] is not None else True
        self.use_caption_weight = (
            config['use_caption_weight'] if config['use_caption_weight'] is not None else True
        )

        self.proj_video = self._build_projection(config['image_dim'])
        self.proj_text = self._build_projection(config['text_dim'])
        self.proj_audio = self._build_projection(config['audio_dim'])

        # Eq. 3: caption weight, W_1 maps a caption frame to a scalar score and
        # the learnable bias b_1 guarantees a uniform fallback distribution.
        self.caption_score = nn.Linear(self.dim, 1, bias=True)

        self.prompt_encoder = PromptEncoder(self.dim, self.prompt_tokens, self.dropout)
        self.cka_video = PromptedCoAttentionBlock(self.dim, self.num_heads, self.dropout)
        self.cka_audio = PromptedCoAttentionBlock(self.dim, self.num_heads, self.dropout)
        self.cka_text_video = PromptedCoAttentionBlock(self.dim, self.num_heads, self.dropout)
        self.cka_text_audio = PromptedCoAttentionBlock(self.dim, self.num_heads, self.dropout)

    def _build_projection(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, self.dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim, self.dim),
            nn.Dropout(p=self.dropout),
        )

    def caption_weight(self, caption):
        """Eq. 3: ``U = Softmax(tanh(W_1 t^j + b_1))`` over the frame axis."""
        score = torch.tanh(self.caption_score(caption))
        return torch.softmax(score, dim=1)

    @staticmethod
    def _align_weight(weight, length):
        """Resample the caption weight so that it matches ``length`` frames."""
        if weight.size(1) == length:
            return weight
        aligned = F.interpolate(
            weight.transpose(1, 2), size=length, mode='linear', align_corners=False
        ).transpose(1, 2)
        return aligned / aligned.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def forward(self, video, text, audio, prompt, caption=None):
        """Refine the three modalities.

        Args:
            video: ``(B, K_v, image_dim)`` frame-level visual features.
            text: ``(B, K_t, text_dim)`` frame-level textual features.
            audio: ``(B, K_a, audio_dim)`` frame-level acoustic features.
            prompt: ``(B, 4)`` missing prompt.
            caption: optional ``(B, K_c, text_dim)`` embedded-caption features.
                Falls back to ``text`` when not provided.

        Returns:
            tuple of refined features ``(F_v, F_t, F_a)``, each ``(B, dim)``.
        """
        video = self.proj_video(ensure_sequence(video))
        text = self.proj_text(ensure_sequence(text))
        audio = self.proj_audio(ensure_sequence(audio))
        caption = text if caption is None else self.proj_text(ensure_sequence(caption))

        if self.use_caption_weight:
            weight = self.caption_weight(caption)
            video = video * self._align_weight(weight, video.size(1)) * video.size(1)
            audio = audio * self._align_weight(weight, audio.size(1)) * audio.size(1)

        prompt_token = self.prompt_encoder(prompt) if self.use_prompt else None

        fea_video = self.cka_video(video, text, prompt_token)
        fea_audio = self.cka_audio(audio, text, prompt_token)
        fea_text_v = self.cka_text_video(text, video, prompt_token)
        fea_text_a = self.cka_text_audio(text, audio, prompt_token)

        fea_video = fea_video.mean(dim=1)
        fea_audio = fea_audio.mean(dim=1)
        fea_text = 0.5 * (fea_text_v.mean(dim=1) + fea_text_a.mean(dim=1))
        return fea_video, fea_text, fea_audio
