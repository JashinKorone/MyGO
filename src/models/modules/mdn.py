"""Modality Disentangling Network (MDN).

Implements Section 3.3 of the MyGO paper:

* :class:`ContextAwareSharedEncoder` -- Sec. 3.3.1, Eq. 5-6.  A shared
  projection :math:`W^s` plus self-attention yields the video-level shared
  representation :math:`\\mathbf{H}^s`, which is further shaped by an
  event-level supervised contrastive loss :math:`\\mathcal{L}_{ctrs}`.
* :class:`ModalitySpecificDecoupling` -- Sec. 3.3.2, Eq. 7-8.  Per-modality
  projections :math:`W^u_*` produce modality-specific features that are kept
  orthogonal to the shared subspace via :math:`\\mathcal{L}_{o}`.
* :class:`ModalitySpecificEncoder` -- Sec. 3.3.3, Eq. 9-10.  A single gate
  filters noisy specific information, and pairwise transformer encoders measure
  cross-modality misalignment.
* :class:`InterModalityDependencyLearning` -- Sec. 3.3.4, Eq. 11.  Concatenates
  the pairwise inconsistencies with the shared representation into
  :math:`\\mathbf{Z}`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Event-level contrastive consistency learning (Eq. 6).

    Short videos annotated with the same event are treated as positives.  The
    implementation is the supervised contrastive loss of Khosla et al. (2020)
    restricted to the ``H^s`` anchors of a batch.  Samples whose event is
    unique inside the batch simply do not contribute.
    """

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=-1)
        batch_size = features.size(0)
        if batch_size < 2:
            return features.new_zeros(())

        labels = labels.contiguous().view(-1, 1)
        positive_mask = torch.eq(labels, labels.t()).float()
        self_mask = torch.eye(batch_size, device=features.device)
        positive_mask = positive_mask - positive_mask * self_mask

        logits = features @ features.t() / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * (1.0 - self_mask)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        positive_num = positive_mask.sum(dim=1)
        valid = positive_num > 0
        if not torch.any(valid):
            return features.new_zeros(())
        mean_log_prob = (positive_mask * log_prob).sum(dim=1)[valid] / positive_num[valid]
        return -mean_log_prob.mean()


class ContextAwareSharedEncoder(nn.Module):
    """Context-aware shared feature encoder (Eq. 5-6)."""

    def __init__(self, config):
        super().__init__()
        self.dim = config['fea_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']

        # W^s: a single projection shared by all modalities.
        self.shared_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim * 2,
            dropout=self.dropout,
            batch_first=True,
        )
        self.contrastive = SupervisedContrastiveLoss(config['cl_temp'] or 0.2)

    @property
    def weight(self):
        """Expose :math:`W^s` for the orthogonal constraint of Eq. 8."""
        return self.shared_projection.weight

    def forward(self, fea_video, fea_text, fea_audio, event=None, availability=None):
        shared = torch.stack(
            (
                self.shared_projection(fea_video),
                self.shared_projection(fea_audio),
                self.shared_projection(fea_text),
            ),
            dim=1,
        )  # (B, 3, dim), the (v, a, t) order follows Eq. 5.

        key_padding_mask = None
        if availability is not None:
            # ``availability`` is 1 for an available modality; the transformer
            # expects ``True`` for positions that must be ignored.
            order = availability[:, [0, 2, 1]]  # (v, t, a) -> (v, a, t)
            key_padding_mask = order < 0.5
            # Never mask every position, otherwise attention returns NaN.
            all_masked = key_padding_mask.all(dim=1)
            if torch.any(all_masked):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False

        shared = self.self_attn(shared, src_key_padding_mask=key_padding_mask)
        h_shared = shared.mean(dim=1)

        cl_loss = h_shared.new_zeros(())
        if event is not None:
            cl_loss = self.contrastive(h_shared, event)
        return h_shared, cl_loss


class ModalitySpecificDecoupling(nn.Module):
    """Modality-specific feature decoupling (Eq. 7-8)."""

    def __init__(self, config):
        super().__init__()
        self.dim = config['fea_dim']
        self.proj_video = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_text = nn.Linear(self.dim, self.dim, bias=False)
        self.proj_audio = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, fea_video, fea_text, fea_audio):
        return (
            self.proj_video(fea_video),
            self.proj_text(fea_text),
            self.proj_audio(fea_audio),
        )

    def orthogonal_loss(self, shared_weight):
        """Eq. 8: ``||W_s^T W_v^u||_F^2 + ||W_s^T W_t^u||_F^2 + ||W_s^T W_a^u||_F^2``.

        The squared Frobenius norm grows with ``dim^2``, which would dwarf the
        cross-entropy term for the ``dim = 128`` setting of Sec. 4.3.  We
        therefore average over the matrix entries, which only rescales
        :math:`\\beta` and keeps the constraint (and its gradient direction)
        identical.
        """
        loss = shared_weight.new_zeros(())
        for projection in (self.proj_video, self.proj_text, self.proj_audio):
            product = shared_weight.t() @ projection.weight
            loss = loss + product.pow(2).mean()
        return loss


class ModalitySpecificEncoder(nn.Module):
    """Single-gated encoder plus pairwise misalignment encoders (Eq. 9-10)."""

    def __init__(self, config):
        super().__init__()
        self.dim = config['fea_dim']
        self.num_heads = config['num_heads']
        self.dropout = config['dropout']

        # Eq. 9: one shared single-gate keeps the inherent dependencies intact.
        self.gate = nn.Linear(self.dim, self.dim)
        self.encoder_va = self._build_encoder()
        self.encoder_vt = self._build_encoder()
        self.encoder_ta = self._build_encoder()

    def _build_encoder(self):
        return nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.num_heads,
            dim_feedforward=self.dim * 2,
            dropout=self.dropout,
            batch_first=True,
        )

    def gating(self, fea):
        return torch.sigmoid(self.gate(fea)) * fea

    @staticmethod
    def _pair(encoder, left, right):
        pair = torch.stack((left, right), dim=1)
        return encoder(pair).mean(dim=1)

    def forward(self, fea_video, fea_text, fea_audio):
        gated_video = self.gating(fea_video)
        gated_text = self.gating(fea_text)
        gated_audio = self.gating(fea_audio)

        inconsistency_va = self._pair(self.encoder_va, gated_video, gated_audio)
        inconsistency_vt = self._pair(self.encoder_vt, gated_video, gated_text)
        inconsistency_ta = self._pair(self.encoder_ta, gated_text, gated_audio)
        return (
            (gated_video, gated_text, gated_audio),
            (inconsistency_va, inconsistency_ta, inconsistency_vt),
        )


class InterModalityDependencyLearning(nn.Module):
    """Fuse shared and specific evidence into ``Z`` (Eq. 11)."""

    def __init__(self, config):
        super().__init__()
        self.dim = config['fea_dim']
        self.dropout = config['dropout']
        self.fusion = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * 2, self.dim),
        )

    def forward(self, inconsistencies, h_shared):
        inconsistency_va, inconsistency_ta, inconsistency_vt = inconsistencies
        fused = torch.cat((inconsistency_va, inconsistency_ta, inconsistency_vt, h_shared), dim=-1)
        return self.fusion(fused)


class ModalityDisentanglingNetwork(nn.Module):
    """The full MDN of Sec. 3.3."""

    def __init__(self, config):
        super().__init__()
        self.shared_encoder = ContextAwareSharedEncoder(config)
        self.decoupling = ModalitySpecificDecoupling(config)
        self.specific_encoder = ModalitySpecificEncoder(config)
        self.dependency = InterModalityDependencyLearning(config)
        self.disentangle = config['disentangle'] if config['disentangle'] is not None else True

    def forward(self, fea_video, fea_text, fea_audio, event=None, availability=None):
        h_shared, cl_loss = self.shared_encoder(
            fea_video, fea_text, fea_audio, event=event, availability=availability
        )

        if not self.disentangle:
            # -w/o MDN ablation: plain cross-modal transformer fusion.
            fused = self.dependency(
                (h_shared, h_shared, h_shared), h_shared
            )
            zero = h_shared.new_zeros(())
            return fused, h_shared, cl_loss, zero

        specific_video, specific_text, specific_audio = self.decoupling(
            fea_video, fea_text, fea_audio
        )
        _, inconsistencies = self.specific_encoder(
            specific_video, specific_text, specific_audio
        )
        fused = self.dependency(inconsistencies, h_shared)
        orth_loss = self.decoupling.orthogonal_loss(self.shared_encoder.weight)
        return fused, h_shared, cl_loss, orth_loss
