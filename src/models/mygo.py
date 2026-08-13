"""MyGO: Modality-incomplete Fake News Video Detection via Prompt-assisted
Modality Disentangling Model (TOMM 2025).

The model follows Figure 2 of the paper and stacks three modules:

1. **CKA** -- Caption-guided Keyframe Attention (Sec. 3.2) turns frame-level raw
   features into refined modality features by re-weighting frames with the
   embedded captions and by prompting the two-stream co-attention with the
   missing prompt.
2. **MDN** -- Modality Disentangling Network (Sec. 3.3) decomposes the refined
   features into a modality-shared part (constrained by an event-level
   supervised contrastive loss) and modality-specific parts (kept orthogonal to
   the shared subspace), then fuses pairwise inconsistencies with the shared
   representation into the inter-modality dependency ``Z``.
3. **PMA** -- Prompt-assisted Modality Aligning (Sec. 3.4) mines weak modality
   combinations with a zoned prompt global memory and rebalances the per-sample
   classification loss with a regularizer.

Optimisation objective (Eq. 14)::

    L = L'_cls + alpha * L_ctrs + beta * L_o
"""

import torch.nn as nn

from common.abstract import AbstractModel
from models.modules import (
    CaptionGuidedKeyframeAttention,
    ModalityDisentanglingNetwork,
    PromptAssistedModalityAligning,
)
from utils.utils import xavier_normal_initialization


class Classifier(nn.Module):
    """MLP classifier of Eq. 12."""

    def __init__(self, config):
        super().__init__()
        dim = config['fea_dim']
        dropout = config['dropout']
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, 2),
        )

    def forward(self, fea):
        return self.net(fea)


class MyGO(AbstractModel):
    """The full MyGO model."""

    def __init__(self, config, debunk_data=None):
        super().__init__()
        self.config = config
        self.dim = config['fea_dim']
        self.ctrs_loss_wgt = config['ctrs_loss_wgt'] if config['ctrs_loss_wgt'] is not None else 0.3
        self.orth_loss_wgt = config['orth_loss_wgt'] if config['orth_loss_wgt'] is not None else 0.2

        self.cka = CaptionGuidedKeyframeAttention(config)
        self.mdn = ModalityDisentanglingNetwork(config)
        self.pma = PromptAssistedModalityAligning(config)
        self.classifier = Classifier(config)

        # Eq. 12 -- kept un-reduced so that PMA can re-weight each sample.
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.apply(xavier_normal_initialization)

    # --------------------------------------------------------------- lifecycle
    def pre_epoch_processing(self):
        return None

    def post_epoch_processing(self):
        """Freeze the epoch counter of the prompt memory (Fig. 5)."""
        self.pma.step_epoch()
        return self.pma.summary()

    # ----------------------------------------------------------------- forward
    @staticmethod
    def _read_prompt(batch):
        """Fetch the 4-digit missing prompt of Eq. 2 from a batch."""
        prompt = batch.get('prompt', batch.get('masker'))
        if prompt is None:
            raise KeyError("batch must provide either 'prompt' or 'masker'")
        return prompt.float()

    @staticmethod
    def _availability(prompt):
        """``1`` for an available modality, in the (video, text, audio) order.

        The last three prompt digits mark *missing* modalities, hence the
        complement.
        """
        return 1.0 - prompt[:, 1:4]

    def forward(self, batch):
        prompt = self._read_prompt(batch)
        availability = self._availability(prompt)

        fea_video, fea_text, fea_audio = self.cka(
            video=batch['frames'],
            text=batch['text'],
            audio=batch['audioframes'],
            prompt=prompt,
            caption=batch.get('caption'),
        )

        fused, h_shared, ctrs_loss, orth_loss = self.mdn(
            fea_video,
            fea_text,
            fea_audio,
            event=batch.get('event'),
            availability=availability,
        )
        logits = self.classifier(fused)
        return {
            'logits': logits,
            'fused': fused,
            'shared': h_shared,
            'ctrs_loss': ctrs_loss,
            'orth_loss': orth_loss,
            'prompt': prompt,
        }

    def predict(self, batch):
        return self.forward(batch)['logits']

    # -------------------------------------------------------------------- loss
    def calculate_loss(self, batch):
        outputs = self.forward(batch)
        loss_per_sample = self.criterion(outputs['logits'], batch['label'])
        cls_loss, _ = self.pma(outputs['prompt'], loss_per_sample)
        return (
            cls_loss
            + self.ctrs_loss_wgt * outputs['ctrs_loss']
            + self.orth_loss_wgt * outputs['orth_loss']
        )
