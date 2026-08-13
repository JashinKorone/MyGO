"""Building blocks of MyGO.

The three modules follow Section 3 of the paper
"MyGO: Modality-incomplete Fake News Video Detection via Prompt-assisted
Modality Disentangling Model" (TOMM 2025):

* :mod:`cka` -- Caption-guided Keyframe Attention (Sec. 3.2, Eq. 3-4)
* :mod:`mdn` -- Modality Disentangling Network (Sec. 3.3, Eq. 5-12)
* :mod:`pma` -- Prompt-assisted Modality Aligning (Sec. 3.4, Eq. 13-14)
"""

from models.modules.cka import CaptionGuidedKeyframeAttention
from models.modules.mdn import (
    ContextAwareSharedEncoder,
    InterModalityDependencyLearning,
    ModalityDisentanglingNetwork,
    ModalitySpecificEncoder,
    SupervisedContrastiveLoss,
)
from models.modules.pma import PromptAssistedModalityAligning, PromptGlobalMemory

__all__ = [
    'CaptionGuidedKeyframeAttention',
    'ContextAwareSharedEncoder',
    'InterModalityDependencyLearning',
    'ModalityDisentanglingNetwork',
    'ModalitySpecificEncoder',
    'SupervisedContrastiveLoss',
    'PromptAssistedModalityAligning',
    'PromptGlobalMemory',
]
