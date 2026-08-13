"""Prompt-assisted Modality Aligning (PMA).

Implements Section 3.4 of the MyGO paper.  PMA is *plug-and-play*: it consumes
the batch missing prompts :math:`P` together with the per-sample classification
loss :math:`\\mathcal{L}_{cls}` and returns the balanced loss
:math:`\\mathcal{L}'_{cls}` of Eq. 13.

It has two parts:

``PromptGlobalMemory`` (Sec. 3.4.1)
    A dynamically weighted memory of *weak* modality combinations.  Per batch
    the Top-:math:`K` samples with the highest classification loss are treated
    as candidates and their prompts are merged into the memory.  The memory is
    partitioned into three zones (Fig. 5):

    * **Warm-up stage** -- the first ``warmup_epochs`` epochs still merge
      candidates, but the loss regularizer stays deactivated;
    * **Inactive memory** -- older epochs, whose counts receive geometrically
      *decaying* weights so that outdated statistics fade away;
    * **Active memory** -- a fixed-size sliding window over the newest epochs,
      whose counts receive linearly *increasing* weights and therefore react
      quickly to distribution shifts.

``PromptAssistedModalityAligning`` (Sec. 3.4.2)
    The loss regularizer :math:`[\\mathcal{R}] = u \\circ [\\mathcal{C}] / M`,
    which up-weights samples whose modality combination is frequently recorded
    as weak.
"""

from collections import Counter, deque

import torch
import torch.nn as nn

PROMPT_BITS = torch.tensor([8, 4, 2, 1])


def prompt_to_code(prompt):
    """Map a ``(B, 4)`` binary missing prompt to an integer id in ``[0, 16)``."""
    bits = PROMPT_BITS.to(device=prompt.device, dtype=torch.long)
    return (prompt.long() * bits).sum(dim=-1)


def code_to_prompt(code):
    """Inverse of :func:`prompt_to_code`, returns a 4-digit string."""
    return format(int(code), '04b')


class PromptGlobalMemory:
    """Zoned, dynamically weighted memory of weak modality combinations."""

    def __init__(self, top_k=10, active_window=5, decay=0.9, warmup_epochs=5,
                 max_epochs_kept=50):
        self.top_k = top_k
        self.active_window = max(1, active_window)
        self.decay = decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs_kept = max_epochs_kept

        self.current = Counter()
        self.epoch_counters = deque(maxlen=max_epochs_kept)
        self.epoch = 0

    # ------------------------------------------------------------------ merge
    def merge(self, prompt, loss_per_sample):
        """Merge the Top-K highest-loss prompts of a batch into the memory."""
        if prompt.numel() == 0:
            return
        k = min(self.top_k, loss_per_sample.size(0))
        if k <= 0:
            return
        _, indices = torch.topk(loss_per_sample.detach(), k=k, largest=True)
        codes = prompt_to_code(prompt[indices])
        self.current.update(codes.tolist())

    def step_epoch(self):
        """Freeze the counter of the finished epoch and open a new one."""
        self.epoch_counters.append(self.current)
        self.current = Counter()
        self.epoch += 1

    # ---------------------------------------------------------------- weights
    def zone_weights(self):
        """Return the per-epoch weight of every stored counter.

        The newest ``active_window`` counters form the *active* memory and get
        linearly increasing weights in ``(0, 1]``; the remaining counters form
        the *inactive* memory and get geometrically decaying weights.
        """
        total = len(self.epoch_counters)
        weights = []
        for offset, _ in enumerate(self.epoch_counters):
            age = total - offset - 1  # 0 == newest
            if age < self.active_window:
                weights.append((self.active_window - age) / self.active_window)
            else:
                weights.append(self.decay ** (age - self.active_window + 1))
        return weights

    def weighted_counts(self, include_current=True):
        """Aggregate the zones into a single ``{code: weighted count}`` table."""
        counts = Counter()
        for weight, counter in zip(self.zone_weights(), self.epoch_counters):
            for code, value in counter.items():
                counts[code] += weight * value
        if include_current:
            for code, value in self.current.items():
                counts[code] += value
        return counts

    def is_active(self):
        """The regularizer is deactivated during the warm-up stage."""
        return self.epoch >= self.warmup_epochs

    # ------------------------------------------------------------------- misc
    def summary(self, top=5):
        counts = self.weighted_counts()
        if not counts:
            return 'PMA memory: empty'
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
        body = ', '.join('{}:{:.1f}'.format(code_to_prompt(c), v) for c, v in ranked)
        return 'PMA memory (epoch {}, {} prompts) top weak combinations -> {}'.format(
            self.epoch, len(counts), body
        )

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'current': dict(self.current),
            'epoch_counters': [dict(c) for c in self.epoch_counters],
        }

    def load_state_dict(self, state):
        self.epoch = state['epoch']
        self.current = Counter(state['current'])
        self.epoch_counters = deque(
            (Counter(c) for c in state['epoch_counters']), maxlen=self.max_epochs_kept
        )


class PromptAssistedModalityAligning(nn.Module):
    """Prompt global memory + loss regularizer (Eq. 13).

    The module is stateless w.r.t. gradients -- it only rescales an existing
    per-sample loss -- which is what makes it attachable to any backbone
    (see Sec. 4.8 of the paper, where PMA is plugged into TKCM / TwtrD /
    SVFEND).
    """

    def __init__(self, config):
        super().__init__()
        self.enabled = config['use_pma'] if config['use_pma'] is not None else True
        self.use_regularizer = (
            config['use_pma_loss'] if config['use_pma_loss'] is not None else True
        )
        self.weight = config['pma_weight'] if config['pma_weight'] is not None else 1.0
        self.max_scale = config['pma_max_scale'] if config['pma_max_scale'] is not None else 3.0
        self.memory = PromptGlobalMemory(
            top_k=config['prompt_top_k'] or 10,
            active_window=config['pma_active_window'] or 5,
            decay=config['pma_decay'] or 0.9,
            warmup_epochs=config['warmup_epochs'] if config['warmup_epochs'] is not None else 5,
        )

    # --------------------------------------------------------------- lifecycle
    def step_epoch(self):
        if self.enabled:
            self.memory.step_epoch()

    def summary(self):
        return self.memory.summary() if self.enabled else None

    # ------------------------------------------------------------- regularizer
    def regularization_term(self, prompt):
        """Compute ``[R] = u * [C] / M`` for the prompts inside a batch."""
        counts = self.memory.weighted_counts()
        ones = torch.ones(prompt.size(0), device=prompt.device)
        if not counts:
            return ones
        normalizer = max(counts.values())
        if normalizer <= 0:
            return ones
        codes = prompt_to_code(prompt).tolist()
        values = [counts.get(code, 0.0) / normalizer for code in codes]
        scale = ones + self.weight * torch.tensor(
            values, device=prompt.device, dtype=torch.float
        )
        return scale.clamp(min=1.0, max=self.max_scale)

    def forward(self, prompt, loss_per_sample):
        """Return ``(balanced loss, regularization term)``.

        ``loss_per_sample`` must be *un-reduced* so that the memory can rank
        the samples and the regularizer can act element-wise.
        """
        if not self.enabled:
            return loss_per_sample.mean(), None

        self.memory.merge(prompt, loss_per_sample)
        if not (self.use_regularizer and self.memory.is_active()):
            return loss_per_sample.mean(), None

        scale = self.regularization_term(prompt)
        return (scale * loss_per_sample).mean(), scale
