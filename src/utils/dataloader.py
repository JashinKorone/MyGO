"""Batch collation.

Each batch is the tuple ``C_i = {V_i, T_i, A_i, P_i}`` of Eq. 2 plus the label
and the event id required by the event-level contrastive loss (Eq. 6).
"""

import numpy as np
import torch


def _init_fn(worker_id):
    np.random.seed(2024 + worker_id)


def _stack(values):
    return torch.stack([torch.as_tensor(v) for v in values])


def collate_fn(batch):
    collated = {
        'label': _stack([item['label'] for item in batch]),
        'event': _stack([item['event'] for item in batch]),
        'audioframes': _stack([item['audioframes'] for item in batch]),
        'frames': _stack([item['frames'] for item in batch]),
        'text': _stack([item['text'] for item in batch]),
        'prompt': _stack([item['prompt'] for item in batch]),
    }
    if 'caption' in batch[0]:
        collated['caption'] = _stack([item['caption'] for item in batch])
    return collated
