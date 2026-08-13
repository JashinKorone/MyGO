"""Smoke test with synthetic FakeSV+ style batches.

Runs on CPU without the real dataset and checks that

* CKA / MDN / PMA produce finite tensors and gradients for every one of the
  seven modality combinations of Table 2;
* the PMA prompt global memory stays deactivated during the warm-up stage and
  starts re-weighting the loss afterwards;
* the ablation switches (-w/o CKA / MDN / PMA / PMA loss) are runnable.

Usage::

    cd src && python -m tests.test_mygo
"""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mygo import MyGO  # noqa: E402
from models.modules.pma import PromptGlobalMemory, prompt_to_code  # noqa: E402
from models.simplefusion import SimpleFusion  # noqa: E402
from models.svfend import SVFEND  # noqa: E402

BASE_CONFIG = {
    'fea_dim': 32,
    'dropout': 0.1,
    'num_heads': 4,
    'prompt_tokens': 1,
    'audio_dim': 64,
    'image_dim': 64,
    'text_dim': 64,
    'cl_temp': 0.2,
    'ctrs_loss_wgt': 0.3,
    'orth_loss_wgt': 0.2,
    'use_pma': True,
    'use_pma_loss': True,
    'use_prompt': True,
    'use_caption_weight': True,
    'disentangle': True,
    'prompt_top_k': 4,
    'warmup_epochs': 1,
    'pma_active_window': 2,
    'pma_decay': 0.9,
    'pma_weight': 1.0,
    'pma_max_scale': 3.0,
}


class DictConfig(dict):
    """Mimic ``utils.configurator.Config``, returning ``None`` for absent keys."""

    def __getitem__(self, key):
        return self.get(key, None)


def make_config(**overrides):
    config = DictConfig(BASE_CONFIG)
    config.update(overrides)
    return config


def make_batch(config, prompts, num_frames=6):
    batch_size = len(prompts)
    prompt = torch.tensor(prompts, dtype=torch.int)
    keep = 1.0 - prompt[:, 1:4].float()
    frames = torch.randn(batch_size, num_frames, config['image_dim']) * keep[:, 0].view(-1, 1, 1)
    text = torch.randn(batch_size, num_frames, config['text_dim']) * keep[:, 1].view(-1, 1, 1)
    audio = torch.randn(batch_size, num_frames, config['audio_dim']) * keep[:, 2].view(-1, 1, 1)
    return {
        'frames': frames,
        'text': text,
        'audioframes': audio,
        'prompt': prompt,
        'label': torch.randint(0, 2, (batch_size,)),
        'event': torch.randint(0, 3, (batch_size,)),
    }


def all_prompts():
    """The 2^M - 1 = 7 modality combinations, encoded as missing prompts."""
    prompts = []
    for missing in itertools.product((0, 1), repeat=3):
        if sum(missing) == 3:
            continue  # at least one available modality
        prompts.append([int(any(missing))] + list(missing))
    return prompts


def test_forward_all_combinations():
    torch.manual_seed(0)
    config = make_config()
    model = MyGO(config)
    batch = make_batch(config, all_prompts())
    outputs = model(batch)
    assert outputs['logits'].shape == (7, 2)
    assert torch.isfinite(outputs['logits']).all()
    assert torch.isfinite(outputs['ctrs_loss']) and outputs['ctrs_loss'] >= 0
    assert torch.isfinite(outputs['orth_loss']) and outputs['orth_loss'] >= 0
    print('[ok] forward pass over all 7 modality combinations')


def test_backward():
    torch.manual_seed(0)
    config = make_config()
    model = MyGO(config)
    batch = make_batch(config, all_prompts() * 2)
    loss = model.calculate_loss(batch)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, 'no gradient produced'
    assert all(torch.isfinite(g).all() for g in grads)
    print('[ok] backward pass, loss = {:.4f}'.format(loss.item()))


def test_pma_warmup_and_regularizer():
    torch.manual_seed(0)
    config = make_config(warmup_epochs=2)
    model = MyGO(config)
    batch = make_batch(config, all_prompts() * 2)

    assert not model.pma.memory.is_active(), 'regularizer must sleep during warm-up'
    for _ in range(2):
        model.calculate_loss(batch)
        model.post_epoch_processing()
    assert model.pma.memory.is_active(), 'regularizer must wake up after warm-up'

    prompt = batch['prompt'].float()
    scale = model.pma.regularization_term(prompt)
    assert scale.shape == (prompt.size(0),)
    assert torch.all(scale >= 1.0) and torch.all(scale <= config['pma_max_scale'])
    print('[ok] PMA warm-up + loss regularizer, scale range = [{:.3f}, {:.3f}]'.format(
        scale.min().item(), scale.max().item()))


def test_memory_zone_weights():
    memory = PromptGlobalMemory(top_k=2, active_window=2, decay=0.5, warmup_epochs=0)
    prompt = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 1]], dtype=torch.int)
    for _ in range(5):
        memory.merge(prompt, torch.tensor([2.0, 1.0]))
        memory.step_epoch()
    weights = memory.zone_weights()
    assert len(weights) == 5
    # newest epochs (active memory) are weighted higher than the oldest ones
    assert weights[-1] >= weights[-2] >= weights[0]
    counts = memory.weighted_counts()
    assert prompt_to_code(prompt)[0].item() in counts
    print('[ok] zoned memory weights = ' + ', '.join('{:.3f}'.format(w) for w in weights))


def test_ablations():
    prompts = all_prompts()
    variants = {
        '-w/o CKA': {'use_caption_weight': False, 'use_prompt': False},
        '-w/o MDN': {'disentangle': False},
        '-w/o PMA': {'use_pma': False},
        '-w/o PMA loss': {'use_pma_loss': False},
        '-w/o ctrs loss': {'ctrs_loss_wgt': 0.0},
        '-w/o orth loss': {'orth_loss_wgt': 0.0},
    }
    for name, override in variants.items():
        torch.manual_seed(0)
        config = make_config(**override)
        model = MyGO(config)
        loss = model.calculate_loss(make_batch(config, prompts))
        assert torch.isfinite(loss), name + ' produced a non-finite loss'
        print('[ok] {:<16} loss = {:.4f}'.format(name, loss.item()))


def test_baselines():
    prompts = all_prompts()
    for cls in (SVFEND, SimpleFusion):
        torch.manual_seed(0)
        config = make_config()
        model = cls(config)
        loss = model.calculate_loss(make_batch(config, prompts))
        assert torch.isfinite(loss)
        print('[ok] baseline {:<13} loss = {:.4f}'.format(cls.__name__, loss.item()))


if __name__ == '__main__':
    test_forward_all_combinations()
    test_backward()
    test_pma_warmup_and_regularizer()
    test_memory_zone_weights()
    test_ablations()
    test_baselines()
    print('\nAll smoke tests passed.')
