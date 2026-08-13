"""Evaluation metrics.

Accuracy and macro F1 are the two metrics reported in the paper (Eq. 16);
precision / recall / AUC are logged as auxiliary information.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    result = {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    # AUC is undefined when a split happens to contain a single class.
    result['auc'] = roc_auc_score(y_true, y_pred, average='macro') if len(set(y_true.tolist())) > 1 else 0.0
    return result


def combination_metrics(y_true, y_pred, prompts):
    """Per modality-combination accuracy / F1, i.e. Table 5 of the paper.

    Args:
        y_true / y_pred: iterables of int labels.
        prompts: list of 4-digit missing prompts (list/tuple of ints).

    Returns:
        ``{prompt string: {'acc': .., 'f1': .., 'num': ..}}``
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    keys = [''.join(str(int(d)) for d in prompt) for prompt in prompts]

    grouped = {}
    for key in sorted(set(keys)):
        index = [i for i, k in enumerate(keys) if k == key]
        sub_true, sub_pred = y_true[index], y_pred[index]
        grouped[key] = {
            'acc': accuracy_score(sub_true, sub_pred),
            'f1': f1_score(sub_true, sub_pred, average='macro', zero_division=0),
            'num': len(index),
        }
    return grouped


def prompt_to_modalities(prompt_str):
    """Human readable modality combination of a missing prompt.

    ``prompt_str`` is ``[any_missing, video, text, audio]`` with ``1`` marking a
    missing modality, so ``1001`` -> ``Text+Video``.
    """
    names = ('Video', 'Text', 'Audio')
    available = [name for name, digit in zip(names, prompt_str[1:]) if digit == '0']
    return '+'.join(available) if available else 'None'
