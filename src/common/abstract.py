import os

import numpy as np
import torch
import torch.nn as nn


class AbstractModel(nn.Module):
    r"""Base class for all models.

    Subclasses implement :meth:`calculate_loss` and :meth:`predict`.  The two
    epoch hooks are used by MyGO to drive the PMA prompt global memory
    (Sec. 3.4.1): ``post_epoch_processing`` closes the memory zone of the
    finished epoch and returns a short log line.
    """

    def pre_epoch_processing(self):
        return None

    def post_epoch_processing(self):
        return None

    def calculate_loss(self, data):
        r"""Calculate the training loss for a batch of data."""
        raise NotImplementedError

    def predict(self, data):
        r"""Return the class logits of a batch of data."""
        raise NotImplementedError

    def save_best(self, save_dir='./checkpoints'):
        r"""Persist the current parameters as the best checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, '{}.pth'.format(self.__class__.__name__))
        torch.save(self.state_dict(), path)
        return path

    def __str__(self):
        """Model prints with number of trainable parameters."""
        params = sum([np.prod(p.size()) for p in self.parameters()])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
