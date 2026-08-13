"""Model registry.

``MyGO`` is the model proposed in the paper; the remaining entries are the
baselines used in Sec. 4.4 and the PMA plug-in variants of Sec. 4.8.
"""

from models.mygo import MyGO
from models.simplefusion import SimpleFusion
from models.svfend import SVFEND

__all__ = ['MyGO', 'SVFEND', 'SimpleFusion']
