"""RCAN implementation adapted for Gaofen-2 super-resolution.

This package bundles the model definition, datasets, and training utilities.
"""

from .model import RCAN
from .datasets import ImagePairDataset, SingleImageDataset
from .lit_module import RCANLightningModule

__all__ = [
    "RCAN",
    "ImagePairDataset",
    "SingleImageDataset",
    "RCANLightningModule",
]
