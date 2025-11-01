"""PyTorch Lightning data module for Gaofen-2 RCAN."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import ImagePairDataset, PatchConfig


@dataclass
class DataConfig:
    train_hr: str
    train_lr: Optional[str] = None
    val_hr: Optional[str] = None
    val_lr: Optional[str] = None
    scale: int = 4
    batch_size: int = 8
    num_workers: int = 4
    patch_size: int = 128
    augment: bool = True


class RCANDataModule(pl.LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        patch = PatchConfig(patch_size=self.config.patch_size, augment=self.config.augment)
        if stage in (None, "fit"):
            if not self.config.train_hr:
                raise ValueError("train_hr must be specified")
            self.train_dataset = ImagePairDataset(
                self.config.train_hr,
                self.config.train_lr,
                scale=self.config.scale,
                patch_config=patch,
                is_train=True,
            )
        if stage in (None, "fit", "validate") and self.config.val_hr:
            self.val_dataset = ImagePairDataset(
                self.config.val_hr,
                self.config.val_lr,
                scale=self.config.scale,
                patch_config=PatchConfig(patch_size=self.config.patch_size, augment=False),
                is_train=False,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("setup must be called before train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
