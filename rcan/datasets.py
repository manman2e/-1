"""Datasets for Gaofen-2 super-resolution training and inference."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass
class PatchConfig:
    patch_size: int = 128
    augment: bool = True


class ImagePairDataset(Dataset):
    """Dataset for paired LR/HR images stored on disk."""

    def __init__(
        self,
        hr_dir: Path | str,
        lr_dir: Optional[Path | str] = None,
        scale: int = 4,
        transform: Optional[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None,
        patch_config: Optional[PatchConfig] = None,
        is_train: bool = True,
    ) -> None:
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.scale = scale
        self.transform = transform
        self.patch_config = patch_config or PatchConfig()
        self.is_train = is_train

        self.hr_files = self._collect_files(self.hr_dir)
        if not self.hr_files:
            raise FileNotFoundError(f"No images found in {self.hr_dir}")
        if self.lr_dir:
            self.lr_files = self._collect_files(self.lr_dir)
            if len(self.hr_files) != len(self.lr_files):
                raise ValueError("HR and LR directory must contain the same number of images")
        else:
            self.lr_files = [None] * len(self.hr_files)

    def __len__(self) -> int:
        return len(self.hr_files)

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor

    def _random_crop(self, hr: torch.Tensor, lr: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.is_train:
            return hr, lr
        ps = self.patch_config.patch_size
        _, h, w = hr.shape
        if h < ps or w < ps:
            return hr, lr
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        hr_patch = hr[:, top : top + ps, left : left + ps]
        if lr is not None:
            lr_ps = ps // self.scale
            lr_top = top // self.scale
            lr_left = left // self.scale
            lr_patch = lr[:, lr_top : lr_top + lr_ps, lr_left : lr_left + lr_ps]
        else:
            lr_patch = None
        return hr_patch, lr_patch

    def _augment(self, hr: torch.Tensor, lr: Optional[torch.Tensor]) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.is_train or not self.patch_config.augment:
            return hr, lr
        if random.random() < 0.5:
            hr = torch.flip(hr, dims=[2])
            if lr is not None:
                lr = torch.flip(lr, dims=[2])
        if random.random() < 0.5:
            hr = torch.flip(hr, dims=[1])
            if lr is not None:
                lr = torch.flip(lr, dims=[1])
        return hr, lr

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        hr_path = self.hr_files[idx]
        hr = self._load(hr_path)
        if self.lr_dir:
            lr_path = self.lr_files[idx]
            lr = self._load(lr_path)
        else:
            lr = F.interpolate(hr.unsqueeze(0), scale_factor=1 / self.scale, mode="bicubic", align_corners=False).squeeze(0)
        hr, lr = self._random_crop(hr, lr)
        hr, lr = self._augment(hr, lr)
        if self.transform:
            lr, hr = self.transform(lr, hr)
        return {"lr": lr, "hr": hr, "name": hr_path.name}

    @staticmethod
    def _collect_files(directory: Path) -> list[Path]:
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        files: Iterable[Path] = directory.iterdir()
        return sorted([p for p in files if p.is_file() and p.suffix.lower() in exts])


class SingleImageDataset(Dataset):
    """Dataset for inference on arbitrary images."""

    def __init__(self, image_paths: Sequence[Path | str]):
        self.image_paths = [Path(p) for p in image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return {"lr": tensor, "name": path.name, "path": str(path)}
