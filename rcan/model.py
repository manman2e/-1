"""PyTorch implementation of RCAN model.

The architecture is adapted from lornatang's public RCAN implementation but
restructured into modular building blocks suitable for Lightning training.
This variant adds configuration hooks so multi-spectral imagery (e.g. Gaofen-2
with RGB+NIR bands) can be trained without hand editing the model definition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class RCANConfig:
    scale: int = 4
    n_colors: int = 4
    n_feats: int = 64
    n_resgroups: int = 10
    n_resblocks: int = 20
    reduction: int = 16
    rgb_range: float = 1.0
    channel_mean: tuple[float, ...] = ()
    channel_std: tuple[float, ...] = ()


class MeanShift(nn.Conv2d):
    """Normalize images by subtracting/adding a per-channel mean shift."""

    def __init__(
        self,
        n_channels: int,
        rgb_range: float = 1.0,
        channel_mean: tuple[float, ...] | list[float] | None = None,
        channel_std: tuple[float, ...] | list[float] | None = None,
        sign: int = -1,
    ) -> None:
        super().__init__(n_channels, n_channels, kernel_size=1)
        if channel_mean is None:
            channel_mean = [0.0] * n_channels
        if channel_std is None:
            channel_std = [1.0] * n_channels
        if len(channel_mean) != n_channels or len(channel_std) != n_channels:
            raise ValueError("channel_mean and channel_std must match n_channels")
        std = torch.tensor(channel_std, dtype=torch.float32)
        weight = torch.eye(n_channels).view(n_channels, n_channels, 1, 1)
        self.weight.data = weight / std.view(n_channels, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.tensor(channel_mean, dtype=torch.float32) / std
        for p in self.parameters():
            p.requires_grad = False


class CALayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, n_feat: int, kernel_size: int = 3, reduction: int = 16, bias: bool = True, bn: bool = False, act: Optional[nn.Module] = nn.ReLU(True)):
        super().__init__()
        modules = []
        for i in range(2):
            modules.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias))
            if bn:
                modules.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules.append(act if act else nn.Identity())
        self.body = nn.Sequential(*modules)
        self.ca = CALayer(n_feat, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = self.ca(res)
        return res + x


class ResidualGroup(nn.Module):
    def __init__(self, n_feat: int, n_resblocks: int, kernel_size: int = 3, reduction: int = 16):
        super().__init__()
        modules = [RCAB(n_feat, kernel_size, reduction) for _ in range(n_resblocks)]
        modules.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2)))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        return res + x


def default_upsampler(scale: int, n_feat: int) -> nn.Module:
    modules = []
    if scale in (2, 4, 8):
        for _ in range(int(scale).bit_length() - 1):
            modules += [nn.Conv2d(n_feat, 4 * n_feat, kernel_size=3, padding=1), nn.PixelShuffle(2), nn.ReLU(True)]
    elif scale == 3:
        modules += [nn.Conv2d(n_feat, 9 * n_feat, kernel_size=3, padding=1), nn.PixelShuffle(3), nn.ReLU(True)]
    else:
        raise ValueError(f"Unsupported scale factor: {scale}")
    return nn.Sequential(*modules)


class RCAN(nn.Module):
    def __init__(self, config: RCANConfig):
        super().__init__()
        self.config = config
        channel_mean = config.channel_mean or (0.0,) * config.n_colors
        channel_std = config.channel_std or (1.0,) * config.n_colors
        self.sub_mean = MeanShift(
            config.n_colors,
            rgb_range=config.rgb_range,
            channel_mean=channel_mean,
            channel_std=channel_std,
            sign=-1,
        )
        self.add_mean = MeanShift(
            config.n_colors,
            rgb_range=config.rgb_range,
            channel_mean=channel_mean,
            channel_std=channel_std,
            sign=1,
        )

        self.head = nn.Conv2d(config.n_colors, config.n_feats, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualGroup(config.n_feats, config.n_resblocks, reduction=config.reduction) for _ in range(config.n_resgroups)],
            nn.Conv2d(config.n_feats, config.n_feats, kernel_size=3, padding=1),
        )
        self.upsampler = default_upsampler(config.scale, config.n_feats)
        self.tail = nn.Conv2d(config.n_feats, config.n_colors, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.upsampler(res)
        x = self.tail(x)
        x = self.add_mean(x)
        return x
