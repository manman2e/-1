"""Lightning module wrapping RCAN for Gaofen-2 training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
import pytorch_lightning as pl

from .model import RCAN, RCANConfig


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


class RCANLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: RCANConfig,
        optim_config: Optional[OptimConfig] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = RCAN(config)
        self.loss_fn = loss_fn or nn.L1Loss()
        self.optim_config = optim_config or OptimConfig()

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        return self.model(lr)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        preds = self.forward(batch["lr"])
        loss = self.loss_fn(preds, batch["hr"])
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        preds = self.forward(batch["lr"])
        loss = self.loss_fn(preds, batch["hr"])
        psnr = self._psnr(preds.detach(), batch["hr"].detach())
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/psnr", psnr, prog_bar=True)
        # Lightning does not expose slash-separated metrics for checkpoint filename templates,
        # so duplicate the value under a sanitized key.
        self.log("val_psnr", psnr, prog_bar=False)
        return {"val_loss": loss, "val_psnr": psnr}

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:
        lr = batch["lr"]
        with torch.no_grad():
            sr = self.model(lr)
        return {"sr": sr, "name": batch.get("name"), "path": batch.get("path")}

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.optim_config.lr,
            betas=self.optim_config.betas,
            weight_decay=self.optim_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    @staticmethod
    def _psnr(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mse = torch.mean((sr - hr) ** 2)
        return 10 * torch.log10(1.0 / (mse + eps))
