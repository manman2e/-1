"""Train RCAN on Gaofen-2 satellite imagery."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from rcan.datamodule import DataConfig, RCANDataModule
from rcan.lit_module import OptimConfig, RCANLightningModule
from rcan.model import RCANConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RCAN on Gaofen-2 data")
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming training")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPU devices to use")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Training accelerator (gpu/cpu)")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = RCANConfig(**cfg["model"])
    optim_cfg = OptimConfig(**cfg.get("optimizer", {}))
    data_cfg = DataConfig(**cfg["data"])

    module = RCANLightningModule(model_cfg, optim_config=optim_cfg)
    datamodule = RCANDataModule(data_cfg)

    callbacks = [
        ModelCheckpoint(
            monitor="val/psnr",
            mode="max",
            save_last=True,
            save_top_k=3,
            filename="rcan-{epoch:02d}-{val_psnr:.2f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.get("trainer", {}).get("max_epochs", 100),
        accelerator=args.accelerator,
        devices=args.devices,
        precision=cfg.get("trainer", {}).get("precision", 32),
        callbacks=callbacks,
        default_root_dir=cfg.get("trainer", {}).get("default_root_dir", "runs"),
        gradient_clip_val=cfg.get("trainer", {}).get("grad_clip", 0.0),
        log_every_n_steps=cfg.get("trainer", {}).get("log_every_n_steps", 50),
        check_val_every_n_epoch=cfg.get("trainer", {}).get("val_interval", 1),
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
