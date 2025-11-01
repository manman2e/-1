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
    parser.add_argument("--train-hr", type=Path, help="Override training HR directory")
    parser.add_argument("--train-lr", type=Path, help="Override training LR directory")
    parser.add_argument("--val-hr", type=Path, help="Override validation HR directory")
    parser.add_argument("--val-lr", type=Path, help="Override validation LR directory")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--num-workers", type=int, help="Override dataloader worker count")
    parser.add_argument("--patch-size", type=int, help="Override training patch size")
    parser.add_argument("--max-epochs", type=int, help="Override max epochs from config")
    parser.add_argument("--precision", type=int, help="Override precision from config")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_cfg = RCANConfig(**cfg["model"])
    optim_cfg = OptimConfig(**cfg.get("optimizer", {}))

    data_dict = cfg["data"].copy()
    if args.train_hr:
        data_dict["train_hr"] = str(args.train_hr)
    if args.train_lr:
        data_dict["train_lr"] = str(args.train_lr)
    if args.val_hr:
        data_dict["val_hr"] = str(args.val_hr)
    if args.val_lr:
        data_dict["val_lr"] = str(args.val_lr)
    if args.batch_size:
        data_dict["batch_size"] = args.batch_size
    if args.num_workers is not None:
        data_dict["num_workers"] = args.num_workers
    if args.patch_size:
        data_dict["patch_size"] = args.patch_size
    data_cfg = DataConfig(**data_dict)

    module = RCANLightningModule(model_cfg, optim_config=optim_cfg)
    datamodule = RCANDataModule(data_cfg)

    callbacks = [
        ModelCheckpoint(
            monitor="val_psnr",
            mode="max",
            save_last=True,
            save_top_k=3,
            filename="rcan-{epoch:02d}-{val_psnr:.2f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer_cfg = cfg.get("trainer", {}).copy()
    if args.max_epochs:
        trainer_cfg["max_epochs"] = args.max_epochs
    if args.precision:
        trainer_cfg["precision"] = args.precision

    trainer = pl.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 100),
        accelerator=args.accelerator,
        devices=args.devices,
        precision=trainer_cfg.get("precision", 32),
        callbacks=callbacks,
        default_root_dir=trainer_cfg.get("default_root_dir", "runs"),
        gradient_clip_val=trainer_cfg.get("grad_clip", 0.0),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        check_val_every_n_epoch=trainer_cfg.get("val_interval", 1),
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
