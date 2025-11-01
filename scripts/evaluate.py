"""Evaluate a trained RCAN checkpoint on Gaofen-2 validation data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import torch
from tqdm import tqdm

from rcan.datamodule import DataConfig, RCANDataModule
from rcan.lit_module import RCANLightningModule
from rcan.model import RCANConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RCAN checkpoint")
    parser.add_argument("config", type=Path, help="Path to YAML configuration file")
    parser.add_argument("checkpoint", type=Path, help="Path to trained checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Dataset split to evaluate")
    parser.add_argument("--val-hr", type=Path, help="Override validation HR directory from config")
    parser.add_argument("--val-lr", type=Path, help="Override validation LR directory from config")
    parser.add_argument("--batch-size", type=int, help="Override validation batch size")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = RCANConfig(**cfg["model"])
    module = RCANLightningModule.load_from_checkpoint(args.checkpoint, config=model_cfg)
    module.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)

    data_dict = cfg["data"].copy()
    if args.val_hr:
        data_dict["val_hr"] = str(args.val_hr)
    if args.val_lr:
        data_dict["val_lr"] = str(args.val_lr)
    if args.batch_size:
        data_dict["batch_size"] = args.batch_size
    data_cfg = DataConfig(**data_dict)
    if model_cfg.n_colors != data_cfg.num_channels:
        raise ValueError(
            f"Model expects {model_cfg.n_colors} channels but datamodule is configured for {data_cfg.num_channels}"
        )
    dm = RCANDataModule(data_cfg)
    stage = "validate" if args.split == "val" else "fit"
    dm.setup(stage=stage)
    dataloader = dm.val_dataloader() if args.split == "val" else dm.train_dataloader()
    if dataloader is None:
        raise RuntimeError(f"No dataloader available for split {args.split}")

    psnrs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            sr = module(lr).clamp(0.0, 1.0)
            mse = torch.mean((sr - hr) ** 2).item()
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8)).item()
            psnrs.append(psnr)
    if psnrs:
        mean_psnr = sum(psnrs) / len(psnrs)
        print(f"Mean PSNR: {mean_psnr:.2f} dB over {len(psnrs)} samples")
    else:
        print("No samples found in the selected split.")


if __name__ == "__main__":
    main()
