"""Run RCAN inference on Gaofen-2 (or other) imagery."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import yaml
import torch
from tqdm import tqdm

from rcan.datasets import SingleImageDataset
from rcan.io import write_image
from rcan.lit_module import RCANLightningModule
from rcan.model import RCANConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Super-resolve images using a trained RCAN model")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    parser.add_argument("input", type=Path, help="Directory of images or a single image file")
    parser.add_argument("output", type=Path, help="Directory to save super-resolved images")
    parser.add_argument("--config", type=Path, help="Optional YAML config to override checkpoint hyperparameters")
    parser.add_argument("--tile-size", type=int, default=0, help="Optional tiling size for large images")
    parser.add_argument("--overlap", type=int, default=32, help="Overlap size when tiling")
    parser.add_argument("--half", action="store_true", help="Use half precision during inference")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_images(input_path: Path) -> Iterable[Path]:
    exts = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    if input_path.is_file():
        if input_path.suffix.lower() in exts:
            yield input_path
    else:
        for path in sorted(input_path.iterdir()):
            if path.is_file() and path.suffix.lower() in exts:
                yield path


def tile_image(image: torch.Tensor, tile_size: int, overlap: int) -> list[tuple[torch.Tensor, tuple[int, int], tuple[int, int]]]:
    _, h, w = image.shape
    stride = max(tile_size - overlap, 1)
    tiles = []
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)
            tile = image[:, top:bottom, left:right]
            tiles.append((tile, (top, left), (bottom - top, right - left)))
    return tiles


def merge_tiles(tiles: list[tuple[torch.Tensor, tuple[int, int], tuple[int, int]]], shape: tuple[int, int, int], scale: int) -> torch.Tensor:
    c, h, w = shape
    sr = torch.zeros((c, h * scale, w * scale), dtype=tiles[0][0].dtype, device=tiles[0][0].device)
    count = torch.zeros_like(sr)
    for tile, (top, left), (lh, lw) in tiles:
        top_sr = top * scale
        left_sr = left * scale
        sr[:, top_sr : top_sr + lh * scale, left_sr : left_sr + lw * scale] += tile
        count[:, top_sr : top_sr + lh * scale, left_sr : left_sr + lw * scale] += 1
    sr /= count.clamp(min=1.0)
    return sr


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        model_cfg = RCANConfig(**cfg["model"])
        module = RCANLightningModule.load_from_checkpoint(args.checkpoint, config=model_cfg)
    else:
        module = RCANLightningModule.load_from_checkpoint(args.checkpoint)
    module.eval()
    module.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)
    if args.half:
        module.model.half()

    input_path = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [p for p in iter_images(input_path)]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_path}")
    expected_channels = module.model.config.n_colors
    dataset = SingleImageDataset(image_paths, num_channels=expected_channels)

    for sample in tqdm(dataset, desc="Super-resolving"):
        lr = sample["lr"].unsqueeze(0).to(device)
        if args.half:
            lr = lr.half()
        if args.tile_size > 0:
            tiles = tile_image(lr.squeeze(0), args.tile_size, args.overlap)
            sr_tiles = []
            for tile, offset, size in tiles:
                sr_tile = module.model(tile.unsqueeze(0)).squeeze(0)
                sr_tiles.append((sr_tile, offset, size))
            sr = merge_tiles(sr_tiles, lr.squeeze(0).shape, module.model.config.scale)
        else:
            sr = module.model(lr).squeeze(0)
        write_image(sr, output_dir / sample["name"], meta=sample.get("meta"))


if __name__ == "__main__":
    main()
