"""Generate low-resolution counterparts from high-resolution imagery."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rcan.io import read_image, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample HR images to produce LR training data")
    parser.add_argument("input", type=Path, help="Directory containing high-resolution images")
    parser.add_argument("output", type=Path, help="Directory to store the generated low-resolution images")
    parser.add_argument("--scale", type=int, default=4, help="Downscale factor (e.g., 4 for 4x)")
    parser.add_argument(
        "--mode",
        type=str,
        default="bicubic",
        choices=["bicubic", "bilinear", "nearest", "area"],
        help="Interpolation mode used when downsampling",
    )
    parser.add_argument("--suffix", type=str, default="", help="Optional suffix appended to output filenames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in the output directory")
    return parser.parse_args()


def collect_images(directory: Path) -> Iterable[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts])


def ensure_divisible(tensor: torch.Tensor, scale: int) -> tuple[torch.Tensor, tuple[int, int] | None]:
    _, h, w = tensor.shape
    new_h = h - (h % scale)
    new_w = w - (w % scale)
    if new_h == h and new_w == w:
        return tensor, None
    return tensor[:, :new_h, :new_w], (new_h, new_w)


def downsample(tensor: torch.Tensor, scale: int, mode: str) -> torch.Tensor:
    kwargs = {}
    if mode in {"bicubic", "bilinear"}:
        kwargs["align_corners"] = False
    return F.interpolate(tensor.unsqueeze(0), scale_factor=1 / scale, mode=mode, **kwargs).squeeze(0)


def main() -> None:
    args = parse_args()
    if args.scale <= 1:
        raise ValueError("Scale factor must be greater than 1")

    if not args.input.exists() or not args.input.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    images = list(collect_images(args.input))
    if not images:
        raise FileNotFoundError(f"No supported images found in {args.input}")

    for path in tqdm(images, desc="Generating LR"):
        tensor, meta = read_image(path)
        orig_h, orig_w = tensor.shape[1], tensor.shape[2]
        tensor, cropped_shape = ensure_divisible(tensor, args.scale)
        lr = downsample(tensor, args.scale, args.mode)

        meta = meta.copy()
        meta["height"] = lr.shape[1]
        meta["width"] = lr.shape[2]

        out_name = path.stem + args.suffix + path.suffix if args.suffix else path.name
        out_path = args.output / out_name
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output file already exists: {out_path}")

        write_image(lr, out_path, meta=meta)
        if cropped_shape is not None:
            new_h, new_w = cropped_shape
            tqdm.write(
                f"Cropped {path.name} from {orig_h}x{orig_w} to {new_h}x{new_w} before downsampling"
            )


if __name__ == "__main__":
    main()
