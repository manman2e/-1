"""I/O helpers for multi-band raster imagery."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
import tifffile


def read_image(path: Path | str) -> tuple[torch.Tensor, Dict[str, object]]:
    """Load an image into a normalized CHW tensor along with metadata."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    else:
        with Image.open(path) as image:
            array = np.asarray(image)

    if array.ndim == 2:
        array = array[..., None]
    elif (
        array.ndim == 3
        and array.shape[0] <= 8
        and array.shape[0] < array.shape[1]
        and array.shape[0] < array.shape[2]
    ):
        array = np.transpose(array, (1, 2, 0))

    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape {array.shape} for {path}")

    orig_dtype = np.dtype(array.dtype)
    if np.issubdtype(orig_dtype, np.integer):
        scale = float(np.iinfo(orig_dtype).max)
        array = array.astype(np.float32) / scale
        dtype_kind = "i"
    else:
        scale = 1.0
        array = array.astype(np.float32)
        dtype_kind = "f"

    tensor = torch.from_numpy(array).permute(2, 0, 1)
    meta: Dict[str, object] = {
        "dtype": orig_dtype.str,
        "dtype_kind": dtype_kind,
        "scale": scale,
        "channels": tensor.shape[0],
        "height": tensor.shape[1],
        "width": tensor.shape[2],
    }
    return tensor, meta


def write_image(tensor: torch.Tensor, path: Path | str, meta: dict[str, object] | None = None) -> None:
    """Persist an image tensor using the metadata captured by :func:`read_image`."""

    path = Path(path)
    array = tensor.clamp(0.0, 1.0).cpu().numpy()
    array = np.transpose(array, (1, 2, 0))

    suffix = path.suffix.lower()
    if meta:
        dtype = np.dtype(meta.get("dtype", "float32"))
        scale = float(meta.get("scale", 1.0))
        dtype_kind = meta.get("dtype_kind", "f")
    else:
        dtype = np.uint8
        scale = 255.0
        dtype_kind = "i"

    if dtype_kind == "i":
        array = np.round(array * scale).clip(0, scale)
        array = array.astype(dtype)
    else:
        array = array.astype(dtype)

    if array.shape[2] == 1:
        array = array[:, :, 0]

    if suffix in {".tif", ".tiff"} or (meta and meta.get("channels", 1) > 3):
        tifffile.imwrite(str(path), array)
    else:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        image.save(path)
