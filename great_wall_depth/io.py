from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - informative message
    raise ImportError(
        "open3d is required for loading Great Wall datasets. Install it via 'pip install open3d'."
    ) from exc


def load_geometry(path: Path, voxel_size: float | None = None) -> np.ndarray:
    """Load a point cloud or mesh and return sampled points as an ``(N, 3)`` array."""

    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".ply", ".pcd", ".xyz", ".pts", ".las", ".laz"}:
        cloud = o3d.io.read_point_cloud(str(path))
    elif suffix in {".obj", ".stl", ".off", ".fbx", ".gltf", ".glb"}:
        mesh = o3d.io.read_triangle_mesh(str(path))
        if not mesh.has_triangles():
            raise ValueError(f"Mesh at {path} has no faces")
        cloud = mesh.sample_points_poisson_disk(number_of_points=2_000_000)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    if len(cloud.points) == 0:
        raise ValueError("Input geometry contains no points")

    if voxel_size and voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size)

    return np.asarray(cloud.points)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def write_depth_png(path: Path, depth: np.ndarray, max_depth: float) -> None:
    """Save a depth map as a 16-bit PNG."""

    import imageio.v2 as imageio

    clipped = np.clip(depth, 0, max_depth)
    scaled = np.round((clipped / max_depth) * (2**16 - 1)).astype(np.uint16)
    imageio.imwrite(path, scaled)
