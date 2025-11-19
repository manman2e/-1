from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np


Vector3 = Tuple[float, float, float]


@dataclass
class CameraConfig:
    """Camera configuration shared across generated depth maps."""

    image_width: int = 1024
    image_height: int = 768
    horizontal_fov_deg: float = 70.0
    near_clip: float = 0.1
    far_clip: float = 200.0
    lateral_offset: float = 8.0
    vertical_offset: float = 4.0
    top_view_height: float = 60.0

    def intrinsics(self) -> Tuple[float, float, float, float]:
        """Return focal lengths and principal point (fx, fy, cx, cy)."""

        fov_rad = np.deg2rad(self.horizontal_fov_deg)
        fx = 0.5 * self.image_width / np.tan(fov_rad / 2.0)
        fy = fx  # square pixels assumption
        cx = (self.image_width - 1) / 2.0
        cy = (self.image_height - 1) / 2.0
        return fx, fy, cx, cy


@dataclass
class PipelineConfig:
    """Configuration container for the full processing pipeline."""

    input_path: Path
    output_dir: Path
    world_up: Vector3 = (0.0, 0.0, 1.0)
    voxel_size: float = 0.2
    long_axis_step: float = 1.0
    min_points_per_bin: int = 200
    perpendicular_threshold: float = 12.0
    smoothing_window: int = 9
    chunk_length: float = 150.0
    chunk_overlap: float = 30.0
    save_intermediate: bool = True
    camera: CameraConfig = field(default_factory=CameraConfig)

    def validate(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Image width/height must be positive")
        if self.camera.near_clip <= 0:
            raise ValueError("Near clipping distance must be positive")
        if self.camera.far_clip <= self.camera.near_clip:
            raise ValueError("Far clip must be greater than near clip")
        if self.long_axis_step <= 0:
            raise ValueError("long_axis_step must be positive")
        if self.min_points_per_bin <= 0:
            raise ValueError("min_points_per_bin must be positive")
        if self.chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.perpendicular_threshold <= 0:
            raise ValueError("perpendicular_threshold must be positive")

    @property
    def image_width(self) -> int:
        return self.camera.image_width

    @property
    def image_height(self) -> int:
        return self.camera.image_height

    def ensure_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    @classmethod
    def from_paths(
        cls,
        input_path: str | Path,
        output_dir: str | Path,
        **kwargs,
    ) -> "PipelineConfig":
        cfg_kwargs = dict(kwargs)
        if "camera" in cfg_kwargs and isinstance(cfg_kwargs["camera"], dict):
            cfg_kwargs["camera"] = CameraConfig(**cfg_kwargs["camera"])
        return cls(Path(input_path), Path(output_dir), **cfg_kwargs)

    def as_dict(self) -> dict:
        return {
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "world_up": tuple(self.world_up),
            "voxel_size": self.voxel_size,
            "long_axis_step": self.long_axis_step,
            "min_points_per_bin": self.min_points_per_bin,
            "perpendicular_threshold": self.perpendicular_threshold,
            "smoothing_window": self.smoothing_window,
            "chunk_length": self.chunk_length,
            "chunk_overlap": self.chunk_overlap,
            "save_intermediate": self.save_intermediate,
            "camera": {
                "image_width": self.camera.image_width,
                "image_height": self.camera.image_height,
                "horizontal_fov_deg": self.camera.horizontal_fov_deg,
                "near_clip": self.camera.near_clip,
                "far_clip": self.camera.far_clip,
                "lateral_offset": self.camera.lateral_offset,
                "vertical_offset": self.camera.vertical_offset,
                "top_view_height": self.camera.top_view_height,
            },
        }
