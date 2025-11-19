from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from .cameras import generate_poses
from .centerline import CenterlineResult, compute_centerline
from .config import PipelineConfig
from .depth_renderer import render_depth_map
from .geometry import normalize, resample_polyline
from .io import load_geometry, write_depth_png, write_json, write_numpy
from .segments import Segment, split_into_segments


class PipelineState:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.points = np.zeros((0, 3))
        self.centerline: CenterlineResult | None = None
        self.segments: List[Segment] = []


def _prepare_points(config: PipelineConfig) -> np.ndarray:
    points = load_geometry(config.input_path, voxel_size=config.voxel_size)
    return points


def _compute_centerline(state: PipelineState) -> CenterlineResult:
    config = state.config
    centerline = compute_centerline(
        state.points,
        world_up=np.asarray(config.world_up, dtype=float),
        step=config.long_axis_step,
        min_points=config.min_points_per_bin,
        smoothing_window=config.smoothing_window,
    )
    state.centerline = centerline
    return centerline


def _split_segments(state: PipelineState) -> List[Segment]:
    config = state.config
    centerline = state.centerline
    if centerline is None:
        raise RuntimeError("Centerline must be computed before segmenting")
    segments = split_into_segments(
        state.points,
        centerline.basis.to_local(state.points),
        centerline.local_points,
        centerline.world_points,
        segment_length=config.chunk_length,
        overlap=config.chunk_overlap,
        lateral_threshold=config.perpendicular_threshold,
    )
    state.segments = segments
    return segments


def _sample_tangents(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.tile(np.array([1.0, 0.0, 0.0]), (len(points), 1))
    tangents = np.zeros_like(points)
    tangents[1:-1] = points[2:] - points[:-2]
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    return normalize(tangents)


def run_pipeline(config: PipelineConfig) -> Dict[str, Path]:
    config.validate()
    config.ensure_output_dir()
    state = PipelineState(config)
    state.points = _prepare_points(config)
    centerline = _compute_centerline(state)
    world_up = np.asarray(config.world_up, dtype=float)
    resampled_world = resample_polyline(centerline.world_points, config.long_axis_step)
    resampled_local = centerline.basis.to_local(resampled_world)
    centerline = CenterlineResult(world_points=resampled_world, local_points=resampled_local, basis=centerline.basis)
    state.centerline = centerline

    if config.save_intermediate:
        write_numpy(config.output_dir / "centerline_world.npy", centerline.world_points)
        write_numpy(config.output_dir / "centerline_local.npy", centerline.local_points)

    segments = _split_segments(state)
    if not segments:
        raise RuntimeError("No segments created; adjust chunk_length/perpendicular_threshold")

    intrinsics = config.camera.intrinsics()
    outputs: Dict[str, Path] = {}

    for segment in segments:
        tangents = _sample_tangents(segment.centerline_world)
        for idx, (point, tangent) in enumerate(zip(segment.centerline_world, tangents)):
            poses = generate_poses(point, tangent, world_up, config.camera)
            for pose in poses:
                depth = render_depth_map(segment.world_points, pose, intrinsics, config.camera)
                base = config.output_dir / f"segment_{segment.index:03d}" / pose.name
                depth_path = base / f"sample_{idx:04d}_depth.npy"
                write_numpy(depth_path, depth)
                png_path = base / f"sample_{idx:04d}_depth.png"
                write_depth_png(png_path, depth, config.camera.far_clip)
                meta_path = base / f"sample_{idx:04d}_meta.json"
                write_json(
                    meta_path,
                    {
                        "segment_index": segment.index,
                        "sample_index": idx,
                        "pose": {
                            "name": pose.name,
                            "position": pose.position.tolist(),
                            "forward": pose.forward.tolist(),
                            "up": pose.up.tolist(),
                            "right": pose.right.tolist(),
                        },
                        "intrinsics": {
                            "fx": intrinsics[0],
                            "fy": intrinsics[1],
                            "cx": intrinsics[2],
                            "cy": intrinsics[3],
                            "width": config.image_width,
                            "height": config.image_height,
                            "near": config.camera.near_clip,
                            "far": config.camera.far_clip,
                        },
                    },
                )
                outputs[f"segment_{segment.index:03d}_{pose.name}_{idx:04d}"] = depth_path

    summary_path = config.output_dir / "summary.json"
    write_json(
        summary_path,
        {
            "config": config.as_dict(),
            "outputs": {k: str(v) for k, v in outputs.items()},
        },
    )
    outputs["summary"] = summary_path
    return outputs
