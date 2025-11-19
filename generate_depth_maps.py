from __future__ import annotations

import argparse
import json
from pathlib import Path

from great_wall_depth import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate depth maps for a Great Wall 3D dataset")
    parser.add_argument("input", type=Path, help="Path to the input point cloud or mesh (PLY/PCD/OBJ/etc.)")
    parser.add_argument("output", type=Path, help="Directory where outputs will be stored")
    parser.add_argument("--voxel", type=float, default=0.2, help="Voxel downsample size in meters")
    parser.add_argument("--step", type=float, default=1.0, help="Sampling step along the long axis (meters)")
    parser.add_argument("--min-points", type=int, default=200, help="Minimum points per bin when estimating the centerline")
    parser.add_argument("--perpendicular", type=float, default=12.0, help="Maximum lateral distance from the centerline for segment points")
    parser.add_argument("--smoothing", type=int, default=9, help="Moving window size for centerline smoothing")
    parser.add_argument("--chunk-length", type=float, default=150.0, help="Segment length in meters")
    parser.add_argument("--chunk-overlap", type=float, default=30.0, help="Segment overlap in meters")
    parser.add_argument("--width", type=int, default=1024, help="Depth image width")
    parser.add_argument("--height", type=int, default=768, help="Depth image height")
    parser.add_argument("--fov", type=float, default=70.0, help="Camera horizontal field of view in degrees")
    parser.add_argument("--near", type=float, default=0.1, help="Near clipping distance")
    parser.add_argument("--far", type=float, default=200.0, help="Far clipping distance")
    parser.add_argument("--lateral", type=float, default=8.0, help="Camera offset from the centerline for A/B views")
    parser.add_argument("--vertical", type=float, default=4.0, help="Camera vertical offset for A/B views")
    parser.add_argument("--top-height", type=float, default=60.0, help="Height above the centerline for top-down views")
    parser.add_argument("--world-up", type=float, nargs=3, default=(0.0, 0.0, 1.0), help="World up vector")
    parser.add_argument("--no-intermediate", action="store_true", help="Disable saving intermediate centerline arrays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_kwargs = dict(
        image_width=args.width,
        image_height=args.height,
        horizontal_fov_deg=args.fov,
        near_clip=args.near,
        far_clip=args.far,
        lateral_offset=args.lateral,
        vertical_offset=args.vertical,
        top_view_height=args.top_height,
    )
    config = PipelineConfig.from_paths(
        input_path=args.input,
        output_dir=args.output,
        voxel_size=args.voxel,
        long_axis_step=args.step,
        min_points_per_bin=args.min_points,
        perpendicular_threshold=args.perpendicular,
        smoothing_window=args.smoothing,
        chunk_length=args.chunk_length,
        chunk_overlap=args.chunk_overlap,
        world_up=tuple(args.world_up),
        save_intermediate=not args.no_intermediate,
        camera=camera_kwargs,
    )

    outputs = run_pipeline(config)
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
