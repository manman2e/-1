from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .geometry import cumulative_lengths


@dataclass
class Segment:
    index: int
    world_points: np.ndarray
    local_points: np.ndarray
    centerline_world: np.ndarray
    centerline_local: np.ndarray
    centerline_distances: np.ndarray


def split_into_segments(
    all_points_world: np.ndarray,
    all_points_local: np.ndarray,
    centerline_local: np.ndarray,
    centerline_world: np.ndarray,
    segment_length: float,
    overlap: float,
    lateral_threshold: float,
) -> List[Segment]:
    """Split the dataset into overlapping segments aligned with the centerline."""

    distances = cumulative_lengths(centerline_world)
    if len(distances) == 0:
        return []

    segments: List[Segment] = []
    start_distance = 0.0
    idx = 0
    while start_distance < distances[-1]:
        end_distance = min(start_distance + segment_length, distances[-1])
        mask_centerline = (distances >= start_distance) & (distances <= end_distance)
        if mask_centerline.sum() < 2:
            start_distance += max(segment_length - overlap, segment_length)
            continue

        seg_centerline_local = centerline_local[mask_centerline]
        seg_centerline_world = centerline_world[mask_centerline]
        min_u = seg_centerline_local[:, 0].min()
        max_u = seg_centerline_local[:, 0].max()
        margin = segment_length * 0.5
        mask_points = (
            (all_points_local[:, 0] >= min_u - margin)
            & (all_points_local[:, 0] <= max_u + margin)
            & (np.abs(all_points_local[:, 1]) <= lateral_threshold)
        )
        seg_points_world = all_points_world[mask_points]
        seg_points_local = all_points_local[mask_points]
        if len(seg_points_world) == 0:
            start_distance += max(segment_length - overlap, segment_length)
            continue

        segments.append(
            Segment(
                index=idx,
                world_points=seg_points_world,
                local_points=seg_points_local,
                centerline_world=seg_centerline_world,
                centerline_local=seg_centerline_local,
                centerline_distances=distances[mask_centerline],
            )
        )
        idx += 1
        start_distance += max(segment_length - overlap, segment_length)
    return segments
