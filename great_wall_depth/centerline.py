from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .geometry import OrientedBasis, estimate_long_axis, smooth_polyline


@dataclass
class CenterlineResult:
    world_points: np.ndarray
    local_points: np.ndarray
    basis: OrientedBasis


def compute_centerline(
    points: np.ndarray,
    world_up: np.ndarray,
    step: float,
    min_points: int,
    smoothing_window: int,
) -> CenterlineResult:
    """Estimate a smoothed centerline along the Great Wall dataset."""

    basis = estimate_long_axis(points, world_up)
    local = basis.to_local(points)

    u = local[:, 0]
    v = local[:, 1]
    z = local[:, 2]

    min_u, max_u = np.min(u), np.max(u)
    bins = np.arange(min_u, max_u + step, step)
    centers = []
    for start, end in zip(bins[:-1], bins[1:]):
        mask = (u >= start) & (u < end)
        if mask.sum() < min_points:
            continue
        seg_u = u[mask]
        seg_v = v[mask]
        seg_z = z[mask]
        center_u = 0.5 * (start + end)
        center_v = np.median(seg_v)
        center_z = np.percentile(seg_z, 75)
        centers.append([center_u, center_v, center_z])

    if not centers:
        raise RuntimeError("Unable to compute centerline; increase min_points or adjust voxel size")

    centers_local = np.asarray(centers)
    smoothed_local = smooth_polyline(centers_local, smoothing_window)
    world_points = basis.to_world(smoothed_local)

    return CenterlineResult(world_points=world_points, local_points=smoothed_local, basis=basis)
