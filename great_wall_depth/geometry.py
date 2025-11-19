from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

Vector3 = np.ndarray


@dataclass
class OrientedBasis:
    origin: Vector3
    basis: np.ndarray  # shape (3, 3) columns are basis vectors

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.basis

    def to_world(self, points: np.ndarray) -> np.ndarray:
        return points @ self.basis.T + self.origin


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def estimate_long_axis(points: np.ndarray, world_up: np.ndarray) -> OrientedBasis:
    """Estimate the long-axis-aligned basis for the point cloud."""

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3)")

    centroid = points.mean(axis=0)
    centered = points - centroid
    xy = centered[:, :2]
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_vec_xy = eigvecs[:, np.argmax(eigvals)]
    long_vec = normalize(np.array([long_vec_xy[0], long_vec_xy[1], 0.0]))
    up = normalize(world_up)
    short_vec = normalize(np.cross(up, long_vec))
    basis = np.stack([long_vec, short_vec, up], axis=1)
    return OrientedBasis(origin=centroid, basis=basis)


def cumulative_lengths(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0)
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1, keepdims=True)
    lengths = np.concatenate([np.zeros((1, 1)), diffs], axis=0)
    return np.cumsum(lengths[:, 0])


def smooth_polyline(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(points) < 3:
        return points
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    padded = np.pad(points, ((window // 2, window // 2), (0, 0)), mode="edge")
    smoothed = np.empty_like(points)
    for i in range(3):
        smoothed[:, i] = np.convolve(padded[:, i], kernel, mode="valid")
    return smoothed


def resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) == 0:
        return points
    if spacing <= 0:
        return points
    lengths = cumulative_lengths(points)
    if lengths[-1] == 0:
        return points[[0]]
    target_lengths = np.arange(0.0, lengths[-1] + spacing, spacing)
    resampled = np.empty((len(target_lengths), 3))
    resampled_idx = 0
    for s in target_lengths:
        idx = np.searchsorted(lengths, s)
        if idx == 0:
            resampled[resampled_idx] = points[0]
        elif idx >= len(points):
            resampled[resampled_idx] = points[-1]
        else:
            t = (s - lengths[idx - 1]) / (lengths[idx] - lengths[idx - 1] + 1e-8)
            resampled[resampled_idx] = points[idx - 1] * (1 - t) + points[idx] * t
        resampled_idx += 1
    return resampled
