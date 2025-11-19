from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import CameraConfig
from .geometry import normalize


@dataclass
class CameraPose:
    name: str
    position: np.ndarray
    forward: np.ndarray
    up: np.ndarray
    right: np.ndarray

    def extrinsic_matrix(self) -> np.ndarray:
        rotation = np.stack([self.right, self.up, self.forward], axis=0)
        translation = -rotation @ self.position
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        return extrinsic


def _build_pose(
    name: str,
    position: np.ndarray,
    target: np.ndarray,
    world_up: np.ndarray,
) -> CameraPose:
    forward = normalize(target - position)
    right = normalize(np.cross(forward, world_up))
    if np.linalg.norm(right) < 1e-6:
        # fallback for near-vertical views
        arbitrary = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(arbitrary, forward)) > 0.9:
            arbitrary = np.array([0.0, 1.0, 0.0])
        right = normalize(np.cross(forward, arbitrary))
    up = normalize(np.cross(right, forward))
    return CameraPose(name=name, position=position, forward=forward, up=up, right=right)


def generate_poses(
    center_point: np.ndarray,
    tangent: np.ndarray,
    world_up: np.ndarray,
    config: CameraConfig,
) -> List[CameraPose]:
    tangent = normalize(tangent)
    side = normalize(np.cross(world_up, tangent))
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    center = center_point
    elevated_target = center + world_up * config.vertical_offset

    poses: List[CameraPose] = []
    pos_a = center + side * config.lateral_offset + world_up * config.vertical_offset
    poses.append(_build_pose("A", pos_a, elevated_target, world_up))

    pos_b = center - side * config.lateral_offset + world_up * config.vertical_offset
    poses.append(_build_pose("B", pos_b, elevated_target, world_up))

    top_pos = center + world_up * config.top_view_height
    poses.append(_build_pose("Top", top_pos, center, tangent))
    return poses
