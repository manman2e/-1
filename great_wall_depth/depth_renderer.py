from __future__ import annotations

import numpy as np


def project_points(
    points: np.ndarray,
    camera_position: np.ndarray,
    rotation: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float,
    far: float,
    width: int,
    height: int,
) -> np.ndarray:
    translated = points - camera_position
    cam_points = translated @ rotation.T
    mask = cam_points[:, 2] > near
    cam_points = cam_points[mask]
    if len(cam_points) == 0:
        return np.full((height, width), fill_value=far, dtype=np.float32)

    xs = fx * cam_points[:, 0] / cam_points[:, 2] + cx
    ys = fy * cam_points[:, 1] / cam_points[:, 2] + cy
    zs = cam_points[:, 2]
    valid = (
        (xs >= 0)
        & (xs < width)
        & (ys >= 0)
        & (ys < height)
        & (zs <= far)
    )
    xs = xs[valid].astype(np.int32)
    ys = ys[valid].astype(np.int32)
    zs = zs[valid]

    depth = np.full((height, width), fill_value=far, dtype=np.float32)
    if len(zs) == 0:
        return depth
    np.minimum.at(depth, (ys, xs), zs)
    return depth


def render_depth_map(points: np.ndarray, pose, intrinsics, config) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    rotation = np.stack([pose.right, pose.up, pose.forward], axis=0)
    depth = project_points(
        points,
        camera_position=pose.position,
        rotation=rotation,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        near=config.near_clip,
        far=config.far_clip,
        width=config.image_width,
        height=config.image_height,
    )
    return depth
