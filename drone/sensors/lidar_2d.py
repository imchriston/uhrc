# sensors/lidar_2d.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class BoxObstacle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

def ray_aabb_intersect_2d(p: np.ndarray, d: np.ndarray, box: BoxObstacle) -> float | None:
    """
    Ray-AABB intersection in 2D.
    Ray: p + t d, t>=0. Returns smallest positive t, or None if no hit.
    """
    eps = 1e-9
    tmin, tmax = -1e18, 1e18

    # X slabs
    if abs(d[0]) < eps:
        if p[0] < box.xmin or p[0] > box.xmax:
            return None
    else:
        tx1 = (box.xmin - p[0]) / d[0]
        tx2 = (box.xmax - p[0]) / d[0]
        tmin = max(tmin, min(tx1, tx2))
        tmax = min(tmax, max(tx1, tx2))

    # Y slabs
    if abs(d[1]) < eps:
        if p[1] < box.ymin or p[1] > box.ymax:
            return None
    else:
        ty1 = (box.ymin - p[1]) / d[1]
        ty2 = (box.ymax - p[1]) / d[1]
        tmin = max(tmin, min(ty1, ty2))
        tmax = min(tmax, max(ty1, ty2))

    if tmax < 0.0 or tmin > tmax:
        return None

    t_hit = tmin if tmin >= 0.0 else tmax
    return t_hit if t_hit >= 0.0 else None

def lidar_scan_xy(
    pos_xy: np.ndarray,
    yaw: float,
    boxes: list[BoxObstacle],
    angles: np.ndarray,
    r_max: float = 10.0,
) -> np.ndarray:
    """
    2D lidar scan in XY plane around yaw.
    angles: ray angles relative to body-forward (radians), e.g. [-pi/2 ... +pi/2]
    """
    ranges = np.full((len(angles),), r_max, dtype=np.float64)

    cy, sy = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sy],
                  [sy,  cy]], dtype=np.float64)

    for i, a in enumerate(angles):
        # body ray direction (forward = +x_body)
        d_body = np.array([np.cos(a), np.sin(a)], dtype=np.float64)
        d = R @ d_body  # world direction
        best = r_max
        for b in boxes:
            t_hit = ray_aabb_intersect_2d(pos_xy, d, b)
            if t_hit is not None and 0.0 <= t_hit < best:
                best = t_hit
        ranges[i] = best

    return ranges
