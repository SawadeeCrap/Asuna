"""Inverse kinematics helpers (world space)."""
from __future__ import annotations

import math

import numpy as np

from ..util.mathutil import normalize, project_on_plane


def soft_distance(dist: float, reach: float, softness: float = 0.03) -> float:
    """Andy Nicholas' soft IK: approach full extension asymptotically (no knee pop)."""
    ds = reach * (1.0 - softness)
    if dist <= ds or softness <= 0.0:
        return dist
    span = reach - ds
    return ds + span * (1.0 - math.exp(-(dist - ds) / span))


def two_bone(root: np.ndarray, target: np.ndarray, l1: float, l2: float, pole: np.ndarray,
             softness: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a two-bone chain. Returns (mid joint, effector position, plane normal).

    ``pole`` is a direction (not a point): the mid joint bends toward it.
    """
    d = target - root
    dist = float(np.linalg.norm(d))
    dhat = normalize(d, np.array([0.0, 0.0, -1.0]))
    reach = l1 + l2
    dist = soft_distance(dist, reach, softness)
    dist = min(max(dist, abs(l1 - l2) + 1e-5), reach - 1e-6)
    pole_perp = project_on_plane(pole, dhat)
    if float(np.linalg.norm(pole_perp)) < 1e-8:
        alt = np.array([1.0, 0.0, 0.0]) if abs(dhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        pole_perp = project_on_plane(alt, dhat)
    b = normalize(pole_perp)
    n = normalize(np.cross(dhat, b))
    cos_a = (l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)
    cos_a = min(1.0, max(-1.0, cos_a))
    sin_a = math.sqrt(max(0.0, 1.0 - cos_a * cos_a))
    mid = root + dhat * (l1 * cos_a) + b * (l1 * sin_a)
    eff = root + dhat * dist
    return mid, eff, n


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix taking direction a onto direction b."""
    a = normalize(a)
    b = normalize(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees: any perpendicular axis.
        axis = normalize(np.cross(a, np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def clamp_cone(direction: np.ndarray, axis: np.ndarray, max_angle: float) -> np.ndarray:
    """Limit ``direction`` to a cone of half-angle ``max_angle`` around ``axis``."""
    d = normalize(direction)
    a = normalize(axis)
    c = float(np.dot(d, a))
    ang = math.acos(min(1.0, max(-1.0, c)))
    if ang <= max_angle:
        return d
    perp = normalize(d - a * c, np.array([1.0, 0.0, 0.0]))
    return a * math.cos(max_angle) + perp * math.sin(max_angle)
