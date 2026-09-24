"""Small linear-algebra helpers shared by every subsystem.

Conventions used throughout Myrmex:

* World space is right-handed, Z up, metres, seconds (identical to Blender).
* A character's *body frame* is x = forward, y = left, z = up.
* Rotation matrices are 3x3 numpy arrays whose *columns* are the local axes
  expressed in the parent space (so ``R @ v_local == v_parent``).
* Quaternions are ``(w, x, y, z)`` like Blender.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-9
TAU = 2.0 * math.pi
UP = np.array([0.0, 0.0, 1.0])
X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------- scalars

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def inv_lerp(a: float, b: float, x: float) -> float:
    if abs(b - a) < EPS:
        return 0.0
    return (x - a) / (b - a)


def remap(x: float, a0: float, a1: float, b0: float, b1: float, clamp_out: bool = True) -> float:
    t = inv_lerp(a0, a1, x)
    if clamp_out:
        t = clamp(t, 0.0, 1.0)
    return b0 + (b1 - b0) * t


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = clamp(inv_lerp(e0, e1, x), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smootherstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (a + math.pi) % TAU - math.pi


def angle_lerp(a: float, b: float, t: float) -> float:
    return a + wrap_angle(b - a) * t


def frac(x: float) -> float:
    return x - math.floor(x)


def phase_diff(a: float, b: float) -> float:
    """Signed shortest difference b - a between two cycle phases in [0, 1)."""
    d = (b - a) % 1.0
    return d - 1.0 if d > 0.5 else d


def bump(x: float, center: float, width: float) -> float:
    """Smooth compact bump: 1 at ``center``, 0 beyond ``+-width``."""
    if width <= 0.0:
        return 0.0
    u = (x - center) / width
    if u <= -1.0 or u >= 1.0:
        return 0.0
    return (1.0 - u * u) ** 2


def exp_decay(dt: float, halflife: float) -> float:
    """Blend factor so that a first-order filter reaches 50% after ``halflife`` s."""
    if halflife <= 1e-6:
        return 1.0
    return 1.0 - 0.5 ** (dt / halflife)


def softclip(x: float, limit: float) -> float:
    """tanh-like soft limiter that is linear near zero."""
    if limit <= 0.0:
        return 0.0
    return limit * math.tanh(x / limit)


# --------------------------------------------------------------------------- vectors

def vec(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=float)


def length(v: np.ndarray) -> float:
    return float(math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2))


def length2d(v: np.ndarray) -> float:
    return float(math.hypot(float(v[0]), float(v[1])))


def normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = length(v)
    if n < EPS:
        return (fallback if fallback is not None else X_AXIS).astype(float).copy()
    return v / n


def project_on_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - n * float(np.dot(v, n))


def rotate2d(x: float, y: float, a: float) -> tuple[float, float]:
    c, s = math.cos(a), math.sin(a)
    return c * x - s * y, s * x + c * y


def heading_vec(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw), 0.0])


# --------------------------------------------------------------------------- rotations

def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_rotation(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Body attitude from heading, pitch (nose up positive) and roll (right side down positive)."""
    return rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll)


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    a = normalize(axis)
    x, y, z = float(a[0]), float(a[1]), float(a[2])
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotvec_matrix(rv: np.ndarray) -> np.ndarray:
    ang = length(rv)
    if ang < 1e-12:
        return np.eye(3)
    return axis_angle_matrix(rv / ang, ang)


def align_up(R: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Minimal rotation that tilts frame ``R`` so that its z axis becomes ``up``."""
    z = R[:, 2]
    up = normalize(up, UP)
    axis = np.cross(z, up)
    s = length(axis)
    c = float(np.dot(z, up))
    if s < 1e-9:
        return R.copy()
    ang = math.atan2(s, c)
    return axis_angle_matrix(axis / s, ang) @ R


def frame_from_forward_up(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Body-style frame: columns (x=forward, y=left, z=up), orthonormalised."""
    f = normalize(forward)
    left = np.cross(up, f)
    if length(left) < 1e-6:
        left = np.cross(UP if abs(f[2]) < 0.9 else Y_AXIS, f)
    left = normalize(left)
    upv = np.cross(f, left)
    return np.column_stack([f, left, upv])


def frame_from_y_z(y_axis: np.ndarray, z_hint: np.ndarray) -> np.ndarray:
    """Bone-style frame: y along the bone, z as close as possible to ``z_hint``."""
    y = normalize(y_axis)
    z = project_on_plane(z_hint, y)
    if length(z) < 1e-6:
        alt = UP if abs(y[2]) < 0.9 else X_AXIS
        z = project_on_plane(alt, y)
    z = normalize(z)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    M = u @ vt
    if np.linalg.det(M) < 0:
        u[:, -1] *= -1
        M = u @ vt
    return M


def yaw_of(R: np.ndarray) -> float:
    f = R[:, 0]
    return math.atan2(float(f[1]), float(f[0]))


# --------------------------------------------------------------------------- quaternions (w, x, y, z)

def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(c) for c in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quats_from_matrices(Rs: np.ndarray) -> np.ndarray:
    """Vectorised rotation-matrix (..., 3, 3) -> quaternion (..., 4) conversion."""
    Rs = np.asarray(Rs, dtype=float)
    shape = Rs.shape[:-2]
    m = Rs.reshape(-1, 3, 3)
    n = m.shape[0]
    q = np.empty((n, 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    # Robust branch selection (Shepperd's method).
    diag = np.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], axis=1)
    k = np.argmax(diag, axis=1)
    for case in range(4):
        idx = np.nonzero(k == case)[0]
        if idx.size == 0:
            continue
        mm = m[idx]
        if case == 0:
            s = np.sqrt(np.maximum(tr[idx] + 1.0, 1e-12)) * 2.0
            q[idx, 0] = 0.25 * s
            q[idx, 1] = (mm[:, 2, 1] - mm[:, 1, 2]) / s
            q[idx, 2] = (mm[:, 0, 2] - mm[:, 2, 0]) / s
            q[idx, 3] = (mm[:, 1, 0] - mm[:, 0, 1]) / s
        elif case == 1:
            s = np.sqrt(np.maximum(1.0 + mm[:, 0, 0] - mm[:, 1, 1] - mm[:, 2, 2], 1e-12)) * 2.0
            q[idx, 0] = (mm[:, 2, 1] - mm[:, 1, 2]) / s
            q[idx, 1] = 0.25 * s
            q[idx, 2] = (mm[:, 0, 1] + mm[:, 1, 0]) / s
            q[idx, 3] = (mm[:, 0, 2] + mm[:, 2, 0]) / s
        elif case == 2:
            s = np.sqrt(np.maximum(1.0 + mm[:, 1, 1] - mm[:, 0, 0] - mm[:, 2, 2], 1e-12)) * 2.0
            q[idx, 0] = (mm[:, 0, 2] - mm[:, 2, 0]) / s
            q[idx, 1] = (mm[:, 0, 1] + mm[:, 1, 0]) / s
            q[idx, 2] = 0.25 * s
            q[idx, 3] = (mm[:, 1, 2] + mm[:, 2, 1]) / s
        else:
            s = np.sqrt(np.maximum(1.0 + mm[:, 2, 2] - mm[:, 0, 0] - mm[:, 1, 1], 1e-12)) * 2.0
            q[idx, 0] = (mm[:, 1, 0] - mm[:, 0, 1]) / s
            q[idx, 1] = (mm[:, 0, 2] + mm[:, 2, 0]) / s
            q[idx, 2] = (mm[:, 1, 2] + mm[:, 2, 1]) / s
            q[idx, 3] = 0.25 * s
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(shape + (4,))


def matrices_from_quats(qs: np.ndarray) -> np.ndarray:
    qs = np.asarray(qs, dtype=float)
    w, x, y, z = qs[..., 0], qs[..., 1], qs[..., 2], qs[..., 3]
    R = np.empty(qs.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_continuity(qs: np.ndarray) -> np.ndarray:
    """Flip quaternion signs along axis 0 so consecutive samples stay in the same hemisphere."""
    out = np.array(qs, dtype=float, copy=True)
    for i in range(1, out.shape[0]):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] = -out[i]
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(a, b))
    if d < 0.0:
        b = -b
        d = -d
    if d > 0.9995:
        q = a + (b - a) * t
        return q / np.linalg.norm(q)
    th0 = math.acos(d)
    th = th0 * t
    s0 = math.sin(th0 - th) / math.sin(th0)
    s1 = math.sin(th) / math.sin(th0)
    return a * s0 + b * s1


# --------------------------------------------------------------------------- 4x4 transforms

def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def inv_transform(M: np.ndarray) -> np.ndarray:
    R = M[:3, :3]
    t = M[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def batch_inv_rigid(M: np.ndarray) -> np.ndarray:
    """Inverse of a batch (..., 4, 4) of rigid transforms."""
    R = M[..., :3, :3]
    t = M[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(M)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


# --------------------------------------------------------------------------- geometry

def convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain; returns hull in counter-clockwise order."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def support_margin(point: tuple[float, float], hull: list[tuple[float, float]]) -> float:
    """Signed distance of ``point`` to the boundary of a CCW convex polygon (positive = inside)."""
    n = len(hull)
    if n == 0:
        return -1e9
    if n == 1:
        return -math.hypot(point[0] - hull[0][0], point[1] - hull[0][1])
    if n == 2:
        (ax, ay), (bx, by) = hull
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < EPS else clamp(((point[0] - ax) * dx + (point[1] - ay) * dy) / L2, 0.0, 1.0)
        return -math.hypot(point[0] - (ax + t * dx), point[1] - (ay + t * dy))
    best = 1e9
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        L = math.hypot(ex, ey)
        if L < EPS:
            continue
        # Inward normal for CCW polygon is (-ey, ex).
        d = ((point[0] - ax) * (-ey) + (point[1] - ay) * ex) / L
        best = min(best, d)
    return best


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through points. Returns (centroid, unit normal pointing up)."""
    c = points.mean(axis=0)
    if points.shape[0] < 3:
        return c, UP.copy()
    A = points - c
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    n = normalize(n, UP)
    # Guard against degenerate (collinear) configurations.
    if n[2] < 0.3:
        n = UP.copy()
    return c, n
