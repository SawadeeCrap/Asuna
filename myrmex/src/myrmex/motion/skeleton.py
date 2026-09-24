"""Skeleton rest data and pose evaluation in *rest-world delta* form.

Every bone's posed world matrix is written as

    M_b(t) = D_b(t) @ M_b(rest)

where ``D_b`` is a rigid "delta" transform in world space.  Forward
kinematics composes deltas down the hierarchy:

    D_b = D_parent @ Rot(Q_b, about the bone's rest head)

with ``Q_b`` a rotation expressed in *world-at-rest axes*.  Because rotations
are described in the character's rest frame (forward/left/up) rather than in
bone-local axes, the motion code never depends on how a particular rig
rolled its bones – which is what makes one motion engine drive any
auto-rigged or hand-made skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rig.rigdesc import RigDescription
from ..util.mathutil import frame_from_y_z, normalize


@dataclass
class SkeletonRest:
    names: list[str]
    parents: np.ndarray            # (B,) parent index or -1
    heads: np.ndarray              # (B, 3) rest head positions (world == armature space)
    tails: np.ndarray              # (B, 3)
    rest: np.ndarray               # (B, 4, 4) rest world matrices (y along bone)
    order: list[int]               # parents before children
    index: dict[str, int]
    forward: np.ndarray
    left: np.ndarray
    up: np.ndarray

    @classmethod
    def from_rig(cls, rd: RigDescription) -> "SkeletonRest":
        names = [b.name for b in rd.bones]
        index = {n: i for i, n in enumerate(names)}
        parents = np.array([index[b.parent] if b.parent else -1 for b in rd.bones], dtype=np.int64)
        heads = np.array([b.head for b in rd.bones], dtype=float)
        tails = np.array([b.tail for b in rd.bones], dtype=float)
        rest = np.zeros((len(names), 4, 4))
        for i, b in enumerate(rd.bones):
            R = frame_from_y_z(tails[i] - heads[i], np.asarray(b.roll_ref, dtype=float))
            rest[i, :3, :3] = R
            rest[i, :3, 3] = heads[i]
            rest[i, 3, 3] = 1.0
        order: list[int] = []
        seen: set[int] = set()

        def visit(i: int) -> None:
            if i in seen:
                return
            if parents[i] >= 0:
                visit(int(parents[i]))
            seen.add(i)
            order.append(i)

        for i in range(len(names)):
            visit(i)
        fwd = normalize(np.asarray(rd.forward, dtype=float))
        up = normalize(np.asarray(rd.up, dtype=float))
        left = normalize(np.cross(up, fwd))
        return cls(names, parents, heads, tails, rest, order, index, fwd, left, up)

    def length(self, name: str) -> float:
        i = self.index[name]
        return float(np.linalg.norm(self.tails[i] - self.heads[i]))

    def head(self, name: str) -> np.ndarray:
        return self.heads[self.index[name]].copy()

    def tail(self, name: str) -> np.ndarray:
        return self.tails[self.index[name]].copy()


class Pose:
    """World-space deltas for all bones, evaluated by forward kinematics."""

    def __init__(self, sk: SkeletonRest):
        self.sk = sk
        n = len(sk.names)
        self.local_rot = np.tile(np.eye(3), (n, 1, 1))      # Q_b in rest-world axes
        self.override = [None] * n                           # absolute world deltas (IK results)
        self.root_delta = np.eye(4)                          # delta applied to parentless bones
        self.delta = np.tile(np.eye(4), (n, 1, 1))
        self._local = np.tile(np.eye(4), (n, 1, 1))
        self._order = [int(i) for i in sk.order]
        self._parents = [int(p) for p in sk.parents]

    def reset(self) -> None:
        self.local_rot[:] = np.eye(3)
        self.override = [None] * len(self.sk.names)
        self.root_delta = np.eye(4)

    def set_rot(self, name: str, R: np.ndarray) -> None:
        self.local_rot[self.sk.index[name]] = R

    def rotate(self, name: str, R: np.ndarray) -> None:
        """Pre-multiply an additional rest-world-axes rotation onto a bone."""
        i = self.sk.index[name]
        self.local_rot[i] = R @ self.local_rot[i]

    def set_world_delta(self, name: str, D: np.ndarray) -> None:
        self.override[self.sk.index[name]] = D

    def solve(self) -> np.ndarray:
        sk = self.sk
        Q = self.local_rot
        # Local deltas for all bones at once: rotate about the bone head.
        local = self._local
        local[:, :3, :3] = Q
        local[:, :3, 3] = sk.heads - np.einsum("bij,bj->bi", Q, sk.heads)
        delta = self.delta
        for i in self._order:
            ov = self.override[i]
            if ov is not None:
                delta[i] = ov
                continue
            p = self._parents[i]
            np.matmul(self.root_delta if p < 0 else delta[p], local[i], out=delta[i])
        return delta

    def world_matrices(self) -> np.ndarray:
        return np.einsum("bij,bjk->bik", self.delta, self.sk.rest)

    def world_head(self, name: str) -> np.ndarray:
        i = self.sk.index[name]
        D = self.delta[i]
        return D[:3, :3] @ self.sk.heads[i] + D[:3, 3]

    def world_tail(self, name: str) -> np.ndarray:
        i = self.sk.index[name]
        D = self.delta[i]
        return D[:3, :3] @ self.sk.tails[i] + D[:3, 3]

    def apply(self, name: str, p_rest: np.ndarray) -> np.ndarray:
        """Where a rest-space point rigidly attached to ``name`` currently is."""
        D = self.delta[self.sk.index[name]]
        return D[:3, :3] @ p_rest + D[:3, 3]

    def parent_delta(self, name: str) -> np.ndarray:
        p = int(self.sk.parents[self.sk.index[name]])
        return self.root_delta if p < 0 else self.delta[p]


def delta_aligning(rest_head: np.ndarray, rest_dir: np.ndarray, rest_up: np.ndarray,
                   new_head: np.ndarray, new_dir: np.ndarray, new_up: np.ndarray) -> np.ndarray:
    """Rigid world delta mapping a rest bone frame (dir, up-hint) onto a new one."""
    F0 = frame_from_y_z(rest_dir, rest_up)
    F1 = frame_from_y_z(new_dir, new_up)
    R = F1 @ F0.T
    D = np.eye(4)
    D[:3, :3] = R
    D[:3, 3] = new_head - R @ rest_head
    return D
