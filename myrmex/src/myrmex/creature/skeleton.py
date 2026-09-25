"""The bony skeleton of the material: a minimum spanning tree through each body's nodes.

Replaces the strut lattice (random neighbour links that looked like flying shards).  A spanning tree
has no crossings and no shortcuts through empty space - read along its branches it is a spine with
ribs and limbs.  Every tree edge owns a stable slot that *grows* (strength 0 -> ossification) when the
edge appears and *dissolves* when it goes, so the links mutate smoothly instead of popping.  The
renderer turns every slot into an articulated bone link (see blender/myrmex_blender/creature.py).
"""
from __future__ import annotations

import math

import numpy as np


def mst_edges(x: np.ndarray, owner: np.ndarray, max_len: float) -> list[tuple[int, int]]:
    """Prim's minimum spanning tree inside every body (edges longer than ``max_len`` are left out)."""
    out: list[tuple[int, int]] = []
    for g in np.unique(owner):
        idx = np.nonzero(owner == g)[0]
        m = len(idx)
        if m < 2:
            continue
        P = x[idx]
        d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(2))
        in_tree = np.zeros(m, bool)
        in_tree[0] = True
        best = d[0].copy()
        parent = np.zeros(m, int)
        best[0] = np.inf
        for _ in range(m - 1):
            j = int(np.argmin(np.where(in_tree, np.inf, best)))
            if not np.isfinite(best[j]):
                break
            in_tree[j] = True
            if best[j] <= max_len:
                a, b = int(idx[parent[j]]), int(idx[j])
                out.append((min(a, b), max(a, b)))
            closer = (d[j] < best) & ~in_tree
            best[closer] = d[j][closer]
            parent[closer] = j
    return out


class Skeleton:
    def __init__(self, max_links: int, rate: float = 15.0, grow: float = 0.35, fade: float = 0.5):
        self.links = np.full((max_links, 3), -1.0)
        self.links[:, 2] = 0.0
        self.target = np.zeros(max_links)
        self.slot: dict[tuple[int, int], int] = {}
        self.period = 1.0 / rate
        self.acc = self.period
        self.grow, self.fade = grow, fade

    def update(self, x: np.ndarray, owner: np.ndarray, oss: np.ndarray, dt: float, max_len: float) -> np.ndarray:
        """``oss``: ossification per node (0 liquid .. 1 bone).  Returns (max_links, 3): i, j, strength."""
        self.acc += dt
        L = self.links
        if self.acc >= self.period:
            self.acc = 0.0
            want = {e: float(min(oss[e[0]], oss[e[1]])) for e in mst_edges(x, owner, max_len)}
            for e, s in self.slot.items():
                self.target[s] = want.get(e, 0.0)
            free = [s for s in range(len(L)) if L[s, 0] < 0]
            for e, w in want.items():
                if e in self.slot or not free or w < 0.05:
                    continue
                s = free.pop(0)
                self.slot[e] = s
                L[s, 0], L[s, 1], L[s, 2], self.target[s] = e[0], e[1], 0.0, w
        used = L[:, 0] >= 0
        cur = L[:, 2]
        tau = np.where(self.target > cur, self.grow, self.fade)
        cur[used] += (self.target[used] - cur[used]) * np.minimum(1.0, dt / tau[used])
        gone = used & (self.target <= 0.0) & (cur < 0.01)
        if gone.any():
            for s in np.nonzero(gone)[0]:
                self.slot.pop((int(L[s, 0]), int(L[s, 1])), None)
            L[gone, 0:2] = -1.0
            L[gone, 2] = 0.0
        return L.copy()


def phase_of(slot: int) -> float:
    """A stable per-link mutation phase (golden-ratio spread)."""
    return 2 * math.pi * ((slot * 0.6180339887) % 1.0)


__all__ = ["Skeleton", "mst_edges", "phase_of"]
