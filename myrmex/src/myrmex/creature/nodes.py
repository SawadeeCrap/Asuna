"""Internal control network: point masses pulled by springs toward targets (inertia, overshoot, settling)."""
from __future__ import annotations

import numpy as np

CORE, PRIMARY, SECONDARY, APPENDAGE = 0, 1, 2, 3


class NodeSystem:
    """Semi-implicit Euler on per-node second-order springs: k = m(2πf)², c = 2ζm(2πf).

    Every node pursues its target with its own frequency / damping, so input becomes
    force -> acceleration -> motion -> overshoot -> damping -> rest, never a teleport.
    """

    def __init__(self, n: int):
        self.n = n
        self.pos = np.zeros((n, 3))
        self.vel = np.zeros((n, 3))
        self.target = np.zeros((n, 3))
        self.ext = np.zeros((n, 3))          # external forces this step (impulses / N)
        self.mass = np.ones(n)
        self.freq = np.ones(n)
        self.zeta = np.full(n, 0.7)
        self.radius = np.zeros(n)
        self.gravity_w = np.zeros(n)
        self.kind = np.zeros(n, np.int8)
        self.anchor = np.full(n, -1, np.int16)

    def impulse(self, idx, dv: np.ndarray) -> None:
        self.vel[idx] += dv

    def step(self, dt: float, gravity: float, max_speed: float, repel: np.ndarray | None = None,
             repel_k: float = 0.0, ground: bool = True) -> None:
        w = 2.0 * np.pi * self.freq
        acc = (w * w)[:, None] * (self.target - self.pos) - (2.0 * self.zeta * w)[:, None] * self.vel
        acc += self.ext / self.mass[:, None]
        acc[:, 2] -= gravity * self.gravity_w
        if repel is not None and len(repel) > 1 and repel_k > 0:
            P = self.pos[repel]
            d = P[:, None, :] - P[None, :, :]
            dist = np.maximum(np.linalg.norm(d, axis=2), 1e-3)
            r = self.radius[repel]
            overlap = np.maximum(0.0, 0.7 * (r[:, None] + r[None, :]) - dist)
            np.fill_diagonal(overlap, 0.0)
            push = (overlap / dist)[:, :, None] * d
            acc[repel] += repel_k * push.sum(axis=1)
        self.vel += acc * dt
        sp = np.linalg.norm(self.vel, axis=1)
        fast = sp > max_speed
        if fast.any():
            self.vel[fast] *= (max_speed / sp[fast])[:, None]
        self.pos += self.vel * dt
        if ground:
            floor = 0.35 * self.radius
            below = self.pos[:, 2] < floor
            if below.any():
                self.pos[below, 2] = floor[below]
                self.vel[below, 2] = np.abs(self.vel[below, 2]) * 0.15
                self.vel[below, :2] *= 0.92                   # friction: contacts grip
        self.ext[:] = 0.0

    def center_of_mass(self) -> np.ndarray:
        m = self.mass * (self.radius > 1e-4)
        return (self.pos * m[:, None]).sum(axis=0) / max(m.sum(), 1e-9)
