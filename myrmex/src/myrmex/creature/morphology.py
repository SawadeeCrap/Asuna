"""Morphology: continuous blends between body configurations, and temporary appendages.

A morphology is a vector of shape descriptors, never a mesh: the body *moves toward* it.
Appendages are chains of low-mass nodes grown out of the same material (their volume is
taken from the core by the MassField) and retracted back into it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..util.mathutil import smootherstep

KEYS = ("spread", "elong", "flat", "lift", "lean", "budget", "spike", "legs", "fins", "noise")
MORPHS = {
    "COMPACT":      (0.55, 1.0, 1.0, 0.55, 0.00, 0.06, 0.0, 0.0, 0.0, 0.20),
    "ELONGATED":    (0.80, 2.0, 0.75, 0.60, 0.10, 0.12, 0.1, 0.0, 0.0, 0.25),
    "QUADRUPED":    (0.85, 1.5, 0.70, 0.95, 0.05, 0.10, 0.0, 1.0, 0.0, 0.20),
    "MULTI":        (0.70, 1.1, 0.90, 0.75, 0.00, 0.42, 0.2, 0.0, 0.0, 0.30),
    "AERODYNAMIC":  (0.90, 2.4, 0.45, 1.15, 0.25, 0.20, 0.0, 0.0, 1.0, 0.20),
    "DEFENSIVE":    (0.50, 0.9, 1.10, 0.40, -0.05, 0.25, 1.0, 0.0, 0.0, 0.15),
    "AGGRESSIVE":   (0.90, 1.6, 0.80, 0.75, 0.45, 0.30, 0.5, 0.0, 0.0, 0.30),
    "EXPLOSIVE":    (1.70, 1.2, 1.00, 0.85, 0.00, 0.35, 0.8, 0.0, 0.0, 0.80),
    "COLLAPSED":    (1.10, 1.3, 0.30, 0.15, 0.00, 0.02, 0.0, 0.0, 0.0, 0.35),
    "REORGANIZING": (0.75, 1.3, 0.80, 0.50, 0.00, 0.15, 0.2, 0.0, 0.0, 0.90),
}


def morph_vec(name: str) -> np.ndarray:
    return np.array(MORPHS[name], float)


class Morphology:
    def __init__(self, transition_time: float, rng):
        self.name = "COMPACT"
        self.vec = morph_vec("COMPACT")
        self.src = self.vec.copy()
        self.dst = self.vec.copy()
        self.t0 = -1e9
        self.T = transition_time
        self.rng = rng
        self.drift_phase = np.array([rng.uniform(0, 100) for _ in KEYS])

    def set_target(self, name: str, t: float, T: float | None = None) -> None:
        if name not in MORPHS:
            return
        self.src = self.vec.copy()
        self.dst = morph_vec(name)
        self.name = name
        self.t0 = t
        self.T = T or self.T

    def progress(self, t: float) -> float:
        return float(np.clip((t - self.t0) / max(self.T, 1e-3), 0.0, 1.0))

    def update(self, t: float, mutation: float, instability: float) -> np.ndarray:
        u = smootherstep(self.progress(t))
        base = self.src + (self.dst - self.src) * u
        # Mutation: slow coherent drift of every descriptor; instability: faster wobble.
        ph = self.drift_phase
        drift = 0.18 * mutation * np.sin(0.07 * t + ph) + 0.12 * instability * np.sin(0.9 * t + 1.7 * ph)
        self.vec = base * (1.0 + drift)
        return self.vec

    def __getitem__(self, key: str) -> float:
        return float(self.vec[KEYS.index(key)])


ARCHETYPES = {   # length, thickness, freq, zeta, wave, curl, taper, gravity, flat, tip
    "TENDRIL": (0.9, 0.075, 2.2, 0.18, 0.55, 0.30, 0.85, 0.3, 0.0, 1.0),
    "SPIKE":   (0.45, 0.070, 6.0, 0.60, 0.02, 0.00, 0.95, 0.0, 0.0, 1.0),
    "LIMB":    (0.85, 0.095, 3.5, 0.45, 0.05, 0.15, 0.55, 1.0, 0.0, 1.0),
    "SENSOR":  (0.65, 0.035, 4.0, 0.30, 0.30, 0.10, 0.30, 0.0, 0.0, 2.2),
    "BLADE":   (0.60, 0.090, 5.0, 0.50, 0.05, 0.20, 0.90, 0.0, 1.0, 1.0),
    "WHIP":    (1.20, 0.050, 1.8, 0.12, 0.90, 0.40, 0.90, 0.4, 0.0, 1.0),
    "CABLE":   (1.00, 0.045, 1.2, 0.35, 0.10, 0.00, 0.30, 1.5, 0.0, 1.0),
    "ARM":     (0.90, 0.090, 2.8, 0.35, 0.20, 0.50, 0.50, 0.3, 0.0, 1.0),
    "FIN":     (0.55, 0.100, 3.0, 0.40, 0.35, 0.10, 0.80, 0.0, 1.0, 1.0),
}
ARCH_KEYS = ("length", "thick", "freq", "zeta", "wave", "curl", "taper", "gravity", "flat", "tip")


@dataclass
class Appendage:
    slot: int
    nodes: np.ndarray
    archetype: str = "TENDRIL"
    origin: int = 0
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    activation: float = 0.0
    growth: float = 0.0
    growth_v: float = 0.0
    phase: float = 0.0
    born: float = 0.0
    scale: float = 1.0

    def spec(self, key: str) -> float:
        return ARCHETYPES[self.archetype][ARCH_KEYS.index(key)]


class AppendageSystem:
    def __init__(self, pool: np.ndarray, rng):
        self.items = [Appendage(i, pool[i]) for i in range(len(pool))]
        self.rng = rng

    @property
    def active(self) -> list[Appendage]:
        return [a for a in self.items if a.activation > 0.0 or a.growth > 0.02]

    def spawn(self, archetype: str, origin: int, direction: np.ndarray, t: float, scale: float = 1.0) -> bool:
        free = [a for a in self.items if a.activation == 0.0 and a.growth < 0.05]
        if not free or archetype not in ARCHETYPES:
            return False
        a = free[0]
        a.archetype, a.origin, a.activation, a.born, a.scale = archetype, origin, 1.0, t, scale
        d = np.asarray(direction, float)
        a.direction = d / max(np.linalg.norm(d), 1e-9)
        a.phase = self.rng.uniform(0.0, 2 * math.pi)
        return True

    def retract(self, a: Appendage) -> None:
        a.activation = 0.0

    def retract_all(self) -> None:
        for a in self.items:
            a.activation = 0.0

    def update(self, dt: float, fluidity: float) -> None:
        # Growth is itself a spring: emerging material overshoots slightly, retraction is viscous.
        for a in self.items:
            f = 0.55 + 0.6 * fluidity
            w = 2 * math.pi * f
            a.growth_v += dt * (w * w * (a.activation - a.growth) - 2 * 0.62 * w * a.growth_v)
            a.growth = min(1.25, max(0.0, a.growth + dt * a.growth_v))

    def volumes(self, segments: int, size: float) -> np.ndarray:
        out = np.zeros(len(self.items) * segments)
        for a in self.items:
            if a.growth <= 0.01:
                continue
            L = a.spec("length") * a.scale * size * a.growth
            r = a.spec("thick") * size
            for i in range(segments):
                taper = 1.0 - a.spec("taper") * i / segments
                out[a.slot * segments + i] = math.pi * (r * taper) ** 2 * (L / segments) / (size ** 3) * 1.4
        return out
