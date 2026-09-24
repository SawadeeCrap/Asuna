"""Attention and gaze: saccades, fixations and orienting to sound sources.

Real eyes/heads do not glide continuously: they *jump* (saccade) and *hold*
(fixation).  The gaze target changes in discrete jumps whose rate depends on
curiosity and agitation; the motor's head springs turn the jumps into
natural head turns with slight overshoot.

Each semantic instrument group is given a virtual direction around the
creature (seeded), so salient events can make it briefly turn toward "where
the snare is" – a subtle cue that it is *listening*, not being driven.
"""
from __future__ import annotations

import math

import numpy as np

from ..util.rng import Rng


class Gaze:
    def __init__(self, rng: Rng, eye_height: float, groups: tuple[str, ...]):
        self.rng = rng
        self.eye_h = eye_height
        self.target_local = np.array([3.0, 0.0, 0.0])     # (forward, left, up offset) relative to body
        self.next_saccade = 0.5
        self.hold_until = 0.0
        self.sources = {}
        for g in groups:
            ang = rng.uniform(-2.4, 2.4)
            elev = rng.uniform(-0.25, 0.35)
            self.sources[g] = (ang, elev)
        self.weight = 1.0

    def orient(self, t: float, group: str | None, strength: float) -> None:
        if group not in self.sources:
            return
        ang, elev = self.sources[group]
        ang = max(-1.3, min(1.3, ang))
        d = 2.5
        self.target_local = np.array([d * math.cos(ang), d * math.sin(ang), d * math.tan(elev)])
        self.hold_until = t + 0.35 + 0.6 * strength
        self.next_saccade = self.hold_until + self.rng.uniform(0.3, 1.2)

    def look_local(self, t: float, fwd: float, left: float, up: float, hold: float = 0.8) -> None:
        self.target_local = np.array([fwd, left, up])
        self.hold_until = t + hold
        self.next_saccade = t + hold + self.rng.uniform(0.2, 0.8)

    def update(self, t: float, curiosity: float, agitation: float, walking: bool, pitch: float,
               wander_scale: float = 1.0) -> None:
        if t < self.hold_until or t < self.next_saccade:
            return
        r = self.rng
        if walking and r.chance(0.65):
            # Mostly look where we go, with small offsets.
            self.target_local = np.array([4.0, r.normal(0.0, 0.5), r.normal(-0.3, 0.25)])
        else:
            spread = (0.5 + 1.1 * curiosity + 0.6 * agitation) * wander_scale
            ang = r.normal(0.0, 0.55 * spread)
            elev = r.normal(-0.05 + 0.35 * (pitch - 0.5), 0.18 * spread)
            d = r.uniform(1.5, 5.0)
            self.target_local = np.array([d * math.cos(ang), d * math.sin(ang), d * math.tan(elev)])
        hold = r.uniform(0.4, 2.8) * (1.4 - 0.8 * agitation) * (0.8 + 0.4 * (1.0 - curiosity))
        self.hold_until = t + hold
        self.next_saccade = self.hold_until

    def world_target(self, pos: np.ndarray, fwd: np.ndarray, left: np.ndarray) -> np.ndarray:
        f, l, u = self.target_local
        return pos + fwd * f + left * l + np.array([0.0, 0.0, self.eye_h + u])
