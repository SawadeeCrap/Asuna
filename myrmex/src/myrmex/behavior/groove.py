"""Soft entrainment to the beat (the difference between "dancing" and "being clocked").

A :class:`GrooveOscillator` has its own phase and natural rate.  It is
*coupled* to the musical beat (Kuramoto coupling) with a strength that grows
with rhythmic regularity and energy.  Weak coupling = the body drifts and
catches the beat now and then; strong coupling = it locks, with a personal
timing offset (laid back or eager) that belongs to the character's seed.
The subdivision (half-time / beat / double-time) changes with energy using
hysteresis, never on every bar.
"""
from __future__ import annotations

import math

from ..util.rng import Rng


class GrooveOscillator:
    def __init__(self, rng: Rng, lag_beats: float | None = None):
        self.phase = rng.random()                 # in cycles
        self.mult = 1.0                           # cycles per beat
        self.lag = lag_beats if lag_beats is not None else rng.uniform(-0.07, 0.03)
        self.drift = rng.uniform(-0.03, 0.03)     # natural-rate mismatch
        self.locked = 0.0
        self._rng = rng

    def update(self, dt: float, beat: float, bpm: float, coupling: float, energy: float) -> float:
        # Subdivision with hysteresis.
        if self.mult == 1.0 and energy < 0.25:
            self.mult = 0.5
        elif self.mult == 0.5 and energy > 0.4:
            self.mult = 1.0
        elif self.mult == 1.0 and energy > 0.92 and bpm < 105:
            self.mult = 2.0
        elif self.mult == 2.0 and energy < 0.75:
            self.mult = 1.0
        bps = bpm / 60.0
        target = beat * self.mult - self.lag * self.mult
        err = math.sin(2.0 * math.pi * (target - self.phase))
        k = 3.5 * coupling
        self.phase += dt * (self.mult * bps * (1.0 + self.drift) + k * err / (2.0 * math.pi) * bps * 2.0)
        self.locked += ((1.0 - abs(err)) * coupling - self.locked) * min(1.0, dt * 1.5)
        return self.phase % 1.0

    def bounce(self, sharpness: float = 0.5) -> float:
        """0..1, maximal on the (oscillator's) beat; sharper with ``sharpness``."""
        c = 0.5 * (1.0 + math.cos(2.0 * math.pi * (self.phase % 1.0)))
        return c ** (1.0 + 2.0 * sharpness)

    def sway(self) -> float:
        """-1..1 side-to-side at half the oscillator rate."""
        return math.sin(math.pi * self.phase)
