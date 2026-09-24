"""Novelty and habituation: the anti-repetition machinery.

* :class:`RecencyTracker` remembers what the creature did recently (behaviours,
  strategies, reaction types, gestures, headings) and returns a multiplicative
  penalty for repeating it.  It shapes *weights* – it never forbids – so the
  music remains the main driver and variety is gentle and weighted.
* :class:`Habituation` models the biological tendency to respond less to a
  repeated identical stimulus and to respond strongly again when something
  new happens (dishabituation).
"""
from __future__ import annotations

import math
from collections import deque


class RecencyTracker:
    def __init__(self, half_life: float = 20.0, strength: float = 1.0, maxlen: int = 64):
        self.half_life = half_life
        self.strength = strength
        self.items: deque[tuple[float, str]] = deque(maxlen=maxlen)

    def add(self, t: float, key: str) -> None:
        self.items.append((t, key))

    def score(self, t: float, key: str) -> float:
        """Recency-weighted count of ``key`` (1.0 for 'just now')."""
        s = 0.0
        for ti, k in self.items:
            if k == key:
                s += 0.5 ** ((t - ti) / self.half_life)
        return s

    def penalty(self, t: float, key: str, novelty: float = 1.0) -> float:
        return math.exp(-self.strength * novelty * self.score(t, key))

    def last(self) -> str | None:
        return self.items[-1][1] if self.items else None

    def streak(self, key: str) -> int:
        n = 0
        for _, k in reversed(self.items):
            if k != key:
                break
            n += 1
        return n


class Habituation:
    def __init__(self, drop: float = 0.14, recover_tau: float = 7.0, floor: float = 0.12):
        self.gain: dict[str, float] = {}
        self.drop = drop
        self.tau = recover_tau
        self.floor = floor

    def get(self, key: str) -> float:
        return self.gain.get(key, 1.0)

    def stimulate(self, key: str, surprise: float = 0.0) -> float:
        g = self.get(key)
        if surprise > 0.45:
            g = max(g, 0.75 + 0.25 * surprise)          # dishabituation
        out = g
        g = max(self.floor, g - self.drop * (1.0 - surprise))
        self.gain[key] = g
        return out

    def update(self, dt: float) -> None:
        a = 1.0 - math.exp(-dt / self.tau)
        for k, g in self.gain.items():
            self.gain[k] = g + (1.0 - g) * a
