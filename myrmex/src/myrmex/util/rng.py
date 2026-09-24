"""Deterministic, platform-independent random streams.

Every subsystem draws from its *own named stream* derived from the master
seed.  Changing how often one subsystem consumes random numbers therefore
never reshuffles the others: tweaking a mapping does not change which gait
variation a leg gets, and re-rendering with the same seed reproduces the same
performance bit-for-bit (the generator is pure Python, so it does not depend
on the numpy version that ships with a particular Blender build).
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

MASK64 = (1 << 64) - 1


def stable_hash64(*parts: object) -> int:
    """Stable 64-bit hash of arbitrary printable parts (unlike built-in ``hash``)."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little")


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def hash_unit(seed: int, *coords: int) -> float:
    """Hash integer lattice coordinates to a float in [0, 1)."""
    x = seed & MASK64
    for c in coords:
        x = splitmix64(x ^ ((c * 0xD6E8FEB86659FD93) & MASK64))
    return (splitmix64(x) >> 11) * (1.0 / (1 << 53))


class Rng:
    """Small xorshift-style generator seeded through SplitMix64."""

    __slots__ = ("_s", "_gauss_spare")

    def __init__(self, seed: int):
        self._s = splitmix64(seed & MASK64) or 0x1234567
        self._gauss_spare: float | None = None

    def _next(self) -> int:
        # xorshift64* – fast and good enough for animation variation.
        x = self._s
        x ^= (x >> 12)
        x ^= (x << 25) & MASK64
        x ^= (x >> 27)
        self._s = x
        return (x * 0x2545F4914F6CDD1D) & MASK64

    def random(self) -> float:
        return (self._next() >> 11) * (1.0 / (1 << 53))

    def uniform(self, lo: float = 0.0, hi: float = 1.0) -> float:
        return lo + (hi - lo) * self.random()

    def randint(self, lo: int, hi: int) -> int:
        """Integer in [lo, hi] inclusive."""
        return lo + int(self.random() * (hi - lo + 1)) if hi >= lo else lo

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        if self._gauss_spare is not None:
            z = self._gauss_spare
            self._gauss_spare = None
            return mu + sigma * z
        while True:
            u = 2.0 * self.random() - 1.0
            v = 2.0 * self.random() - 1.0
            s = u * u + v * v
            if 0.0 < s < 1.0:
                break
        f = math.sqrt(-2.0 * math.log(s) / s)
        self._gauss_spare = v * f
        return mu + sigma * u * f

    def chance(self, p: float) -> bool:
        return self.random() < p

    def choice(self, items: Sequence):
        return items[min(int(self.random() * len(items)), len(items) - 1)]

    def weighted_index(self, weights: Sequence[float]) -> int:
        total = sum(max(0.0, w) for w in weights)
        if total <= 0.0:
            return self.randint(0, len(weights) - 1)
        r = self.random() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc += max(0.0, w)
            if r < acc:
                return i
        return len(weights) - 1

    def lognormal_factor(self, spread: float) -> float:
        """Multiplicative jitter centred on 1 (``spread`` ~ relative std-dev)."""
        return math.exp(self.normal(0.0, spread))

    def beta_like(self, a: float, b: float) -> float:
        """Cheap Kumaraswamy draw – a smooth skewed distribution on [0, 1]."""
        u = min(max(self.random(), 1e-12), 1.0 - 1e-12)
        return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)


class RngStreams:
    """Factory of named, independent random streams derived from one seed."""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._cache: dict[str, Rng] = {}

    def stream(self, name: str) -> Rng:
        rng = self._cache.get(name)
        if rng is None:
            rng = Rng(stable_hash64(self.seed, name))
            self._cache[name] = rng
        return rng

    def fresh(self, name: str) -> Rng:
        """A new generator (not cached) – restarting the same sequence each call."""
        return Rng(stable_hash64(self.seed, name))

    def subseed(self, name: str) -> int:
        return stable_hash64(self.seed, "subseed", name) & 0x7FFFFFFF
