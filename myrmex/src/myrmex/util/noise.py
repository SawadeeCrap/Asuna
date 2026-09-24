"""Correlated procedural variation.

Independent white noise on every joint reads as twitching.  Organic motion
comes from a *few* slow latent processes that many body parts share, each
part seeing them with its own weight and time delay.  This module provides:

* :class:`Noise1D` – smooth, stateless 1-D gradient noise (fBm), sampled at
  any time, so reading it "in the past" (a delayed copy) is free.
* :class:`LatentField` – K latent noise channels mixed into J named outputs
  with per-output delays: the head sees the body's drift 80 ms later, the
  abdomen 250 ms later, and so on.
* :class:`OrnsteinUhlenbeck` – mean-reverting random walk for stateful drift.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .rng import Rng, stable_hash64

_TABLE = 4096


def _fade(t: np.ndarray | float):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


class Noise1D:
    """Stateless fractal gradient noise. Output roughly in [-1, 1]."""

    def __init__(self, seed: int, frequency: float = 0.5, octaves: int = 3,
                 lacunarity: float = 2.137, gain: float = 0.5):
        rng = Rng(stable_hash64("noise1d", seed))
        self.frequency = float(frequency)
        self.octaves = max(1, int(octaves))
        self.lacunarity = lacunarity
        self.gain = gain
        self._grad = np.array([[rng.uniform(-1.0, 1.0) for _ in range(_TABLE)]
                               for _ in range(self.octaves)])
        self._offset = np.array([rng.uniform(0.0, 1000.0) for _ in range(self.octaves)])
        amp = 1.0
        norm = 0.0
        for _ in range(self.octaves):
            norm += amp
            amp *= gain
        # Perlin 1-D output peaks near +-0.5; rescale to about +-1.
        self._norm = 2.0 / norm

    def sample(self, t: float) -> float:
        total = 0.0
        amp = 1.0
        freq = self.frequency
        for o in range(self.octaves):
            x = t * freq + self._offset[o]
            i = math.floor(x)
            f = x - i
            g0 = self._grad[o, i % _TABLE]
            g1 = self._grad[o, (i + 1) % _TABLE]
            u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
            total += amp * (g0 * f + (g1 * (f - 1.0) - g0 * f) * u)
            amp *= self.gain
            freq *= self.lacunarity
        return total * self._norm

    def sample_many(self, ts: np.ndarray) -> np.ndarray:
        ts = np.asarray(ts, dtype=float)
        total = np.zeros_like(ts)
        amp = 1.0
        freq = self.frequency
        for o in range(self.octaves):
            x = ts * freq + self._offset[o]
            i = np.floor(x)
            f = x - i
            ii = i.astype(np.int64)
            g0 = self._grad[o, ii % _TABLE]
            g1 = self._grad[o, (ii + 1) % _TABLE]
            u = _fade(f)
            total += amp * (g0 * f + (g1 * (f - 1.0) - g0 * f) * u)
            amp *= self.gain
            freq *= self.lacunarity
        return total * self._norm


class LatentField:
    """A handful of shared latent processes mixed into many named outputs.

    ``outputs`` maps an output name to ``(weights, delay_seconds)`` where
    ``weights`` has one entry per latent channel.  Missing weights are drawn
    from the seed so that every seed yields a differently "wired" body while
    remaining coherent.
    """

    def __init__(self, seed: int, frequencies: Sequence[float], outputs: Sequence[str],
                 delays: dict[str, float] | None = None,
                 weights: dict[str, Sequence[float]] | None = None,
                 octaves: int = 2):
        self.channels = [Noise1D(stable_hash64("latent", seed, k), f, octaves=octaves)
                         for k, f in enumerate(frequencies)]
        self.names = list(outputs)
        self.index = {n: i for i, n in enumerate(self.names)}
        rng = Rng(stable_hash64("latent-mix", seed))
        K = len(self.channels)
        J = len(self.names)
        W = np.zeros((J, K))
        D = np.zeros(J)
        for j, name in enumerate(self.names):
            if weights and name in weights:
                w = np.asarray(weights[name], dtype=float)
                W[j, : len(w)] = w[:K]
            else:
                # Structured random wiring: one dominant channel plus minor leakage.
                dominant = rng.randint(0, K - 1)
                for k in range(K):
                    W[j, k] = rng.uniform(-0.35, 0.35)
                W[j, dominant] = rng.choice([-1.0, 1.0]) * rng.uniform(0.7, 1.0)
            n = float(np.linalg.norm(W[j]))
            if n > 1e-9:
                W[j] /= n
            D[j] = (delays or {}).get(name, rng.uniform(0.0, 0.15))
        self.W = W
        self.D = D

    def sample(self, t: float) -> dict[str, float]:
        vals = self.sample_array(t)
        return {n: float(vals[i]) for i, n in enumerate(self.names)}

    def sample_array(self, t: float) -> np.ndarray:
        ts = t - self.D
        L = np.stack([ch.sample_many(ts) for ch in self.channels], axis=1)  # (J, K)
        return np.einsum("jk,jk->j", self.W, L)


class OrnsteinUhlenbeck:
    """Mean-reverting random walk; deterministic for a fixed dt sequence."""

    def __init__(self, rng: Rng, theta: float = 1.0, sigma: float = 0.3, mu: float = 0.0,
                 x0: float = 0.0):
        self.rng = rng
        self.theta = theta
        self.sigma = sigma
        self.mu = mu
        self.x = x0

    def step(self, dt: float) -> float:
        self.x += self.theta * (self.mu - self.x) * dt + self.sigma * math.sqrt(dt) * self.rng.normal()
        return self.x
