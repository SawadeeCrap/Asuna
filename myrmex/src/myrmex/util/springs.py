"""Physically flavoured filters: the "muscle tone" of every procedural channel.

Nothing in Myrmex jumps straight to a target.  Targets are *pursued* by
second-order systems whose three parameters read like animation vocabulary:

* ``f``   natural frequency in Hz – how quickly the channel responds,
* ``zeta`` damping – < 1 wobbles/overshoots, 1 is critically damped,
* ``r``   initial response – r < 0 winds up *against* the change first
  (anticipation), r > 1 overshoots on the way (snap), r = 0 eases in.

(After t3ssel8r, "Giving Personality to Procedural Animations using Math",
2022, with the semi-implicit stability clamp.)  Impulses are injected as
velocity kicks, so an event produces compression -> release -> overshoot ->
settle instead of a teleport.
"""
from __future__ import annotations

import math

import numpy as np


class SecondOrder:
    """Second-order dynamics tracking a (scalar or vector) target."""

    __slots__ = ("k1", "k2", "k3", "y", "yd", "xp", "f", "zeta", "r")

    def __init__(self, f: float, zeta: float, r: float, x0=0.0):
        self.y = np.array(x0, dtype=float) if np.ndim(x0) else float(x0)
        self.yd = np.zeros_like(self.y) if np.ndim(x0) else 0.0
        self.xp = np.array(x0, dtype=float) if np.ndim(x0) else float(x0)
        self.set_params(f, zeta, r)

    def set_params(self, f: float, zeta: float, r: float) -> None:
        f = max(1e-3, float(f))
        self.f, self.zeta, self.r = f, zeta, r
        w = 2.0 * math.pi * f
        self.k1 = zeta / (math.pi * f)
        self.k2 = 1.0 / (w * w)
        self.k3 = r * zeta / w

    def reset(self, x) -> None:
        if np.ndim(x):
            self.y = np.array(x, dtype=float)
            self.yd = np.zeros_like(self.y)
            self.xp = np.array(x, dtype=float)
        else:
            self.y = float(x)
            self.yd = 0.0
            self.xp = float(x)

    def update(self, dt: float, x, xd=None):
        if dt <= 0.0:
            return self.y
        if xd is None:
            xd = (x - self.xp) / dt
        self.xp = x.copy() if isinstance(x, np.ndarray) else x
        k2 = max(self.k2, dt * dt / 2.0 + dt * self.k1 / 2.0, dt * self.k1)
        self.y = self.y + dt * self.yd
        self.yd = self.yd + dt * (x + self.k3 * xd - self.y - self.k1 * self.yd) / k2
        return self.y

    def kick(self, dv) -> None:
        """Inject an instantaneous velocity change (an impulse divided by mass)."""
        self.yd = self.yd + dv


class Spring:
    """Critically damped spring with exact integration (D. Holden, 2021)."""

    __slots__ = ("x", "v", "halflife")

    def __init__(self, halflife: float, x0=0.0):
        self.halflife = halflife
        self.x = np.array(x0, dtype=float) if np.ndim(x0) else float(x0)
        self.v = np.zeros_like(self.x) if np.ndim(x0) else 0.0

    def update(self, dt: float, goal, halflife: float | None = None):
        hl = self.halflife if halflife is None else halflife
        y = (4.0 * 0.69314718056) / (hl + 1e-5) / 2.0
        j0 = self.x - goal
        j1 = self.v + j0 * y
        eydt = math.exp(-y * dt)
        self.x = eydt * (j0 + j1 * dt) + goal
        self.v = eydt * (self.v - j1 * y * dt)
        return self.x

    def kick(self, dv) -> None:
        self.v = self.v + dv


class Envelope:
    """Asymmetric one-pole follower (fast attack, slow release, or vice versa)."""

    __slots__ = ("value", "attack", "release")

    def __init__(self, attack: float, release: float, x0: float = 0.0):
        self.value = x0
        self.attack = attack
        self.release = release

    def update(self, dt: float, x: float) -> float:
        tau = self.attack if x > self.value else self.release
        a = 1.0 - math.exp(-dt / max(tau, 1e-4))
        self.value += (x - self.value) * a
        return self.value


class Impulse:
    """A decaying bump: attack -> peak -> exponential decay.  Used for event gains."""

    __slots__ = ("value", "decay")

    def __init__(self, decay: float):
        self.value = 0.0
        self.decay = decay

    def trigger(self, amount: float) -> None:
        self.value = max(self.value, amount)

    def add(self, amount: float) -> None:
        self.value += amount

    def update(self, dt: float) -> float:
        self.value *= math.exp(-dt / max(self.decay, 1e-4))
        return self.value
