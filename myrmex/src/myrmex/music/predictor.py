"""Causal groove learner: anticipation, surprise, omission and instrument entries.

For every semantic group the predictor keeps a decaying histogram of onsets
over bar positions (16 steps per bar by default), normalised by the number
of bars in which that group was *active*.  After a couple of bars of a
repeating pattern it can tell

* how likely an onset is at each upcoming step (-> **anticipation**: the body
  prepares 150–350 ms before an expected hit),
* how unexpected an onset was (-> **surprise**: startle, dishabituation) –
  only for groups whose pattern is established; within a bar a group can
  only surprise once,
* when a strongly expected onset did **not** happen (-> **omission**: the
  creature hesitates, as if listening for the missing beat) – at most once
  per group and bar,
* when an instrument (re)enters after silence (-> **entry**: an orienting
  response, not a startle).

It learns only from the past, identically offline and live, so the character
genuinely "gets into" a groove over the first bars instead of knowing the
score in advance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Expectation:
    group: str
    time_until: float
    probability: float
    strength: float


class GroovePredictor:
    def __init__(self, steps: int = 16, decay: float = 0.8, groups: tuple[str, ...] = ()):
        self.steps = steps
        self.decay = decay
        self.H: dict[str, np.ndarray] = {}
        self.V: dict[str, np.ndarray] = {}
        self.active: dict[str, float] = {}         # decayed count of active bars
        self.silent_bars: dict[str, int] = {}
        self.bar_hits: dict[str, np.ndarray] = {}
        self.surprised: dict[str, bool] = {}
        self.omitted: dict[str, bool] = {}
        self.cur_bar = -1
        self.last_step_checked = -1
        for g in groups:
            self._ensure(g)

    def _ensure(self, g: str) -> None:
        if g not in self.H:
            self.H[g] = np.zeros(self.steps)
            self.V[g] = np.full(self.steps, 0.5)
            self.active[g] = 0.0
            self.silent_bars[g] = 99
            self.bar_hits[g] = np.zeros(self.steps)
            self.surprised[g] = False
            self.omitted[g] = False

    def step_of(self, bar_phase: float) -> int:
        return int(math.floor(bar_phase * self.steps + 0.5)) % self.steps

    def learned(self, g: str) -> float:
        return min(1.0, self.active.get(g, 0.0) / 2.5)

    def probability(self, g: str, step: int) -> float:
        if g not in self.H or self.active[g] < 0.8:
            return 0.0
        h = self.H[g]
        p = h[step] + 0.35 * max(h[(step - 1) % self.steps], h[(step + 1) % self.steps])
        return float(min(1.0, p / self.active[g]))

    def confidence(self, g: str) -> float:
        if g not in self.H or self.active[g] < 0.8:
            return 0.0
        p = self.H[g] / max(self.active[g], 1e-6)
        hits = p[p > 0.1]
        return float(np.clip(np.mean(hits) * self.learned(g), 0.0, 1.0)) if len(hits) else 0.0

    # ------------------------------------------------------------------ updates
    def advance(self, bar: int, bar_phase: float) -> list[tuple[str, float]]:
        """Call every tick. Returns omission events (group, strength)."""
        out: list[tuple[str, float]] = []
        if bar != self.cur_bar:
            if self.cur_bar >= 0:
                self._check_omissions(self.steps, out)
                for g in self.H:
                    was_active = self.bar_hits[g].max() > 0.0
                    self.H[g] = self.H[g] * self.decay + (np.minimum(self.bar_hits[g], 1.0) if was_active else 0.0)
                    self.active[g] = self.active[g] * self.decay + (1.0 if was_active else 0.0)
                    self.silent_bars[g] = 0 if was_active else self.silent_bars[g] + 1
                    self.bar_hits[g][:] = 0.0
                    self.surprised[g] = False
                    self.omitted[g] = False
            self.cur_bar = bar
            self.last_step_checked = -1
        cur = int(bar_phase * self.steps - 0.34)
        self._check_omissions(cur + 1, out)
        return out

    def _check_omissions(self, upto: int, out: list) -> None:
        for s in range(self.last_step_checked + 1, min(upto, self.steps)):
            for g in self.H:
                if self.omitted[g] or self.silent_bars[g] > 1 or self.learned(g) < 0.9:
                    continue
                p = self.probability(g, s)
                if p > 0.75 and self.bar_hits[g][s] == 0.0 and self.bar_hits[g][(s - 1) % self.steps] == 0.0 \
                        and self.bar_hits[g][(s + 1) % self.steps] == 0.0:
                    out.append((g, p * float(self.V[g][s])))
                    self.omitted[g] = True
            self.last_step_checked = s

    def observe(self, g: str, bar: int, bar_phase: float, velocity: float) -> tuple[float, bool]:
        """Register an onset; returns (surprise 0..1, is_entry)."""
        self._ensure(g)
        if bar != self.cur_bar:
            self.advance(bar, bar_phase)
        s = self.step_of(bar_phase)
        entry = self.silent_bars[g] >= 2 and self.bar_hits[g].max() == 0.0
        p = self.probability(g, s)
        learned = self.learned(g)
        off = abs(bar_phase * self.steps - round(bar_phase * self.steps)) * 2.0
        surprise = learned * (1.0 - p) * (0.6 + 0.4 * velocity) + 0.4 * learned * off * velocity
        if self.surprised[g]:
            surprise *= 0.25
        if surprise > 0.4:
            self.surprised[g] = True
        self.bar_hits[g][s] = max(self.bar_hits[g][s], velocity)
        self.V[g][s] += (velocity - self.V[g][s]) * 0.3
        return float(min(1.0, surprise)), entry

    def expectations(self, bar_phase: float, seconds_per_bar: float, horizon: float = 0.6) -> list[Expectation]:
        out: list[Expectation] = []
        step_dur = seconds_per_bar / self.steps
        cur = bar_phase * self.steps
        n_ahead = int(horizon / max(step_dur, 1e-3)) + 1
        for k in range(1, n_ahead + 1):
            s_abs = math.floor(cur) + k
            dt = (s_abs - cur) * step_dur
            if dt > horizon:
                break
            s = s_abs % self.steps
            for g in self.H:
                if self.silent_bars[g] > 1:
                    continue
                p = self.probability(g, s)
                if p > 0.3:
                    out.append(Expectation(g, dt, p, p * float(self.V[g][s]) * self.learned(g)))
        return out

    def anticipation(self, bar_phase: float, seconds_per_bar: float, lead: float = 0.3) -> dict[str, float]:
        out: dict[str, float] = {}
        for e in self.expectations(bar_phase, seconds_per_bar, horizon=lead * 1.5):
            ramp = max(0.0, 1.0 - e.time_until / lead)
            val = e.strength * ramp * ramp
            if val > out.get(e.group, 0.0):
                out[e.group] = val
        return out
