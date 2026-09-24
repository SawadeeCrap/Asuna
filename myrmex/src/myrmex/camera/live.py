"""Live cinematographer: the offline shot grammar, made causal.

The offline :mod:`cinematographer` knows the whole performance and smooths with
zero phase.  Live, the future is unknown - but a walking subject is very
predictable: position is extrapolated along the walking direction, so the
camera leads instead of lagging.  Cuts happen on downbeats of the live clock
(every 2-4 bars, earlier on a phrase change or when a trigger asks for it);
within a shot the camera glides on critically damped springs, across a cut it
jumps (that is what a cut is).
"""
from __future__ import annotations

import math

import numpy as np

from ..realtime.protocol import CameraState
from ..util.noise import Noise1D
from ..util.rng import Rng, stable_hash64
from ..util.springs import Spring
from .cinematographer import ESTABLISHING, SECTION_SHOTS, SHOTS

UP = np.array([0.0, 0.0, 1.0])


class LiveCinematographer:
    def __init__(self, height: float = 1.7, seed: int = 0, bars_min: int = 2, bars_max: int = 4):
        self.h = height
        self.sc = height / 1.7
        self.rng = Rng(stable_hash64("live-camera", seed))
        self.bars_min, self.bars_max = bars_min, bars_max
        self.kind = None
        self.side = self.rng.choice([-1.0, 1.0])
        self.shot_id = 0
        self.shot_start_t = 0.0
        self.shot_start_bar = 0
        self.shot_len_bars = 4
        self.recent: list[str] = []
        self.pos_s = Spring(0.35)
        self.tgt_s = Spring(0.25)
        self.fwd = None
        self.subj = None
        self.subj_v = np.zeros(3)
        self.n1 = Noise1D(seed + 11, 0.07, 2)
        self.n2 = Noise1D(seed + 12, 0.05, 2)
        self.last_bar = None
        self.pending_cut = False
        self.forced_kind: str | None = None
        self.state: CameraState | None = None

    # ------------------------------------------------------------------ decisions
    def request_cut(self, kind: str | None = None) -> None:
        self.pending_cut = True
        if kind in SHOTS:
            self.forced_kind = kind

    def _choose(self, label: str, energy: float) -> str:
        if self.forced_kind:
            k, self.forced_kind = self.forced_kind, None
            return k
        table = ESTABLISHING if self.kind is None else SECTION_SHOTS.get(label, SECTION_SHOTS["groove"])
        names, w = [], []
        for k, base in table.items():
            st = SHOTS[k]
            fit = 1.0 if st.energy[0] <= energy <= st.energy[1] else 0.4
            pen = 0.0 if k == self.kind else (0.55 if k in self.recent[-3:] else 1.0)
            if "close" in k:
                pen *= 0.6                       # live: the walk is the star, close-ups are accents
                if self.kind and "close" in self.kind:
                    pen *= 0.3
            names.append(k)
            w.append(base * fit * pen)
        if sum(w) <= 0:
            return "front_dolly"
        return names[self.rng.weighted_index(w)]

    def _cut(self, t: float, bar: int, label: str, energy: float) -> None:
        self.kind = self._choose(label, energy)
        if self.kind in ("front_dolly", "rear_follow") and self.rng.chance(0.3):
            self.side = -self.side
        self.recent.append(self.kind)
        self.shot_id += 1
        self.shot_start_t = t
        self.shot_start_bar = bar
        energetic = label in ("drop", "peak", "return") or energy > 0.7
        if "close" in self.kind:
            self.shot_len_bars = self.rng.randint(1, 2)                 # accents, not the whole chorus
        else:
            hi = max(self.bars_min, self.bars_max - (2 if energetic else 0))
            self.shot_len_bars = self.rng.randint(self.bars_min, hi)
        self.pending_cut = False
        self._snap = True

    # ------------------------------------------------------------------ update
    def update(self, t: float, dt: float, pelvis: np.ndarray, head: np.ndarray, feet: np.ndarray,
               heading: float, beat: float, beats_per_bar: float, label: str = "groove",
               energy: float = 0.5, phrase: bool = False) -> CameraState:
        sc = self.sc
        # Subject frame: predicted, smoothed path and walking direction.
        p = np.asarray(pelvis, float)
        if self.subj is None:
            self.subj = p.copy()
            self.fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
        v = (p - self.subj) / max(dt, 1e-4)
        v[2] = 0.0
        self.subj_v += (v - self.subj_v) * (1.0 - math.exp(-dt / 0.6))
        self.subj = p
        lead = self.subj_v * 0.25                              # look where she will be
        S = p + lead
        spd = float(np.linalg.norm(self.subj_v[:2]))
        f_goal = self.subj_v / spd if spd > 0.15 else np.array([math.cos(heading), math.sin(heading), 0.0])
        self.fwd += (f_goal - self.fwd) * (1.0 - math.exp(-dt / 1.2))
        self.fwd[2] = 0.0
        self.fwd /= max(np.linalg.norm(self.fwd), 1e-9)
        F = self.fwd
        L = np.cross(UP, F)
        # Cuts on downbeats.
        bar = int(beat // max(beats_per_bar, 1.0))
        new_bar = self.last_bar is not None and bar != self.last_bar
        self.last_bar = bar
        self._snap = False
        if self.kind is None:
            self._cut(t, bar, label, energy)
        elif new_bar:
            age = bar - self.shot_start_bar
            if self.pending_cut or age >= self.shot_len_bars or (phrase and age >= 1):
                self._cut(t, bar, label, energy)
        st = SHOTS[self.kind]
        u = min(1.0, (t - self.shot_start_t) / max(self.shot_len_bars * beats_per_bar * 0.5, 1.0))
        ang = st.drift * self.n1.sample(t)
        fr = math.cos(ang) * F + math.sin(ang) * L
        lr = math.cos(ang) * L - math.sin(ang) * F
        push = 1.0 - st.push * (u * u * (3 - 2 * u))
        if self.kind == "wide_orbit":
            oa = 0.12 * t + self.shot_id
            r = math.hypot(st.fwd, st.left) * push
            off = (math.cos(oa) * F + math.sin(oa) * L) * r * sc + UP * st.up * sc
        else:
            off = (fr * st.fwd + lr * st.left * self.side) * push * sc + UP * (st.up * sc + 0.06 * self.n2.sample(t))
        ground = S * np.array([1.0, 1.0, 0.0])
        pos_goal = ground + off
        if st.target == "head":
            tgt_goal = np.asarray(head, float) + lead
        elif st.target == "feet":
            tgt_goal = np.asarray(feet, float) * np.array([1.0, 1.0, 0.0]) + lead + UP * (st.target_up + 0.08) * sc
        else:
            tgt_goal = ground + UP * st.target_up * sc
        if self._snap or self.state is None:
            self.pos_s.x, self.pos_s.v = pos_goal.copy(), np.zeros(3)
            self.tgt_s.x, self.tgt_s.v = tgt_goal.copy(), np.zeros(3)
        pos = self.pos_s.update(dt, pos_goal)
        tgt = self.tgt_s.update(dt, tgt_goal)
        focus_pt = np.asarray(head, float) if st.target == "head" else (
            np.asarray(feet, float) if st.target == "feet" else 0.6 * np.asarray(head, float) + 0.4 * tgt)
        self.state = CameraState(pos.copy(), tgt.copy(), st.lens, float(np.linalg.norm(pos - focus_pt)), st.fstop,
                                 self.shot_id, self.kind)
        return self.state


__all__ = ["LiveCinematographer"]
