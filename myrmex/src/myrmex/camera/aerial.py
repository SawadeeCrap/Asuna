"""Aerial cinematographer for the flying Mimetic Polyalloy: nine camera behaviours.

OBSERVE (wide, stable) · FOLLOW (behind, matches speed) · APPROACH (slow push-in) ·
RETREAT (pull out as it expands) · ORBIT (circle while it hovers / reconfigures) ·
LOCK (shoulder-locked, rotation-stable) · TRACK (side-on, parallel) ·
IMPACT (short shake + fast reframe on a kick / obstacle) · RECOVERY (settles after impact).

Same interface as LiveCinematographer (``mode``, ``manual``, ``request_cut``, ``update``), so
the app's camera controls work unchanged; the nine shot buttons map onto the nine modes.
Automatic mode chooses by the organism's intent and material state; cuts land on bars.
"""
from __future__ import annotations

import math

import numpy as np

from ..realtime.protocol import CameraState
from ..util.noise import Noise1D
from ..util.rng import Rng, stable_hash64
from ..util.springs import Spring

MODES = ("observe", "follow", "approach", "retreat", "orbit", "lock", "track", "impact", "recovery")
UP = np.array([0.0, 0.0, 1.0])
APP_BUTTONS = ("front_dolly", "front_low", "three_quarter", "side_track", "rear_follow", "hips_close", "feet_close",
               "face_close", "wide_orbit")
# distance (x size), height offset, side angle (rad, 0 = behind), lens, pos halflife, target halflife
SHOT = {
    "observe": (7.0, 1.2, 2.2, 35.0, 0.9, 0.5),
    "follow": (3.4, 0.6, 0.25, 32.0, 0.35, 0.2),
    "approach": (4.5, 0.2, 2.6, 50.0, 0.6, 0.3),
    "retreat": (3.0, 0.5, 2.9, 28.0, 0.6, 0.3),
    "orbit": (4.2, 0.4, 0.0, 40.0, 0.4, 0.25),
    "lock": (2.6, 0.35, 0.6, 30.0, 0.12, 0.08),
    "track": (4.0, 0.0, 1.57, 45.0, 0.3, 0.2),
    "impact": (2.8, 0.2, 2.4, 26.0, 0.08, 0.05),
    "recovery": (4.6, 0.6, 2.0, 35.0, 0.7, 0.45),
}
BY_INTENT = {"CRUISE": {"follow": 3, "track": 2, "lock": 1.5, "observe": 1},
             "HOVER": {"orbit": 3, "approach": 2, "observe": 1},
             "EXPLORE": {"observe": 2, "track": 2, "orbit": 1},
             "DISPLAY": {"approach": 2.5, "orbit": 2, "retreat": 1.5},
             "REFORM": {"recovery": 3, "observe": 1},
             "EVADE": {"impact": 2, "lock": 1},
             "HUNT": {"follow": 3, "lock": 2, "track": 2},
             "ENVELOP": {"approach": 3, "orbit": 2},
             "PERCH": {"track": 2, "orbit": 2, "approach": 1},
             "FORMATION": {"observe": 2, "retreat": 2, "follow": 1},
             "MERGE": {"observe": 3, "orbit": 1}}


class AerialCinematographer:
    def __init__(self, size: float = 1.6, seed: int = 0, bars_min: int = 2, bars_max: int = 4):
        self.size = size
        self.rng = Rng(stable_hash64("aerial-camera", seed))
        self.bars_min, self.bars_max = bars_min, bars_max
        self.kind: str | None = None
        self.shot_id, self.shot_start_t, self.shot_start_bar, self.shot_len = 0, 0.0, 0, 4
        self.pos_s, self.tgt_s = Spring(0.4), Spring(0.25)
        self.side = self.rng.choice([-1.0, 1.0])
        self.fwd = None
        self.subj_v = np.zeros(3)
        self.prev = None
        self.last_bar = None
        self.pending_cut = False
        self.forced_kind: str | None = None
        self.shake = 0.0
        self.n1, self.n2, self.n3 = Noise1D(seed + 21, 0.9, 2), Noise1D(seed + 22, 1.1, 2), Noise1D(seed + 23, 0.06, 2)
        self.state: CameraState | None = None
        self.mode = "auto"
        self.manual = {"distance": 1.0, "height": 0.0, "orbit": 0.0, "lens": 0.0, "smooth": 0.35}
        self.recent: list[str] = []
        self.last_impact_cut = -1e9
        self.last_suggest = -1e9

    def request_cut(self, kind: str | None = None) -> None:
        self.pending_cut = True
        if kind in MODES:
            self.forced_kind = kind
        elif kind in APP_BUTTONS:                          # the app's shot buttons 1-9 send humanoid names
            self.forced_kind = MODES[APP_BUTTONS.index(kind)]

    def impact(self, strength: float = 1.0, t: float = 0.0, reframe: bool = False) -> None:
        """A physical event happened: shake now; big ones may reframe (automatic mode, not too often)."""
        self.shake = max(self.shake, 0.12 * min(1.5, strength))
        if (reframe and self.mode == "auto" and self.kind != "impact" and t - self.last_impact_cut > 10.0
                and self.rng.chance(0.45)):
            self.last_impact_cut = t
            self.pending_cut, self.forced_kind = True, "impact"

    def suggest(self, kind: str, t: float) -> None:
        """A story moment (the flock splits, prey is caught, it lands): reframe, not too often."""
        if self.mode == "auto" and kind in MODES and t - self.last_suggest > 8.0:
            self.last_suggest = t
            self.pending_cut, self.forced_kind = True, kind

    def _choose(self, intent: str) -> str:
        if self.forced_kind:
            k, self.forced_kind = self.forced_kind, None
            return k
        if self.kind == "impact":
            return "recovery"
        table = BY_INTENT.get(intent, BY_INTENT["CRUISE"])
        names = list(table)
        w = [table[k] * (0.2 if k == self.kind else (0.6 if k in self.recent[-3:] else 1.0)) for k in names]
        return names[self.rng.weighted_index(w)]

    def _cut(self, t: float, bar: int, intent: str) -> None:
        self.kind = self._choose(intent)
        self.recent.append(self.kind)
        self.shot_id += 1
        self.shot_start_t, self.shot_start_bar = t, bar
        self.shot_len = 1 if self.kind in ("impact",) else self.rng.randint(self.bars_min, self.bars_max)
        if self.rng.chance(0.35):
            self.side = -self.side
        self.pending_cut = False
        self._snap = self.kind != "recovery"

    def update(self, t: float, dt: float, com: np.ndarray, heading: float, beat: float, beats_per_bar: float,
               intent: str = "CRUISE", extent: float = 1.0) -> CameraState:
        com = np.asarray(com, float)
        if self.prev is None:
            self.prev = com.copy()
            self.fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
        v = (com - self.prev) / max(dt, 1e-4)
        self.prev = com.copy()
        self.subj_v += (v - self.subj_v) * (1.0 - math.exp(-dt / 0.5))
        spd = float(np.linalg.norm(self.subj_v[:2]))
        f_goal = np.array([*(self.subj_v[:2] / spd), 0.0]) if spd > 0.3 else np.array([math.cos(heading), math.sin(heading), 0.0])
        self.fwd += (f_goal - self.fwd) * (1.0 - math.exp(-dt / 1.0))
        self.fwd /= max(np.linalg.norm(self.fwd), 1e-9)
        bar = int(beat // max(beats_per_bar, 1.0))
        new_bar = self.last_bar is not None and bar != self.last_bar
        self.last_bar = bar
        self._snap = False
        if self.kind is None:
            self._cut(t, bar, intent)
        elif self.mode == "manual":
            if self.pending_cut:
                self._cut(t, bar, intent)
        elif self.pending_cut and self.forced_kind == "impact":
            self._cut(t, bar, intent)                        # impacts do not wait for the bar
        elif new_bar or (self.kind == "impact" and t - self.shot_start_t > 1.2):
            age = bar - self.shot_start_bar
            if self.pending_cut or age >= self.shot_len or self.kind == "impact":
                self._cut(t, bar, intent)
        dist, up, ang, lens, hl_p, hl_t = SHOT[self.kind]
        u = min(1.0, (t - self.shot_start_t) / max(self.shot_len * beats_per_bar * 0.5, 1.0))
        ease = u * u * (3 - 2 * u)
        scale = self.size * (0.7 + 0.5 * min(2.5, extent))
        if self.kind == "approach":
            dist *= 1.0 - 0.45 * ease
        elif self.kind == "retreat":
            dist *= 1.0 + 0.8 * ease
        if self.kind == "orbit":
            ang = 0.35 * (t - self.shot_start_t) + self.shot_id
        ang = ang * self.side + 0.15 * self.n3.sample(t)
        mn = self.manual
        ang += mn["orbit"]
        F = self.fwd
        L = np.cross(UP, F)
        back = -(math.cos(ang) * F + math.sin(ang) * L)
        if self.kind == "orbit":
            back = math.cos(ang) * F + math.sin(ang) * L
        lead = self.subj_v * (0.35 if self.kind in ("follow", "lock", "track") else 0.15)
        S = com + lead
        off = back * dist * scale * mn["distance"] + UP * (up * scale + mn["height"])
        pos_goal = S + off
        pos_goal[2] = max(0.3, pos_goal[2])
        tgt_goal = S.copy()
        self.pos_s.halflife = hl_p * (mn["smooth"] / 0.35)
        self.tgt_s.halflife = hl_t
        if self._snap or self.state is None:
            self.pos_s.x, self.pos_s.v = pos_goal.copy(), np.zeros(3)
            self.tgt_s.x, self.tgt_s.v = tgt_goal.copy(), np.zeros(3)
        pos = self.pos_s.update(dt, pos_goal)
        tgt = self.tgt_s.update(dt, tgt_goal)
        if self.shake > 1e-3:                               # impact: short, decaying handheld shake
            k = self.shake
            pos = pos + np.array([self.n1.sample(t * 9), self.n2.sample(t * 9), self.n1.sample(t * 7 + 5)]) * k
            tgt = tgt + np.array([self.n2.sample(t * 8 + 3), self.n1.sample(t * 8 + 9), 0.0]) * 0.5 * k
            self.shake *= math.exp(-dt / 0.25)
        self.state = CameraState(pos.copy(), tgt.copy(), mn["lens"] or lens, float(np.linalg.norm(pos - com)), 2.8,
                                 self.shot_id, self.kind)
        return self.state


__all__ = ["AerialCinematographer", "MODES"]
