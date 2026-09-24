"""Runway director: continuous, beat-locked, confident walking (the default for bipeds).

The creature does not wander – it *travels*.  It walks forward along a
runway, every heel strike lands on the musical grid (like a good dancer
simply walking across a dance floor), and the music shapes *how* it walks:

* tempo      -> footfall grid (one step per beat, half-time when slow or very fast);
* sections   -> intensity: sultry half-time intro, building stride in builds,
                full strut on drops, slowed walk / pose in breaks, final pose in the outro;
* energy     -> stride length, swagger (hip sway), arm swing;
* accents    -> hip pops (snare), bounce (kick), shoulder shimmy (hats),
                glances toward surprising sounds;
* phrases    -> rare pose stops (freeze before a drop, burst on the drop),
                flourishes while walking (hand on hip, hair touch, shoulder roll, side glance).

Variation is correlated: stride, crossover and hip amplitude drift over steps
with smooth noise, so no two steps are identical, yet the walk stays coherent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..motion.command import ArmGesture, MotionCommand
from ..util.mathutil import clamp, smootherstep, wrap_angle
from ..util.noise import Noise1D
from .novelty import RecencyTracker

SECTION_PROFILE = {             # energy bias, subdivision (steps per beat), stride scale, sway
    "intro": (0.35, 0.5, 0.85, 0.85),
    "build": (0.6, 1.0, 0.95, 0.95),
    "drop": (1.0, 1.0, 1.1, 1.2),
    "peak": (0.95, 1.0, 1.08, 1.15),
    "groove": (0.75, 1.0, 1.0, 1.0),
    "return": (0.85, 1.0, 1.05, 1.1),
    "break": (0.3, 0.5, 0.85, 0.9),
    "outro": (0.3, 0.5, 0.8, 0.85),
    "silence": (0.2, 0.5, 0.8, 0.8),
}

FLOURISHES = ("hand_hip", "hair_touch", "shoulder_roll", "side_glance", "chin_up", "none")


@dataclass
class _Pose:
    t0: float
    duration: float
    side: float
    kind: str


@dataclass
class _Flourish:
    t0: float
    duration: float
    kind: str
    side: str


class RunwayDirector:
    def __init__(self, rng, height: float, heading: float, seed: int = 0):
        self.rng = rng
        self.height = height
        self.heading0 = heading
        self.heading = heading
        self.mult = 1.0
        self.pose: _Pose | None = None
        self.flourish: _Flourish | None = None
        self.flourish_hist = RecencyTracker(half_life=30.0, strength=1.2)
        self.lag = rng.uniform(0.005, 0.03)           # personal "laid back" feel (s)
        self.n_stride = Noise1D(seed * 7 + 1, 0.11, octaves=2)
        self.n_sway = Noise1D(seed * 7 + 2, 0.08, octaves=2)
        self.n_head = Noise1D(seed * 7 + 3, 0.05, octaves=2)
        self.n_arm = Noise1D(seed * 7 + 4, 0.09, octaves=2)
        self.last_phrase_t = -100.0
        self.next_flourish_t = rng.uniform(6.0, 12.0)
        self.walking = False
        self.label = "intro"
        self._decided_boundary = None
        # Live macro controls (set by the realtime session from CV / CC / OSC):
        #   hold (bool)  stand and pose instead of walking (e.g. transport stopped)
        #   energy, stride, sway (0..1)  override the section's intensity / stride / hip sway
        #   pose (str)   one-shot pose request, flourish (str) one-shot flourish request
        self.live: dict = {}
        self._hold_flip_t = 0.0

    # ------------------------------------------------------------------ helpers
    def _subdivision(self, bpm: float, section_mult: float) -> float:
        m = section_mult
        if bpm * m > 150.0:
            m *= 0.5
        if bpm * m < 55.0:
            m *= 2.0
        # Hysteresis: only switch subdivision when clearly needed.
        if m != self.mult:
            self.mult = m
        return self.mult

    def _start_pose(self, t: float, beats: float, beat_dur: float, kind: str | None = None) -> None:
        self.pose = _Pose(t, beats * beat_dur, self.rng.choice([-1.0, 1.0]),
                          kind or self.rng.choice(["hip_out", "look_back", "hand_hip", "lean"]))

    def _maybe_flourish(self, t: float, d, beat_dur: float) -> None:
        if self.flourish is not None and t - self.flourish.t0 > self.flourish.duration:
            self.flourish = None
        if self.flourish is not None or t < self.next_flourish_t or self.pose is not None:
            return
        weights = []
        for k in FLOURISHES:
            base = {"hand_hip": 1.0 + d.confidence, "hair_touch": 0.8 + 0.5 * d.arousal,
                    "shoulder_roll": 0.6 + d.groove, "side_glance": 0.9 + d.curiosity,
                    "chin_up": 0.6 + d.confidence, "none": 1.2}[k]
            weights.append(base * self.flourish_hist.penalty(t, k))
        k = FLOURISHES[self.rng.weighted_index(weights)]
        self.flourish_hist.add(t, k)
        beats = self.rng.choice([4.0, 6.0, 8.0])
        self.flourish = _Flourish(t, beats * beat_dur, k, self.rng.choice(["l", "r"]))
        self.next_flourish_t = t + beats * beat_dur + self.rng.uniform(4.0, 11.0) * (1.3 - 0.6 * d.arousal)

    # ------------------------------------------------------------------ main
    def apply(self, ctx, cmd: MotionCommand, events, structure=None) -> None:
        t, d, fr = ctx.t, ctx.drives, ctx.fr
        beat_dur = 60.0 / max(fr.tempo, 30.0)
        label = ctx.section
        prof = SECTION_PROFILE.get(label, SECTION_PROFILE["groove"])
        e_bias, sec_mult, stride_s, sway_s = prof
        live = self.live
        if live.get("energy") is not None:
            e_bias = float(live["energy"])
            sec_mult = 0.5 if e_bias < 0.3 else 1.0
        if live.get("stride") is not None:
            stride_s *= 0.7 + 0.6 * float(live["stride"])
        if live.get("sway") is not None:
            sway_s *= 0.6 + 0.9 * float(live["sway"])
        energy = clamp(0.5 * e_bias + 0.5 * d.arousal, 0.0, 1.0)
        # ---------------------------------------------------- live holds / requests
        if live.get("hold"):
            if self.pose is None or self.pose.kind != "hold":
                self.pose = _Pose(t, 1e9, self.rng.choice([-1.0, 1.0]), "hold")
                self._hold_flip_t = t + self.rng.uniform(3.0, 7.0)
            elif t > self._hold_flip_t:
                # Shift the weight to the other hip now and then: waiting, but alive.
                self.pose = _Pose(t, 1e9, -self.pose.side, "hold")
                self._hold_flip_t = t + self.rng.uniform(3.0, 8.0)
        elif self.pose is not None and self.pose.kind == "hold":
            self.pose = None
        req = live.pop("pose", None)
        if req and self.pose is None:
            self._start_pose(t, 4.0, beat_dur, req if req in ("hip_out", "look_back", "hand_hip", "lean") else None)
        req = live.pop("flourish", None)
        if req and self.pose is None:
            self.flourish = _Flourish(t, 6.0 * beat_dur, req if req in FLOURISHES else "hand_hip",
                                      self.rng.choice(["l", "r"]))
        # ---------------------------------------------------- structural moments
        for ev in events:
            if ev.type in ("phrase",) and t - self.last_phrase_t > 4.0 * beat_dur:
                self.last_phrase_t = t
                new_label = ev.data.get("label", label)
                if new_label in ("break", "outro") and self.rng.chance(0.65):
                    self._start_pose(t, self.rng.choice([2.0, 4.0]) if new_label == "break" else 8.0,
                                     beat_dur, "look_back" if new_label == "break" else "hip_out")
                elif new_label in ("drop", "peak") and self.pose is not None:
                    self.pose = None                                  # burst out of a held pose
            elif ev.type == "silence_start" and ev.strength > 0.35 and self.pose is None:
                self._start_pose(t, 3.0, beat_dur, "hip_out")
            elif ev.type == "drop":
                self.pose = None
        # Freeze in a pose over the last beats before a known drop (tension -> release).
        if structure is not None and self.pose is None:
            nb = structure.next_boundary(t)
            nxt = structure.section_at(nb + 1e-3) if nb is not None else None
            if nxt is not None and nxt.label in ("drop", "peak") and label == "build" and \
                    0.0 < nb - t < 2.0 * beat_dur and self._decided_boundary != nb:
                self._decided_boundary = nb
                if self.rng.chance(0.7):
                    self._start_pose(t, max((nb - t) / beat_dur, 0.5), beat_dur, "hip_out")
        if self.pose is not None and t - self.pose.t0 > self.pose.duration:
            self.pose = None
        self._maybe_flourish(t, d, beat_dur)
        # ---------------------------------------------------- walking parameters
        m = self._subdivision(fr.tempo, sec_mult)
        period = beat_dur / m
        H = self.height / 1.7
        stride = 0.42 * H * stride_s * (0.82 + 0.3 * energy) * (1.0 + 0.07 * self.n_stride.sample(t))
        speed = stride / period
        self.heading = self.heading0 + 0.05 * self.n_head.sample(t * 0.5)
        cmd.heading = self.heading
        cmd.energy = max(cmd.energy, 0.45 + 0.5 * energy)
        cmd.hip_sway = clamp(sway_s * (0.75 + 0.35 * d.confidence + 0.15 * self.n_sway.sample(t)), 0.3, 1.5)
        cmd.arm_swing = clamp(0.75 + 0.45 * energy + 0.1 * self.n_arm.sample(t), 0.4, 1.4)
        cmd.step_height = 0.8 + 0.3 * energy
        cmd.stride_scale = 1.0
        cmd.lean = -0.01 + 0.02 * d.tension
        beat_now = fr.beat
        ref_beat = math.floor(beat_now)
        cmd.step_ref = t + (ref_beat - beat_now) * beat_dur          # time of the current beat
        cmd.step_period = period
        cmd.step_offset = self.lag
        if self.pose is None:
            self.walking = True
            cmd.speed = speed
        else:
            self.walking = False
            if self.pose.kind == "hold":
                env = smootherstep((t - self.pose.t0) / 0.8)
            else:
                p = clamp((t - self.pose.t0) / max(self.pose.duration, 1e-3), 0.0, 1.0)
                env = smootherstep(p / 0.15) * smootherstep((1.0 - p) / 0.2)
            cmd.speed = 0.0
            cmd.step_period = None
            side = self.pose.side
            cmd.weight_bias = 0.95 * side * env
            cmd.side_lean = -0.07 * side * env
            cmd.hip_sway = 1.4
            if self.pose.kind == "look_back":
                ctx.gaze.look_local(t, 1.0, 2.2 * side, 0.1, hold=0.25)
                cmd.twist = 0.3 * side * env
            elif self.pose.kind == "hand_hip":
                s = "l" if side > 0 else "r"
                cmd.gestures = {s: ArmGesture(np.array([0.02, 0.2 * side * H, 0.02 * H]), env, 0.95, 0.4)}
            elif self.pose.kind == "lean":
                cmd.lean = -0.1 * env
                cmd.head_nod = 0.08 * env
        # ---------------------------------------------------- flourishes while walking
        f = self.flourish
        if f is not None and self.pose is None:
            p = clamp((t - f.t0) / max(f.duration, 1e-3), 0.0, 1.0)
            env = smootherstep(p / 0.25) * smootherstep((1.0 - p) / 0.25)
            sg = 1.0 if f.side == "l" else -1.0
            if f.kind == "hand_hip":
                cmd.gestures = {f.side: ArmGesture(np.array([0.02, 0.21 * sg * H, 0.03 * H]), env, 0.95, 0.4)}
            elif f.kind == "hair_touch":
                cmd.gestures = {f.side: ArmGesture(np.array([0.03, 0.13 * sg * H, 0.63 * H]), env * 0.95, 0.85, 0.6)}
            elif f.kind == "shoulder_roll":
                cmd.shoulder_raise = 0.35 * env * (0.5 + 0.5 * math.sin(2.0 * math.pi * fr.beat * 0.5))
            elif f.kind == "side_glance":
                if 0.15 < p < 0.75:
                    ctx.gaze.look_local(t, 2.0, 1.6 * sg, 0.05, hold=0.2)
            elif f.kind == "chin_up":
                cmd.head_nod = 0.09 * env
        # Gaze: straight down the runway unless something else took it.
        if self.pose is None and (f is None or f.kind != "side_glance"):
            if t > ctx.gaze.hold_until:
                ctx.gaze.look_local(t, 6.0, 0.15 * self.n_head.sample(t * 3.0), 0.05, hold=0.4)
