"""Fast reactions to individual musical events – with physical duration.

A note never *sets* a pose.  It may start a short reaction (dip, nod, jolt,
shrug, twist, sway, tilt, flinch, orient, breath, freeze) whose envelope has
an attack and a decay; the result is *added* to the command and the motor's
springs turn it into compression -> release -> overshoot -> settle.

Whether a reaction happens at all depends on salience (velocity, accent,
surprise), attention and habituation; which reaction happens is drawn from a
per-group pool with a recency penalty, so identical notes never produce
identical responses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..motion.command import MotionCommand
from ..util.mathutil import clamp
from .novelty import Habituation, RecencyTracker

POOLS = {
    "kick": [("dip", 1.0), ("nod", 0.6), ("sway", 0.25)],
    "snare": [("jolt", 1.0), ("shrug", 0.6), ("twist", 0.6), ("nod", 0.35)],
    "hats": [("nod", 0.5), ("tilt", 0.25), ("shrug", 0.15)],
    "perc": [("twist", 0.5), ("nod", 0.5), ("tilt", 0.4), ("orient", 0.3)],
    "bass": [("sway", 1.0), ("dip", 0.4)],
    "melody": [("tilt", 0.6), ("orient", 0.6), ("breath", 0.3)],
    "harmony": [("breath", 0.6), ("tilt", 0.2)],
    "texture": [("breath", 0.5), ("orient", 0.3)],
    "fx": [("flinch", 1.0), ("orient", 1.0), ("twist", 0.3)],
}

SHAPES = {   # attack, decay (s)
    "dip": (0.035, 0.16), "nod": (0.04, 0.2), "jolt": (0.025, 0.14), "shrug": (0.05, 0.25),
    "twist": (0.04, 0.22), "sway": (0.08, 0.35), "tilt": (0.07, 0.4), "flinch": (0.03, 0.45),
    "orient": (0.0, 0.0), "breath": (0.4, 1.2), "freeze": (0.02, 0.5),
}


@dataclass
class Active:
    kind: str
    t0: float
    amp: float
    sign: float = 1.0
    delay: float = 0.0

    def value(self, t: float) -> float:
        a, d = SHAPES[self.kind]
        x = t - self.t0 - self.delay
        if x < 0.0:
            return 0.0
        rise = 1.0 - math.exp(-x / max(a, 1e-3)) if a > 0 else 1.0
        return self.amp * rise * math.exp(-x / max(d, 1e-3))

    def alive(self, t: float) -> bool:
        _, d = SHAPES[self.kind]
        return t - self.t0 - self.delay < 5.0 * d + 0.05


@dataclass
class Reactions:
    rng: object
    habituation: Habituation = field(default_factory=Habituation)
    recency: RecencyTracker = field(default_factory=lambda: RecencyTracker(half_life=4.0, strength=0.8))
    active: list[Active] = field(default_factory=list)
    last_time: dict = field(default_factory=dict)
    vel_mean: dict = field(default_factory=dict)
    count: int = 0

    def trigger(self, t: float, group: str, velocity: float, surprise: float, drives, mods: dict,
                gaze=None, anticipated: float = 0.0) -> str | None:
        if group not in POOLS:
            return None
        vm = self.vel_mean.get(group, velocity)
        accent = clamp((velocity - vm) / max(vm, 0.1), 0.0, 1.0)
        self.vel_mean[group] = vm + (velocity - vm) * 0.15
        if t - self.last_time.get(group, -1.0) < 0.11:
            return None
        salience = velocity * (0.55 + 0.45 * accent) + 0.9 * surprise
        h = self.habituation.get(group)
        resp = float(mods.get("responsiveness", 1.0)) * float(mods.get("reaction_gain", 1.0))
        p = clamp(salience * h * resp * (0.35 + 0.65 * drives.attention) * (0.6 + 0.6 * drives.arousal), 0.0, 1.0)
        if not self.rng.chance(p):
            return None
        gain = self.habituation.stimulate(group, surprise)
        pool = POOLS[group]
        weights = [w * self.recency.penalty(t, f"{group}:{k}") for k, w in pool]
        kind = pool[self.rng.weighted_index(weights)][0]
        if surprise > 0.6 and group in ("fx", "perc", "snare"):
            kind = "flinch"
        amp = clamp(salience * gain * (0.6 + 0.6 * drives.arousal) * resp, 0.0, 1.3)
        # Anticipated hits get a smaller, better-timed response (the body was ready).
        amp *= 1.0 - 0.35 * anticipated
        delay = self.rng.uniform(0.0, 0.045) * (1.0 - 0.6 * anticipated)
        sign = self.rng.choice([-1.0, 1.0])
        self.last_time[group] = t
        self.recency.add(t, f"{group}:{kind}")
        self.count += 1
        if kind == "orient":
            if gaze is not None:
                gaze.orient(t, group, salience)
            return kind
        self.active.append(Active(kind, t, amp, sign, delay))
        if kind == "flinch" and gaze is not None:
            gaze.orient(t, group, 1.0)
        return kind

    def freeze(self, t: float, strength: float) -> None:
        self.active.append(Active("freeze", t, clamp(strength, 0.0, 1.0)))

    def update(self, t: float, dt: float) -> None:
        self.habituation.update(dt)
        self.active = [a for a in self.active if a.alive(t)]

    def apply(self, t: float, cmd: MotionCommand, anticipation: dict | None = None, groove: float = 0.0) -> None:
        for a in self.active:
            v = a.value(t)
            k = a.kind
            if k == "dip":
                cmd.crouch += 0.2 * v
                cmd.head_nod -= 0.03 * v
            elif k == "nod":
                cmd.head_nod -= 0.13 * v
            elif k == "jolt":
                cmd.lean -= 0.07 * v
                cmd.shoulder_raise += 0.35 * v
                cmd.crouch += 0.05 * v
            elif k == "shrug":
                cmd.shoulder_raise += 0.7 * v
            elif k == "twist":
                cmd.twist += 0.13 * v * a.sign
            elif k == "sway":
                cmd.weight_bias += 0.35 * v * a.sign
            elif k == "tilt":
                cmd.side_lean += 0.06 * v * a.sign
            elif k == "flinch":
                cmd.lean -= 0.12 * v
                cmd.crouch += 0.12 * v
                cmd.shoulder_raise += 0.8 * v
                cmd.tension = max(cmd.tension, 0.6 + 0.4 * v)
            elif k == "breath":
                cmd.lean -= 0.025 * v
                cmd.head_nod += 0.03 * v
            elif k == "freeze":
                cmd.tension = max(cmd.tension, 0.5 + 0.5 * v)
                cmd.speed *= 1.0 - 0.7 * v
        # Anticipatory preparation: a slight pre-load before strongly expected low hits.
        if anticipation and groove > 0.2:
            pre = max(anticipation.get("kick", 0.0), 0.6 * anticipation.get("snare", 0.0))
            cmd.crouch += 0.05 * pre * groove
