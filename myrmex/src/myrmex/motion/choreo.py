"""Scripted command timelines (tests, previews, manual choreography).

A :class:`CommandScript` is a list of timed segments.  Each segment sets some
:class:`MotionCommand` fields from its start time on; numeric fields are
eased from the previous value over ``blend`` seconds.  The motor's own
springs add the physical follow-through on top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..util.mathutil import smootherstep
from .command import ArmGesture, MotionCommand

_NUMERIC = ("speed", "lateral", "turn_rate", "stride_scale", "cadence_scale", "step_height", "sharpness",
            "crouch", "rise", "lean", "side_lean", "twist", "hip_sway", "weight_bias", "shoulder_raise",
            "head_nod", "energy", "tension", "arm_swing", "gaze_weight")


@dataclass
class Segment:
    t: float
    values: dict
    blend: float = 0.6
    label: str = ""


@dataclass
class CommandScript:
    segments: list[Segment] = field(default_factory=list)

    def at(self, t: float, values: dict | None = None, blend: float = 0.6, label: str = "", **kw) -> "CommandScript":
        v = dict(values or {})
        v.update(kw)
        self.segments.append(Segment(float(t), v, blend, label))
        self.segments.sort(key=lambda s: s.t)
        return self

    def label(self, t: float) -> str:
        lab = ""
        for s in self.segments:
            if s.t <= t and s.label:
                lab = s.label
        return lab

    def command(self, t: float) -> MotionCommand:
        cmd = MotionCommand()
        current: dict = {}
        for seg in self.segments:
            if seg.t > t:
                break
            prev = dict(current)
            current.update(seg.values)
            if seg.blend > 0 and t < seg.t + seg.blend:
                a = smootherstep((t - seg.t) / seg.blend)
                for k, v in seg.values.items():
                    if k in _NUMERIC and k in prev:
                        current[k] = prev[k] + (v - prev[k]) * a
                    elif k in _NUMERIC:
                        base = getattr(MotionCommand(), k)
                        current[k] = base + (v - base) * a
        for k, v in current.items():
            if k == "gaze_target" and v is not None:
                cmd.gaze_target = np.asarray(v, dtype=float)
            elif k == "gestures":
                cmd.gestures = {s: (g if isinstance(g, ArmGesture) else ArmGesture(np.asarray(g[0], float), *g[1:]))
                                for s, g in v.items()}
            elif hasattr(cmd, k):
                setattr(cmd, k, v)
        return cmd
