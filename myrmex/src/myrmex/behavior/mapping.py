"""Configurable music -> motion mappings (no mapping is hard-coded in the engine).

A mapping reads one feature from the :class:`~myrmex.music.features.ControlFrame`
(``energy``, ``hats_density``, ``bass_activity``, ``pitch`` ...), shapes it with
a curve and writes it into a *modulation slot*.  Slots are named parameters
that behaviours and the motor read (``cadence_scale``, ``stride_scale``,
``bounce_amount``, ``reaction_gain``, ``turn_curvature``, ``gaze_wander``,
...).  ``mode`` "mul" multiplies the slot (neutral 1), "add" adds (neutral 0).

Configuration lives in JSON (comments allowed, see ``configs/mappings``)::

    {"hat_density": {"source": "hats_density", "target": "cadence_scale",
                     "curve": "smoothstep", "amount": 0.35, "mode": "mul"}}
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..util.mathutil import clamp, smoothstep

CURVES = {
    "linear": lambda x: x,
    "smoothstep": lambda x: smoothstep(0.0, 1.0, x),
    "ease_in": lambda x: x * x,
    "ease_out": lambda x: 1.0 - (1.0 - x) ** 2,
    "sqrt": lambda x: math.sqrt(max(x, 0.0)),
    "exp": lambda x: (math.exp(3.0 * x) - 1.0) / (math.exp(3.0) - 1.0),
    "bipolar": lambda x: 2.0 * x - 1.0,
    "threshold": lambda x: 1.0 if x > 0.5 else 0.0,
    "invert": lambda x: 1.0 - x,
}

SLOTS_MUL = ("cadence_scale", "stride_scale", "step_height", "arm_swing", "walk_speed", "bounce_amount",
             "sway_amount", "gesture_rate", "reaction_gain", "responsiveness", "gaze_wander", "hip_sway",
             "behavior_tempo", "turn_rate")
SLOTS_ADD = ("crouch", "lean", "side_lean", "tension", "energy", "sharpness", "turn_curvature",
             "head_nod", "shoulder_raise", "hesitation", "weight_bias", "gaze_height", "variation")


@dataclass
class Mapping:
    name: str
    source: str
    target: str
    curve: str = "linear"
    amount: float = 0.5
    mode: str = "auto"
    input_range: tuple[float, float] = (0.0, 1.0)
    smooth: float = 0.25              # seconds (one-pole)
    enabled: bool = True
    value: float = 0.0

    def resolved_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        return "mul" if self.target in SLOTS_MUL else "add"


@dataclass
class MappingSet:
    mappings: list[Mapping] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "MappingSet":
        ms = []
        for name, m in d.items():
            if name.startswith("_") or not isinstance(m, dict):
                continue
            ms.append(Mapping(name=name, source=m.get("source", name), target=m["target"],
                              curve=m.get("curve", "linear"), amount=float(m.get("amount", 0.5)),
                              mode=m.get("mode", "auto"),
                              input_range=tuple(m.get("input_range", (0.0, 1.0))),
                              smooth=float(m.get("smooth", 0.25)), enabled=bool(m.get("enabled", True))))
        return cls(ms)

    def to_dict(self) -> dict:
        return {m.name: {"source": m.source, "target": m.target, "curve": m.curve, "amount": m.amount,
                         "mode": m.mode, "input_range": list(m.input_range), "smooth": m.smooth,
                         "enabled": m.enabled} for m in self.mappings}

    def evaluate(self, features: dict[str, float], dt: float) -> dict[str, float]:
        mods: dict[str, float] = {s: 1.0 for s in SLOTS_MUL}
        mods.update({s: 0.0 for s in SLOTS_ADD})
        for m in self.mappings:
            if not m.enabled or m.source not in features:
                continue
            lo, hi = m.input_range
            x = clamp((float(features[m.source]) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
            y = CURVES.get(m.curve, CURVES["linear"])(x)
            a = 1.0 - math.exp(-dt / max(m.smooth, 1e-3))
            m.value += (y - m.value) * a
            if m.resolved_mode() == "mul":
                mods[m.target] = mods.get(m.target, 1.0) * (1.0 + m.amount * (m.value - 0.5) * 2.0)
            else:
                mods[m.target] = mods.get(m.target, 0.0) + m.amount * m.value
        return mods


DEFAULT_MAPPINGS = {
    "_comment": "source feature -> modulation slot. Edit freely; see docs/BEHAVIOR_MAPPING.md",
    "hat_density": {"source": "hats_density", "target": "cadence_scale", "curve": "smoothstep", "amount": 0.18},
    "perc_density": {"source": "perc_density", "target": "variation", "curve": "linear", "amount": 0.4},
    "bass_weight": {"source": "bass_activity", "target": "crouch", "curve": "ease_in", "amount": 0.12, "smooth": 0.6},
    "kick_bounce": {"source": "kick_activity", "target": "bounce_amount", "curve": "smoothstep", "amount": 0.35},
    "snare_reaction": {"source": "snare_density", "target": "reaction_gain", "curve": "linear", "amount": 0.2},
    "energy_intensity": {"source": "energy", "target": "walk_speed", "curve": "smoothstep", "amount": 0.35, "smooth": 1.0},
    "velocity_sharpness": {"source": "velocity", "target": "sharpness", "curve": "linear", "amount": 0.5},
    "regularity_groove": {"source": "rhythm", "target": "sway_amount", "curve": "smoothstep", "amount": 0.25, "smooth": 1.5},
    "pitch_gaze": {"source": "pitch", "target": "gaze_height", "curve": "bipolar", "amount": 0.35, "smooth": 0.8},
    "pitch_curvature": {"source": "pitch_velocity", "target": "turn_curvature", "curve": "linear",
                        "input_range": [-1.0, 1.0], "amount": 0.0, "mode": "add"},
    "silence_hesitation": {"source": "silence", "target": "hesitation", "curve": "ease_in", "amount": 0.8, "smooth": 0.3},
    "novelty_wander": {"source": "novelty", "target": "gaze_wander", "curve": "linear", "amount": 0.4},
    "trend_tension": {"source": "trend", "target": "tension", "curve": "linear", "input_range": [0.0, 0.6],
                      "amount": 0.25, "smooth": 1.5},
}
