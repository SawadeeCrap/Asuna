"""The interface between *behaviour* (what the creature wants) and *motors* (how its body does it).

Behaviour layers write a :class:`MotionCommand` every tick.  Motors never
consume it raw: every field is pursued through springs, so behaviour can
change its mind abruptly while the body still moves continuously.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np


@dataclass
class ArmGesture:
    """A hand target in the character frame (x = forward, y = left, z = up, metres from the pelvis)."""
    target: np.ndarray
    weight: float = 1.0
    elbow_out: float = 0.3          # 0 = elbow down/back, 1 = elbow out to the side
    hand_pitch: float = 0.0         # radians, wrist flexion


@dataclass
class MotionCommand:
    # ------------------------------------------------------------ locomotion
    speed: float = 0.0              # desired speed along the heading (m/s)
    lateral: float = 0.0            # desired sideways speed (m/s), + = to the left
    heading: float | None = None    # absolute target heading (rad); None -> use turn_rate
    turn_rate: float = 0.0          # rad/s when heading is None
    stride_scale: float = 1.0       # longer / shorter steps
    cadence_scale: float = 1.0      # quicker / slower swings
    step_height: float = 1.0        # foot clearance multiplier
    sharpness: float = 0.5          # 0 = soft/lazy easing, 1 = crisp/snappy
    hold_step: bool = False         # hesitation: freeze a swinging foot mid-air
    step_period: float | None = None  # beat-locked stepping: seconds between footfalls (None = need-driven)
    step_ref: float | None = None     # an absolute time on the footfall grid (e.g. a beat time)
    step_offset: float = 0.0          # micro-timing of footfalls (+ = laid back)
    # ------------------------------------------------------------ posture
    crouch: float = 0.0             # 0..1 bend knees / lower the pelvis
    rise: float = 0.0               # 0..1 rise onto the toes / stretch tall
    lean: float = 0.0               # forward (+) / backward (-) torso lean, rad
    side_lean: float = 0.0          # lateral lean, rad (+ = toward the left)
    twist: float = 0.0              # upper-body yaw relative to the pelvis, rad
    hip_sway: float = 0.6           # style: pelvic rotation / obliquity amount
    weight_bias: float = 0.0        # standing weight shift, -1 right .. +1 left
    shoulder_raise: float = 0.0     # 0..1 tension in the shoulders
    head_nod: float = 0.0           # additive head pitch, rad
    # ------------------------------------------------------------ dynamics / tone
    energy: float = 0.5             # vigour: amplitudes, cadence, spring stiffness
    tension: float = 0.3            # muscle tone: less sway, stiffer, raised shoulders
    arm_swing: float = 1.0
    # ------------------------------------------------------------ attention
    gaze_target: np.ndarray | None = None
    gaze_weight: float = 1.0
    # ------------------------------------------------------------ gestures
    gestures: dict[str, ArmGesture] = field(default_factory=dict)   # key: "l" / "r"
    # ------------------------------------------------------------ impulses (consumed each tick)
    impulses: list[tuple[str, np.ndarray]] = field(default_factory=list)

    def copy(self) -> "MotionCommand":
        c = MotionCommand()
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                v = v.copy()
            elif isinstance(v, dict):
                v = dict(v)
            elif isinstance(v, list):
                v = list(v)
            setattr(c, f.name, v)
        return c
