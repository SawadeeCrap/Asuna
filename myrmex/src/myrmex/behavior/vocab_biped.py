"""Procedural behaviour vocabulary for bipeds.

Every behaviour is a small *generator* with sampled parameters – never a
clip.  It writes targets into a :class:`~myrmex.motion.command.MotionCommand`;
the motor's springs and constraints make them physical.  Durations are
measured in beats and softly aligned to the beat grid, so behaviour changes
feel phrased without being clocked.

Affinity functions express when a behaviour "makes sense" given the drives;
the selector multiplies them by strategy weights and novelty penalties.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..motion.command import ArmGesture, MotionCommand
from ..util.mathutil import clamp, lerp, smootherstep, smoothstep, wrap_angle


@dataclass
class Ctx:
    t: float
    dt: float
    fr: object
    drives: object
    mods: dict
    rng: object
    pos: np.ndarray
    psi: float
    speed: float
    stage_center: np.ndarray
    stage_radius: float
    beat_dur: float
    groove: object
    gaze: object
    section: str
    strategy: str
    anticipation: dict = field(default_factory=dict)
    height: float = 1.7
    events: list = field(default_factory=list)

    def m(self, key: str, default: float = 1.0) -> float:
        return float(self.mods.get(key, default))


def envelope(p: float, attack: float = 0.25, release: float = 0.25) -> float:
    """0 -> 1 -> 0 over normalised progress p with smooth attack / release."""
    a = smootherstep(p / max(attack, 1e-3))
    r = smootherstep((1.0 - p) / max(release, 1e-3))
    return min(a, r)


class Behavior:
    name = "base"
    locomotion = False
    min_beats = 2.0
    max_beats = 8.0
    interruptible = True

    def __init__(self, ctx: Ctx, **params):
        self.t0 = ctx.t
        self.params = params
        beats = ctx.rng.uniform(self.min_beats, self.max_beats) / max(ctx.m("behavior_tempo"), 0.2)
        dur = beats * ctx.beat_dur
        # Soft alignment: with probability ~ groove, end on a beat boundary.
        if ctx.rng.chance(0.3 + 0.6 * ctx.drives.groove):
            phase = ctx.fr.beat_phase if ctx.fr is not None else 0.0
            end_beat = math.ceil(beats + phase)
            dur = (end_beat - phase) * ctx.beat_dur
        self.duration = max(0.35, dur)
        self.setup(ctx)

    # -------------------------------------------------------------- overridables
    @staticmethod
    def affinity(ctx: Ctx) -> float:
        return 1.0

    def setup(self, ctx: Ctx) -> None:
        pass

    def apply(self, ctx: Ctx, cmd: MotionCommand) -> None:
        pass

    # -------------------------------------------------------------- helpers
    def progress(self, ctx: Ctx) -> float:
        return clamp((ctx.t - self.t0) / self.duration, 0.0, 1.0)

    def finished(self, ctx: Ctx) -> bool:
        return ctx.t - self.t0 >= self.duration

    def stage_heading(self, ctx: Ctx, spread: float) -> float:
        """A heading that explores but keeps the creature inside its stage."""
        to_c = ctx.stage_center - ctx.pos
        dist = float(np.linalg.norm(to_c[:2]))
        center_yaw = math.atan2(float(to_c[1]), float(to_c[0])) if dist > 1e-3 else ctx.psi
        edge = smoothstep(0.35, 0.9, dist / max(ctx.stage_radius, 1e-3))
        wander = ctx.psi + ctx.rng.normal(0.0, spread)
        return wander + wrap_angle(center_yaw - wander) * edge


# ============================================================================ standing
class Idle(Behavior):
    name = "idle"
    min_beats, max_beats = 4.0, 12.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.6 + 1.3 * (1.0 - d.arousal) + 0.3 * d.curiosity - 0.8 * d.boredom

    def setup(self, ctx):
        self.side = ctx.rng.choice([-1.0, 1.0]) * ctx.rng.uniform(0.3, 0.8)
        self.lean = ctx.rng.normal(0.0, 0.03)

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        cmd.speed = 0.0
        cmd.weight_bias = self.side * smootherstep(p * 2.0)
        cmd.lean = self.lean + 0.03 * (ctx.drives.tension - 0.4)


class WeightShift(Behavior):
    name = "weight_shift"
    min_beats, max_beats = 2.0, 6.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.7 + 0.6 * d.groove + 0.4 * (1.0 - d.arousal)

    def setup(self, ctx):
        self.side = ctx.rng.choice([-1.0, 1.0])
        self.back = ctx.rng.chance(0.4)
        self.amount = ctx.rng.uniform(0.6, 1.0)
        self.tilt = ctx.rng.uniform(0.02, 0.07)

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        w = smootherstep(p / 0.35) if not self.back else envelope(p, 0.35, 0.35)
        cmd.speed = 0.0
        cmd.weight_bias = self.side * self.amount * w
        cmd.side_lean = -self.side * self.tilt * w
        cmd.hip_sway = 1.0


class Groove(Behavior):
    name = "groove"
    min_beats, max_beats = 4.0, 16.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.3 + 2.6 * d.groove * (0.4 + d.arousal) + 0.4 * d.confidence - 1.0 * d.boredom

    def setup(self, ctx):
        r = ctx.rng
        self.style = r.choice(["bounce", "sway", "bounce_sway", "shoulders", "subtle", "hips"])
        self.bounce_w = {"bounce": 1.0, "sway": 0.3, "bounce_sway": 0.8, "shoulders": 0.4, "subtle": 0.35,
                         "hips": 0.5}[self.style] * r.uniform(0.8, 1.2)
        self.sway_w = {"bounce": 0.25, "sway": 1.0, "bounce_sway": 0.8, "shoulders": 0.3, "subtle": 0.3,
                       "hips": 1.0}[self.style] * r.uniform(0.8, 1.2)
        self.shoulder_w = 1.0 if self.style == "shoulders" else r.uniform(0.0, 0.3)
        self.nod_w = r.uniform(0.2, 1.0)
        self.twist_w = r.uniform(0.0, 0.6)
        self.facing = self.stage_heading(ctx, 0.6) if r.chance(0.3) else None

    def apply(self, ctx, cmd):
        d = ctx.drives
        p = self.progress(ctx)
        env = envelope(p, 0.15, 0.2)
        g = ctx.groove
        amp = (0.35 + 0.8 * d.arousal) * env
        b = g.bounce(0.3 + 0.5 * d.arousal)
        sw = g.sway()
        cmd.speed = 0.0
        if self.facing is not None:
            cmd.heading = self.facing
        cmd.crouch = 0.05 + 0.16 * self.bounce_w * amp * b * ctx.m("bounce_amount")
        cmd.weight_bias = 0.85 * self.sway_w * amp * sw * ctx.m("sway_amount")
        cmd.side_lean = -0.05 * self.sway_w * amp * sw
        cmd.twist = 0.08 * self.twist_w * amp * math.sin(2.0 * math.pi * g.phase * 0.5 + 0.7)
        cmd.head_nod = -0.10 * self.nod_w * amp * b
        cmd.shoulder_raise = 0.5 * self.shoulder_w * amp * b
        cmd.hip_sway = 1.2
        cmd.arm_swing = 0.3


class Pose(Behavior):
    name = "pose"
    min_beats, max_beats = 2.0, 5.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.15 + 0.8 * d.confidence + 0.4 * d.arousal

    def setup(self, ctx):
        r = ctx.rng
        self.side = r.choice([-1.0, 1.0])
        self.kind = r.choice(["hip_hand", "look_over_shoulder", "lean_back", "reach_up", "contrapposto"])
        self.head_yaw = r.uniform(0.4, 0.9) * self.side
        h = ctx.height
        s = "l" if self.side > 0 else "r"
        if self.kind == "hip_hand":
            self.gestures = {s: ArmGesture(np.array([0.02, 0.2 * self.side * h / 1.7, 0.02]), 1.0, 0.95, 0.4)}
        elif self.kind == "reach_up":
            self.gestures = {s: ArmGesture(np.array([0.12, 0.12 * self.side, 0.62 * h / 1.7]), 1.0, 0.6, -0.2)}
        elif self.kind == "lean_back":
            self.gestures = {}
        else:
            self.gestures = {}

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        env = envelope(p, 0.12, 0.25)
        cmd.speed = 0.0
        cmd.weight_bias = 0.9 * self.side * env
        cmd.side_lean = -0.06 * self.side * env
        if self.kind == "lean_back":
            cmd.lean = -0.12 * env
        if self.kind in ("look_over_shoulder", "contrapposto"):
            cmd.twist = 0.25 * self.head_yaw * env
            fwd = 2.0 * math.cos(self.head_yaw * 1.4)
            left = 2.0 * math.sin(self.head_yaw * 1.4)
            ctx.gaze.look_local(ctx.t, fwd, left, 0.05, hold=0.3)
        cmd.hip_sway = 1.3
        if self.gestures:
            cmd.gestures = {k: ArmGesture(g.target, g.weight * env, g.elbow_out, g.hand_pitch)
                            for k, g in self.gestures.items()}


class Gesture(Behavior):
    name = "gesture"
    min_beats, max_beats = 2.0, 6.0
    KINDS = ("raise_hand", "reach", "open_arms", "hand_hip", "hair_touch", "sweep", "low_flow")

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return (0.25 + 0.6 * d.arousal + 0.5 * d.confidence) * ctx.m("gesture_rate")

    def setup(self, ctx):
        r = ctx.rng
        kind = self.params.get("kind")
        if kind is None:
            weights = [ctx.mods.get("_gesture_penalty_" + k, 1.0) for k in self.KINDS]
            kind = self.KINDS[r.weighted_index(weights)]
        self.kind = kind
        self.side = r.choice(["l", "r"])
        self.sign = 1.0 if self.side == "l" else -1.0
        self.amp = r.uniform(0.75, 1.1)

    def targets(self, ctx, p):
        h = ctx.height / 1.7
        s = self.sign
        k = self.kind
        if k == "raise_hand":
            return {self.side: ArmGesture(np.array([0.18, 0.2 * s, 0.75 * h * self.amp]), 1.0, 0.55, -0.3)}
        if k == "reach":
            return {self.side: ArmGesture(np.array([0.55 * h * self.amp, 0.12 * s, 0.35 * h]), 1.0, 0.3, 0.1)}
        if k == "open_arms":
            return {"l": ArmGesture(np.array([0.12, 0.55 * h * self.amp, 0.36 * h]), 1.0, 0.8, -0.2),
                    "r": ArmGesture(np.array([0.12, -0.55 * h * self.amp, 0.36 * h]), 1.0, 0.8, -0.2)}
        if k == "hand_hip":
            return {self.side: ArmGesture(np.array([0.02, 0.2 * s * h, 0.02 * h]), 1.0, 0.95, 0.4)}
        if k == "hair_touch":
            return {self.side: ArmGesture(np.array([0.04, 0.12 * s * h, 0.62 * h]), 1.0, 0.85, 0.6)}
        if k == "sweep":
            lat = lerp(0.5 * s, -0.15 * s, smootherstep(p))
            return {self.side: ArmGesture(np.array([0.4 * h, lat * h, 0.3 * h]), 1.0, 0.5, 0.0)}
        # low_flow: hand drifts outward at waist height
        return {self.side: ArmGesture(np.array([0.25 * h, (0.35 + 0.1 * math.sin(p * 6.0)) * s * h, 0.12 * h]),
                                      1.0, 0.6, 0.2)}

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        env = envelope(p, 0.3, 0.3)
        cmd.speed = min(cmd.speed, 0.25)
        tg = self.targets(ctx, p)
        cmd.gestures = {k: ArmGesture(g.target, g.weight * env, g.elbow_out, g.hand_pitch) for k, g in tg.items()}
        if self.kind in ("reach", "raise_hand"):
            f, l, u = tg[self.side].target
            ctx.gaze.look_local(ctx.t, 2.0, 2.0 * l, 0.6 * u, hold=0.2)


class LookAround(Behavior):
    name = "look_around"
    min_beats, max_beats = 4.0, 8.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.3 + 1.8 * d.curiosity + 1.0 * d.suspense + 0.6 * d.boredom

    def setup(self, ctx):
        r = ctx.rng
        n = r.randint(2, 4)
        self.points = []
        side = r.choice([-1.0, 1.0])
        for i in range(n):
            ang = side * r.uniform(0.5, 1.3) * (1 if i % 2 == 0 else -0.8)
            self.points.append((ang, r.uniform(-0.15, 0.25)))
        self.twist_follow = r.uniform(0.15, 0.4)

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        k = min(int(p * len(self.points)), len(self.points) - 1)
        ang, el = self.points[k]
        cmd.speed = 0.0
        ctx.gaze.look_local(ctx.t, 3.0 * math.cos(ang), 3.0 * math.sin(ang), 3.0 * el, hold=0.25)
        cmd.twist = self.twist_follow * ang * envelope(p, 0.15, 0.25)


class Lean(Behavior):
    name = "lean"
    min_beats, max_beats = 2.0, 6.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.3 + 0.6 * d.tension + 0.4 * d.curiosity

    def setup(self, ctx):
        r = ctx.rng
        self.fwd = r.choice([0.14, -0.1, 0.08])
        self.side = r.normal(0.0, 0.05)
        self.crouch = r.uniform(0.0, 0.2) if self.fwd > 0 else 0.0

    def apply(self, ctx, cmd):
        env = envelope(self.progress(ctx), 0.3, 0.3)
        cmd.speed = 0.0
        cmd.lean = self.fwd * env
        cmd.side_lean = self.side * env
        cmd.crouch = max(cmd.crouch, self.crouch * env)


class Crouch(Behavior):
    name = "crouch"
    min_beats, max_beats = 2.0, 6.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.15 + 1.4 * d.tension * (1.0 - 0.5 * d.fatigue) + (0.6 if ctx.section == "build" else 0.0)

    def setup(self, ctx):
        self.depth = ctx.rng.uniform(0.35, 0.75)
        self.lean = ctx.rng.uniform(0.05, 0.2)

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        env = envelope(p, 0.4, 0.2)
        cmd.speed = 0.0
        cmd.crouch = self.depth * env
        cmd.lean = self.lean * env
        cmd.arm_swing = 0.2


class Rise(Behavior):
    name = "rise"
    min_beats, max_beats = 1.5, 4.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.1 + 0.5 * d.arousal + 0.8 * d.startle

    def setup(self, ctx):
        self.amount = ctx.rng.uniform(0.5, 0.9)
        self.arms = ctx.rng.chance(0.5)

    def apply(self, ctx, cmd):
        env = envelope(self.progress(ctx), 0.25, 0.35)
        cmd.speed = 0.0
        cmd.rise = self.amount * env
        cmd.lean = -0.06 * env
        cmd.head_nod = 0.12 * env
        if self.arms:
            h = ctx.height / 1.7
            cmd.gestures = {"l": ArmGesture(np.array([0.1, 0.25, 0.8 * h]), env, 0.5, -0.3),
                            "r": ArmGesture(np.array([0.1, -0.25, 0.8 * h]), env, 0.5, -0.3)}


# ============================================================================ locomotion
class Walk(Behavior):
    name = "walk"
    locomotion = True
    min_beats, max_beats = 4.0, 16.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.4 + 1.8 * d.arousal * (1.0 - 0.7 * d.fatigue) + 0.7 * d.curiosity - 0.3 * d.suspense

    def setup(self, ctx):
        r = ctx.rng
        self.heading = self.stage_heading(ctx, 1.0)
        self.speed = lerp(0.4, 1.1, ctx.drives.arousal) * r.uniform(0.85, 1.15)
        self.curv = r.normal(0.0, 0.25)
        self.wiggle = r.uniform(0.0, 0.35)
        self.phase = r.uniform(0.0, 6.28)

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        env = smootherstep(p / 0.1) * smootherstep((1.0 - p) / 0.08)
        self.heading += (self.curv + ctx.m("turn_curvature", 0.0)) * ctx.dt
        self.heading += 0.25 * self.wiggle * math.sin(ctx.t * 0.9 + self.phase) * ctx.dt
        # Steer back inside the stage.
        to_c = ctx.stage_center - ctx.pos
        dist = float(np.linalg.norm(to_c[:2]))
        if dist > 0.75 * ctx.stage_radius:
            cy = math.atan2(float(to_c[1]), float(to_c[0]))
            self.heading += wrap_angle(cy - self.heading) * min(1.0, ctx.dt * 1.2)
        cmd.speed = self.speed * ctx.m("walk_speed") * env
        cmd.heading = self.heading
        cmd.arm_swing = 1.0
        cmd.lean = 0.02 + 0.03 * ctx.drives.arousal


class Turn(Behavior):
    name = "turn"
    min_beats, max_beats = 2.0, 4.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.35 + 0.7 * d.curiosity + 0.7 * d.agitation + 0.9 * d.boredom

    def setup(self, ctx):
        r = ctx.rng
        target = self.params.get("heading")
        if target is None:
            delta = r.choice([-1.0, 1.0]) * r.uniform(0.9, 2.8)
            target = self.stage_heading(ctx, 0.3) if r.chance(0.3) else ctx.psi + delta
        self.target = target
        self.start = ctx.psi
        self.counter = r.uniform(0.03, 0.08)

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        d = wrap_angle(self.target - self.start)
        # Anticipation: a small counter-twist before the turn, gaze leads the body.
        if p < 0.15:
            cmd.twist = -math.copysign(self.counter, d) * smoothstep(0.0, 0.15, p)
        ctx.gaze.look_local(ctx.t, 3.0 * math.cos(d * 0.9), 3.0 * math.sin(d * 0.9), 0.0, hold=0.2)
        cmd.speed = 0.0
        cmd.heading = self.start + d * smootherstep(clamp((p - 0.1) / 0.8, 0.0, 1.0))


class Spin(Turn):
    name = "spin"
    min_beats, max_beats = 1.5, 3.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.05 + 0.5 * d.arousal * d.confidence

    def setup(self, ctx):
        super().setup(ctx)
        self.target = ctx.psi + ctx.rng.choice([-1.0, 1.0]) * 2.0 * math.pi * ctx.rng.uniform(0.85, 1.0)

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        d = self.target - self.start
        cmd.speed = 0.0
        cmd.heading = self.start + d * smootherstep(p)
        cmd.rise = 0.3 * envelope(p, 0.2, 0.3)
        cmd.energy = 1.0


class Sidestep(Behavior):
    name = "sidestep"
    locomotion = True
    min_beats, max_beats = 2.0, 5.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.3 + 0.6 * d.groove + 0.5 * d.agitation

    def setup(self, ctx):
        self.v = ctx.rng.choice([-1.0, 1.0]) * ctx.rng.uniform(0.2, 0.4)

    def apply(self, ctx, cmd):
        env = envelope(self.progress(ctx), 0.15, 0.2)
        cmd.speed = 0.0
        cmd.lateral = self.v * env
        cmd.side_lean = 0.04 * math.copysign(1.0, self.v) * env


class Backstep(Behavior):
    name = "backstep"
    locomotion = True
    min_beats, max_beats = 1.5, 3.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.15 + 0.6 * d.agitation + 0.3 * d.tension

    def apply(self, ctx, cmd):
        env = envelope(self.progress(ctx), 0.15, 0.25)
        cmd.speed = -0.32 * env
        cmd.lean = -0.05 * env


class Hesitate(Behavior):
    name = "hesitate"
    min_beats, max_beats = 1.0, 3.0

    @staticmethod
    def affinity(ctx):
        d = ctx.drives
        return 0.15 + 2.0 * ctx.m("hesitation", 0.0) + 1.4 * d.suspense + 0.3 * d.tension

    def setup(self, ctx):
        self.hold = ctx.speed > 0.2 and ctx.rng.chance(0.7)
        self.fix = (ctx.rng.uniform(1.5, 4.0), ctx.rng.normal(0.0, 0.8), ctx.rng.normal(0.0, 0.2))

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        cmd.speed = 0.0
        cmd.hold_step = self.hold and p < 0.8
        cmd.tension = max(cmd.tension, 0.85)
        f, l, u = self.fix
        ctx.gaze.look_local(ctx.t, f, l, u, hold=0.2)


class Recoil(Behavior):
    name = "recoil"
    locomotion = True
    min_beats, max_beats = 1.0, 2.0

    @staticmethod
    def affinity(ctx):
        return 0.02 + 1.5 * ctx.drives.startle

    def setup(self, ctx):
        self.side = ctx.rng.choice([-1.0, 1.0])

    def apply(self, ctx, cmd):
        p = self.progress(ctx)
        env = envelope(p, 0.08, 0.5)
        cmd.speed = -0.45 * env
        cmd.lean = -0.14 * env
        cmd.crouch = 0.2 * env
        cmd.shoulder_raise = 0.8 * env
        cmd.tension = 1.0
        h = ctx.height / 1.7
        cmd.gestures = {"l": ArmGesture(np.array([0.3 * h, 0.15, 0.3 * h]), 0.8 * env, 0.3, 0.3),
                        "r": ArmGesture(np.array([0.3 * h, -0.15, 0.3 * h]), 0.8 * env, 0.3, 0.3)}


VOCABULARY = {cls.name: cls for cls in (Idle, WeightShift, Groove, Pose, Gesture, LookAround, Lean, Crouch,
                                        Rise, Walk, Turn, Spin, Sidestep, Backstep, Hesitate, Recoil)}

STRATEGIES = {
    "still": {"idle": 3.0, "look_around": 2.0, "weight_shift": 2.0, "lean": 1.0, "gesture": 0.4, "hesitate": 0.6,
              "turn": 0.5},
    "sway": {"groove": 4.0, "weight_shift": 1.5, "gesture": 1.0, "turn": 0.6, "pose": 0.6, "idle": 0.4,
             "sidestep": 0.8},
    "stroll": {"walk": 4.0, "look_around": 1.2, "turn": 1.0, "idle": 0.6, "gesture": 0.3, "hesitate": 0.4},
    "travel": {"walk": 5.0, "turn": 1.0, "sidestep": 0.6, "pose": 0.3, "spin": 0.2},
    "display": {"pose": 2.0, "gesture": 2.6, "groove": 2.2, "turn": 0.8, "spin": 0.5, "lean": 0.8, "rise": 0.8},
    "agitated": {"walk": 2.0, "turn": 2.0, "sidestep": 1.4, "backstep": 1.0, "hesitate": 1.4, "look_around": 1.4},
    "suspense": {"crouch": 2.0, "hesitate": 1.5, "look_around": 1.2, "lean": 1.2, "groove": 0.8, "idle": 0.6},
}

SECTION_STRATEGIES = {
    "intro": {"still": 3.0, "stroll": 2.0, "sway": 1.0},
    "build": {"suspense": 3.0, "sway": 2.0, "stroll": 1.0},
    "drop": {"display": 3.0, "travel": 2.0, "sway": 2.0, "agitated": 0.8},
    "peak": {"display": 2.5, "travel": 2.0, "sway": 2.5, "agitated": 1.0},
    "break": {"still": 2.0, "stroll": 2.0, "suspense": 1.0},
    "return": {"sway": 3.0, "travel": 2.0, "display": 1.5},
    "groove": {"sway": 3.0, "stroll": 2.0, "display": 1.0, "travel": 1.0},
    "outro": {"still": 3.0, "stroll": 1.0},
    "silence": {"still": 4.0, "suspense": 1.5},
}
