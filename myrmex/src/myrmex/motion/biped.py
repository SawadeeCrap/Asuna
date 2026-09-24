"""Procedural biped motor: organic walking, standing and posing for any humanoid rig.

Design principles
-----------------
* **Feet are real contacts.**  A planted foot is anchored in the world at a
  pivot (heel -> flat -> ball) and never slides.  Heel strike, roll-over and
  heel rise are rotations about those pivots.
* **Stepping is need-driven, not clock-driven.**  A foot lifts when it trails
  its ideal support position by more than a step length (or when the body
  turned away from it), and only after a minimum double-support time.  Walk,
  turn-in-place, side-steps, back-steps, starts, stops and fidget-steps all
  come out of the same rule; cadence emerges from speed and stride length.
* **The pelvis is an inverted pendulum.**  Its height is limited by what the
  loaded leg can reach, so the familiar vertical bob, the dip at double
  support and the heel rise of the trailing leg *emerge* from leg geometry
  instead of being drawn as sine waves.
* **Everything else follows through springs** (second-order dynamics):
  pelvic sway/rotation/obliquity, spine counter-rotation, pendulum arms,
  gaze-stabilised head, breathing and correlated micro-motion.
* **Arms respect the body** via torso ellipses and thigh capsules derived
  from the actual mesh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..util.mathutil import (UP, axis_angle_matrix, clamp, lerp, normalize, rot_z, smootherstep,
                             smoothstep, wrap_angle)
from ..util.noise import LatentField
from ..util.rng import RngStreams
from ..util.springs import SecondOrder
from .bodyplan import BipedPlan, LegPlan
from .command import MotionCommand
from .ik import two_bone
from .skeleton import Pose, delta_aligning

G = 9.81


def rot_about(point: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """4x4 rotation about an axis through ``point``."""
    R = axis_angle_matrix(axis, angle)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = point - R @ point
    return M


@dataclass
class FootState:
    leg: LegPlan
    fdir0: np.ndarray                 # rest foot forward (ground plane)
    lat0: np.ndarray                  # rest foot lateral axis (pitch axis)
    C0: np.ndarray                    # rest foot centre on the ground
    planted: bool = True
    pivot: str = "heel"               # heel | ball
    pivot_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0                  # delta yaw vs rest foot
    pitch: float = 0.0                # + = heel up (plantarflex), - = toes up
    toe_bend: float = 0.0
    touchdown_time: float = -10.0
    liftoff_time: float = -10.0
    load: float = 0.5
    # swing
    swing_t: float = 0.0
    swing_T: float = 0.4
    lo_ankle: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lo_yaw: float = 0.0
    lo_pitch: float = 0.0
    tgt_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tgt_yaw: float = 0.0
    td_pitch: float = -0.2
    clearance: float = 0.08
    hold: float = 0.0                 # hesitation amount 0..1
    steps: int = 0

    def rotation(self, yaw: float | None = None, pitch: float | None = None) -> np.ndarray:
        y = self.yaw if yaw is None else yaw
        p = self.pitch if pitch is None else pitch
        return rot_z(y) @ axis_angle_matrix(self.lat0, p)

    def rest_pivot(self, name: str) -> np.ndarray:
        leg = self.leg
        return {"heel": leg.heel, "ball": leg.ball_pt, "toe": leg.toe}[name]

    def delta(self) -> np.ndarray:
        """Rigid world delta of the foot bone."""
        if self.planted:
            R = self.rotation()
            P0 = self.rest_pivot(self.pivot)
            D = np.eye(4)
            D[:3, :3] = R
            D[:3, 3] = self.pivot_world - R @ P0
            return D
        raise RuntimeError("delta() of a swinging foot is computed by the motor")

    def set_pivot(self, name: str, D: np.ndarray) -> None:
        """Switch pivot keeping the current pose (continuity)."""
        P0 = self.rest_pivot(name)
        self.pivot = name
        self.pivot_world = D[:3, :3] @ P0 + D[:3, 3]


class BipedMotor:
    """Turns :class:`MotionCommand` streams into poses for a :class:`BipedPlan`."""

    def __init__(self, plan: BipedPlan, seed: int = 0, style: dict | None = None):
        self.plan = plan
        self.sk = plan.sk
        self.pose = Pose(self.sk)
        self.rng = RngStreams(seed)
        self.style = {
            "stance_width": 1.0,        # scale of the rest foot spacing
            "step_length": 1.0,
            "toe_out": math.radians(4.0),
            "heel_strike": math.radians(14.0),
            "toe_off": math.radians(38.0),
            "knee_softness": 0.03,
            "pelvis_drop": 0.015,       # nominal knee flex (fraction of leg length)
            "hip_rotation": math.radians(7.0),
            "hip_drop": math.radians(5.0),
            "lateral_sway": 0.45,       # fraction of half stance width
            "arm_swing": math.radians(20.0),
            "arm_hang": math.radians(9.0),
            "elbow_bend": math.radians(12.0),
            "counter_rotation": 0.8,
            "breath_rate": 0.24,
            "micro": 1.0,
        }
        if plan.high_heels:
            self.style.update({"heel_strike": math.radians(4.0), "toe_off": math.radians(26.0),
                               "stance_width": 0.55, "hip_rotation": math.radians(9.0),
                               "hip_drop": math.radians(6.5), "step_length": 0.8})
        if style:
            self.style.update(style)
        self.t = 0.0
        self.events: list[tuple[float, str, dict]] = []
        # ------------------------------------------------ root / locomotion state
        fwd0 = self.sk.forward
        self.psi0 = math.atan2(float(fwd0[1]), float(fwd0[0]))
        self.psi = self.psi0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.psi_dyn = SecondOrder(1.0, 0.8, 0.0, self.psi0)
        self.vel_dyn = SecondOrder(1.1, 0.9, 0.0, np.zeros(3))
        self.speed_ref = 0.0
        # ------------------------------------------------ feet
        self.feet: dict[str, FootState] = {}
        for s, leg in plan.legs.items():
            fd = leg.toe - leg.heel
            fd[2] = 0.0
            fdir0 = normalize(fd, fwd0)
            lat0 = normalize(np.cross(UP, fdir0))
            C0 = 0.5 * (leg.heel + leg.ball_pt)
            C0[2] = 0.0
            fs = FootState(leg, fdir0, lat0, C0)
            fs.planted = True
            fs.pivot = "heel"
            fs.pivot_world = leg.heel.copy()
            fs.touchdown_time = -1.0
            self.feet[s] = fs
        # ------------------------------------------------ pelvis / body springs
        pe = plan.pelvis_height
        self.pelvis_z = SecondOrder(2.8, 0.8, 0.0, pe)
        self.pelvis_lat = SecondOrder(1.4, 0.7, 0.0, 0.0)
        self.pelvis_fwd = SecondOrder(1.6, 0.8, 0.0, 0.0)
        self.pelvis_yaw = SecondOrder(1.6, 0.6, 0.0, 0.0)
        self.pelvis_roll = SecondOrder(1.8, 0.55, 0.0, 0.0)
        self.pelvis_pitch = SecondOrder(1.5, 0.7, 0.0, 0.0)
        self.chest_rot = SecondOrder(1.5, 0.6, 0.0, np.zeros(3))      # (lean, side, twist)
        self.head_rot = SecondOrder(2.6, 0.62, 0.15, np.zeros(2))     # (yaw, pitch) relative to chest
        self.arm_flex = {s: SecondOrder(1.55, 0.42, 0.0, 0.0) for s in plan.arms}
        self.elbow_flex = {s: SecondOrder(2.1, 0.5, 0.0, 0.0) for s in plan.arms}
        self.arm_abd = {s: SecondOrder(1.8, 0.7, 0.0, 0.0) for s in plan.arms}
        self.gesture_w = {s: SecondOrder(1.6, 0.85, 0.0, 0.0) for s in plan.arms}
        self.gesture_pos = {s: SecondOrder(1.8, 0.75, 0.0, np.zeros(3)) for s in plan.arms}
        self.cmd_crouch = SecondOrder(1.2, 0.8, 0.0, 0.0)
        self.cmd_rise = SecondOrder(1.2, 0.8, 0.0, 0.0)
        self.weight = SecondOrder(0.9, 0.85, 0.0, 0.0)
        self.impulse = SecondOrder(3.0, 0.35, 0.0, np.zeros(3))         # body jolts (x fwd, y left, z up)
        self.breath_phase = self.rng.stream("breath").random()
        self.micro = LatentField(self.rng.subseed("micro"), [0.07, 0.17, 0.33, 0.71, 1.4],
                                 ["p_lat", "p_fwd", "p_yaw", "p_roll", "lean", "side", "twist", "h_yaw",
                                  "h_pitch", "arm_l", "arm_r", "elb_l", "elb_r", "weight"],
                                 delays={"p_lat": 0.0, "lean": 0.08, "side": 0.1, "twist": 0.12,
                                         "h_yaw": 0.22, "h_pitch": 0.2, "arm_l": 0.3, "arm_r": 0.33})
        self._arm_rest_abd = {s: self._rest_abduction(a) for s, a in plan.arms.items()}
        self._step_len = 0.0
        self._zlim_hist: list[tuple[float, float]] = []
        self._low_water = plan.pelvis_height
        self._last_pelvis_delta = np.eye(4)
        self.contact_debug: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    @property
    def dpsi(self) -> float:
        return wrap_angle(self.psi - self.psi0)

    def char_axes(self, psi: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = wrap_angle((self.psi if psi is None else psi) - self.psi0)
        R = rot_z(d)
        return R @ self.sk.forward, R @ self.sk.left

    def _rest_abduction(self, arm) -> float:
        v = arm.elbow - arm.shoulder
        lat = float(np.dot(v, self.sk.left)) * arm.sign
        return math.atan2(max(lat, 0.0), max(-float(v[2]), 1e-6))

    def home_center(self, foot: FootState, pos: np.ndarray, psi: float) -> np.ndarray:
        """Neutral ground position of a foot's centre for a root at pos/psi."""
        C0 = foot.C0.copy()
        lat = float(np.dot(C0, self.sk.left))
        C0 = C0 + self.sk.left * lat * (self.style["stance_width"] - 1.0)
        return pos + rot_z(wrap_angle(psi - self.psi0)) @ C0

    def step_length(self, speed: float, cmd: MotionCommand) -> float:
        L = self.plan.leg_length
        vref = math.sqrt(G * L) * 0.45
        base = L * 0.82 * math.sqrt(max(speed, 0.0) / vref) if speed > 1e-3 else 0.0
        base *= self.style["step_length"] * cmd.stride_scale
        return clamp(base, 0.0, 0.95 * L * self.style["step_length"])

    def swing_duration(self, cmd: MotionCommand, dist: float) -> float:
        L = self.plan.leg_length
        base = 0.36 * math.sqrt(L / 0.85)
        base *= 1.12 - 0.25 * cmd.energy
        base *= 1.0 + 0.35 * max(0.0, dist / max(L, 1e-3) - 0.6)
        return clamp(base / max(cmd.cadence_scale, 0.3), 0.22, 0.7)

    # ------------------------------------------------------------------ main update
    def update(self, dt: float, cmd: MotionCommand) -> Pose:
        self.t += dt
        self._update_root(dt, cmd)
        self._update_steps(dt, cmd)
        pelvis_D = self._update_pelvis(dt, cmd)
        self._solve_legs(pelvis_D, cmd)
        self._update_upper_body(dt, cmd, pelvis_D)
        self.pose.solve()
        return self.pose

    # ------------------------------------------------------------------ root
    def _update_root(self, dt: float, cmd: MotionCommand) -> None:
        if cmd.heading is not None:
            target = self.psi + wrap_angle(cmd.heading - self.psi)
        else:
            target = self.psi_dyn.y + cmd.turn_rate * 0.6
        self.psi_dyn.set_params(0.7 + 0.8 * cmd.energy, 0.65 + 0.3 * cmd.tension, 0.0)
        prev = self.psi
        self.psi = float(self.psi_dyn.update(dt, target))
        rate = (self.psi - prev) / max(dt, 1e-6)
        max_rate = 1.5 + 3.0 * cmd.energy
        if abs(rate) > max_rate:
            self.psi = prev + math.copysign(max_rate * dt, rate)
            self.psi_dyn.y = self.psi
        fwd, left = self.char_axes()
        L = self.plan.leg_length
        vmax = 1.1 * math.sqrt(G * L) * self.style["step_length"] * (0.7 + 0.5 * cmd.cadence_scale)
        sp = clamp(cmd.speed, -0.4 * vmax, vmax)
        lat = clamp(cmd.lateral, -0.35 * vmax, 0.35 * vmax)
        v_des = fwd * sp + left * lat
        if cmd.hold_step:
            v_des = v_des * 0.05
        self.vel_dyn.set_params(0.8 + 0.9 * cmd.energy, 0.95, 0.0)
        v_prev = self.vel.copy()
        v = self.vel_dyn.update(dt, v_des)
        acc = (v - v_prev) / max(dt, 1e-6)
        amax = 1.2 + 2.8 * cmd.energy
        na = float(np.linalg.norm(acc))
        if na > amax:
            v = v_prev + acc * (amax / na) * dt
            self.vel_dyn.y = v
        self.vel = np.array([v[0], v[1], 0.0])
        self.pos = self.pos + self.vel * dt
        self.speed_ref = float(np.linalg.norm(self.vel))

    # ------------------------------------------------------------------ stepping
    def _foot_delta(self, f: FootState) -> np.ndarray:
        if f.planted:
            return f.delta()
        return self._swing_delta(f)

    def _swing_delta(self, f: FootState) -> np.ndarray:
        s = f.swing_t
        # Horizontal progress starts after a vertical peel-off and ends with a vertical set-down.
        h = smootherstep(clamp((s - 0.06) / 0.86, 0.0, 1.0))
        yaw = f.lo_yaw + wrap_angle(f.tgt_yaw - f.lo_yaw) * smootherstep(s)
        p_mid = math.radians(8.0)
        if s < 0.45:
            pitch = lerp(f.lo_pitch, p_mid, smoothstep(0.0, 0.45, s))
        else:
            pitch = lerp(p_mid, f.td_pitch, smoothstep(0.45, 1.0, s))
        R = f.rotation(yaw, pitch)
        # Touchdown pose: heel at the target heel point with td_pitch about the heel.
        R_td = f.rotation(f.tgt_yaw, f.td_pitch)
        heel_td = f.tgt_center + rot_z(f.tgt_yaw) @ (f.leg.heel - f.C0)
        heel_td[2] = 0.0
        ankle_td = heel_td + R_td @ (f.leg.ankle - f.leg.heel)
        ankle = f.lo_ankle + (ankle_td - f.lo_ankle) * h
        a_, b_ = 0.75, 1.25
        peak = (a_ / (a_ + b_)) ** a_ * (b_ / (a_ + b_)) ** b_
        lift = (s ** a_) * ((1.0 - s) ** b_) / peak
        ankle = ankle + UP * f.clearance * lift
        D = np.eye(4)
        D[:3, :3] = R
        D[:3, 3] = ankle - R @ f.leg.ankle
        return D

    def _predict(self, horizon: float, cmd: MotionCommand) -> tuple[np.ndarray, float]:
        """Root position/heading after ``horizon`` s, integrating along the current arc."""
        w = float(self.psi_dyn.yd) if not isinstance(self.psi_dyn.yd, np.ndarray) else 0.0
        v = self.vel
        if abs(w) < 1e-3:
            pos = self.pos + v * horizon
        else:
            a = w * horizon
            perp = np.array([-v[1], v[0], 0.0])
            pos = self.pos + v * (math.sin(a) / w) + perp * ((1.0 - math.cos(a)) / w)
        return pos, self.psi + w * horizon

    def _predict_dir(self, horizon: float) -> np.ndarray:
        w = float(self.psi_dyn.yd) if not isinstance(self.psi_dyn.yd, np.ndarray) else 0.0
        if self.speed_ref < 1e-4:
            return self.char_axes()[0]
        return rot_z(w * horizon) @ (self.vel / self.speed_ref)

    def _update_steps(self, dt: float, cmd: MotionCommand) -> None:
        speed = self.speed_ref
        vdir = self.vel / speed if speed > 1e-4 else self.char_axes()[0]
        step_len = self.step_length(speed, cmd)
        self._step_len = step_len
        swinging = [f for f in self.feet.values() if not f.planted]
        # ---- advance swings
        for f in swinging:
            hold_target = 1.0 if cmd.hold_step else 0.0
            f.hold += (hold_target - f.hold) * min(1.0, dt * (12.0 if hold_target else 6.0))
            rate = (1.0 - 0.97 * f.hold) / max(f.swing_T, 1e-3)
            f.swing_t = min(1.0, f.swing_t + rate * dt)
            if f.swing_t < 0.72:
                remaining = (1.0 - f.swing_t) * f.swing_T
                c, y = self._target_for(f, remaining, cmd, step_len, vdir)
                k = min(1.0, dt * 10.0)
                f.tgt_center = f.tgt_center + (c - f.tgt_center) * k
                f.tgt_yaw = f.tgt_yaw + wrap_angle(y - f.tgt_yaw) * k
            if f.swing_t >= 1.0:
                self._touchdown(f, cmd)
        # ---- decide lift-off
        planted = [f for f in self.feet.values() if f.planted]
        if len(planted) == 2 and not cmd.hold_step:
            ds_min = clamp(0.16 - 0.1 * speed / max(math.sqrt(G * self.plan.leg_length), 1e-3), 0.05, 0.16)
            candidates = []
            for f in planted:
                other = [o for o in planted if o is not f][0]
                urgent = f.pivot == "ball" and f.pitch >= 0.92 * self.style["toe_off"] * 1.6
                if self.t - other.touchdown_time < (0.4 * ds_min if urgent else ds_min):
                    continue
                if self.t - f.touchdown_time < 0.25:
                    continue
                need = self._need(f, cmd, step_len, vdir)
                if f.pivot == "ball" and f.pitch >= 0.92 * self.style["toe_off"] * 1.6:
                    need = max(need, 1.0)          # trailing leg at its limit: go now
                candidates.append((need, f))
            if candidates:
                need, f = max(candidates, key=lambda x: x[0])
                if need >= 1.0:
                    self._liftoff(f, cmd, step_len, vdir)

    def _target_for(self, f: FootState, horizon: float, cmd: MotionCommand, step_len: float,
                    vdir: np.ndarray) -> tuple[np.ndarray, float]:
        pos, psi = self._predict(horizon, cmd)
        home = self.home_center(f, pos, psi)
        lead = 0.45 * step_len
        target = home + self._predict_dir(horizon) * lead
        # Capture-point style correction toward the velocity the body actually has.
        target[2] = 0.0
        side = f.leg.sign
        yaw = wrap_angle(psi - self.psi0) + side * self.style["toe_out"] * 0.5
        return target, yaw

    def _need(self, f: FootState, cmd: MotionCommand, step_len: float, vdir: np.ndarray) -> float:
        T = self.swing_duration(cmd, step_len)
        pos, psi = self._predict(T, cmd)
        target = self.home_center(f, pos, psi) + self._predict_dir(T) * 0.45 * step_len
        cur = self._foot_center_world(f)
        dist = float(np.linalg.norm((target - cur)[:2]))
        yaw_err = abs(wrap_angle((wrap_angle(psi - self.psi0)) - f.yaw))
        moving = self.speed_ref > 0.05
        dist_thr = max(0.8 * step_len, 0.045 * self.plan.leg_length / 0.85 * 1.0) if moving \
            else 0.075 * self.plan.leg_length
        need_d = dist / max(dist_thr, 1e-3)
        need_y = yaw_err / math.radians(24.0)
        return max(need_d, need_y)

    def _foot_center_world(self, f: FootState) -> np.ndarray:
        D = self._foot_delta(f)
        C = D[:3, :3] @ f.C0 + D[:3, 3]
        C[2] = 0.0
        return C

    def _liftoff(self, f: FootState, cmd: MotionCommand, step_len: float, vdir: np.ndarray) -> None:
        D = f.delta()
        f.planted = False
        f.swing_t = 0.0
        f.lo_ankle = D[:3, :3] @ f.leg.ankle + D[:3, 3]
        f.lo_yaw = f.yaw
        f.lo_pitch = f.pitch
        cur = D[:3, :3] @ f.C0 + D[:3, 3]
        T_guess = self.swing_duration(cmd, step_len)
        tc, ty = self._target_for(f, T_guess, cmd, step_len, vdir)
        dist = float(np.linalg.norm((tc - cur)[:2]))
        f.swing_T = self.swing_duration(cmd, dist) * self.rng.stream(f"swing_{f.leg.side}").lognormal_factor(0.06)
        f.tgt_center, f.tgt_yaw = tc, ty
        L = self.plan.leg_length
        f.clearance = (0.045 + 0.05 * cmd.energy) * cmd.step_height * (L / 0.85) \
            * self.rng.stream(f"clear_{f.leg.side}").lognormal_factor(0.12)
        if dist < 0.12 * L:
            f.clearance *= 0.6
        f.td_pitch = -self.style["heel_strike"] * clamp(dist / max(0.5 * L, 1e-3), 0.25, 1.0)
        f.liftoff_time = self.t
        f.hold = 0.0
        self.events.append((self.t, "liftoff", {"foot": f.leg.side}))

    def _touchdown(self, f: FootState, cmd: MotionCommand) -> None:
        D = self._swing_delta(f)
        f.planted = True
        f.yaw = f.tgt_yaw
        f.pitch = f.td_pitch
        f.toe_bend = 0.0
        f.set_pivot("heel", D)
        f.pivot_world[2] = 0.0
        f.touchdown_time = self.t
        f.steps += 1
        speed = self.speed_ref
        # Weight acceptance: a small vertical dip proportional to speed and energy.
        self.impulse.kick(np.array([0.0, 0.0, -(0.05 + 0.25 * speed) * (0.4 + 0.6 * cmd.energy)]))
        self.events.append((self.t, "touchdown", {"foot": f.leg.side, "speed": speed}))

    # ------------------------------------------------------------------ pelvis
    def _update_pelvis(self, dt: float, cmd: MotionCommand) -> np.ndarray:
        plan = self.plan
        fwd, left = self.char_axes()
        feet = self.feet
        self._update_foot_roll(dt, cmd)
        # Load distribution: single support -> the planted foot; double support -> transfer
        # toward the most recently landed foot.
        planted = [f for f in feet.values() if f.planted]
        if len(planted) == 1:
            load_l = 1.0 if planted[0].leg.side == "l" else 0.0
        else:
            fl, fr = feet["l"], feet["r"]
            newer = fl if fl.touchdown_time > fr.touchdown_time else fr
            since = self.t - newer.touchdown_time
            w_new = smoothstep(0.0, 0.14, since)
            base_l = 0.5 + 0.5 * clamp(self.weight.y, -1.0, 1.0)
            if self.speed_ref > 0.05 or since < 0.3:
                load_l = w_new if newer is fl else 1.0 - w_new
                load_l = lerp(base_l, load_l, smoothstep(0.02, 0.2, self.speed_ref) if since > 0.3 else 1.0)
            else:
                load_l = base_l
        self.weight.update(dt, cmd.weight_bias)
        feet["l"].load, feet["r"].load = load_l, 1.0 - load_l
        # Horizontal pelvis position: root + lateral sway toward the loaded foot.
        cl = self._foot_center_world(feet["l"])
        cr = self._foot_center_world(feet["r"])
        mid = 0.5 * (cl + cr)
        half_w = 0.5 * abs(float(np.dot(cl - cr, left)))
        sway = (load_l - 0.5) * 2.0 * half_w * self.style["lateral_sway"] * (1.0 - 0.4 * cmd.tension)
        mc = self.micro.sample(self.t)
        amp = self.style["micro"] * (0.4 + 0.6 * cmd.energy) * (1.0 + 0.8 * cmd.tension)
        lat_off = float(self.pelvis_lat.update(dt, sway + 0.004 * amp * mc["p_lat"]))
        standing = 1.0 - smoothstep(0.03, 0.25, self.speed_ref)
        # When standing, the pelvis settles over the feet (not the drifting root).
        base_xy = lerp(self.pos, mid, standing)
        fwd_off = float(self.pelvis_fwd.update(dt, 0.004 * amp * mc["p_fwd"] - 0.02 * cmd.lean))
        p_xy = base_xy + left * lat_off + fwd * fwd_off
        # Rotations.
        fa = {s: float(np.dot(self._foot_center_world(f) - p_xy, fwd)) for s, f in feet.items()}
        sl = max(self._step_len, 0.1 * plan.leg_length)
        rot = clamp((fa["l"] - fa["r"]) / sl, -1.2, 1.2)
        yaw_t = -self.style["hip_rotation"] * cmd.hip_sway * rot * (1.0 - 0.5 * cmd.tension)
        roll_t = self.style["hip_drop"] * cmd.hip_sway * (load_l - 0.5) * 2.0 * (1.0 - 0.5 * cmd.tension)
        yaw = float(self.pelvis_yaw.update(dt, yaw_t + 0.02 * amp * mc["p_yaw"]))
        roll = float(self.pelvis_roll.update(dt, roll_t + 0.015 * amp * mc["p_roll"]))
        pitch = float(self.pelvis_pitch.update(dt, 0.05 * self.speed_ref + 0.35 * cmd.lean
                                               + 0.25 * self.cmd_crouch.y))
        R = rot_z(self.dpsi + yaw) @ axis_angle_matrix(self.sk.forward, -roll) \
            @ axis_angle_matrix(self.sk.left, pitch)
        # Height: desired, then limited by what the loaded legs can reach.
        crouch = float(self.cmd_crouch.update(dt, cmd.crouch))
        rise = float(self.cmd_rise.update(dt, cmd.rise))
        L = plan.leg_length
        h_stand = plan.pelvis_height * (1.0 - self.style["pelvis_drop"])
        # Walking: ride lower so the double-support dip stays a gentle bob, not a stumble.
        reach = L * (1.0 - self.style["knee_softness"] * 1.3)
        ankle_z0 = float(np.mean([lg.ankle[2] for lg in plan.legs.values()]))
        hip_dz = float(np.mean([lg.hip[2] for lg in plan.legs.values()])) - float(plan.hip_center[2])
        half = 0.55 * self._step_len
        h_ds = ankle_z0 + math.sqrt(max(reach * reach - half * half, 0.5 * reach * reach)) - hip_dz
        bob = (0.010 + 0.014 * cmd.energy) * L / 0.85
        walking = smoothstep(0.05, 0.45, self.speed_ref)
        # Adaptive ride height: stay at most `bob` above the lowest reachable height of the
        # last gait cycle, so the double-support dip becomes a gentle, even bob.
        z_lim_now = self._reach_limited_height(R, p_xy, plan.hip_center, 1e9, cmd, soft=True, probe=True)
        self._zlim_hist.append((self.t, z_lim_now))
        while self._zlim_hist and self.t - self._zlim_hist[0][0] > 1.1:
            self._zlim_hist.pop(0)
        low = min(z for _, z in self._zlim_hist)
        self._low_water += (low - self._low_water) * min(1.0, dt * 2.5)
        h_walk = min(h_stand, h_ds + bob, self._low_water + bob)
        h_des = lerp(h_stand, h_walk, walking) - crouch * 0.28 * L + rise * 0.04 * L
        imp = self.impulse.update(dt, np.zeros(3))
        h_des += float(imp[2]) * 0.1
        h0 = plan.pelvis_height
        hip_center0 = plan.hip_center
        z = self._reach_limited_height(R, p_xy, hip_center0, h_des, cmd)
        # Anticipate the double-support low point: while a foot swings toward its target,
        # descend smoothly to the height that will be reachable once it lands.
        for f in feet.values():
            if not f.planted:
                z_td = self._touchdown_height(f, R, p_xy, hip_center0, h_des)
                w = smoothstep(0.35, 0.95, f.swing_t)
                z = min(z, lerp(z, z_td + 0.004 * L, w))
        z = float(self.pelvis_z.update(dt, z))
        # Hard safety: never overstretch a planted leg.
        z = min(z, self._reach_limited_height(R, p_xy, hip_center0, 1e9, cmd, soft=False))
        self.pelvis_z.y = z
        D = np.eye(4)
        D[:3, :3] = R
        pelvis_world = np.array([p_xy[0], p_xy[1], z])
        D[:3, 3] = pelvis_world - R @ hip_center0
        self._last_pelvis_delta = D
        self.pose.reset()
        self.pose.set_world_delta(plan.pelvis, D)
        if plan.root:
            Dr = np.eye(4)
            Rr = rot_z(self.dpsi)
            Dr[:3, :3] = Rr
            Dr[:3, 3] = np.array([self.pos[0], self.pos[1], 0.0])
            self.pose.set_world_delta(plan.root, Dr)
        self.contact_debug = {"load_l": load_l, "h": z - h0}
        return D

    def _update_foot_roll(self, dt: float, cmd: MotionCommand) -> None:
        """Heel pivot roll-in after strike, then relax toward flat (or a commanded rise)."""
        rise_pitch = float(self.cmd_rise.y) * math.radians(28.0)
        for f in self.feet.values():
            if not f.planted:
                continue
            if f.pivot == "heel":
                if f.pitch < 0.0:
                    f.pitch = min(0.0, f.pitch + dt * (0.9 + 1.4 * cmd.energy) * self.style["heel_strike"] / 0.12)
                if f.pitch >= 0.0:
                    f.pitch = 0.0
                    f.set_pivot("ball", f.delta())
            elif f.pitch > rise_pitch:
                f.pitch = max(rise_pitch, f.pitch - dt * 1.6)
                f.toe_bend = min(f.pitch, math.radians(50.0))
            elif f.pitch < rise_pitch:
                f.pitch = min(rise_pitch, f.pitch + dt * 1.2)
                f.toe_bend = min(f.pitch, math.radians(50.0))

    def _reach_limited_height(self, R: np.ndarray, p_xy: np.ndarray, hip_center0: np.ndarray,
                              h_des: float, cmd: MotionCommand, soft: bool = True, probe: bool = False) -> float:
        """Highest pelvis z (<= h_des) at which every planted leg can reach its ankle.

        Trailing legs may instead lift their heel (rotate about the ball) – this is
        where the heel rise of terminal stance comes from.
        """
        z = h_des
        fwd, _ = self.char_axes()
        if self.speed_ref > 0.05:
            fwd = self.vel / self.speed_ref
        for f in self.feet.values():
            if not f.planted:
                continue
            leg = f.leg
            reach = leg.length * (1.0 - self.style["knee_softness"] * (1.3 if soft else 1.0)) * (1.0 if soft else 0.999)
            hip_off = R @ (leg.hip - hip_center0)
            D = f.delta()
            ankle = D[:3, :3] @ leg.ankle + D[:3, 3]
            hip_xy = np.array([p_xy[0], p_xy[1], 0.0]) + np.array([hip_off[0], hip_off[1], 0.0])
            ahead = float(np.dot(hip_xy - ankle, fwd))
            trailing = ahead > 0.1 * leg.length and f.pivot != "heel" and self.speed_ref > 0.05
            if soft and trailing:
                # Try to satisfy reach at h_des by raising the heel.
                pitch = self._heel_rise_for(f, hip_xy, hip_off[2], min(z, self.pelvis_z.y + 0.05), reach)
                if probe:
                    pitch = max(f.pitch, self.style["toe_off"] * 1.6)
                    R_f = f.rotation(pitch=pitch)
                    ankle = f.pivot_world + R_f @ (leg.ankle - leg.ball_pt)
                    d_xy = float(np.linalg.norm((hip_xy - ankle)[:2]))
                    zmax = ankle[2] - hip_off[2] if d_xy >= reach else \
                        ankle[2] + math.sqrt(reach * reach - d_xy * d_xy) - hip_off[2]
                    z = min(z, zmax)
                    continue
                if pitch is not None:
                    f.pitch = pitch
                    f.toe_bend = min(pitch, math.radians(50.0))
                    D = f.delta()
                    ankle = D[:3, :3] @ leg.ankle + D[:3, 3]
            d_xy = float(np.linalg.norm((hip_xy - ankle)[:2]))
            if d_xy >= reach:
                zmax = ankle[2] - hip_off[2]
            else:
                zmax = ankle[2] + math.sqrt(reach * reach - d_xy * d_xy) - hip_off[2]
            z = min(z, zmax)
        return z

    def _touchdown_height(self, f: FootState, R: np.ndarray, p_xy: np.ndarray,
                          hip_center0: np.ndarray, h_des: float) -> float:
        """Pelvis height that the swinging foot's leg will allow at heel strike."""
        leg = f.leg
        reach = leg.length * (1.0 - self.style["knee_softness"] * 1.3)
        R_td = f.rotation(f.tgt_yaw, f.td_pitch)
        heel_td = f.tgt_center + rot_z(f.tgt_yaw) @ (leg.heel - f.C0)
        heel_td[2] = 0.0
        ankle = heel_td + R_td @ (leg.ankle - leg.heel)
        remaining = (1.0 - f.swing_t) * f.swing_T
        hip_off = R @ (leg.hip - hip_center0)
        hip_xy = np.array([p_xy[0], p_xy[1], 0.0]) + self.vel * remaining + np.array([hip_off[0], hip_off[1], 0.0])
        d_xy = float(np.linalg.norm((hip_xy - ankle)[:2]))
        if d_xy >= reach:
            return float(ankle[2] - hip_off[2])
        return min(h_des, float(ankle[2] + math.sqrt(reach * reach - d_xy * d_xy) - hip_off[2]))

    def _heel_rise_for(self, f: FootState, hip_xy: np.ndarray, hip_dz: float, z: float,
                       reach: float) -> float | None:
        """Foot pitch about the ball that makes the ankle reachable from a hip at height z."""
        hip = np.array([hip_xy[0], hip_xy[1], z + hip_dz])
        base = f.pitch

        def dist_at(p: float) -> float:
            R = f.rotation(pitch=p)
            ankle = f.pivot_world + R @ (f.leg.ankle - f.leg.ball_pt)
            return float(np.linalg.norm(hip - ankle))

        if dist_at(max(base, 0.0)) <= reach:
            # Relax back toward flat when possible.
            relaxed = max(0.0, base - 0.02)
            return relaxed if dist_at(relaxed) <= reach else base
        hi = self.style["toe_off"] * 1.6
        if dist_at(hi) > reach:
            return hi
        lo = max(base, 0.0)
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            if dist_at(mid) > reach:
                lo = mid
            else:
                hi = mid
        return hi

    # ------------------------------------------------------------------ legs
    def _solve_legs(self, pelvis_D: np.ndarray, cmd: MotionCommand) -> None:
        pose = self.pose
        fwd_c, left_c = self.char_axes()
        for s, f in self.feet.items():
            leg = f.leg
            Dfoot = self._foot_delta(f)
            ankle_t = Dfoot[:3, :3] @ leg.ankle + Dfoot[:3, 3]
            hip = pelvis_D[:3, :3] @ leg.hip + pelvis_D[:3, 3]
            foot_fwd = Dfoot[:3, :3] @ f.fdir0
            pole = normalize(normalize(foot_fwd * np.array([1, 1, 0])) + 0.5 * fwd_c
                             + leg.sign * left_c * 0.12)
            knee, eff, n = two_bone(hip, ankle_t, leg.l_thigh, leg.l_calf, pole, self.style["knee_softness"])
            # Rest frames with the same construction (pole = forward).
            fwd0 = self.sk.forward
            d0 = normalize(leg.ankle - leg.hip)
            b0 = normalize(fwd0 - d0 * float(np.dot(fwd0, d0)))
            n0 = normalize(np.cross(d0, b0))
            Dt = delta_aligning(leg.hip, leg.knee - leg.hip, n0, hip, knee - hip, n)
            Dc = delta_aligning(leg.knee, leg.ankle - leg.knee, n0, knee, eff - knee, n)
            pose.set_world_delta(leg.thigh, Dt)
            pose.set_world_delta(leg.calf, Dc)
            # The foot follows the solved ankle (identical unless soft IK engaged).
            Df = Dfoot.copy()
            Df[:3, 3] += eff - ankle_t
            pose.set_world_delta(leg.foot, Df)
            if leg.ball:
                bend = -f.toe_bend if f.planted else -max(0.0, f.lo_pitch * (1.0 - smoothstep(0.0, 0.35, f.swing_t)))
                pose.set_rot(leg.ball, axis_angle_matrix(f.lat0, bend))

    # ------------------------------------------------------------------ upper body
    def _update_upper_body(self, dt: float, cmd: MotionCommand, pelvis_D: np.ndarray) -> None:
        plan, pose, sk = self.plan, self.pose, self.sk
        mc = self.micro.sample(self.t)
        amp = self.style["micro"] * (0.4 + 0.6 * cmd.energy) * (1.0 + 0.8 * cmd.tension)
        # ---- spine: counter-rotation, lateral compensation, lean, breathing
        pel_yaw = float(self.pelvis_yaw.y)
        pel_roll = float(self.pelvis_roll.y)
        self.breath_phase += dt * self.style["breath_rate"] * (0.8 + 0.6 * cmd.energy + 0.5 * cmd.tension)
        breath = math.sin(2.0 * math.pi * self.breath_phase)
        acc_lean = -0.03 * float(np.dot(self.vel_dyn.yd if isinstance(self.vel_dyn.yd, np.ndarray) else np.zeros(3),
                                        self.char_axes()[0]))
        target = np.array([
            cmd.lean + 0.04 * self.speed_ref + acc_lean + 0.012 * breath + 0.015 * amp * mc["lean"],
            cmd.side_lean + 0.8 * pel_roll + 0.012 * amp * mc["side"],
            cmd.twist - self.style["counter_rotation"] * pel_yaw + 0.02 * amp * mc["twist"],
        ])
        self.chest_rot.set_params(1.2 + 1.2 * cmd.tension, 0.55 + 0.25 * cmd.tension, 0.0)
        lean, side, twist = self.chest_rot.update(dt, target)
        imp = self.impulse.y
        lean = lean + float(imp[0]) * 0.4
        spine_bones = [n for n in plan.spine if n not in (plan.pelvis, plan.neck, plan.head)]
        weights = np.linspace(0.8, 1.2, len(spine_bones)) if spine_bones else []
        weights = np.asarray(weights) / (np.sum(weights) if len(spine_bones) else 1.0)
        fwd0, left0 = sk.forward, sk.left
        for w, name in zip(weights, spine_bones):
            R = axis_angle_matrix(UP, twist * w) @ axis_angle_matrix(fwd0, -side * w) \
                @ axis_angle_matrix(left0, lean * w)
            pose.set_rot(name, R)
        # ---- head: gaze (world target) stabilised against the chest motion
        pose.solve()
        chest_D = pose.delta[sk.index[plan.chest]]
        if plan.head:
            head_pos = pose.world_head(plan.head)
            fwd_w, left_w = self.char_axes()
            if cmd.gaze_target is not None:
                look = normalize(cmd.gaze_target - head_pos)
                look = normalize(lerp(fwd_w, look, clamp(cmd.gaze_weight, 0.0, 1.0)))
            else:
                look = fwd_w
            # Express the look direction in the chest's rest frame.
            local = chest_D[:3, :3].T @ look
            yaw_t = math.atan2(float(np.dot(local, left0)), float(np.dot(local, fwd0)))
            pitch_t = math.asin(clamp(float(local[2]), -1.0, 1.0))
            yaw_t = clamp(yaw_t + 0.03 * amp * mc["h_yaw"], -1.2, 1.2)
            pitch_t = clamp(pitch_t + cmd.head_nod + 0.02 * amp * mc["h_pitch"], -0.6, 0.5)
            self.head_rot.set_params(2.0 + 2.2 * cmd.energy, 0.55 + 0.2 * cmd.tension, 0.1)
            hy, hp = self.head_rot.update(dt, np.array([yaw_t, pitch_t]))
            for name, frac in ((plan.neck, 0.4), (plan.head, 0.6)):
                if name:
                    # pitch: positive = look up -> rotate about left axis negatively.
                    pose.set_rot(name, axis_angle_matrix(UP, hy * frac) @ axis_angle_matrix(left0, -hp * frac))
        # ---- arms
        fwd_c, _ = self.char_axes()
        p_head = pelvis_D[:3, :3] @ plan.hip_center + pelvis_D[:3, 3]
        sl = max(self._step_len, 0.15 * plan.leg_length)
        for s, arm in plan.arms.items():
            foot = self.feet.get(s)
            fa = 0.0
            if foot is not None:
                fa = float(np.dot(self._foot_center_world(foot) - p_head, fwd_c)) / (0.5 * sl)
            swing_amp = self.style["arm_swing"] * cmd.arm_swing * (0.35 + 0.65 * smoothstep(0.0, 1.2, self.speed_ref)) \
                * (0.6 + 0.6 * cmd.energy) * (1.0 - 0.4 * cmd.tension)
            flex_t = -swing_amp * clamp(fa, -1.3, 1.3) + 0.03 * amp * mc[f"arm_{s}"]
            self.arm_flex[s].set_params(1.3 + 0.8 * cmd.tension, 0.38 + 0.2 * cmd.tension, 0.0)
            flex = float(self.arm_flex[s].update(dt, flex_t))
            elbow_t = self.style["elbow_bend"] * (1.0 + 0.6 * cmd.tension) + 0.45 * max(0.0, flex) \
                + 0.04 * amp * mc[f"elb_{s}"]
            elbow = float(self.elbow_flex[s].update(dt, elbow_t))
            abd_rest = self._arm_rest_abd[s]
            abd_t = self.style["arm_hang"] + 0.12 * cmd.shoulder_raise
            abd = float(self.arm_abd[s].update(dt, abd_t - abd_rest))
            sign = arm.sign
            # Gesture blend (IK toward a hand target in the character frame).
            g = cmd.gestures.get(s)
            gw = float(self.gesture_w[s].update(dt, g.weight if g else 0.0))
            if arm.clavicle:
                pose.set_rot(arm.clavicle, axis_angle_matrix(fwd0, sign * 0.18 * cmd.shoulder_raise
                                                             + sign * 0.02 * breath))
            self._pose_arm(arm, flex, elbow, abd, sign)
            if gw > 0.01 and g is not None:
                self._blend_gesture(arm, g, gw, dt)
            self._resolve_arm_collisions(arm, flex, elbow, abd, sign, gw)

    def _pose_arm(self, arm, flex: float, elbow: float, abd: float, sign: float) -> None:
        sk = self.sk
        R_up = axis_angle_matrix(sk.left, -flex) @ axis_angle_matrix(sk.forward, sign * abd)
        self.pose.set_rot(arm.upper, R_up)
        self.pose.set_rot(arm.lower, axis_angle_matrix(sk.left, -elbow))
        self.pose.set_rot(arm.hand, axis_angle_matrix(sk.left, -0.25 * elbow))

    def _blend_gesture(self, arm, g, gw: float, dt: float) -> None:
        pose, sk, plan = self.pose, self.sk, self.plan
        pose.solve()
        pel = pose.delta[sk.index[plan.pelvis]]
        R_char = pel[:3, :3]
        base = pel[:3, :3] @ plan.hip_center + pel[:3, 3]
        tgt_local = self.gesture_pos[arm.side].update(dt, np.asarray(g.target, dtype=float))
        f0, l0 = sk.forward, sk.left
        target = base + R_char @ (f0 * tgt_local[0] + l0 * tgt_local[1] + UP * tgt_local[2])
        shoulder = pose.world_head(arm.upper)
        wrist_now = pose.world_head(arm.hand)
        target = wrist_now + (target - wrist_now) * gw
        pole = normalize(R_char @ (-f0 * (1.0 - g.elbow_out) + l0 * arm.sign * g.elbow_out - UP * 0.3))
        elbow, eff, n = two_bone(shoulder, target, arm.l_upper, arm.l_lower, pole, 0.02)
        d0 = normalize(arm.wrist - arm.shoulder)
        pole0 = normalize(-f0 * 0.7 + l0 * arm.sign * 0.3 - UP * 0.3)
        b0 = normalize(pole0 - d0 * float(np.dot(pole0, d0)))
        n0 = normalize(np.cross(d0, b0))
        Du = delta_aligning(arm.shoulder, arm.elbow - arm.shoulder, n0, shoulder, elbow - shoulder, n)
        Dl = delta_aligning(arm.elbow, arm.wrist - arm.elbow, n0, elbow, eff - elbow, n)
        pose.set_world_delta(arm.upper, Du)
        pose.set_world_delta(arm.lower, Dl)
        pose.set_world_delta(arm.hand, Dl @ rot_about(arm.wrist, l0, -g.hand_pitch))

    def _resolve_arm_collisions(self, arm, flex, elbow, abd, sign, gw) -> None:
        plan, pose, sk = self.plan, self.pose, self.sk
        if not plan.ellipses and not plan.capsules:
            return
        for _ in range(4):
            pose.solve()
            push = 0.0
            pts = [pose.world_head(arm.lower), pose.world_head(arm.hand),
                   0.5 * (pose.world_head(arm.lower) + pose.world_head(arm.hand)),
                   0.5 * (pose.world_head(arm.hand) + pose.world_tail(arm.hand)), pose.world_tail(arm.hand)]
            shoulder = pose.world_head(arm.upper)
            margin = 0.012 * plan.height
            for p in pts:
                for e in plan.ellipses:
                    D = pose.delta[sk.index[e.bone]]
                    c = D[:3, :3] @ e.center + D[:3, 3]
                    Rl = D[:3, :3]
                    rel = Rl.T @ (p - c)
                    if abs(float(rel[2])) > 0.03 * plan.height:
                        continue
                    lx = float(np.dot(rel, sk.left))
                    fx = float(np.dot(rel, sk.forward))
                    rho = math.sqrt((lx / max(e.half_left + margin, 1e-4)) ** 2 + (fx / max(e.half_fwd + margin, 1e-4)) ** 2)
                    if rho < 1.0 and lx * sign > -0.3 * e.half_left:
                        depth = (1.0 - rho) * (e.half_left + margin)
                        push = max(push, depth / max(float(np.linalg.norm(p - shoulder)), 1e-3))
                for cap in plan.capsules:
                    D = pose.delta[sk.index[cap.bone]]
                    a = D[:3, :3] @ cap.a + D[:3, 3]
                    b = D[:3, :3] @ cap.b + D[:3, 3]
                    ab = b - a
                    tt = clamp(float(np.dot(p - a, ab)) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0)
                    q = a + ab * tt
                    d = float(np.linalg.norm(p - q))
                    r = cap.radius + margin
                    side_ok = float(np.dot(p - q, D[:3, :3] @ sk.left)) * sign > -0.5 * r
                    if d < r and side_ok:
                        push = max(push, (r - d) / max(float(np.linalg.norm(p - shoulder)), 1e-3))
            if push < 1e-4:
                break
            abd = abd + push * 1.1
            self.arm_abd[arm.side].y = abd
            if gw < 0.5:
                self._pose_arm(arm, flex, elbow, abd, sign)
            else:
                break
