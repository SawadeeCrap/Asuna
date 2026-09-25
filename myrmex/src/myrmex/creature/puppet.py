"""Direct, synchronous control of a creature by a hand (the Hand Glove): the engine side.

Parameters (MIDI CC -> "expansion", "fluidity" ...) only nudge the organism's behaviour, slowly,
through its morphological inertia - a hand turning does not *look* like anything.  A puppet link
acts on the body itself:

* orientation: the hand's rotation (relative to a neutral pose) rotates the body - every node is
  turned rigidly with the hand each step and the shape targets are turned with it, so the body
  follows without lag while its own dynamics (waves, limbs, flight) go on;
* fingers: five sectors around the body's long axis are five fingers - extending one pushes that
  side out into a limb or spike, curling it pulls the material in and forward like a closing claw;
  the open hand spreads the whole body, the fist compacts it;
* position, spin, material, energy and a sculpting mode (fingers blend the forms) are set by the
  presets in realtime/glove.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SECTORS = np.pi / 2 + (np.arange(5) - 2) * (2 * np.pi / 5)      # middle finger on top, thumb / pinky below


@dataclass
class GloveControl:
    active: bool = False
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))     # hand orientation relative to neutral
    fingers: np.ndarray = field(default_factory=lambda: np.full(5, 0.5))   # extension 0 curled .. 1 extended
    finger_mode: str = "limbs"            # limbs | morph | strings | none
    amount: float = 1.0                   # intensity of the deformation
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))   # x steer, y altitude, z depth (-1..1)
    scale: float = 1.0
    grip: float = 0.0                     # 0..1: how firmly the hand holds the body
    spin: float = 0.0                     # rad/s about the body's vertical axis (conductor)
    material: float | None = None         # -1 fluid .. +1 hard / bone
    energy: float | None = None           # extra arousal 0..1
    stretch: np.ndarray = field(default_factory=lambda: np.ones(3))   # length, width, height factors
    twist: float = 0.0                    # rad over the body's length (screw)
    waves: np.ndarray = field(default_factory=lambda: np.zeros(5))    # travelling-wave amplitude per finger
    wave_speed: float = 1.0
    pulse: float = 0.0                    # beat-locked breathing amplitude 0..1
    scatter: float = 0.0                  # 0..1: the material comes apart
    freeze: float = 0.0                   # 0..1: motion held
    wind: np.ndarray = field(default_factory=lambda: np.zeros(3))     # flow force (m/s^2, world)
    formation_turn: float = 0.0           # rad: a flock's formation turns round its lead
    formation_spread: float = 1.0         # flock slot distance factor
    swarm_release: float = 0.0            # hive: nanomachines let out 0..1
    swarm_pull: float = 0.0               # hive: nanomachines called back 0..1
    lines: float | None = None            # light-line / glow intensity 0..1
    angvel: np.ndarray = field(default_factory=lambda: np.zeros(3))   # rad/s thrown into the body (its frame)
    rays: int = 0                         # radial symmetry: number of rays round the long axis
    ray_phase: float = 0.0
    ray_len: float = 0.0
    point: np.ndarray | None = None       # a place on the stage (world, m) the organism is led to
    orbit: float = 0.0                    # rad/s: circle round ``point``
    orbit_radius: float = 3.0
    speed: float = 1.0                    # cruise factor
    flock: int = 0                        # colony: how many bodies the hand wants (0 = the organism decides)
    cloud: np.ndarray | None = None       # hive: where the released nanomachines gather (lead body frame, m)
    cloud_swirl: float = 0.0              # rad/s round that point


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis * angle."""
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    a = math.acos(c)
    if a < 1e-6:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.linalg.norm(v)
    if s < 1e-9:                                         # ~180 degrees
        w, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        return V[:, int(np.argmax(w))] * a
    return v / s * a


def from_rotvec(r: np.ndarray) -> np.ndarray:
    a = float(np.linalg.norm(r))
    if a < 1e-9:
        return np.eye(3)
    k = r / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """x forward: roll about x, pitch about y, yaw about z (applied roll -> pitch -> yaw)."""
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), \
        math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def finger_deform(loc: np.ndarray, ext: np.ndarray, amount: float, openness: float | None = None) -> np.ndarray:
    """Five sectors around the long (x) axis follow five fingers; local body coordinates in and out."""
    if amount <= 0:
        return loc
    y, z = loc[:, 1], loc[:, 2]
    phi = np.arctan2(z, y)
    r = np.hypot(y, z)
    d = (phi[:, None] - SECTORS[None, :] + np.pi) % (2 * np.pi) - np.pi
    w = np.exp(-(d / 0.6) ** 2)
    e = (w * (2.0 * np.asarray(ext)[None, :] - 1.0)).sum(1) / np.maximum(w.sum(1), 1e-6)       # -1 curled .. 1 out
    r2 = r * (1.0 + 0.9 * amount * e)
    curl = np.maximum(0.0, -e) * amount
    out = loc.copy()
    out[:, 0] = loc[:, 0] + curl * 0.45 * r                      # curled fingers close forward like a claw
    out[:, 1] = r2 * np.cos(phi)
    out[:, 2] = r2 * np.sin(phi)
    op = float(np.mean(ext)) if openness is None else openness
    return out * (0.8 + 0.4 * op * amount + (1.0 - amount) * 0.2)


class GloveDriver:
    """Held by an engine: the current control, the accumulated body rotation, the per-step increment."""

    def __init__(self):
        self.ctrl = GloveControl()
        self.G = np.eye(3)
        self.W = np.eye(3)                                # orientation thrown by the hand (flywheel)
        self.spin_angle = 0.0

    def set(self, ctrl: GloveControl | None) -> None:
        self.ctrl = ctrl or GloveControl()

    @property
    def active(self) -> bool:
        return self.ctrl.active

    def begin(self, dt: float) -> np.ndarray:
        """Advance the body rotation; returns the increment (rotate the nodes by it)."""
        c = self.ctrl
        if c.active:
            if c.spin:
                self.spin_angle += c.spin * dt
            else:                                        # a spin that stopped settles back (shortest way)
                self.spin_angle = ((self.spin_angle + math.pi) % (2 * math.pi) - math.pi) * math.exp(-dt / 1.2)
            if np.any(np.abs(c.angvel) > 1e-4):
                self.W = self.W @ from_rotvec(np.asarray(c.angvel, float) * dt)
            else:
                self.W = from_rotvec(rotvec(self.W) * math.exp(-dt / 1.5))
            cz, sz = math.cos(self.spin_angle), math.sin(self.spin_angle)
            target = c.rot @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]]) @ self.W
        else:                                            # no hand: ease back to the organism's own pose
            self.spin_angle = ((self.spin_angle + math.pi) % (2 * math.pi) - math.pi) * math.exp(-dt / 0.8)
            self.W = from_rotvec(rotvec(self.W) * math.exp(-dt / 0.6))
            target = from_rotvec(rotvec(self.G) * math.exp(-dt / 0.6))
        dR = target @ self.G.T
        self.G = target
        return dR

    def rigid(self, x: np.ndarray, v: np.ndarray, P: np.ndarray, dR: np.ndarray, idx=None) -> None:
        """Turn nodes (all, or ``idx``) about ``P`` with the hand - in place."""
        if np.allclose(dR, np.eye(3), atol=1e-7):
            return
        g = max(0.0, min(1.0, self.ctrl.grip)) if self.ctrl.active else 1.0
        if g < 1.0:
            dR = from_rotvec(rotvec(dR) * g)
        sl = slice(None) if idx is None else idx
        x[sl] = P + (x[sl] - P) @ dR.T
        v[sl] = v[sl] @ dR.T

    def local(self, loc: np.ndarray, t: float = 0.0, beat: float = 0.0, limbs: bool = True) -> np.ndarray:
        """The hand's deformations of a body in its own frame (x forward)."""
        c = self.ctrl
        if not c.active:
            return loc
        out = finger_deform(loc, c.fingers, c.amount) if (limbs and c.finger_mode == "limbs") else loc.copy()
        span = max(float(np.ptp(out[:, 0])), 1e-3)
        xs = (out[:, 0] - out[:, 0].mean()) / span                       # -0.5 .. 0.5 along the body
        if c.finger_mode == "strings":                                   # five strings along the body, tail .. head
            anchors = -0.4 + 0.2 * np.arange(5)
            w = np.exp(-((xs[:, None] - anchors[None, :]) / 0.13) ** 2)
            out[:, 2] += 0.55 * span * c.amount * (w * (1.0 - np.asarray(c.fingers))[None, :]).sum(1)
        if c.rays > 0 and c.ray_len > 1e-3:                              # radial symmetry round the long axis
            phi = np.arctan2(out[:, 2], out[:, 1])
            lobe = np.maximum(0.0, np.cos(c.rays * phi - c.ray_phase)) ** 3
            f = 1.0 + c.ray_len * (1.6 * lobe - 0.35)
            out[:, 1] *= f
            out[:, 2] *= f
        if abs(c.twist) > 1e-4:                                          # a screw along the length
            a = c.twist * xs
            ca, sa = np.cos(a), np.sin(a)
            y, z = out[:, 1].copy(), out[:, 2].copy()
            out[:, 1], out[:, 2] = ca * y - sa * z, sa * y + ca * z
        if np.any(c.waves > 1e-3):                                       # each finger its own wave
            for k in range(5):
                if c.waves[k] > 1e-3:
                    ph = 2 * np.pi * ((k + 1) * xs) - t * c.wave_speed * (2.0 + 0.7 * k)
                    out[:, 1 + (k % 2)] += 0.22 * span * c.waves[k] * np.sin(ph)
        if c.pulse > 1e-3:                                               # breathing on the beat
            out *= 1.0 + 0.28 * c.pulse * math.sin(math.pi * beat) ** 2
        return out * c.stretch * c.scale

    def damp(self, v: np.ndarray, dt: float) -> None:
        """Freeze: the hand holds the motion (in place)."""
        if self.ctrl.active and self.ctrl.freeze > 1e-3:
            v *= math.exp(-self.ctrl.freeze * 7.0 * dt)

    def wind_acc(self, x: np.ndarray, P: np.ndarray, R: np.ndarray) -> np.ndarray | float:
        """A flow from the hand (``wind`` in the body frame): the outer material streams like a flag or a
        comet's tail (zero-mean part) and the body drifts a little (the rest)."""
        c = self.ctrl
        if not c.active or not np.any(np.abs(c.wind) > 1e-4):
            return 0.0
        wv = np.asarray(R, float) @ c.wind
        r = np.linalg.norm(x - P, axis=1)
        w = r / max(float(r.max()), 1e-6)
        return wv[None, :] * (1.6 * (w - w.mean()) + 0.12)[:, None]

    def flight(self, v_des: np.ndarray, P: np.ndarray, cruise: float) -> np.ndarray:
        """Leash / orbit / speed: where the hand leads the organism (desired velocity, world)."""
        c = self.ctrl
        if not c.active:
            return v_des
        v_des = v_des.copy()
        v_des[:2] *= c.speed
        if c.point is not None:
            d = np.asarray(c.point, float) - P
            ang = math.atan2(-d[1], -d[0])                               # where it is, seen from the point
            if abs(c.orbit) > 1e-3:
                ang += math.copysign(0.6, c.orbit)                       # lead ahead on the circle
            goal = np.asarray(c.point, float) + np.array([math.cos(ang), math.sin(ang), 0.0]) * c.orbit_radius
            g = goal - P
            g[2] = np.asarray(c.point, float)[2] - P[2]
            tang = np.array([-math.sin(ang), math.cos(ang), 0.0]) * c.orbit * c.orbit_radius
            v = np.clip(1.2 * g, -8.0, 8.0) + tang
            v_des = np.array([v[0], v[1], float(np.clip(v[2], -3.0, 3.0))])
        return v_des


__all__ = ["GloveControl", "GloveDriver", "finger_deform", "euler", "rotvec", "from_rotvec", "SECTORS"]
