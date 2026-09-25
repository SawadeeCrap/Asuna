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
    finger_mode: str = "limbs"            # limbs | morph | none
    amount: float = 1.0                   # intensity of the deformation
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))   # x steer, y altitude, z depth (-1..1)
    scale: float = 1.0
    grip: float = 0.0                     # 0..1: how firmly the hand holds the body
    spin: float = 0.0                     # rad/s about the body's vertical axis (conductor)
    material: float | None = None         # -1 fluid .. +1 hard / bone
    energy: float | None = None           # extra arousal 0..1


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
            self.spin_angle += c.spin * dt
            cz, sz = math.cos(self.spin_angle), math.sin(self.spin_angle)
            target = c.rot @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
        else:                                            # no hand: ease back to the organism's own pose
            self.spin_angle *= math.exp(-dt / 0.8)
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

    def local(self, loc: np.ndarray) -> np.ndarray:
        c = self.ctrl
        if not c.active:
            return loc
        out = finger_deform(loc, c.fingers, c.amount) if c.finger_mode == "limbs" else loc
        return out * c.scale


__all__ = ["GloveControl", "GloveDriver", "finger_deform", "euler", "rotvec", "from_rotvec", "SECTORS"]
