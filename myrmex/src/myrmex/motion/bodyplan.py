"""Body plans: what the motion engine needs to know about a rig.

A :class:`BipedPlan` is derived automatically from a humanoid
:class:`~myrmex.rig.rigdesc.RigDescription` (auto-rigged or mapped from an
existing rig).  It carries rest geometry (joint positions, segment lengths,
foot contact points), proportions used for physically plausible scaling, and
collision proxies (torso ellipses, thigh capsules) that keep swinging arms
out of the hips.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..rig.rigdesc import RigDescription
from .skeleton import SkeletonRest


@dataclass
class LegPlan:
    side: str                     # "l" / "r"
    sign: float                   # +1 left, -1 right
    thigh: str
    calf: str
    foot: str
    ball: str | None
    hip: np.ndarray               # rest positions (world)
    knee: np.ndarray
    ankle: np.ndarray
    heel: np.ndarray              # ground contact points at rest
    ball_pt: np.ndarray
    toe: np.ndarray
    l_thigh: float
    l_calf: float
    foot_yaw_offset: float        # toe-out of the rest foot relative to forward (rad)

    @property
    def length(self) -> float:
        return self.l_thigh + self.l_calf


@dataclass
class ArmPlan:
    side: str
    sign: float
    clavicle: str | None
    upper: str
    lower: str
    hand: str
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    hand_tip: np.ndarray
    l_upper: float
    l_lower: float


@dataclass
class Ellipse:
    bone: str
    center: np.ndarray            # rest world
    half_left: float
    half_fwd: float


@dataclass
class Capsule:
    bone: str
    a: np.ndarray
    b: np.ndarray
    radius: float


@dataclass
class BipedPlan:
    rig: RigDescription
    sk: SkeletonRest
    height: float
    legs: dict[str, LegPlan]
    arms: dict[str, ArmPlan]
    spine: list[str]
    pelvis: str
    chest: str
    neck: str | None
    head: str | None
    root: str | None
    hip_center: np.ndarray
    pelvis_height: float
    heel_rise: float
    ellipses: list[Ellipse] = field(default_factory=list)
    capsules: list[Capsule] = field(default_factory=list)

    @property
    def forward(self) -> np.ndarray:
        return self.sk.forward

    @property
    def left(self) -> np.ndarray:
        return self.sk.left

    @property
    def leg_length(self) -> float:
        return 0.5 * (self.legs["l"].length + self.legs["r"].length)

    @property
    def high_heels(self) -> bool:
        return self.heel_rise > 0.035 * self.height

    @property
    def natural_frequency(self) -> float:
        """Pendulum frequency of the leg (Hz) – drives Froude-consistent cadence."""
        return math.sqrt(9.81 / max(self.leg_length, 0.2)) / (2.0 * math.pi)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_rig(cls, rd: RigDescription, proxy_verts: np.ndarray | None = None) -> "BipedPlan":
        sk = SkeletonRest.from_rig(rd)
        fwd, left = sk.forward, sk.left
        legs: dict[str, LegPlan] = {}
        for c in rd.chains_by_role("leg"):
            s = c.side.lower()
            names = c.bones
            thigh, calf, foot = names[0], names[1], names[2]
            ball = names[3] if len(names) > 3 else None
            hip, knee, ankle = sk.head(thigh), sk.head(calf), sk.head(foot)
            heel = np.asarray(rd.landmarks.get(f"heel_{s}", ankle - fwd * 0.05 * rd.height), dtype=float).copy()
            toe = np.asarray(rd.landmarks.get(f"toe_tip_{s}", sk.tail(ball) if ball else sk.tail(foot)), dtype=float).copy()
            ball_pt = sk.head(ball) if ball else sk.tail(foot)
            heel[2] = max(0.0, heel[2])
            toe[2] = max(0.0, toe[2])
            ball_pt = ball_pt.copy()
            ball_pt[2] = 0.0
            fdir = toe - heel
            fdir[2] = 0.0
            yaw_off = math.atan2(float(np.dot(fdir, left)), float(np.dot(fdir, fwd)))
            legs[s] = LegPlan(s, 1.0 if s == "l" else -1.0, thigh, calf, foot, ball, hip, knee, ankle,
                              heel, ball_pt, toe, sk.length(thigh), sk.length(calf), yaw_off)
        arms: dict[str, ArmPlan] = {}
        for c in rd.chains_by_role("arm"):
            s = c.side.lower()
            b = c.bones
            clav = b[0] if len(b) == 4 else None
            upper, lower, hand = b[-3], b[-2], b[-1]
            arms[s] = ArmPlan(s, 1.0 if s == "l" else -1.0, clav, upper, lower, hand,
                              sk.head(upper), sk.head(lower), sk.head(hand), sk.tail(hand),
                              sk.length(upper), sk.length(lower))
        spine = rd.chains_by_role("spine")[0].bones if rd.chains_by_role("spine") else []
        pelvis = spine[0] if spine else "pelvis"
        chest = next((n for n in spine if rd.bone(n).role == "chest"), spine[-3] if len(spine) >= 3 else pelvis)
        neck = next((n for n in spine if rd.bone(n).role == "neck"), None)
        head = next((n for n in spine if rd.bone(n).role == "head"), None)
        root = "root" if rd.has_bone("root") else None
        hip_center = sk.head(pelvis)
        plan = cls(rd, sk, rd.height, legs, arms, spine, pelvis, chest, neck, head, root, hip_center,
                   float(hip_center[2]), float(rd.params.get("heel_rise", 0.0)))
        if proxy_verts is not None:
            plan.build_collision(proxy_verts)
        elif "collision" in rd.params:
            plan.load_collision(rd.params["collision"])
        return plan

    # ------------------------------------------------------------------ collision proxies
    def build_collision(self, verts: np.ndarray) -> None:
        """Torso ellipses + thigh capsules from the analysis proxy (rest pose)."""
        sk, fwd, left, H = self.sk, self.forward, self.left, self.height
        arm_pts = []
        for a in self.arms.values():
            for n in (a.upper, a.lower, a.hand):
                arm_pts.append((sk.head(n), sk.tail(n)))
        # Exclude vertices close to arm bones.
        V = verts
        keep = np.ones(len(V), bool)
        for h, t in arm_pts:
            ab = t - h
            tt = np.clip(((V - h) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
            d = np.linalg.norm(V - (h + tt[:, None] * ab), axis=1)
            keep &= d > 0.05 * H
        core = V[keep & (np.abs(V @ left) < 0.2 * H)]
        z0 = self.pelvis_height - 0.1 * H
        z1 = float(sk.head(self.chest)[2]) + 0.06 * H
        ellipses = []
        for z in np.linspace(z0, z1, 9):
            ring = core[np.abs(core[:, 2] - z) < 0.015 * H]
            if len(ring) < 12:
                continue
            l = ring @ left
            f = ring @ fwd
            cl = 0.5 * (np.percentile(l, 97) + np.percentile(l, 3))
            cf = 0.5 * (np.percentile(f, 97) + np.percentile(f, 3))
            hl = 0.5 * (np.percentile(l, 97) - np.percentile(l, 3))
            hf = 0.5 * (np.percentile(f, 97) - np.percentile(f, 3))
            center = left * cl + fwd * cf + np.array([0.0, 0.0, z])
            bone = self._spine_bone_at(z)
            ellipses.append(Ellipse(bone, center, float(hl), float(hf)))
        caps = []
        for leg in self.legs.values():
            h, t = leg.hip, leg.knee
            ab = t - h
            tt = ((V - h) @ ab) / max(float(ab @ ab), 1e-9)
            sel = (tt > 0.15) & (tt < 0.85)
            d = np.linalg.norm(V - (h + np.clip(tt, 0, 1)[:, None] * ab), axis=1)
            dd = d[sel & (d < 0.15 * H)]
            r = float(np.percentile(dd, 60)) if len(dd) else 0.06 * H
            caps.append(Capsule(leg.thigh, h.copy(), t.copy(), r))
        self.ellipses = ellipses
        self.capsules = caps
        self.rig.params["collision"] = self.collision_dict()

    def _spine_bone_at(self, z: float) -> str:
        best, bestd = self.pelvis, 1e9
        for n in self.spine:
            if self.rig.bone(n).role in ("neck", "head"):
                continue
            h, t = self.sk.head(n), self.sk.tail(n)
            mid = 0.5 * (h[2] + t[2])
            if h[2] - 1e-6 <= z <= t[2] + 1e-6:
                return n
            if abs(mid - z) < bestd:
                best, bestd = n, abs(mid - z)
        return best

    def collision_dict(self) -> dict:
        return {
            "ellipses": [{"bone": e.bone, "center": e.center.tolist(), "half_left": e.half_left,
                          "half_fwd": e.half_fwd} for e in self.ellipses],
            "capsules": [{"bone": c.bone, "a": c.a.tolist(), "b": c.b.tolist(), "radius": c.radius}
                         for c in self.capsules],
        }

    def load_collision(self, d: dict) -> None:
        self.ellipses = [Ellipse(e["bone"], np.asarray(e["center"]), e["half_left"], e["half_fwd"])
                         for e in d.get("ellipses", [])]
        self.capsules = [Capsule(c["bone"], np.asarray(c["a"]), np.asarray(c["b"]), c["radius"])
                         for c in d.get("capsules", [])]
