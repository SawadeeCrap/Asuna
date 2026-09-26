"""Tensor (creature v14): a tensegrity spine swimming through the air.

A tower of triangular tensegrity prisms (Snelson's needle tower): K stacked rings of three nodes, each
module held by three rigid struts that never touch each other directly and a net of cables that can
only pull.  Struts are hard distance constraints (position-based projection); cables are springs that
go slack when compressed.  The cables are also the muscles - their rest length is actuated - so every
form is a pattern of contraction running through the net, and the physics decides the shape:

    SWIM    a lateral wave travelling head to tail (anguilliform)
    COIL    constant curvature: the spine curls into an arc and a ring, the bending plane turning
    HELIX   curvature whose direction turns along the body: a helical coil
    SPRING  every longitudinal cable contracted: short, wide, twisted, breathing on the beat
    REACH   ring cables contracted, longitudinals released: long and needle-thin
    WHIP    one sharp bend travelling head to tail again and again: a lash

Tension is shown: cables brighten with the load they carry, so the force network is visible.
Events: LASH (a whip crack), COIL (curl into a ring), UNFURL (reach out).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import BionicConfig, BionicEngine

N_RING = 3                       # nodes per ring (triangular prisms)


@dataclass
class TensorConfig(BionicConfig):
    modules: int = 8
    cruise: float = 2.8
    sim_rate: float = 240.0          # stiff cables: small steps


class TensorEngine(BionicEngine):
    VARIANT = "tensor"
    KIND = 0
    REGIMES = ("SWIM", "COIL", "HELIX", "SPRING", "REACH", "WHIP")
    PLAN = {"CRUISE": (("SWIM", "SWIM", "WHIP", "REACH"), 1.0), "HOVER": (("COIL", "HELIX", "SPRING"), 0.15),
            "EXPLORE": (("SWIM", "HELIX", "REACH"), 0.6), "DISPLAY": (("COIL", "HELIX", "WHIP", "SPRING"), 0.3),
            "EVADE": (("SPRING", "COIL"), 1.4), "REFORM": (("SWIM",), 0.5), "STRIKE": (("REACH", "WHIP"), 1.6)}
    EVENTS = BionicEngine.EVENTS + ("LASH", "COIL", "UNFURL")

    def __init__(self, cfg: BionicConfig | None = None):
        super().__init__(cfg or TensorConfig())
        cfg = self.cfg
        K = getattr(cfg, "modules", 8)
        s = cfg.size / 1.8
        self.K = K
        n = (K + 1) * N_RING
        self.n = n
        # straight tower in the body frame: x forward (ring K is the head), rings twisted by 30 degrees each
        length = 3.0 * s
        rings = []
        for k in range(K + 1):
            u = k / K
            rad = (0.2 + 0.2 * math.sin(math.pi * (0.25 + 0.65 * u))) * s     # thin tail, full body, a neck
            psi = k * math.pi / 6.0
            for i in range(N_RING):
                a = psi + 2 * math.pi * i / N_RING
                rings.append((-0.5 * length + length * u, rad * math.cos(a), rad * math.sin(a)))
        local = np.array(rings)
        idx = lambda k, i: k * N_RING + (i % N_RING)                                   # noqa: E731
        struts, rcab, vcab, dcab = [], [], [], []
        for k in range(K):
            for i in range(N_RING):
                struts.append((idx(k, i), idx(k + 1, i + 1)))
                vcab.append((idx(k, i), idx(k + 1, i)))
                dcab.append((idx(k, i + 1), idx(k + 1, i + 2)))                        # saddle cables: stability
        for k in range(K + 1):
            for i in range(N_RING):
                rcab.append((idx(k, i), idx(k, i + 1)))
        self.struts = np.array(struts)
        self.cables = np.array(rcab + vcab + dcab)
        nr, nv = len(rcab), len(vcab)
        self.c_kind = np.array([0] * nr + [1] * nv + [2] * len(dcab))              # 0 ring, 1 long., 2 saddle
        self.c_module = np.array([c // N_RING for c in range(nr)] + [c // N_RING for c in range(nv)] +
                                 [c // N_RING for c in range(len(dcab))]) / max(K, 1)
        mid = 0.5 * (local[self.cables[:, 0]] + local[self.cables[:, 1]])
        self.c_ang = np.arctan2(mid[:, 2], mid[:, 1])                               # side of the body
        self.L_strut = np.linalg.norm(local[self.struts[:, 1]] - local[self.struts[:, 0]], axis=1)
        self.rest0 = np.linalg.norm(local[self.cables[:, 1]] - local[self.cables[:, 0]], axis=1) * 0.93
        self.mass = np.full(n, 1.0 / n)                                              # finite material
        R = self._R()
        self.x = self.P + local @ R.T
        self.v = np.zeros((n, 3))
        self.act = np.zeros(len(self.cables))
        self.tension = np.zeros(len(self.cables))
        self.phase = 0.0
        self.whip = -1.0
        self.plane = 0.0
        self.u_node = np.repeat(np.arange(K + 1) / K, N_RING)
        self.h_min = 0.45 * length / K
        for _ in range(240):                                                          # form finding
            self._physics(1 / 120, np.zeros(3), settle=True)

    # ------------------------------------------------------------------ muscles
    def _activation(self, dt: float) -> None:
        pr, inp, w = self.params.values(), self.inp, self.weights()
        W = dict(zip(self.REGIMES, w))
        bps = max(inp.tempo, 60.0) / 60.0
        freq = (0.5 if inp.playing else 0.35) * bps * (0.6 + 0.8 * pr["speed"]) * (0.7 + 0.6 * self.arousal)
        self.phase += 2 * math.pi * freq * dt
        self.plane += dt * (0.3 + 0.8 * pr["mechanism"])
        amp = (0.18 + 0.22 * pr["mechanism"] + 0.18 * inp.bass) * (0.8 + 0.4 * pr["tendril_activity"])
        u, ang, kind = self.c_module, self.c_ang, self.c_kind
        lon = kind == 1                                         # the longitudinal cables are the bending muscles
        swim = np.sin(2 * math.pi * 1.1 * u - self.phase) * np.cos(ang - math.pi / 2) * (0.35 + 0.65 * (1 - u))
        coil = np.cos(ang - self.plane)
        helix = np.cos(ang - (self.plane + 2 * math.pi * 1.5 * u))
        breathe = 0.5 + 0.5 * math.sin(math.pi * (inp.beat if inp.playing else 2 * self.t))
        side = np.cos(ang - math.pi / 2)
        s_pos = (self.t * 0.9 * bps) % 1.3                     # the WHIP regime: a lash every bar or so
        whip = np.exp(-((1.0 - u - s_pos) / 0.12) ** 2) * side
        if self.whip >= 0.0:                                   # a triggered lash (event or a hard kick)
            self.whip += dt * 1.6 * bps
            if self.whip > 1.3:
                self.whip = -1.0
        lash = np.exp(-((1.0 - u - self.whip) / 0.1) ** 2) * side if self.whip >= 0.0 else 0.0 * u
        a_lon = amp * (W["SWIM"] * 1.8 * swim + W["COIL"] * 1.9 * coil + W["HELIX"] * 1.5 * helix +
                       W["WHIP"] * 1.6 * whip + 1.8 * lash) + \
            W["SPRING"] * (0.12 + 0.07 * breathe) - W["REACH"] * 0.12 + 0.15 * self.kick * (W["SPRING"] + 0.3)
        a_ring = W["REACH"] * 0.28 - W["SPRING"] * 0.12 + 0.05 * inp.high * np.sin(40 * self.t + 7 * u)
        a = np.where(lon, np.clip(a_lon, -0.35, 0.45), np.where(kind == 0, a_ring, 0.0))
        gc = self.glove.ctrl
        if gc.active:
            if gc.finger_mode == "limbs":                          # each finger pulls its side of the body
                side = ((ang + math.pi) / (2 * math.pi) * 5).astype(int) % 5
                a = a + np.where(lon, (1.0 - np.asarray(gc.fingers)[side]) * 0.35 * gc.amount, 0.0)
            a = a + np.where(lon, 0.25 * (1.0 - gc.stretch[0]), -0.2 * (1.0 - gc.stretch[1]))
            if abs(gc.twist) > 1e-3:
                a = a + np.where(kind == 2, 0.08 * gc.twist, 0.0)
        a = a + np.where(kind == 2, 0.12 * (W["HELIX"] - W["REACH"]), 0.0)      # the saddles twist the spine
        tau = 0.08 + 0.4 * (1.0 - pr["reactivity"])
        self.act += (np.clip(a, -0.35, 0.5) - self.act) * min(1.0, dt / tau)

    # ------------------------------------------------------------------ physics
    def _physics(self, dt: float, a_body: np.ndarray, settle: bool = False) -> None:
        pr = self.params.values()
        x, v = self.x, self.v
        x_prev = x.copy()
        ci, cj = self.cables[:, 0], self.cables[:, 1]
        d = x[cj] - x[ci]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        u = d / L[:, None]
        rest = self.rest0 * (1.0 - self.act)
        stretch = L - rest
        kc = (900.0 + 2600.0 * pr["rigidity"]) * (1.2 if settle else 1.0)          # per unit node mass (1/s^2)
        rel = ((v[cj] - v[ci]) * u).sum(1)
        T = np.maximum(np.where(stretch > 0, kc * stretch + 10.0 * rel, 0.0), 0.0)     # cables only pull
        self.tension = T / kc                                                        # (as a strain, for the light)
        acc = np.zeros_like(x)
        np.add.at(acc, ci, T[:, None] * u)
        np.add.at(acc, cj, -T[:, None] * u)
        vcom = v.mean(0)
        acc += -(2.0 + 5.0 * pr["coherence"]) * (v - vcom)                          # internal damping
        acc += a_body
        v += acc * dt
        x += v * dt
        # a module never folds flat: consecutive rings keep a minimum spacing (the struts' reach)
        c = x.reshape(self.K + 1, N_RING, 3).mean(1)
        dc = c[1:] - c[:-1]
        h = np.maximum(np.linalg.norm(dc, axis=1), 1e-6)
        short = np.maximum(0.0, self.h_min - h)
        if short.any():
            push = (0.5 * short / h)[:, None] * dc
            x.reshape(self.K + 1, N_RING, 3)[1:] += push[:, None, :]
            x.reshape(self.K + 1, N_RING, 3)[:-1] -= push[:, None, :]
        si, sj = self.struts[:, 0], self.struts[:, 1]
        for _ in range(6):                                                           # struts: rigid
            d = x[sj] - x[si]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            corr = ((L - self.L_strut) / L * 0.5)[:, None] * d
            np.add.at(x, si, corr)
            np.add.at(x, sj, -corr)
        v[:] = (x - x_prev) / dt

    def _simulate(self, dt: float) -> None:
        x, v = self.x, self.v
        P = x.mean(0)
        vcom = v.mean(0)
        head = x[-N_RING:].mean(0) - x[:N_RING].mean(0)
        axis = head / max(float(np.linalg.norm(head)), 1e-6)
        a = self._flight(dt, P, vcom)
        # the body faces its axis: turn the whole net towards the heading (slowly, as a swimmer does)
        R = self._R()
        want = R[:, 0]
        c = np.cross(axis, want)
        s = float(np.linalg.norm(c))
        if s > 1e-4:
            ang = min(math.asin(min(1.0, s)), 1.2 * dt)
            k = c / s
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Rm = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
            x[:] = P + (x - P) @ Rm.T
            v[:] = v @ Rm.T
        dR = self.glove.begin(dt)
        self.glove.rigid(x, v, P, R @ dR @ R.T)
        pr = self.params.values()
        W = dict(zip(self.REGIMES, self.weights()))
        swim_push = (0.6 + 1.8 * (W["SWIM"] + W["WHIP"])) * (0.5 + pr["mechanism"])      # its own wave drives it
        a = a + axis * swim_push * 0.3
        wind = self.glove.wind_acc(x, P, R)
        self._activation(dt)
        self._physics(dt, a + (wind if not isinstance(wind, float) else 0.0))
        self.glove.damp(v, dt)
        for ob in self.obstacles:
            if ob is not None and self._push_out(x, v, ob).any():
                self.glow = max(self.glow, 0.9)
                self.kick = 1.0
        self.surface += (min(1.0, 0.3 + float(self.tension.mean()) * 0.6) - self.surface) * min(1.0, dt * 3.0)

    # ------------------------------------------------------------------ events
    def _on_kick(self, strength: float) -> None:
        if self.weights()[self.REGIMES.index("WHIP")] > 0.2 or strength > 0.9:
            self.whip = 0.0

    def _apply_event(self, name: str, arg) -> None:
        if name == "LASH":
            self.whip = 0.0
            self.kick = 1.0
            self._log(name)
            return
        if name == "COIL":
            self.goal("COIL", 3.2)
            self._log(name)
            return
        if name == "UNFURL":
            self.goal("REACH", 3.2)
            self._log(name)
            return
        super()._apply_event(name, arg)

    def _threat(self, ob) -> None:
        super()._threat(ob)
        self.goal("SPRING" if self.rng.chance(0.5) else "COIL", 3.0)

    # ------------------------------------------------------------------ state
    def state(self):
        s = self.cfg.size / 1.8
        ns, nc = len(self.struts), len(self.cables)
        M = np.full((ns + nc, 6), -1.0)
        M[:ns, 0:2] = self.struts
        M[:ns, 2] = 0
        M[:ns, 3] = 0.032 * s
        M[:ns, 4] = 0.0
        M[:ns, 5] = np.arange(ns) / max(ns, 1)
        T_ref = 0.25 + float(np.percentile(self.tension, 90))
        M[ns:, 0:2] = self.cables
        M[ns:, 2] = 1 + (self.c_kind == 0)                                            # 1 muscle cable, 2 ring
        M[ns:, 3] = 0.0035 * s
        M[ns:, 4] = np.clip(self.tension / T_ref, 0.0, 1.5)
        M[ns:, 5] = self.act
        radius = np.full(self.n, 0.045 * s)
        return self._state(self.x, radius, M, [self.kick, float(self.tension.mean()), self.phase % (2 * math.pi)],
                           {"total": float(self.mass.sum()), "struts": ns, "cables": nc}, "STRUCTURED")


__all__ = ["TensorEngine", "TensorConfig"]
