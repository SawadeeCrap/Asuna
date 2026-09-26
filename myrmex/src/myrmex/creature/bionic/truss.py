"""Truss (creature v18): a flying variable-geometry truss that remodels itself like bone.

A lattice of 96 hubs and a few hundred struts, flown by thrusters on some of its hubs.  Two laws act
on it at once:

* **Variable geometry** - every strut is an actuator: its rest length is the distance its two hubs have
  in the current target layout, so the whole lattice morphs between machines:

      FUSELAGE  a spindle frame, like an airship's     WING    a swept delta wing
      ARCH      a flying bridge arch                   RING    an annular (ring) wing
      TRIPOD    a lander on three splayed legs         SPINE   a long twisted boom

* **Wolff's law** - the struts are living bone.  Each carries the load the physics gives it (gravity,
  the thrusters' push, manoeuvres, blows, impacts); struts loaded above the average thicken, idle
  ones thin out and are resorbed, slender struts in compression are driven harder (buckling - they
  visibly bow).  The total volume of material is fixed, so bone migrates to where the load paths are:
  after a while the lattice shows you how it is being used.  A strut is reborn where the lattice
  fails to hold its shape.  Thrusters glow; stress glows in the struts.

Events: BLOW (a hammer blow on one hub), ANNEAL (all bone evened out), OVERLOAD (full thrust: the
structure reinforces along the thrust paths).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..puppet import from_rotvec, rotvec
from .base import G, BionicConfig, BionicEngine

N_NODES = 96
K_NN = 7
M_MAX = 520


def _halton(n: int, base: int) -> np.ndarray:
    out = np.zeros(n)
    for i in range(n):
        f, r, k = 1.0, 0.0, i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        out[i] = r
    return out


def layout(name: str, U: np.ndarray, s: float) -> np.ndarray:
    """Hub positions of one machine (body frame: x forward, y left, z up) from fixed node coordinates U."""
    a, b, c = U[:, 0], U[:, 1], U[:, 2]
    if name == "FUSELAGE":
        rho = 0.45 * s * np.sin(math.pi * a) ** 0.6 * (0.72 + 0.28 * c)
        ang = 2 * math.pi * b
        return np.stack([(a - 0.5) * 3.0 * s, rho * np.cos(ang), rho * np.sin(ang)], 1)
    if name == "WING":
        y = (b - 0.5) * 3.2 * s
        ay = np.abs(2 * b - 1)
        chord = 1.4 * s * (1.0 - 0.72 * ay)
        x = -0.55 * np.abs(y) + (a - 0.45) * chord
        z = (c - 0.5) * 0.16 * s * (1.0 - 0.6 * ay) + 0.12 * np.abs(y)
        return np.stack([x + 0.4 * s, y, z], 1)
    if name == "ARCH":
        th = math.pi * a
        rad = 1.3 * s + (c - 0.5) * 0.24 * s
        return np.stack([rad * np.cos(th), (b - 0.5) * 0.5 * s, 0.9 * rad * np.sin(th) - 0.55 * s], 1)
    if name == "RING":
        phi = 2 * math.pi * a
        psi = 2 * math.pi * b
        rm = 0.17 * s * (0.6 + 0.4 * c)
        R0 = 1.0 * s + rm * np.cos(psi)
        return np.stack([1.9 * rm * np.sin(psi), R0 * np.cos(phi), R0 * np.sin(phi)], 1)
    if name == "TRIPOD":
        k = np.minimum((a * 3).astype(int), 2)
        t = a * 3 - k
        ang = 2 * math.pi * k / 3.0
        d = np.stack([0.75 * np.cos(ang), 0.75 * np.sin(ang), -np.ones_like(a)], 1)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        taper = 0.16 * s * (1.0 - 0.65 * t)
        side = np.stack([-np.sin(ang), np.cos(ang), np.zeros_like(a)], 1)
        up = np.cross(d, side)
        pb, pc = (b - 0.5) * 2, (c - 0.5) * 2
        hub = np.array([0.0, 0.0, 0.55 * s])
        return hub + (t * 1.7 * s)[:, None] * d + (taper * pb)[:, None] * side + (taper * pc)[:, None] * up
    # SPINE: a long twisted boom of triangular section
    k = np.minimum((b * 3).astype(int), 2)
    ang = 2 * math.pi * k / 3.0 + 2 * math.pi * 1.2 * a
    rad = 0.2 * s * (0.8 + 0.4 * c) * (1.0 - 0.45 * np.abs(2 * a - 1))
    return np.stack([(a - 0.5) * 3.6 * s, rad * np.cos(ang), rad * np.sin(ang)], 1)


def thrust_weight(name: str, U: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Where the thrusters sit on each machine (weights per hub, not normalised)."""
    a, b = U[:, 0], U[:, 1]
    x = X[:, 0]
    if name in ("FUSELAGE", "SPINE"):
        return (a < 0.1).astype(float)
    if name == "WING":
        return ((np.abs(2 * b - 1) > 0.86) & (x < np.percentile(x, 40))).astype(float) + \
            ((np.abs(2 * b - 1) < 0.06) & (x < np.percentile(x, 50))) * 0.5
    if name == "ARCH":
        return ((a < 0.07) | (a > 0.93)).astype(float)
    if name == "RING":
        return (x < -0.1 * np.ptp(x)).astype(float) * (np.abs(np.cos(2 * math.pi * a * 4)) > 0.8)
    t = a * 3 - np.minimum((a * 3).astype(int), 2)                              # TRIPOD: the feet
    return (t > 0.86).astype(float)


@dataclass
class TrussConfig(BionicConfig):
    cruise: float = 2.4
    sim_rate: float = 240.0


class TrussEngine(BionicEngine):
    VARIANT = "truss"
    KIND = 4
    REGIMES = ("FUSELAGE", "WING", "ARCH", "RING", "TRIPOD", "SPINE")
    PLAN = {"CRUISE": (("FUSELAGE", "WING", "SPINE"), 1.0), "HOVER": (("TRIPOD", "RING", "ARCH"), 0.12),
            "EXPLORE": (("WING", "RING", "SPINE"), 0.6), "DISPLAY": (("ARCH", "RING", "TRIPOD"), 0.3),
            "EVADE": (("SPINE", "WING"), 1.5), "REFORM": (("FUSELAGE",), 0.5), "STRIKE": (("SPINE",), 1.8)}
    EVENTS = BionicEngine.EVENTS + ("BLOW", "ANNEAL", "OVERLOAD")

    def __init__(self, cfg: BionicConfig | None = None):
        super().__init__(cfg or TrussConfig())
        cfg = self.cfg
        self.s = s = cfg.size / 1.8
        n = self.n = N_NODES
        rng = np.random.default_rng(cfg.seed & 0xFFFFFFFF)
        self.U = np.stack([_halton(n, 2), _halton(n, 3), _halton(n, 5)], 1)
        self.U = (self.U + rng.uniform(0, 0.02, self.U.shape)) % 1.0
        self.layouts = {r: layout(r, self.U, s) for r in self.REGIMES}
        self.thrusters = {r: thrust_weight(r, self.U, self.layouts[r]) for r in self.REGIMES}
        self.ei = np.zeros((M_MAX, 2), int)
        self.alive = np.zeros(M_MAX, bool)
        self.dying = np.zeros(M_MAX, bool)
        self.r = np.zeros(M_MAX)
        self.L0 = np.ones(M_MAX)
        self.eps = np.zeros(M_MAX)                    # strain, averaged (what the bone feels)
        self.eps_now = np.zeros(M_MAX)
        self.bow = np.zeros(M_MAX)
        self.r_dead = 0.0045 * s
        self.r_max = 0.07 * s
        self.V_budget = 0.035 * s ** 3 * (0.6 + 0.8 * self.params["density"])
        self.T_body = self._target()
        self._rebuild(initial=True)
        self.x = self.P + (self.T_body - self.T_body.mean(0)) @ self._R().T
        self.v = np.zeros((n, 3))
        self.thrust = np.zeros(n)
        self.thrust_dir = np.array([0.0, 0.0, 1.0])
        self.thrust_mag = 0.0
        self.hit = np.zeros(n)                        # glow of recent blows per hub
        self.remodel_t = self.rebuild_t = 0.0
        self.overload = 0.0
        for _ in range(480):                          # settle under its own weight
            self._physics(1 / 240, np.zeros(3))
        for _ in range(12):
            self._remodel(0.1)

    # ------------------------------------------------------------------ geometry
    def _target(self) -> np.ndarray:
        w = self.weights()
        T = sum(wi * self.layouts[r] for wi, r in zip(w, self.REGIMES))
        gc = self.glove.ctrl
        if gc.active:
            inp = self.inp
            T = self.glove.local(T, self.t, inp.beat if inp.playing else 2.0 * self.t)
        return T

    def _rebuild(self, initial: bool = False) -> None:
        """Candidate struts: the K nearest hubs in the target layout (identity kept for struts that stay)."""
        T = self.T_body
        D = np.linalg.norm(T[:, None, :] - T[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        nn = np.argsort(D, axis=1)[:, :K_NN]
        pairs = {(int(min(i, j)), int(max(i, j))) for i in range(self.n) for j in nn[i]}
        self.pairs = pairs
        have = {tuple(int(q) for q in self.ei[k]): k for k in np.flatnonzero(self.alive)}
        for key, k in have.items():                                             # struts leaving the lattice
            if key not in pairs:
                self.dying[k] = True
            else:
                self.dying[k] = False
        slot = {tuple(int(q) for q in self.ei[k]): k for k in np.flatnonzero(~self.alive) if self.ei[k, 0] != self.ei[k, 1]}
        junk = [k for k in np.flatnonzero(~self.alive) if tuple(int(q) for q in self.ei[k]) not in pairs]
        live_r = self.r[self.alive & ~self.dying]
        r_new = 0.6 * float(live_r.mean()) if len(live_r) else 0.02 * self.s
        for p in pairs:
            if p in have:
                continue
            k = slot.get(p)
            if k is None:
                if not junk:
                    break
                k = junk.pop(0)
            self.ei[k] = p
            self.alive[k], self.dying[k] = True, False
            self.r[k] = r_new if not initial else 1.0
            self.eps[k] = 0.0
            self.L0[k] = float(np.linalg.norm((T if initial else self.x)[p[1]] - (T if initial else self.x)[p[0]]))
        if initial:
            self._normalise()

    def _normalise(self) -> None:
        al = self.alive & ~self.dying
        V = float((math.pi * self.r[al] ** 2 * self.L0[al]).sum())
        if V > 1e-12:
            self.r[al] = np.clip(self.r[al] * math.sqrt(self.V_budget / V), 0.0, self.r_max)

    # ------------------------------------------------------------------ physics
    def _physics(self, dt: float, a_fl: np.ndarray) -> None:
        pr = self.params.values()
        x, v = self.x, self.v
        al = np.flatnonzero(self.alive)
        i, j = self.ei[al, 0], self.ei[al, 1]
        # rest lengths follow the target layout (actuators have a finite speed)
        L_goal = np.linalg.norm(self.T_body[j] - self.T_body[i], axis=1)
        self.L0[al] += np.clip(L_goal - self.L0[al], -0.9 * dt * self.s, 0.9 * dt * self.s)
        L0 = self.L0[al]
        d = x[j] - x[i]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        u = d / L[:, None]
        r = self.r[al]
        A = math.pi * r ** 2
        E = 800.0 * (0.5 + pr["rigidity"])                                      # (per unit density)
        rel = ((v[j] - v[i]) * u).sum(1)
        eps = (L - L0) / L0
        self.eps_now[al] = eps
        m = np.full(self.n, 0.02 / self.n)                                      # hubs
        np.add.at(m, i, 0.5 * A * L0)
        np.add.at(m, j, 0.5 * A * L0)
        mu = np.minimum(m[i], m[j])
        k = np.minimum(E * A / L0, 0.07 * mu / (dt * dt))                       # explicit steps stay stable
        f = k * (L - L0) + 0.3 * np.sqrt(k * mu) * rel                          # axial force (+ tension)
        F = np.zeros_like(x)
        np.add.at(F, i, f[:, None] * u)
        np.add.at(F, j, -f[:, None] * u)
        M = float(m.sum())
        g_eff = 0.35 * G
        w = self.weights()
        thr = sum(wi * self.thrusters[r_] for wi, r_ in zip(w, self.REGIMES))
        if thr.sum() < 1e-6:
            thr = np.ones(self.n)
        thr = thr / thr.sum()
        Fth = M * (a_fl + np.array([0.0, 0.0, g_eff]))
        self.thrust_mag = float(np.linalg.norm(Fth))
        self.thrust_dir = Fth / max(self.thrust_mag, 1e-9)
        F += (0.4 * thr + 0.6 * m / M)[:, None] * Fth[None, :]                     # thrusters + lift over the frame
        if self.overload > 1e-3:                    # a proof load: full thrust against the whole mass, no net force
            X = self.thrust_dir * 2.5 * M * g_eff * self.overload
            F += thr[:, None] * X[None, :] - (m / M)[:, None] * X[None, :]
        self.thrust = thr / max(float(thr.max()), 1e-9) * min(3.0, float(np.linalg.norm(Fth) + 2.5 * M * g_eff *
                                                                             self.overload) / max(M * g_eff, 1e-9))
        F[:, 2] -= m * g_eff
        vcom = (m[:, None] * v).sum(0) / M
        # a faint shape memory: holds the lattice's mechanisms (the modes no strut resists), carries ~nothing else
        com = (m[:, None] * x).sum(0) / M
        R = self._R() @ self.glove.G
        Tw = com + (self.T_body - (m[:, None] * self.T_body).sum(0) / M) @ R.T
        F += m[:, None] * (25.0 + 50.0 * pr["coherence"]) * (Tw - x)
        F -= m[:, None] * (0.6 + 1.2 * pr["coherence"]) * (v - vcom) * 0.5          # structural damping
        v += F / m[:, None] * dt
        dv = v - vcom
        sp = np.linalg.norm(dv, axis=1, keepdims=True)
        v[:] = vcom + dv * np.minimum(1.0, 20.0 / np.maximum(sp, 1e-9))
        x += v * dt

    def _attitude(self, dt: float) -> None:
        """Turn the whole lattice (rigidly - no strut feels it) towards its heading."""
        x, v = self.x, self.v
        com = x.mean(0)
        R = self._R() @ self.glove.G
        A = (self.T_body - self.T_body.mean(0)) @ R.T
        B = x - com
        H = B.T @ A
        U_, _, Vt = np.linalg.svd(H)
        dd = np.sign(np.linalg.det(Vt.T @ U_.T))
        Q = Vt.T @ np.diag([1.0, 1.0, dd]) @ U_.T
        Qf = from_rotvec(rotvec(Q) * min(1.0, dt * 2.5))
        x[:] = com + B @ Qf.T
        v[:] = v @ Qf.T

    def _remodel(self, dt: float) -> None:
        """Wolff's law with a fixed volume of bone: loaded struts grow, idle ones are resorbed."""
        pr = self.params.values()
        al = self.alive & ~self.dying
        k = np.flatnonzero(al)
        if len(k) == 0:
            return
        mag = np.abs(self.eps[k])
        ref = float(mag.mean()) + 1e-9
        L = self.L0[k]
        slender = L / np.maximum(self.r[k] * 22.0, 1e-6)
        comp = np.clip(-self.eps[k] / ref, 0.0, None)
        s = mag / ref + comp * np.clip(slender ** 2 - 1.0, 0.0, 3.0) * 0.35       # buckling drives compression bone
        rate = (0.25 + 1.2 * pr["architecture"]) / (1.0 + 2.0 * pr["rigidity"])
        self.r[k] *= np.exp(np.clip(rate * dt * (s - 1.0), -0.5, 0.5))
        self.bow[:] = 0.0
        self.bow[k] = np.clip((comp - 1.2) * (slender ** 2 - 4.0) * 0.012, 0.0, 0.1) * L    # buckled: it bows
        # resorption: never below four struts per hub (a hub would become a mechanism)
        thin = k[self.r[k] < self.r_dead]
        if len(thin):
            deg = np.bincount(self.ei[al].ravel(), minlength=self.n)
            for q in thin[np.argsort(self.r[thin])]:
                a, b = self.ei[q]
                if deg[a] > 4 and deg[b] > 4:
                    self.alive[q] = False
                    deg[a] -= 1
                    deg[b] -= 1
                else:
                    self.r[q] = self.r_dead
        # rebirth where the lattice fails to hold its shape
        dead = np.flatnonzero(~self.alive)
        if len(dead):
            T = self.T_body
            cand = [q for q in dead if tuple(int(z) for z in self.ei[q]) in self.pairs]
            if cand:
                cand = np.array(cand)
                i, j = self.ei[cand, 0], self.ei[cand, 1]
                Lt = np.linalg.norm(T[j] - T[i], axis=1)
                Lx = np.linalg.norm(self.x[j] - self.x[i], axis=1)
                bad = cand[np.abs(Lx - Lt) / np.maximum(Lt, 1e-6) > 0.08][:6]
                for q in bad:
                    self.alive[q], self.dying[q] = True, False
                    self.r[q] = 1.6 * self.r_dead
                    self.L0[q] = float(np.linalg.norm(self.x[self.ei[q, 1]] - self.x[self.ei[q, 0]]))
                    self.eps[q] = 0.0
        self.V_budget = 0.035 * self.s ** 3 * (0.6 + 0.8 * pr["density"])
        self._normalise()

    def _simulate(self, dt: float) -> None:
        x, v = self.x, self.v
        com, vcom = x.mean(0), v.mean(0)
        a_fl = self._flight(dt, com, vcom)
        dR = self.glove.begin(dt)
        R = self._R()
        self.glove.rigid(x, v, com, R @ dR @ R.T)
        self.T_body = self._target()
        wind = self.glove.wind_acc(x, com, R)
        if not isinstance(wind, float):
            v += wind * dt
        self._physics(dt, a_fl)
        self._attitude(dt)
        self.glove.damp(v, dt)
        for ob in self.obstacles:
            if ob is None:
                continue
            hit = self._push_out(x, v, ob)
            if hit.any():
                self.hit[hit] = 1.0
                self.glow = max(self.glow, 0.8)
        # struts leaving the lattice shrink away
        dy = self.dying & self.alive
        if dy.any():
            self.r[dy] *= math.exp(-dt / 0.3)
            self.alive[dy & (self.r < self.r_dead)] = False
            self.dying[~self.alive] = False
        a = min(1.0, dt / 0.4)
        self.eps += (self.eps_now - self.eps) * a
        self.hit *= math.exp(-dt / 0.6)
        self.overload *= math.exp(-dt / 1.2)
        self.remodel_t += dt
        if self.remodel_t > 0.1:
            self._remodel(self.remodel_t)
            self.remodel_t = 0.0
        self.rebuild_t += dt
        if self.rebuild_t > 1.5:
            self.rebuild_t = 0.0
            self._rebuild()
        k = self.alive & ~self.dying
        ref = float(np.abs(self.eps[k]).mean()) + 1e-9 if k.any() else 1.0
        self.surface += (min(1.0, 0.3 + 0.2 * float(np.abs(self.eps[k] / ref).max(initial=0.0)) / 3.0)
                         - self.surface) * min(1.0, dt * 3.0)

    # ------------------------------------------------------------------ events
    def _blow(self, node: int, strength: float) -> None:
        d = self.nrng.standard_normal(3)
        d /= max(float(np.linalg.norm(d)), 1e-9)
        self.v[node] += d * 3.0 * strength
        self.hit[node] = 1.0

    def _on_kick(self, strength: float) -> None:
        self._blow(int(self.nrng.integers(0, self.n)), strength)

    def _apply_event(self, name: str, arg) -> None:
        if name == "BLOW":
            self._blow(int(self.nrng.integers(0, self.n)), 1.4)
            self.glow = 1.0
        elif name == "ANNEAL":
            al = self.alive & ~self.dying
            self.r[al] = float(self.r[al].mean()) if al.any() else self.r_dead
            self._normalise()
        elif name == "OVERLOAD":
            self.overload = 1.0
        else:
            super()._apply_event(name, arg)
            return
        self._log(name)

    def _threat(self, ob) -> None:
        super()._threat(ob)
        self.goal("SPINE" if self.rng.chance(0.5) else "WING", 3.0)

    # ------------------------------------------------------------------ state
    def state(self):
        M = np.full((M_MAX, 6), -1.0)
        k = np.flatnonzero(self.alive)
        al = self.alive & ~self.dying
        ref = float(np.abs(self.eps[al]).mean()) + 1e-9 if al.any() else 1.0
        M[k, 0:2] = self.ei[k]
        M[k, 2] = self.dying[k]
        M[k, 3] = self.r[k]
        M[k, 4] = np.clip(self.eps[k] / (2.5 * ref), -1.5, 1.5)                  # + tension, - compression
        M[k, 5] = self.bow[k]
        hub = np.zeros(self.n)
        if len(k):
            np.maximum.at(hub, self.ei[k, 0], self.r[k])
            np.maximum.at(hub, self.ei[k, 1], self.r[k])
        hub = np.maximum(hub * 1.35, 0.012 * self.s)
        extra = np.concatenate([self.thrust_dir, [self.thrust_mag, float(al.sum()), self.kick, self.overload],
                                np.clip(self.thrust, 0.0, 3.0), self.hit])
        V = float((math.pi * self.r[al] ** 2 * self.L0[al]).sum())
        return self._state(self.x, hub, M, extra, {"total": V, "budget": self.V_budget, "struts": int(al.sum())},
                           "STRUCTURED")


__all__ = ["TrussEngine", "TrussConfig", "layout", "N_NODES", "M_MAX"]
