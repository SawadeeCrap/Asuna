"""Fold (creature v15): a rigid-foldable Miura-ori sheet that flies, swims and reconfigures by folding.

The body is a sheet of R x C congruent parallelogram panels joined by creases (the Miura-ori pattern).
Its folding is exact rigid-origami kinematics - one fold angle theta moves every panel without
bending any (Schenk & Guest):

    b-edges (across):  dx = S = b cos(theta) tan(g) / sqrt(1 + cos^2(theta) tan^2(g)),  dy = +-V
    a-edges (along):   dy = L = a sqrt(1 - sin^2(theta) sin^2(g)),                       dz = +-H = a sin(theta) sin(g)

theta varies across the sheet (travelling fold waves, beats) and the folded sheet is bent as a whole
(fan, tube, arch, twist, flapping halves); panel edges are hard constraints, so wherever the targets ask
for the impossible the sheet finds its own nearest shape.  Every panel feels the air: pressure drag
along its normal (flutter, gliding, the thrust of a flap).

    GLIDER  nearly flat, flapping wide    PLEAT  folded tight into a bar     TUBE    rolled into a pleated tube
    SHELL   a domed carapace             BELL   a pleated bell (the fold opens along it), pulsing like a jellyfish
    RIBBON  a twisted swimming band

Events: CLAP (a snap fold), FURL (pleat), BLOOM (bell).  Inner faces carry structural colour (Blender).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import BionicConfig, BionicEngine


def miura(theta: np.ndarray, a: float, b: float, g: float) -> np.ndarray:
    """(R+1, C+1) fold angles -> (R+1, C+1, 3) vertex positions (X across, Y along, Z up)."""
    ct2 = np.cos(theta) ** 2 * math.tan(g) ** 2
    S = b * np.cos(theta) * math.tan(g) / np.sqrt(1.0 + ct2)
    V = b / np.sqrt(1.0 + ct2)
    L = a * np.sqrt(1.0 - np.sin(theta) ** 2 * math.sin(g) ** 2)
    H = a * np.sin(theta) * math.sin(g)
    Rn, Cn = theta.shape
    X = np.zeros_like(theta)
    X[:, 1:] = np.cumsum(S[:, :-1], axis=1)
    Y = np.zeros_like(theta)
    Y[1:, :] = np.cumsum(L[:-1, :], axis=0)
    Y += np.where(np.arange(Cn)[None, :] % 2 == 1, V, 0.0)
    Z = np.where(np.arange(Rn)[:, None] % 2 == 1, H, 0.0)
    return np.stack([X, Y, Z], -1)


def _bend(u: np.ndarray, w: np.ndarray, kappa):
    """Bend the (u, w) plane around an axis at distance 1/kappa: u -> arc, w -> radial (kappa may vary)."""
    kappa = np.asarray(kappa, float)
    if np.all(np.abs(kappa) < 1e-4):
        return u, w
    k = np.where(np.abs(kappa) < 1e-4, 1e-4, kappa)
    rho = 1.0 / k - w
    phi = u * k
    return rho * np.sin(phi), 1.0 / k - rho * np.cos(phi)


# regime -> (theta0, bell, tube, arch, twist, flap, wave)
REGIME_PARAMS = {
    "GLIDER": (0.3, 0.0, 0.0, 0.12, 0.0, 0.5, 0.08),
    "PLEAT": (1.3, 0.0, 0.0, 0.0, 0.0, 0.05, 0.08),
    "TUBE": (0.7, 0.0, 0.95, 0.0, 0.5, 0.0, 0.3),
    "SHELL": (0.5, 0.0, 0.45, 0.55, 0.0, 0.12, 0.1),
    "BELL": (0.75, 1.0, 0.95, 0.0, 0.0, 0.0, 0.05),
    "RIBBON": (0.6, 0.0, 0.0, 0.0, 2.8, 0.0, 0.45),
}


@dataclass
class FoldConfig(BionicConfig):
    rows: int = 10
    cols: int = 12
    cruise: float = 2.6
    sim_rate: float = 120.0


class FoldEngine(BionicEngine):
    VARIANT = "fold"
    KIND = 1
    REGIMES = tuple(REGIME_PARAMS)
    PLAN = {"CRUISE": (("GLIDER", "RIBBON", "PLEAT"), 1.0), "HOVER": (("BELL", "SHELL", "GLIDER"), 0.12),
            "EXPLORE": (("RIBBON", "GLIDER", "TUBE"), 0.6), "DISPLAY": (("BELL", "SHELL", "TUBE"), 0.3),
            "EVADE": (("PLEAT", "TUBE"), 1.5), "REFORM": (("GLIDER",), 0.5), "STRIKE": (("PLEAT",), 1.8)}
    EVENTS = BionicEngine.EVENTS + ("CLAP", "FURL", "BLOOM")

    def __init__(self, cfg: BionicConfig | None = None):
        super().__init__(cfg or FoldConfig())
        cfg = self.cfg
        self.R, self.C = getattr(cfg, "rows", 10), getattr(cfg, "cols", 12)
        s = cfg.size / 1.8
        self.a, self.b, self.g = 0.17 * s, 0.15 * s, math.radians(55.0)
        R1, C1 = self.R + 1, self.C + 1
        self.n = R1 * C1
        kk, mm = np.meshgrid(np.arange(R1), np.arange(C1), indexing="ij")
        self.kk, self.mm = kk.ravel(), mm.ravel()
        ij = lambda k, m: k * C1 + m                                                    # noqa: E731
        edges = [(ij(k, m), ij(k, m + 1)) for k in range(R1) for m in range(self.C)] + \
                [(ij(k, m), ij(k + 1, m)) for k in range(self.R) for m in range(C1)] + \
                [(ij(k, m), ij(k + 1, m + 1)) for k in range(self.R) for m in range(self.C)]      # panel diagonals
        self.edges = np.array(edges)
        self.panels = np.array([(ij(k, m), ij(k, m + 1), ij(k + 1, m + 1), ij(k + 1, m))
                                for k in range(self.R) for m in range(self.C)])
        flat = miura(np.zeros((R1, C1)), self.a, self.b, self.g).reshape(-1, 3)
        self.L0 = np.linalg.norm(flat[self.edges[:, 1]] - flat[self.edges[:, 0]], axis=1)
        self.deg = np.bincount(self.edges.ravel(), minlength=self.n).astype(float)
        self.mass = np.full(self.n, 1.0 / self.n)                                      # finite material
        self.phase = self.flap_phase = 0.0
        self.clap = 0.0
        loc = self._local(0.0)
        self.x = self.P + loc @ self._R().T
        self.v = np.zeros((self.n, 3))

    # ------------------------------------------------------------------ the folding
    def _params(self) -> np.ndarray:
        w = self.weights()
        return (w[:, None] * np.array([REGIME_PARAMS[r] for r in self.REGIMES])).sum(0)

    def _local(self, dt: float) -> np.ndarray:
        """Target positions in the body frame (x forward = along the sheet, y across, z up)."""
        pr, inp = self.params.values(), self.inp
        th0, bell, tube, arch, twist, flap, wave = self._params()
        bps = max(inp.tempo, 60.0) / 60.0
        self.phase += 2 * math.pi * dt * (0.35 + 0.5 * pr["mechanism"]) * bps * 0.5
        self.flap_phase += 2 * math.pi * dt * (0.5 * bps if inp.playing else 0.6) * (0.7 + 0.6 * pr["speed"])
        R1, C1 = self.R + 1, self.C + 1
        kn = np.arange(R1)[:, None] / self.R
        beat = inp.beat if inp.playing else 2 * self.t
        th = th0 + 0.18 * inp.bass * math.sin(math.pi * beat) ** 2 \
            + (wave + 0.2 * pr["tendril_activity"] * wave) * np.sin(2 * math.pi * 1.2 * kn - self.phase) \
            + 0.35 * self.clap + 0.25 * self.kick \
            + bell * (1.1 * (0.5 - kn) + 0.3 * math.sin(math.pi * beat) ** 4)           # a bell: pulses on the beat
        th = np.clip(th * (1.25 - 0.5 * pr["expansion"]) + np.zeros((R1, C1)), 0.0, 1.45)
        P3 = miura(th, self.a, self.b, self.g)
        lat, fwd, up = P3[..., 0], P3[..., 1], P3[..., 2]
        width = np.ptp(lat, axis=1)                                                   # each row's own width
        lat = (lat - lat.mean(1, keepdims=True)).ravel()
        fwd, up = (fwd - fwd.mean()).ravel(), (up - up.mean()).ravel()
        W = max(float(np.ptp(lat)), 1e-3)
        Lb = max(float(np.ptp(fwd)), 1e-3)
        row_w = np.repeat(np.maximum(width, 1e-3), C1)
        kap = 2 * math.pi * 0.98 * tube / np.where(bell > 0.05, row_w, W)             # every ring closes on itself
        lat, up = _bend(lat, up, kap)                                                  # rolled into a tube
        fwd, up = _bend(fwd, up, math.pi * 0.9 * arch / Lb)                             # arched along
        ang = twist * fwd / Lb + 0.5 * pr["asymmetry"] * math.sin(0.2 * self.t)
        ca, sa = np.cos(ang), np.sin(ang)
        lat, up = ca * lat - sa * up, sa * lat + ca * up
        fl = (flap + 0.25 * self.clap) * math.sin(self.flap_phase)                       # both halves flap
        fa = fl * np.tanh(lat / (0.12 * W))
        lat, up = np.cos(fa) * lat - np.sin(fa) * up, np.sin(fa) * lat + np.cos(fa) * up
        loc = np.stack([fwd, lat, up], 1)
        return self.glove.local(loc, self.t, inp.beat if inp.playing else 2.0 * self.t)

    # ------------------------------------------------------------------ physics
    def _simulate(self, dt: float) -> None:
        pr = self.params.values()
        x, v = self.x, self.v
        x_prev = x.copy()
        P, vcom = x.mean(0), v.mean(0)
        a_fl = self._flight(dt, P, vcom)
        dR = self.glove.begin(dt)
        R = self._R()
        self.glove.rigid(x, v, P, R @ dR @ R.T)
        T = P + self._local(dt) @ (R @ self.glove.G).T
        k_t = (4.0 + 26.0 * pr["rigidity"]) * (1.0 - 0.5 * pr["fluidity"])
        acc = k_t * (T - x) - (2.0 + 4.0 * pr["coherence"]) * (v - vcom) + a_fl
        # air on every panel: pressure drag along its normal (flutter, glide, the push of a flap)
        p = self.panels
        c0, c1, c2, c3 = x[p[:, 0]], x[p[:, 1]], x[p[:, 2]], x[p[:, 3]]
        nrm = np.cross(c2 - c0, c3 - c1)
        area = np.linalg.norm(nrm, axis=1)
        nrm = nrm / np.maximum(area, 1e-9)[:, None]
        vp = 0.25 * (v[p[:, 0]] + v[p[:, 1]] + v[p[:, 2]] + v[p[:, 3]])
        vn = (vp * nrm).sum(1)
        F = -(0.72 * 0.5 * area * vn * np.abs(vn))[:, None] * nrm                       # 1/2 rho Cd A vn^2
        for q in range(4):
            np.add.at(acc, p[:, q], F * 0.25 / self.mass[p[:, q], None])
        wind = self.glove.wind_acc(x, P, R)
        v += (acc + (wind if not isinstance(wind, float) else 0.0)) * dt
        x += v * dt
        self.project_lengths(x, self.edges, self.L0, 8, 1.5, self.deg)                  # panels stay rigid
        for ob in self.obstacles:
            if ob is not None and self._push_out(x, v, ob).any():
                self.glow = max(self.glow, 0.8)
        v[:] = (x - x_prev) / dt
        self.glove.damp(v, dt)
        self.clap *= math.exp(-dt / 0.35)
        self.surface += (min(1.0, 0.3 + 0.5 * float(np.abs(vn).mean())) - self.surface) * min(1.0, dt * 3.0)

    # ------------------------------------------------------------------ events
    def _on_kick(self, strength: float) -> None:
        if strength > 0.8 and self.rng.chance(0.4):
            self.clap = 1.0

    def _apply_event(self, name: str, arg) -> None:
        if name == "CLAP":
            self.clap = 1.0
            self.glow = 1.0
        elif name == "FURL":
            self.goal("PLEAT", 3.2)
        elif name == "BLOOM":
            self.goal("BELL", 3.2)
        else:
            super()._apply_event(name, arg)
            return
        self._log(name)

    def _threat(self, ob) -> None:
        super()._threat(ob)
        self.goal("PLEAT" if self.rng.chance(0.6) else "TUBE", 3.0)

    # ------------------------------------------------------------------ state
    def state(self):
        folded = float(self._params()[0])
        return self._state(self.x, np.full(self.n, 0.02), np.zeros((0, 6)),
                           [self.R, self.C, self.flap_phase % (2 * math.pi), self.clap, folded],
                           {"total": float(self.mass.sum()), "panels": len(self.panels)}, "STRUCTURED")


__all__ = ["FoldEngine", "FoldConfig", "miura", "REGIME_PARAMS"]
