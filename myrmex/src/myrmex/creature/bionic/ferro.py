"""Ferro (creature v17): a body of ferrofluid led through the air by an invisible magnet.

Black magnetic liquid has no form of its own - the field gives it one.  The organism is one volume of
fluid (conserved exactly, droplets included) and an invisible dipole that flies ahead of it; the
regimes are field configurations and what the fluid does in them:

    CROWN      magnet above: the Rosensweig instability - a hexagonal field of spikes on a dome
    URCHIN     magnet inside: spikes all round, a hedgehog of black needles
    TONGUE     magnet ahead: the fluid is drawn out into a long tongue, spikes combed along the field
    LABYRINTH  field in the plane of a flat disc: the surface folds into a maze of ridges
    FIN        magnet above a thin sheet: a blade whose crest breaks into a serrated edge
    STAR       a rotating field: a flat drop with rotating arms, spikes at their tips

The physics is the real one, simplified:
* spikes appear where the normal field exceeds the critical field Bc and grow as sqrt(B/Bc - 1);
  they vanish only below 0.85 Bc (the transition is hysteretic, as in the real fluid);
* spikes are quasi-particles on the surface that repel each other at the pattern wavelength - the
  hexagonal packing is emergent - and slide up the field gradient;
* each spike is an underdamped capillary oscillator: kicks make them shoot out and quiver;
* the labyrinth is a Swift-Hohenberg pattern (stripes aligned with the in-plane field, spots when
  the symmetry breaks);
* strong pulses tear droplets off the spike tips; the satellites fall back and coalesce, volume and
  momentum conserved.

Events: SPLIT (throw off droplets), SURGE (a field surge), CALM (the field drops: a smooth black mirror).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...util.rng import stable_hash64
from ..puppet import SECTORS
from .base import BionicConfig, BionicEngine
from .meshes import FERRO_GRID, FERRO_HEAD, FERRO_LEN, FERRO_S, FERRO_SAT, ferro_base, fib_sphere, frames

REGIMES = ("CROWN", "URCHIN", "TONGUE", "LABYRINTH", "FIN", "STAR")
# regime -> ellipsoid axes (product 1: volume-true), tongue, arms, labyrinth, spike lean along the field
SHAPE = {
    "CROWN": ((1.3, 1.3, 0.59), 0.0, 0.0, 0.0, 0.3),
    "URCHIN": ((1.0, 1.0, 1.0), 0.0, 0.0, 0.0, 0.0),
    "TONGUE": ((2.2, 0.67, 0.68), 0.9, 0.0, 0.0, 1.0),
    "LABYRINTH": ((1.65, 1.65, 0.37), 0.0, 0.0, 1.0, 0.0),
    "FIN": ((1.9, 0.38, 1.38), 0.15, 0.0, 0.0, 0.5),
    "STAR": ((1.5, 1.5, 0.44), 0.0, 0.5, 0.0, 0.4),
}
# regime -> magnet position (body frame, body radii) and dipole direction (STAR: rotating in the plane)
MAGNET = {
    "CROWN": ((0.0, 0.0, 1.8), (0.0, 0.0, 1.0)),
    "URCHIN": ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "TONGUE": ((2.8, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "LABYRINTH": ((0.0, 0.0, -1.3), (1.0, 0.0, 0.0)),
    "FIN": ((0.0, 0.0, 1.9), (0.0, 0.0, 1.0)),
    "STAR": ((0.0, 0.0, -1.2), (1.0, 0.0, 0.0)),
}
BC = 0.55                                   # critical field (arbitrary units)


class Labyrinth:
    """Swift-Hohenberg on a periodic grid: du/dt = eps u - (1 + lap)^2 u - a (k.f)^2 u + g u^2 - u^3."""

    def __init__(self, n: int, seed: int, waves: float = 5.5):
        self.n = n
        k = np.fft.fftfreq(n, 1.0 / n) / waves                    # in units of the pattern wavenumber
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.k2 = self.KX ** 2 + self.KY ** 2
        self.rng = np.random.default_rng(seed)
        self.u = 0.05 * self.rng.standard_normal((n, n))

    def step(self, dt: float, eps: float, g: float, fdir=(1.0, 0.0), aniso: float = 0.5) -> None:
        kf = (self.KX * fdir[0] + self.KY * fdir[1]) ** 2
        L = eps - (1.0 - self.k2) ** 2 - aniso * kf
        u = self.u
        uh = (np.fft.fft2(u) + dt * np.fft.fft2(g * u * u - u ** 3)) / (1.0 - dt * L)
        self.u = np.real(np.fft.ifft2(uh))

    def stir(self, amount: float) -> None:
        self.u += amount * self.rng.standard_normal(self.u.shape)


@dataclass
class FerroConfig(BionicConfig):
    cruise: float = 2.0
    sim_rate: float = 120.0


class FerroEngine(BionicEngine):
    VARIANT = "ferro"
    KIND = 3
    REGIMES = REGIMES
    PLAN = {"CRUISE": (("TONGUE", "FIN", "STAR"), 1.0), "HOVER": (("CROWN", "URCHIN", "LABYRINTH"), 0.12),
            "EXPLORE": (("TONGUE", "STAR", "FIN"), 0.6), "DISPLAY": (("URCHIN", "CROWN", "STAR", "LABYRINTH"), 0.3),
            "EVADE": (("FIN", "URCHIN"), 1.5), "REFORM": (("CROWN",), 0.4), "STRIKE": (("TONGUE",), 1.8)}
    EVENTS = BionicEngine.EVENTS + ("SPLIT", "SURGE", "CALM")

    def __init__(self, cfg: BionicConfig | None = None):
        super().__init__(cfg or FerroConfig())
        cfg = self.cfg
        self.s = s = cfg.size / 1.8
        self.V_total = 4.0 / 3.0 * math.pi * (0.5 * s) ** 3
        self.V_main = self.V_total
        self.c, self.u = self.P.copy(), np.zeros(3)
        self.fib = fib_sphere(640)
        self.sp_n = np.zeros((FERRO_S, 3))
        self.sp_n[:, 2] = 1.0
        self.sp_h, self.sp_hv = np.zeros(FERRO_S), np.zeros(FERRO_S)
        self.sp_on = np.zeros(FERRO_S, bool)
        self.sp_die = np.zeros(FERRO_S, bool)
        self.sat_x, self.sat_v = np.zeros((FERRO_SAT, 3)), np.zeros((FERRO_SAT, 3))
        self.sat_V, self.sat_t = np.zeros(FERRO_SAT), np.zeros(FERRO_SAT)
        self.B = 0.7
        self.surge = self.calm = 0.0
        self.star_phase = self.ripple_phase = 0.0
        self.k_lobes = 4 + (stable_hash64("ferro-arms", cfg.seed) % 3)
        self.lab = Labyrinth(FERRO_GRID, stable_hash64("ferro-lab", cfg.seed) & 0xFFFFFFFF)
        for _ in range(60):
            self.lab.step(0.6, 0.35, 0.15)
        self.Rb = self._R()
        self.sh = self._shape()
        self.mag_pos, self.mag_dir = self._magnet()
        self.lean = 0.0
        for _ in range(240):                              # the first spikes rise
            self._spikes(1 / 120)

    # ------------------------------------------------------------------ the field and the form
    def _shape(self) -> dict:
        w, gc = self.weights(), self.glove.ctrl
        ax = np.array([SHAPE[r][0] for r in self.REGIMES])
        sc = np.exp((w[:, None] * np.log(ax)).sum(0))
        tear, lobe, ridge, lean = ((w[:, None] * np.array([SHAPE[r][1:] for r in self.REGIMES])).sum(0))
        k = float(self.k_lobes)
        if gc.active:
            sc = sc * np.asarray(gc.stretch, float) * gc.scale
            if gc.rays > 0 and gc.ray_len > 1e-3:                        # the hand's rays become arms
                lobe, k = max(lobe, 0.6 * gc.ray_len), float(gc.rays)
        R = (3.0 * self.V_main / (4.0 * math.pi)) ** (1.0 / 3.0)
        return {"R": R, "sc": sc, "tear": tear, "k": k, "lobe": lobe * (0.6 + 0.5 * min(self.B / BC, 2.0)),
                "phase": self.star_phase, "ridge": ridge, "lean": lean}

    def _magnet(self) -> tuple[np.ndarray, np.ndarray]:
        """The invisible magnet: position (world) and dipole direction (world)."""
        w = self.weights()
        pos = sum(wi * np.array(MAGNET[r][0]) for wi, r in zip(w, self.REGIMES))
        ca, sa = math.cos(1.3 * self.star_phase), math.sin(1.3 * self.star_phase)
        dirs = [np.array((ca, sa, 0.0)) if r == "STAR" else np.array(MAGNET[r][1]) for r in self.REGIMES]
        d = sum(wi * di for wi, di in zip(w, dirs))
        d = d / max(float(np.linalg.norm(d)), 1e-6)
        return self.c + self.Rb @ (pos * self.sh["R"]), self.Rb @ d

    def _g(self, n: np.ndarray) -> np.ndarray:
        """Where the normal field is concentrated, per regime, blended (unit directions, body frame)."""
        w = dict(zip(self.REGIMES, self.weights()))
        nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
        g = w["CROWN"] * np.clip(nz, 0.0, 1.0) ** 0.6 + w["URCHIN"] * 0.95
        g = g + w["TONGUE"] * np.clip(nx + 0.15, 0.0, 1.0) ** 0.8
        g = g + w["FIN"] * np.exp(-(ny / 0.3) ** 2) * np.clip(nz + 0.35, 0.0, 1.0) ** 0.7
        if w["STAR"] > 1e-3:
            phi = np.arctan2(ny, nx)
            g = g + w["STAR"] * (1 - nz ** 2) ** 3 * np.clip(np.cos(self.k_lobes * (phi - self.star_phase)), 0, 1) ** 3
        return g

    def _field(self, dt: float) -> None:
        pr, inp, gc = self.params.values(), self.inp, self.glove.ctrl
        if inp.playing:
            drive = 0.5 + 0.9 * inp.bass + 0.35 * inp.energy
        else:
            drive = 0.8 + 0.1 * math.sin(0.7 * self.t)                    # idling: calm spikes, breathing
        beat = inp.beat if inp.playing else 2.0 * self.t
        if gc.active and gc.pulse > 1e-3:
            drive *= 1.0 + 0.5 * gc.pulse * math.sin(math.pi * beat) ** 2
        target = drive * (0.6 + 0.8 * pr["surface_activity"]) * (1.0 + 1.2 * self.surge) * (1.0 - 0.85 * self.calm) \
            + 0.6 * self.kick
        self.B += (target - self.B) * min(1.0, dt / 0.08)
        self.surge *= math.exp(-dt / 0.6)
        self.calm *= math.exp(-dt / 2.5)
        bps = max(inp.tempo, 60.0) / 60.0
        self.star_phase += dt * (0.5 + 0.9 * pr["mechanism"]) * (0.6 + 0.4 * bps) * (1.0 + 0.6 * self.kick)
        self.ripple_phase += dt * 14.0

    # ------------------------------------------------------------------ spikes
    def _spikes(self, dt: float) -> None:
        pr, gc = self.params.values(), self.glove.ctrl
        sh = self.sh
        R, B = sh["R"], self.B
        over = min(max(B / BC - 1.0, 0.0), 1.5)
        lam = 0.46 * R * (1.0 - 0.2 * min(over, 1.0))
        on = self.sp_on
        n = self.sp_n
        Bn = B * self._g(n)
        H0 = 0.5 * R * (0.6 + 0.8 * pr["tendril_activity"])
        if gc.active and gc.material is not None:
            H0 *= 1.0 + 0.4 * gc.material
        H = np.where(Bn > 0.85 * BC, H0 * np.sqrt(np.maximum(Bn / BC - 1.0, 0.0) + 0.02), 0.0)
        H = np.minimum(H, 0.9 * R)
        if gc.active and gc.finger_mode == "limbs":                       # five sectors follow five fingers
            phi = np.arctan2(n[:, 2], n[:, 1])
            d = (phi[:, None] - SECTORS[None, :] + np.pi) % (2 * np.pi) - np.pi
            ww = np.exp(-(d / 0.6) ** 2)
            f = (ww * (0.2 + 1.6 * np.asarray(gc.fingers))[None, :]).sum(1) / np.maximum(ww.sum(1), 1e-6)
            H = H * (1.0 + gc.amount * (f - 1.0))
        H = np.where(self.sp_die | ~on, 0.0, H)
        om = 2 * math.pi * (2.2 + 1.5 * (1.0 - pr["fluidity"]))
        ze = 0.18 + 0.3 * pr["coherence"]
        self.sp_hv += dt * (om * om * (H - self.sp_h) - 2.0 * ze * om * self.sp_hv)
        self.sp_h += dt * self.sp_hv
        neg = self.sp_h < 0.0
        self.sp_h[neg] = 0.0
        self.sp_hv[neg] = np.maximum(self.sp_hv[neg], 0.0)
        idx = np.flatnonzero(on)
        if len(idx):
            # spikes repel at the wavelength (hexagons emerge) and climb the field gradient
            q = ferro_base(n[idx], R, sh["sc"], sh["tear"], sh["k"], sh["lobe"], sh["phase"])
            D = q[:, None, :] - q[None, :, :]
            dist = np.linalg.norm(D, axis=2) + np.eye(len(idx)) * 1e3
            ov = np.clip(lam - dist, 0.0, None)
            F = ((ov / dist)[:, :, None] * D).sum(1) / max(lam, 1e-6)
            t1, t2 = frames(n[idx])
            e = 0.05
            nn = n[idx]
            gx = self._g(_nrm(nn + e * t1)) - self._g(_nrm(nn - e * t1))
            gy = self._g(_nrm(nn + e * t2)) - self._g(_nrm(nn - e * t2))
            grad = (gx[:, None] * t1 + gy[:, None] * t2) / (2 * e)
            move = 6.0 * F + 0.6 * grad
            move -= (move * nn).sum(1, keepdims=True) * nn
            n[idx] = _nrm(nn + dt * np.clip(move, -8.0, 8.0))
        # deaths: flattened spikes below the field are reabsorbed
        gone = on & (H <= 0.0) & (self.sp_h < 0.004 * R) & (self.sp_hv <= 0.05)
        self.sp_on[gone] = False
        self.sp_die[gone] = False
        # births: where the field is supercritical and there is room (farthest point first)
        gF = self._g(self.fib)
        act = B * gF > BC
        if act.any():
            area = _ellipsoid_area(R, sh["sc"])
            want = min(FERRO_S, int(round(act.mean() * area / (0.866 * lam * lam))))
            have = int(self.sp_on.sum() - (self.sp_on & self.sp_die).sum())
            if have < want and (~self.sp_on).any():
                cand = self.fib[act]
                qc = ferro_base(cand, R, sh["sc"], sh["tear"], sh["k"], sh["lobe"], sh["phase"])
                live = np.flatnonzero(self.sp_on)
                if len(live):
                    ql = ferro_base(n[live], R, sh["sc"], sh["tear"], sh["k"], sh["lobe"], sh["phase"])
                    dmin = np.linalg.norm(qc[:, None, :] - ql[None, :, :], axis=2).min(1)
                else:
                    dmin = np.full(len(cand), 1e3)
                for _ in range(min(2, want - have)):
                    b = int(np.argmax(dmin))
                    if dmin[b] < 0.8 * lam:
                        break
                    free = np.flatnonzero(~self.sp_on)
                    if not len(free):
                        break
                    k = int(free[0])
                    self.sp_on[k], self.sp_die[k] = True, False
                    self.sp_n[k] = cand[b]
                    self.sp_h[k], self.sp_hv[k] = 0.0, 0.3 * H0
                    dmin = np.minimum(dmin, np.linalg.norm(qc - qc[b], axis=1))
            elif have > want + 2:                                        # too many: the weakest sinks back
                live = np.flatnonzero(self.sp_on & ~self.sp_die)
                self.sp_die[live[np.argmin(Bn[live])]] = True
        else:
            self.sp_die |= self.sp_on
        self.lean = sh["lean"] * 0.35 * min(1.5, max(B / BC - 0.5, 0.0))

    # ------------------------------------------------------------------ droplets
    def _eject(self, count: int, speed: float) -> None:
        on = np.flatnonzero(self.sp_on & (self.sp_h > 0.05 * self.sh["R"]))
        for _ in range(count):
            free = np.flatnonzero(self.sat_V <= 0.0)
            vol = self.V_total * self.rng.uniform(0.015, 0.035)
            if not len(free) or self.V_main - vol < 0.6 * self.V_total:
                return
            k = int(free[0])
            if len(on):
                j = int(on[self.rng.randint(0, len(on) - 1)])
                tip = ferro_base(self.sp_n[j:j + 1], self.sh["R"], self.sh["sc"])[0] + self.sp_n[j] * self.sp_h[j]
                d = self.sp_n[j]
            else:
                d = _nrm(self.nrng.standard_normal((1, 3)))[0]
                tip = ferro_base(d[None, :], self.sh["R"], self.sh["sc"])[0]
            self.sat_x[k] = self.c + self.Rb @ tip
            self.sat_v[k] = self.u + self.Rb @ d * speed * self.rng.uniform(0.7, 1.3)
            self.sat_V[k], self.sat_t[k] = vol, 0.0
            self.V_main -= vol
        self._log("SPLIT", None)

    def _satellites(self, dt: float) -> None:
        on = self.sat_V > 0.0
        if not on.any():
            return
        R = self.sh["R"]
        d = self.c - self.sat_x
        dist = np.linalg.norm(d, axis=1)
        a = 9.0 * d - 0.8 * (self.sat_v - self.u)                             # pulled back by the field gradient
        self.sat_v[on] += a[on] * dt
        self.sat_x[on] += self.sat_v[on] * dt
        self.sat_t[on] += dt
        back = on & (dist < 1.05 * R) & (self.sat_t > 0.5)
        if back.any():                                                         # coalescence: volume comes home
            self.V_main += float(self.sat_V[back].sum())
            self.u += (self.sat_V[back][:, None] * (self.sat_v[back] - self.u)).sum(0) / self.V_main
            self.sat_V[back] = 0.0
            self.glow = max(self.glow, 0.5)

    # ------------------------------------------------------------------ physics
    def _simulate(self, dt: float) -> None:
        pr = self.params.values()
        a = self._flight(dt, self.c, self.u)
        self.glove.begin(dt)
        self.Rb = self._R() @ self.glove.G
        wind = self.glove.wind_acc(np.vstack([self.c, self.sat_x]), self.c, self.Rb)
        if not isinstance(wind, float):
            a = a + wind[0]
            self.sat_v += wind[1:] * dt
        self.u += a * dt
        self.c = self.c + self.u * dt
        self._field(dt)
        self.sh = self._shape()
        self._spikes(dt)
        self._satellites(dt)
        w_lab = self.weights()[self.REGIMES.index("LABYRINTH")]
        if w_lab > 0.02 and int(self.t * 120) % 2 == 0:
            md = self.Rb.T @ self.mag_dir
            f = np.array([md[0], md[1]])
            f = f / max(float(np.linalg.norm(f)), 1e-6)
            self.lab.step(0.5 * (0.5 + pr["fluidity"]), 0.25 + 0.3 * min(self.B / BC, 2.0) * w_lab,
                          0.35 * (1.0 - w_lab), (f[0], f[1]), 0.5)
        self.mag_pos, self.mag_dir = self._magnet()
        if self.glove.ctrl.active and self.glove.ctrl.freeze > 1e-3:
            self.sp_hv *= math.exp(-self.glove.ctrl.freeze * 7.0 * dt)
            self.u *= math.exp(-self.glove.ctrl.freeze * 7.0 * dt)
        for ob in self.obstacles:                                               # a splash where it hits
            if ob is None:
                continue
            rel = self.c - ob.pos
            dist = float(np.linalg.norm(rel))
            reach = ob.radius + self.sh["R"] * float(np.max(self.sh["sc"])) * 0.8
            if dist < reach:
                nrm = rel / max(dist, 1e-6)
                self.c = ob.pos + nrm * reach
                vn = float(self.u @ nrm)
                if vn < 0:
                    self.u -= 1.6 * vn * nrm
                self.u += ob.vel * 0.3
                if self.glow < 0.7:
                    self._eject(3, 3.0)
                self.glow = 1.0
                self.kick = 1.0
        self.surface += (min(1.0, 0.25 + 0.5 * min(self.B / BC, 2.0)) - self.surface) * min(1.0, dt * 3.0)

    # ------------------------------------------------------------------ events
    def _on_kick(self, strength: float) -> None:
        H0 = 0.5 * self.sh["R"]
        self.sp_hv[self.sp_on] += 2.5 * H0 * strength                         # the spikes jump
        if strength > 0.85 and self.rng.chance(0.35 + 0.4 * self.weights()[1]):
            self._eject(2, 3.5)                                                # droplets torn off the tips
        if self.weights()[self.REGIMES.index("LABYRINTH")] > 0.2:
            self.lab.stir(0.15 * strength)

    def _apply_event(self, name: str, arg) -> None:
        if name == "SPLIT":
            self._eject(4, 3.5)
            self.glow = 1.0
            return
        if name == "SURGE":
            self.surge = 1.0
            self.calm = 0.0
        elif name == "CALM":
            self.calm = 1.0
            self.surge = 0.0
        else:
            super()._apply_event(name, arg)
            return
        self._log(name)

    def _threat(self, ob) -> None:
        super()._threat(ob)
        self.goal("URCHIN" if self.rng.chance(0.6) else "FIN", 3.0)
        self.surge = max(self.surge, 0.6)

    # ------------------------------------------------------------------ state
    def extra(self) -> np.ndarray:
        sh = self.sh
        R = sh["R"]
        e = np.zeros(FERRO_LEN)
        over = min(max(self.B / BC - 1.0, 0.0), 1.5)
        lam = 0.46 * R * (1.0 - 0.2 * min(over, 1.0))
        e[0], e[1:4], e[4], e[5], e[6], e[7], e[8] = R, sh["sc"], sh["tear"], sh["k"], sh["lobe"], sh["phase"], \
            0.14 * R * sh["ridge"] * (0.6 + 0.4 * min(self.B / BC, 2.0))
        e[9], e[10] = 0.012 * min(1.0, 3.0 * self.inp.high), self.ripple_phase
        e[11] = 1.4 + 2.0 * over
        e[12], e[13] = self.B / BC, self.kick
        e[14], e[15] = int(self.sp_on.sum()), int((self.sat_V > 0).sum())
        e[16:25] = self.Rb.ravel()
        e[25:28], e[28:31], e[31] = self.mag_pos, self.mag_dir, self.B / BC
        H = FERRO_HEAD
        sp = np.zeros((FERRO_S, 8))
        on = self.sp_on
        n = self.sp_n
        nb = _nrm(n / np.asarray(sh["sc"])[None, :])                          # the surface normal of the ellipsoid
        md = self.Rb.T @ self.mag_dir
        ft = md[None, :] - (nb @ md)[:, None] * nb
        ax = _nrm(nb + self.lean * ft)
        sp[:, 0:3] = n
        sp[:, 3] = np.where(on, self.sp_h, 0.0)
        sp[:, 4] = 0.5 * lam / R
        sp[:, 5:8] = ax
        e[H:H + 8 * FERRO_S] = sp.ravel()
        o = H + 8 * FERRO_S
        sat = np.zeros((FERRO_SAT, 4))
        live = self.sat_V > 0
        sat[:, 0:3] = np.where(live[:, None], self.sat_x, self.c)
        sat[:, 3] = np.where(live, (3.0 * self.sat_V / (4.0 * math.pi)) ** (1.0 / 3.0), 0.0)
        e[o:o + 4 * FERRO_SAT] = sat.ravel()
        o += 4 * FERRO_SAT
        u = self.lab.u
        e[o:o + FERRO_GRID * FERRO_GRID] = np.clip(u / 0.8, -1.0, 1.0).ravel()
        return e

    def state(self):
        live = self.sat_V > 0
        pos = np.vstack([self.c, np.where(live[:, None], self.sat_x, self.c)])
        radius = np.concatenate([[self.sh["R"]], np.where(live, (3.0 * self.sat_V / (4.0 * math.pi)) ** (1 / 3), 0.0)])
        reach = self.sh["R"] * float(np.mean(self.sh["sc"])) + 0.6 * float(self.sp_h.max(initial=0.0))
        return self._state(pos, radius, np.zeros((0, 6)), self.extra(),
                           {"total": self.V_total, "main": self.V_main, "spikes": int(self.sp_on.sum())}, "LIQUID",
                           com=self.c, rms=reach)


def _nrm(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _ellipsoid_area(R: float, sc) -> float:
    a, b, c = (R * np.asarray(sc, float)).tolist()
    p = 1.6075
    return 4 * math.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0) ** (1.0 / p)


__all__ = ["FerroEngine", "FerroConfig", "Labyrinth", "BC", "REGIMES"]
