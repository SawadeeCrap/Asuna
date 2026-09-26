"""Arbor (creature v16): a vascular organism that grows its own form and takes it back.

There is no body plan.  The organism is a branching network of vessels grown by *space colonisation*
(Runions et al.): a cloud of attraction points fills the envelope of the current form, every point
pulls on the nearest vessel tip within reach, tips sprout towards the points that pull them, points
that are reached are consumed.  The regimes are only envelopes - the tree itself is emergent:

    SPHERE  a radial star, like a neuron or a dandelion     FAN    a flat gorgonian fan carried like a sail
    SPIRAL  three growing spiral arms                       HALO   spokes carrying a ring (a wheel of vessels)
    CROWN   a cup opening upwards, like a coral             COMET  long vessels streaming behind

The material is finite: thicknesses follow the pipe model (r_parent^2.5 = sum r_child^2.5, between
da Vinci's 2 and Murray's 3) and are scaled so the total vessel volume stays at a fixed budget - a
small tree is thick and stubby, a big one a fine filigree.  In silence the envelope shrinks and
branches that no longer feed retract into their parents (they are *pruned*); music makes it grow.

Physics: each vessel holds its rest direction relative to its parent's (bending propagates down the
branch), with a stiffness growing with r^2 - thick trunks hold, thin tips sag, stream back in flight
and tremble with the highs.  Heartbeats send pressure pulses from the root to the tips: the lumen
dilates and lights up as the pulse passes, and the pressure straightens the branch it runs through
(a hydraulic skeleton - the way a spider extends its legs).  A branch that is hit breaks off and falls
dissolving; the tree grows back into the gap.

Events: SPROUT (a burst of growth), SHED (drop a branch), PULSE (a strong heartbeat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...util.rng import stable_hash64
from .base import BionicConfig, BionicEngine

NMAX = 260                      # vessel nodes (slots)
N_ATTR = 240                    # attraction points
GAMMA = 2.5                     # pipe-model exponent


def _rot_apply(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise: rotate w by the minimal rotation taking unit a to unit b."""
    v = np.cross(a, b)
    c = (a * b).sum(1, keepdims=True)
    return w * c + np.cross(v, w) + v * ((v * w).sum(1, keepdims=True) / np.maximum(1.0 + c, 0.05))


def _unit(d: np.ndarray) -> np.ndarray:
    return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)


@dataclass
class ArborConfig(BionicConfig):
    max_nodes: int = NMAX
    attractors: int = N_ATTR
    cruise: float = 2.2
    sim_rate: float = 120.0


class ArborEngine(BionicEngine):
    VARIANT = "arbor"
    KIND = 2
    REGIMES = ("SPHERE", "FAN", "SPIRAL", "HALO", "CROWN", "COMET")
    PLAN = {"CRUISE": (("COMET", "SPIRAL", "FAN"), 1.0), "HOVER": (("SPHERE", "CROWN", "HALO"), 0.12),
            "EXPLORE": (("FAN", "COMET", "SPIRAL"), 0.6), "DISPLAY": (("CROWN", "HALO", "SPHERE"), 0.3),
            "EVADE": (("COMET",), 1.5), "REFORM": (("SPHERE",), 0.4), "STRIKE": (("COMET",), 1.7)}
    EVENTS = BionicEngine.EVENTS + ("SPROUT", "SHED", "PULSE")
    TROPISM = {"CROWN": (0.0, 0.0, 0.3), "COMET": (-0.35, 0.0, 0.0), "FAN": (0.0, 0.0, 0.08)}

    def __init__(self, cfg: BionicConfig | None = None):
        super().__init__(cfg or ArborConfig())
        cfg = self.cfg
        self.s = s = cfg.size / 1.8
        self.N = N = getattr(cfg, "max_nodes", NMAX)
        self.NA = getattr(cfg, "attractors", N_ATTR)
        self.grng = np.random.default_rng(stable_hash64("arbor-growth", cfg.seed) & 0xFFFFFFFF)
        self.seg, self.d_inf, self.d_kill, self.d_feed = 0.13 * s, 0.9 * s, 0.22 * s, 0.3 * s
        self.r_tip, self.r_ref = 0.011 * s, 0.04 * s
        self.alive = np.zeros(N, bool)
        self.parent = np.full(N, -1)
        self.dirn = np.zeros((N, 3))                   # rest direction from the parent (body frame)
        self.L = np.zeros(N)                           # segment length now (sprouts extend, retracting shrink)
        self.L_goal = np.zeros(N)
        self.r = np.zeros(N)
        self.r_goal = np.zeros(N)
        self.vigor = np.zeros(N)
        self.age = np.zeros(N)
        self.retract = np.zeros(N, bool)
        self.shed = np.zeros(N, bool)                  # dropped branches: falling and dissolving
        self.cut = np.zeros(N, bool)                   # the node where a dropped branch broke off
        self.dying = np.zeros(N)
        self.glow_n = np.zeros(N)
        self.alive[0] = True                           # the heart
        self.Anc = np.zeros((N, N))
        self.levels: list[np.ndarray] = []
        self.depth = np.zeros(N, int)
        self.dist = np.zeros(N)
        self.env_e = 0.35                              # smoothed energy -> envelope size
        self.boost = 0.0                               # SPROUT: the envelope swells for a while
        self.env = self._env_scale()
        self.spin = 0.0
        self.pulses: list[list[float]] = []
        self.last_beat = 0
        self.heart_t = 0.0
        self.grow_t = 0.0
        self.V_used = 0.0
        self.V_max = 0.06 * s ** 3 * (0.6 + 0.8 * self.params["density"])
        self.A = self._sample(self.NA)
        self._topology()
        for _ in range(70):                            # born grown
            self._grow(10, instant=True)
        self.r = self.r_goal.copy()
        self.age[:] = 5.0
        self.pos0 = self.Anc @ (self.dirn * self.L[:, None])
        self.x = self.P + self.pos0 @ self._R().T
        self.v = np.zeros((N, 3))
        self.lay = self.pos0.copy()
        for _ in range(60):
            self._physics(1 / 120, np.zeros(3), self._R())

    # ------------------------------------------------------------------ the envelope
    def _env_scale(self) -> float:
        return 0.5 + 0.65 * self.env_e + 0.3 * (self.params["expansion"] - 0.5) + 0.4 * self.boost

    def _sample(self, n: int) -> np.ndarray:
        """n attraction points from the current blend of envelopes (body frame, x forward, z up)."""
        g, w = self.grng, self.weights()
        regs = g.choice(len(self.REGIMES), size=n, p=w / w.sum())
        out = np.zeros((n, 3))
        E = self.env * self.s
        for q, name in enumerate(self.REGIMES):
            m = regs == q
            k = int(m.sum())
            if not k:
                continue
            u, nr = g.random((k, 3)), g.standard_normal((k, 3))
            if name == "SPHERE":
                pts = _unit(nr) * (E * (0.3 + 0.75 * u[:, :1] ** (1 / 3)))
            elif name == "FAN":
                rho = E * 1.25 * (0.2 + 0.8 * np.sqrt(u[:, 0]))
                phi = np.radians(-30.0 + 240.0 * u[:, 1])
                pts = np.stack([rho * np.cos(phi), 0.05 * E * nr[:, 0], rho * np.sin(phi) + 0.1 * E], 1)   # a sail
            elif name == "SPIRAL":
                th = 3.0 * math.pi * u[:, 0]
                rr = E * 0.22 * np.exp(0.18 * th)
                ang = th + (u[:, 2] * 3).astype(int) * 2 * math.pi / 3 + self.spin
                pts = np.stack([rr * np.cos(ang), rr * np.sin(ang), 0.3 * rr * nr[:, 2] * 0.15], 1) + \
                    0.06 * E * nr * np.array([1.0, 1.0, 0.4])
            elif name == "HALO":
                phi = 2 * math.pi * u[:, 0]
                spoke = u[:, 1] < 0.25                                       # a quarter lies on six spokes
                phi = np.where(spoke, np.round(phi / (math.pi / 3)) * (math.pi / 3), phi)
                rad = np.where(spoke, E * (0.15 + 0.8 * u[:, 2]), E * (1.0 + 0.08 * nr[:, 1]))
                pts = np.stack([0.08 * E * nr[:, 0], rad * np.cos(phi), rad * np.sin(phi)], 1)
            elif name == "CROWN":
                rho = 1.05 * E * np.sqrt(u[:, 0])
                phi = 2 * math.pi * u[:, 1]
                z = 0.25 * E + 0.9 * E * (rho / (1.05 * E)) ** 2
                pts = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], 1) + 0.07 * E * nr
            else:                                                            # COMET
                xx = -E * (0.1 + 2.1 * u[:, 0])
                rr = E * (0.1 + 0.32 * (-xx / E) ** 0.8) * np.sqrt(u[:, 1])
                phi = 2 * math.pi * u[:, 2]
                pts = np.stack([xx, rr * np.cos(phi), rr * np.sin(phi)], 1)
            out[m] = pts
        return out

    # ------------------------------------------------------------------ growth
    def _topology(self) -> None:
        """Ancestry, depth levels, path distance and the pipe-model radii (after the tree changed)."""
        N, al = self.N, self.alive
        par = self.parent
        depth = np.zeros(N, int)
        nodes = np.flatnonzero(al)
        # depth by repeated parent hops (trees are shallow: < 40 levels)
        cur = np.where(al, par, -1)
        for _ in range(64):
            has = cur >= 0
            if not has.any():
                break
            depth[has] += 1
            cur = np.where(has, par[np.maximum(cur, 0)], -1)
        self.depth = depth
        self.levels = [np.flatnonzero(al & (depth == d)) for d in range(1, int(depth[nodes].max(initial=0)) + 1)]
        A = np.zeros((N, N))
        A[0, 0] = 1.0
        for lev in self.levels:
            A[lev] = A[par[lev]]
            A[lev, lev] = 1.0
        self.Anc = A
        # pipe model: tips r_tip, every parent carries its children (attached vessels only)
        att = al & ~self.shed
        kids = np.bincount(par[att & (par >= 0)], minlength=N)
        leaf = att & (kids == 0)
        s = np.where(leaf, self.r_tip ** GAMMA, 0.0)
        for lev in reversed(self.levels):
            lv = lev[att[lev] & ~self.cut[lev]]
            np.add.at(s, par[lv], s[lv])
        rp = s ** (1.0 / GAMMA)
        V = float((math.pi * rp ** 2 * self.L_goal)[att].sum())
        self.V_max = 0.06 * self.s ** 3 * (0.6 + 0.8 * self.params["density"])
        f = math.sqrt(self.V_max / V) if V > 1e-9 else 1.0
        self.r_goal = np.where(att, np.clip(rp * f, 0.005 * self.s, 0.11 * self.s), 0.0)
        self.r_goal[0] = max(self.r_goal[0], 0.06 * self.s)                  # the heart
        self.V_used = float((math.pi * self.r_goal ** 2 * self.L_goal)[att].sum())
        self.leaf = leaf
        self.dist = self.Anc @ self.L_goal

    def _grow(self, budget: int = 8, instant: bool = False) -> None:
        pr, N = self.params.values(), self.N
        att = self.alive & ~self.shed & ~self.retract
        nodes = np.flatnonzero(att)
        if len(nodes) == 0:
            return
        pos = (self.Anc @ (self.dirn * self.L_goal[:, None]))[nodes]
        A = self.A
        D = np.linalg.norm(A[:, None, :] - pos[None, :, :], axis=2)
        near = D.argmin(1)
        dmin = D[np.arange(len(A)), near]
        infl = dmin < self.d_inf
        # vigor: attraction points close enough to feed each subtree (a tip outside the form starves)
        served = np.zeros(N)
        feed = dmin < self.d_feed * min(1.0, self.env)                      # a smaller form feeds less far
        np.add.at(served, nodes[near[feed]], 1.0)
        sub = self.Anc.T @ served
        self.vigor = np.where(att, 0.8 * self.vigor + 0.2 * sub, 0.0)
        # growth: every influenced tip sprouts towards the mean direction of its points
        pull = infl & (dmin > self.d_kill)
        free = np.flatnonzero(~self.alive)
        if len(free) and not pull.any():                                   # nothing in reach: reach out
            a = int(np.argmin(dmin))
            pull = np.zeros(len(A), bool)
            pull[a] = True
        if len(free) and pull.any():
            vec = _unit(A[pull] - pos[near[pull]])
            acc = np.zeros((len(nodes), 3))
            np.add.at(acc, near[pull], vec)
            cnt = np.bincount(near[pull], minlength=len(nodes))
            cand = np.flatnonzero(cnt > 0)
            cand = cand[np.argsort(-cnt[cand])][:min(budget, len(free))]
            w = self.weights()
            trop = sum(w[self.REGIMES.index(k)] * np.array(v) for k, v in self.TROPISM.items())
            new = []
            for c in cand:
                d = _unit(acc[c] / cnt[c] + trop + self.grng.standard_normal(3) * (0.1 + 0.25 * pr["mutation"]))
                p_node = nodes[c]
                if p_node != 0 and float(d @ self.dirn[p_node]) < -0.2:        # no kinks back
                    continue
                q = pos[c] + d * self.seg
                if len(pos) and float(np.min(np.linalg.norm(pos - q, axis=1))) < 0.6 * self.seg:
                    continue
                if new and min(float(np.linalg.norm(q - z)) for z in new) < 0.6 * self.seg:
                    continue
                k = int(free[len(new)])
                new.append(q)
                self.alive[k], self.parent[k], self.dirn[k] = True, p_node, d
                self.L_goal[k] = self.seg
                self.L[k] = self.seg if instant else 0.15 * self.seg
                self.vigor[k], self.age[k], self.r[k] = 1.0, 0.0, self.r_tip * 0.5
                self.retract[k] = self.shed[k] = self.cut[k] = False
                self.dying[k] = 0.0
                if len(new) >= len(free):
                    break
        # consumed points and a slow turnover move to fresh places in the envelope
        redo = (dmin < self.d_kill) | (self.grng.random(len(A)) < 0.08)
        if redo.any():
            A[redo] = self._sample(int(redo.sum()))
        # pruning: starving tips retract into their parents (faster in silence)
        self._topology()
        thr = 0.15 * (1.3 - self.env_e) * (0.7 + 0.6 * (1.0 - pr["density"]))
        starving = self.leaf & (self.vigor < thr) & (self.age > 1.5) & ~self.retract
        starving[0] = False
        if instant:
            self.alive[starving] = False
            self._topology()
        else:
            self.retract |= starving

    def _free(self, idx: np.ndarray) -> None:
        self.alive[idx] = False
        self.retract[idx] = self.shed[idx] = self.cut[idx] = False
        self.parent[idx] = -1
        self.L[idx] = self.L_goal[idx] = self.r[idx] = 0.0

    def _shed(self, k: int) -> None:
        """Drop the branch at node k (autotomy): it falls, dissolving; the tree grows back into the gap."""
        if k <= 0 or not self.alive[k] or self.shed[k]:
            return
        sub = (self.Anc[:, k] > 0) & self.alive
        self.shed[sub] = True
        self.cut[k] = True
        self.dying[sub] = 0.0
        self.glow_n[sub] = 1.0
        self._topology()
        self._log("SHED", None)

    # ------------------------------------------------------------------ physics
    def _physics(self, dt: float, a_fl: np.ndarray, R: np.ndarray) -> None:
        pr = self.params.values()
        x, v = self.x, self.v
        al = self.alive
        idx = np.flatnonzero(al)
        idx = idx[idx != 0]
        if len(idx) == 0:
            v[0] += a_fl * dt
            x[0] += v[0] * dt
            return
        x_prev = x.copy()
        p = self.parent[idx]
        g = self.parent[np.maximum(p, 0)]
        off = (self.lay[idx] - self.lay[p]) @ R.T
        hasg = (p > 0) & (g >= 0) & ~self.cut[p]
        w = off.copy()
        if hasg.any():                                                       # bending propagates down the branch
            a = _unit((self.lay[p[hasg]] - self.lay[g[hasg]]) @ R.T)
            b = _unit(x[p[hasg]] - x[g[hasg]])
            w[hasg] = _rot_apply(a, b, off[hasg])
        T = x[p] + w
        rn = np.clip(self.r[idx] / self.r_ref, 0.3, 2.0)
        glow = self.glow_n[idx]
        stiff = np.clip(rn ** 2, 0.3, 1.5) * (1.0 + (1.0 + 2.0 * pr["mechanism"]) * glow)   # hydraulic stiffening
        k = (60.0 + 240.0 * pr["rigidity"]) * stiff
        sh = self.shed[idx]
        k = np.where(sh, 0.3 * k, k)
        k = np.where(self.cut[idx], 0.0, k)
        c = 2.0 * (0.35 + 0.4 * pr["coherence"]) * np.sqrt(k)
        acc = k[:, None] * (T - x[idx]) - c[:, None] * (v[idx] - v[p]) + np.where(sh[:, None], 0.0, a_fl)
        acc[:, 2] -= np.where(sh, 4.5, 2.5 * (1.2 - pr["rigidity"]) * (1.0 - np.clip(stiff / 1.5, 0.0, 1.0)))
        acc -= (0.25 / rn)[:, None] * v[idx]                                  # air: thin vessels stream back
        hi = self.inp.high
        if hi > 0.05:                                                         # tips tremble with the highs
            acc += self.nrng.standard_normal((len(idx), 3)) * (hi * 12.0 * (1.0 / rn - 0.4)).clip(0.0)[:, None]
        v[idx] += acc * dt
        v[0] += a_fl * dt
        x[al] += v[al] * dt
        for lev in self.levels:                                              # vessels do not stretch
            lv = lev[~self.cut[lev]]
            if len(lv) == 0:
                continue
            pp = self.parent[lv]
            d = x[lv] - x[pp]
            n = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            x[lv] = x[pp] + d * (np.maximum(self.L[lv], 1e-4) / n)[:, None]
        v[al] = (x[al] - x_prev[al]) / dt
        dead = ~al
        x[dead] = x[0]
        v[dead] = 0.0

    def _simulate(self, dt: float) -> None:
        pr, inp = self.params.values(), self.inp
        e = inp.energy if inp.playing else 0.0
        self.env_e += (max(e, 0.3 * self.arousal) - self.env_e) * min(1.0, dt / 2.0)
        self.boost *= math.exp(-dt / 3.0)
        self.env = self._env_scale()
        self.spin += dt * 0.08 * (0.5 + pr["mechanism"])
        # growth ticks: faster with energy, kicks and tendril activity
        self.grow_t += dt * (0.4 + 0.8 * self.env_e + 0.6 * pr["tendril_activity"] + 0.6 * self.kick + 2.0 * self.boost)
        if self.grow_t > 0.14:
            self.grow_t = 0.0
            self._grow(8)
        self.age += dt
        sprout = self.alive & (self.L < self.L_goal) & ~self.retract
        self.L[sprout] = np.minimum(self.L_goal[sprout], self.L[sprout] + dt * self.seg / 0.3)
        if self.retract.any():                                               # retracting tips are drawn back in
            rt = self.retract & self.alive
            self.L[rt] -= dt * self.seg / (0.25 + 0.5 * pr["rigidity"])
            gone = rt & (self.L <= 0.03 * self.seg)
            if gone.any():
                self._free(np.flatnonzero(gone))
                self._topology()
        if self.shed.any():
            sh = self.shed & self.alive
            self.dying[sh] += dt / 1.8
            gone = sh & (self.dying >= 1.0)
            if gone.any():
                self._free(np.flatnonzero(gone))
                self._topology()
        self.r += (self.r_goal - self.r) * min(1.0, dt / 0.4)
        # heartbeats: pressure pulses from the root to the tips
        if inp.playing:
            b = int(math.floor(inp.beat))
            if b != self.last_beat:
                self.last_beat = b
                self._pulse(0.55 + 0.4 * inp.bass)
        else:
            self.heart_t += dt
            if self.heart_t > 1.6:
                self.heart_t = 0.0
                self._pulse(0.45)
        c = 2.2 * self.s * (0.7 + 0.6 * pr["speed"])
        wid = 0.1 * self.s
        glow = np.zeros(self.N)
        live = []
        dmax = float(self.dist[self.alive].max(initial=0.1))
        for pl in self.pulses:
            front = c * (self.t - pl[0])
            if front < dmax + 3 * wid:
                live.append(pl)
                glow = np.maximum(glow, pl[1] * np.exp(-((self.dist - front) / wid) ** 2))
        self.pulses = live[-6:]
        fresh = 0.8 * np.exp(-self.age / 0.6)                                 # growth cones glow as they extend
        tgt = np.where(self.shed, self.glow_n * math.exp(-dt / 0.6), np.maximum(glow, fresh))
        self.glow_n = np.where(self.alive, tgt, 0.0)
        # pose, hand, targets
        x, v = self.x, self.v
        P, vcom = x[0].copy(), v[0].copy()
        a_fl = self._flight(dt, P, vcom)
        dR = self.glove.begin(dt)
        R = self._R()
        self.glove.rigid(x, v, P, R @ dR @ R.T)
        R = R @ self.glove.G
        self.pos0 = self.Anc @ (self.dirn * self.L[:, None])
        idx = np.flatnonzero(self.alive)
        self.lay = self.pos0.copy()
        if self.glove.ctrl.active and len(idx) > 2:
            self.lay[idx] = self.glove.local(self.pos0[idx], self.t, inp.beat if inp.playing else 2.0 * self.t)
        wind = self.glove.wind_acc(x, P, R)
        if not isinstance(wind, float):
            v += wind * dt
        self._physics(dt, a_fl, R)
        self.glove.damp(v, dt)
        for ob in self.obstacles:
            if ob is None:
                continue
            hit = self._push_out(x, v, ob) & self.alive
            if hit.any():
                self.glow = max(self.glow, 0.8)
                hk = np.flatnonzero(hit & (self.depth >= 3) & ~self.shed)
                if len(hk) and self.rng.chance(0.5):
                    self._shed(int(hk[np.argmin(self.depth[hk])]))
        self.surface += (min(1.0, 0.3 + 0.7 * float(glow.max(initial=0.0))) - self.surface) * min(1.0, dt * 3.0)

    def _pulse(self, amp: float) -> None:
        self.pulses.append([self.t, min(1.0, amp * (0.6 + 0.8 * self.params["surface_activity"]))])

    # ------------------------------------------------------------------ events
    def _on_kick(self, strength: float) -> None:
        self._pulse(0.6 + 0.4 * strength)
        self.grow_t = 1.0                                                    # a growth tick now

    def _apply_event(self, name: str, arg) -> None:
        if name == "SPROUT":
            self.boost = 1.0
            self.env = self._env_scale()
            self.A[:] = self._sample(self.NA)
            self._grow(12)
            self.glow = 1.0
        elif name == "SHED":
            cand = np.flatnonzero(self.alive & ~self.shed & (self.depth >= 2) & (self.depth <= 5))
            if len(cand):
                size = self.Anc[:, cand].sum(0)
                ok = size <= 0.3 * self.alive.sum()                          # a limb, never the body
                if ok.any():
                    cand, size = cand[ok], size[ok]
                    self._shed(int(cand[np.argmax(size + self.nrng.uniform(0, 4, len(size)))]))
            return
        elif name == "PULSE":
            self._pulse(1.0)
            self.pulses[-1][1] = 1.0
        else:
            super()._apply_event(name, arg)
            return
        self._log(name)

    def _threat(self, ob) -> None:
        super()._threat(ob)
        self.goal("COMET", 3.0)
        self._pulse(0.9)

    # ------------------------------------------------------------------ state
    def state(self):
        al = self.alive
        N = self.N
        dmax = max(float(self.dist[al].max(initial=0.0)), 1e-3)
        r_draw = np.where(al, self.r * (1.0 + 0.35 * self.glow_n) * (1.0 - np.clip(self.dying, 0.0, 1.0)), 0.0)
        M = np.full((N, 6), -1.0)
        m = al.copy()
        m[0] = False
        m &= ~self.cut
        k = np.flatnonzero(m)
        M[k, 0] = self.parent[k]
        M[k, 1] = k
        M[k, 2] = self.shed[k]
        M[k, 3] = r_draw[k]
        M[k, 4] = self.glow_n[k]
        M[k, 5] = np.clip(self.dist[k] / dmax, 0.0, 1.0)
        pos = np.where(al[:, None], self.x, self.x[0])
        n_alive = int(al.sum())
        live = self.x[al]
        com = live.mean(0)
        return self._state(pos, r_draw, M, [n_alive, dmax, self.V_used / max(self.V_max, 1e-9), self.env,
                                            self.kick, len(self.pulses)],
                           {"total": self.V_used, "budget": self.V_max, "nodes": n_alive}, "STRUCTURED", com=com,
                           rms=float(np.sqrt(((live - com) ** 2).sum(1).mean())))


__all__ = ["ArborEngine", "ArborConfig", "NMAX"]
