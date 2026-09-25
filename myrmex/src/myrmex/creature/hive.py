"""Polyalloy Hive (creature v4): a two-scale material that builds, patterns itself and remembers.

Everything the Polyalloy Colony (v3) does - flock, armour waves, mechanisms, prey - plus:

* **Nanomachine swarm.**  1536 micro-machines ride on the structural nodes, circulating over the surface
  like conveyor belts.  Where the material loosens, is hit or disperses they come off and fly as smoke-like
  streams in a swirling flow, then home back and re-attach.  While the colony is split, couriers stream
  between the bodies: living bridges of material.  (knob *nanoswarm*)
* **Emergent morphogenesis.**  A reaction-diffusion system (Gray-Scott) runs on the elastic network; its
  activator peaks push the surface out into spines and fins that migrate, split and fade - forms that no
  template contains.  Hits seed new peaks where they land; the music tunes the chemistry.  (*pattern*)
* **Living architecture.**  Like army ants building bridges from their own bodies, it leaves part of its
  material behind as a pillar, an arch or a ring gate on its path, flies around and through it, and later
  calls the material back in a stream.  (*architecture*)
* **Phrase memory.**  It fingerprints every 4-bar phrase and remembers the form it took; when a passage
  returns it returns to that form, with variation: the performance gets a shape.  (*memory*)

Procedural and deterministic (seeded); a hand-written controller, no learning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .colony import EVENTS as COLONY_EVENTS
from .colony import G, Body, ColonyConfig, ColonyEngine, ColonyState
from .polyalloy import MATERIAL

STRUCTURES = ("PILLAR", "ARCH", "RING")
EVENTS = COLONY_EVENTS + ("BUILD", "RECALL")


@dataclass
class HiveConfig(ColonyConfig):
    particles: int = 1536
    max_structures: int = 2


@dataclass
class HiveState(ColonyState):
    particles: np.ndarray | None = None     # (Np, 3) nanomachine positions
    rd: np.ndarray | None = None            # (N,) activator of the reaction-diffusion field
    structures: int = 0
    memories: int = 0


def structure_shape(kind: str, U: np.ndarray, s: float) -> np.ndarray:
    """Local shape of a built structure (x along the path, z up; the anchor is its base / centre)."""
    u, v, w = U[:, 0], U[:, 1], U[:, 2]
    if kind == "PILLAR":                                   # a twisted column on the ground
        h = 3.2 * s
        r = 0.22 * s * (1.0 + 0.6 * np.exp(-((u - 0.03) / 0.08) ** 2) + 0.4 * np.exp(-((u - 0.97) / 0.06) ** 2))
        a = 2 * np.pi * v + 2.5 * u
        return np.stack([r * np.cos(a), r * np.sin(a), u * h], 1)
    if kind == "ARCH":                                     # a parabolic arch across the path
        span, h = 3.4 * s, 2.4 * s
        y = (u - 0.5) * span
        z = h * (1.0 - (2 * u - 1) ** 2)
        a = 2 * np.pi * v
        rr = 0.16 * s * np.sqrt(w)
        return np.stack([rr * np.cos(a), y + rr * np.sin(a) * 0.3, z + rr * np.sin(a)], 1)
    R, tube = 1.25 * s, 0.13 * s * np.sqrt(w)              # RING: a gate facing the path
    a, b = 2 * np.pi * u, 2 * np.pi * v
    return np.stack([tube * np.sin(b), (R + tube * np.cos(b)) * np.cos(a), (R + tube * np.cos(b)) * np.sin(a)], 1)


class HiveEngine(ColonyEngine):
    EVENTS = EVENTS

    def __init__(self, cfg: HiveConfig | None = None):
        cfg = cfg or HiveConfig()
        super().__init__(cfg)
        self.n_fly = cfg.max_bodies
        for _ in range(cfg.max_structures):                  # structure slots after the flying bodies
            self.bodies.append(Body(0.0))
        self.kind_of: dict[int, str] = {}
        self.anchor: dict[int, np.ndarray] = {}
        self.born_at: dict[int, float] = {}
        self.last_build = -1e9
        self.build_req: tuple[str | None, float] | None = None
        n, Np = self.n, cfg.particles
        r = self.rs.stream("hive-swarm")
        self.p_host = np.array([r.randint(0, n - 1) for _ in range(Np)])
        d = self.nrng.standard_normal((Np, 3))
        d[:, 0] = np.abs(d[:, 0]) + 0.6                       # mostly on the outside
        self.p_dir = d / np.linalg.norm(d, axis=1, keepdims=True)
        self.p_phase = self.nrng.uniform(0, 2 * np.pi, Np)
        self.p_x = self.x[self.p_host] + self.nrng.standard_normal((Np, 3)) * 0.2
        self.p_v = np.zeros((Np, 3))
        self.p_bound = np.ones(Np, bool)
        self.p_free_t = np.zeros(Np)
        self.p_courier = np.zeros(Np, bool)
        self.rd_u = np.ones(n)
        self.rd_v = np.zeros(n)
        self.rd_v[self.nrng.choice(n, 6, replace=False)] = 0.5
        self.memory: list[tuple[np.ndarray, np.ndarray, str]] = []
        self.phrase = None
        self.acc = np.zeros(6)
        self.acc_n = 0
        self._r = np.full(n, 0.3)
        self._nrm = np.zeros((n, 3))
        self._nrm[:, 2] = 1.0

    # ------------------------------------------------------------------ bodies: flying vs structures
    def _alive(self) -> list[int]:
        return [k for k in range(self.n_fly) if self.bodies[k].alive]

    def _structures(self) -> list[int]:
        return [k for k in range(self.n_fly, len(self.bodies)) if self.bodies[k].alive]

    def _physics_bodies(self) -> list[int]:
        return self._alive() + self._structures()

    def _is_static(self, k: int) -> bool:
        return k >= self.n_fly

    def _static_step(self, k: int, idx: np.ndarray, dt: float):
        b = self.bodies[k]
        b.P, b.vel = self.x[idx].mean(0), self.v[idx].mean(0)
        s = self.cfg.size * 0.8
        loc = structure_shape(self.kind_of[k], self.U[idx], s)
        ch, sh = math.cos(b.heading), math.sin(b.heading)
        R = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1.0]])
        tgt = self.anchor[k] + loc @ R.T
        a = -b.vel / 0.5 + np.array([0.0, 0.0, G])          # hold: no drift, gravity cancelled
        return a, tgt

    # ------------------------------------------------------------------ living architecture
    def _build(self, kind: str | None = None) -> bool:
        free = [k for k in range(self.n_fly, len(self.bodies)) if not self.bodies[k].alive]
        lead = self.bodies[0]
        if not free or len(self._alive()) > 1 or lead.intent in ("ENVELOP", "EVADE"):
            return False
        k = free[0]
        kind = kind if kind in STRUCTURES else STRUCTURES[self.ev_rng.randint(0, len(STRUCTURES) - 1)]
        idx = np.nonzero(self.own == 0)[0]
        m = int(len(idx) * (0.16 + 0.12 * self.params["architecture"]))
        if len(idx) - m < 48:
            return False
        R = lead.R()
        rel = (self.x[idx] - lead.P) @ R
        donors = idx[np.argsort(rel[:, 0])[:m]]               # the tail detaches and flies to the site
        fwd = np.array([math.cos(lead.heading), math.sin(lead.heading), 0.0])
        side = np.array([-fwd[1], fwd[0], 0.0]) * self.ev_rng.uniform(-2.5, 2.5)
        site = lead.P + fwd * self.ev_rng.uniform(6.0, 9.0) + side
        site[2] = 0.0 if kind in ("PILLAR", "ARCH") else max(2.0, lead.P[2])
        b = self.bodies[k]
        b.__init__(lead.heading)
        b.alive, b.intent = True, "STRUCTURE"
        b.mat_goal = np.array(MATERIAL["HIGH_STIFFNESS"])
        self.own[donors] = k
        self.kind_of[k], self.anchor[k], self.born_at[k] = kind, site, self.t
        self._renormalize(k)
        self._renormalize(0)
        e = self.edges
        self.alive &= self.own[e[:, 0]] == self.own[e[:, 1]]
        self.last_build = self.t
        self._set_intent(lead, "PATROL")
        self._log("BUILD", kind)
        return True

    def _recall(self, k: int | None = None) -> None:
        ks = [k] if k is not None else self._structures()
        for kk in ks:
            idx = self.own == kk
            self.own[idx] = 0
            self.bodies[kk].alive = False
            e = self.edges
            self.alive &= ~(idx[e[:, 0]] | idx[e[:, 1]])        # they stream back on their own
        if ks:
            self._renormalize(0)
            self._log("RECALL", len(ks))

    def _renormalize(self, k: int) -> None:
        if k == 0 and len(self._alive()) == 1 and not self._structures():
            self.U = self.U0.copy()
            return
        idx = np.nonzero(self.own == k)[0]
        if len(idx) == 0:
            return
        for c in range(3):
            rank = np.argsort(np.argsort(self.U0[idx, c]))
            self.U[idx, c] = (rank + 0.5) / len(idx)

    def _v_des(self, k, b, lead, dt, pr, sb, base, alt_goal):
        st = self._structures()
        if k == 0 and b.intent == "PATROL" and st:
            kk = st[int(self.t / 9.0) % len(st)]              # visit the structures in turn
            a = self.anchor[kk]
            kind = self.kind_of[kk]
            h = self.bodies[kk].heading
            axis = np.array([math.cos(h), math.sin(h), 0.0])
            if kind == "RING":                                 # fly through the gate, loop, again
                phase = ((self.t - self.born_at[kk]) * 0.25) % 1.0
                goal = a + axis * (8.0 * phase - 4.0) + np.array([0.0, 0.0, 0.0])
            else:                                              # circle around it
                ang = 0.6 * self.t
                r = 3.4 * self.cfg.size / 1.8
                goal = a + np.array([math.cos(ang) * r, math.sin(ang) * r, 0.0])
                goal[2] = max(1.8, a[2] + (2.2 if kind == "PILLAR" else 3.0))
            d = goal - b.P
            dn = float(np.linalg.norm(d))
            return d / max(dn, 1e-6) * min(base * 1.4 + 1.5, 1.8 * dn + 0.5)
        return super()._v_des(k, b, lead, dt, pr, sb, base, alt_goal)

    # ------------------------------------------------------------------ events and behaviour
    def trigger_event(self, name: str, arg=None) -> bool:
        if name.upper() not in EVENTS:
            return False
        self._pending.append((name.upper(), arg))
        return True

    def _apply_event(self, name: str, arg) -> None:
        if name == "BUILD":                                  # waits (up to 8 s) until it can build
            self.build_req = (arg if isinstance(arg, str) else None, self.t + 8.0)
            return
        if name == "RECALL":
            self._recall()
            return
        if name in ("IMPULSE", "WAVE", "OBSTACLE"):            # hits seed new pattern peaks
            seeds = self.nrng.choice(self.n, 3, replace=False)
            self.rd_v[seeds], self.rd_u[seeds] = 0.5, 0.25
        super()._apply_event(name, arg)

    def _think(self, dt: float) -> None:
        super()._think(dt)
        pr, inp, lead = self.params.values(), self.inp, self.bodies[0]
        st = self._structures()
        if self.build_req is not None:
            if self._build(self.build_req[0]) or self.t > self.build_req[1]:
                self.build_req = None
        # Build on calm-to-rising music, call everything back on a drop or after a while.
        if (pr["architecture"] > 0.2 and 0.3 < inp.energy < 0.82 and self.t - self.last_build > 18.0 and
                len(self._alive()) == 1 and lead.intent in ("CRUISE", "EXPLORE", "DISPLAY", "HOVER") and
                self.ev_rng.chance(dt * 0.25 * pr["architecture"])):
            self._build()
        for k in st:
            if (inp.energy > 0.86 and self.t - self.born_at[k] > 6.0) or self.t - self.born_at[k] > 26.0 + 20.0 * \
                    pr["architecture"]:
                self._recall(k)
        if lead.intent == "PATROL" and (not self._structures() or lead.intent_t > 16.0):
            self._set_intent(lead, "CRUISE")
        self._phrase_memory(inp, pr)

    def _phrase_memory(self, inp, pr) -> None:
        """Every 4 bars: recognise the passage (fingerprint) and return to the form it had, or learn it."""
        self.acc += (inp.bass, inp.mid, inp.high, inp.energy, inp.transient, inp.spectral_flux)
        self.acc_n += 1
        ph = int(inp.beat // 16.0) if inp.playing else self.phrase
        if ph == self.phrase:
            return
        prev, self.phrase = self.phrase, ph
        if prev is None or self.acc_n < 10:
            self.acc, self.acc_n = np.zeros(6), 0
            return
        fp = self.acc / self.acc_n
        fp = fp / max(float(np.linalg.norm(fp)), 1e-6)
        self.acc, self.acc_n = np.zeros(6), 0
        lead = self.bodies[0]
        if self.memory:
            dists = [float(np.linalg.norm(fp - m[0])) for m in self.memory]
            j = int(np.argmin(dists))
            if dists[j] < 0.09 and self.ev_rng.chance(0.3 + 0.7 * pr["memory"]) and \
                    lead.intent in ("CRUISE", "HOVER", "EXPLORE", "DISPLAY"):
                _, z_goal, intent = self.memory[j]
                self._set_intent(lead, intent)
                lead.z_goal = z_goal + self.nrng.standard_normal(len(z_goal)) * 0.25 * pr["mutation"]
                self._log("REMEMBER", intent)
                return
        self.memory.append((fp, lead.z_goal.copy(), lead.intent if lead.intent in
                            ("CRUISE", "HOVER", "EXPLORE", "DISPLAY") else "CRUISE"))
        self.memory = self.memory[-24:]
        self._log("LEARN", None)

    # ------------------------------------------------------------------ morphogenesis
    def _post_targets(self, T: np.ndarray, dt: float) -> np.ndarray:
        pr, inp = self.params.values(), self.inp
        e, al = self.edges, self.alive
        n = self.n
        P = np.array([b.P for b in self.bodies])
        rel = self.x - P[self.own]
        rn = np.linalg.norm(rel, axis=1)
        self._nrm = rel / np.maximum(rn, 1e-6)[:, None]
        if al.any():
            i, j = e[al, 0], e[al, 1]
            deg = np.bincount(np.concatenate([i, j]), minlength=n).astype(float)
            F = 0.028 + 0.022 * pr["pattern"] * (0.5 + 0.5 * inp.energy)
            kk = 0.057 + 0.006 * (inp.high - 0.5)
            u, v = self.rd_u, self.rd_v
            inv = 1.0 / np.maximum(deg, 1.0)
            for _ in range(3):
                du, dv = u[j] - u[i], v[j] - v[i]
                lu = (np.bincount(i, du, n) - np.bincount(j, du, n)) * inv
                lv = (np.bincount(i, dv, n) - np.bincount(j, dv, n)) * inv
                uvv = u * v * v
                u = np.clip(u + 0.5 * lu - uvv + F * (1 - u), 0, 1)
                v = np.clip(v + 0.25 * lv + uvv - (F + kk) * v, 0, 1)
            if v.max() < 0.05:                                 # the pattern died out: a new seed
                v[int(self.nrng.integers(0, n))] = 0.5
            self.rd_u, self.rd_v = u, v
        ext = np.clip((self.rd_v - 0.12) / 0.3, 0, 1) * pr["pattern"]
        fly = self.own < self.n_fly
        T[fly] += self._nrm[fly] * (ext[fly] * 0.4 * self.cfg.size)[:, None]
        return T

    # ------------------------------------------------------------------ nanomachines
    def _step(self, dt: float) -> None:
        super()._step(dt)
        self._swarm_step(dt)

    def _swarm_step(self, dt: float) -> None:
        pr, inp, cfg = self.params.values(), self.inp, self.cfg
        n, Np = self.n, len(self.p_x)
        x, v, h = self.x, self.v, self.p_host
        nrm = self._nrm
        r0 = 0.62 * cfg.size * (1.0 / n) ** (1 / 3) * 1.9
        disp = np.clip(self.m[:, 5] + 0.5 * self.local, 0, 1)
        self._r = r0 * (1.0 - 0.45 * disp)
        up = np.where(np.abs(nrm[:, 2:3]) > 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 0, 1.0]]))
        e1 = np.cross(nrm, up)
        e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-6)
        e2 = np.cross(nrm, e1)
        swirl = self.p_phase + self.t * (0.4 + 1.6 * inp.energy)       # circulation over the surface
        d = self.p_dir
        c, s = np.cos(swirl), np.sin(swirl)
        d1, d2 = d[:, 1] * c - d[:, 2] * s, d[:, 1] * s + d[:, 2] * c
        tgt = x[h] + (nrm[h] * d[:, 0:1] + e1[h] * d1[:, None] + e2[h] * d2[:, None]) * (0.95 * self._r[h])[:, None]
        b = self.p_bound
        px, pv = self.p_x, self.p_v
        # bound: stiff spring to the surface point
        pv[b] += (400.0 * (tgt[b] - px[b]) - 40.0 * (pv[b] - v[h[b]])) * dt
        # detach where the material loosens, is hit or scattered
        ns = pr["nanoswarm"]
        rate = ns * (3.0 * disp[h] + 2.5 * self.local[h] + (4.0 if self.t < self.turb_until else 0.0)) + 0.02
        go = b & (self.nrng.random(Np) < rate * dt)
        if go.any():
            b[go] = False
            self.p_free_t[go] = 0.0
            pv[go] += nrm[h[go]] * (1.2 + 2.5 * inp.energy) + self.nrng.standard_normal((int(go.sum()), 3)) * 0.8
        # couriers: while the colony is split, material streams between the bodies
        bodies = self._alive()
        if len(bodies) > 1:
            cour = b & (self.nrng.random(Np) < dt * 0.1 * ns)
            if cour.any():
                others = np.nonzero(self.own[h[cour]][:, None] != np.array(bodies)[None, :])
                pick = np.array(bodies)[self.nrng.integers(0, len(bodies), int(cour.sum()))]
                idx = np.nonzero(cour)[0]
                for q, kk in zip(idx, pick):
                    cand = np.nonzero(self.own == kk)[0]
                    if len(cand) and self.own[h[q]] != kk:
                        h[q] = cand[int(self.nrng.integers(0, len(cand)))]
                        b[q], self.p_courier[q], self.p_free_t[q] = False, True, 0.0
                del others
        # free: swirl flow, drag, then home back to the host once the material there holds together
        f = ~b
        if f.any():
            q = px[f]
            tt = self.t
            flow = np.stack([np.sin(1.3 * q[:, 1] + 0.9 * tt) + np.cos(1.7 * q[:, 2] - 0.6 * tt),
                             np.sin(1.1 * q[:, 2] + 0.7 * tt) + np.cos(1.5 * q[:, 0] + 0.8 * tt),
                             0.6 * (np.sin(1.2 * q[:, 0] - 0.5 * tt) + np.cos(1.4 * q[:, 1] + 0.4 * tt))], 1)
            self.p_free_t[f] += dt
            coh = self.m[h[f], 0]
            # home once the flight is over and the material there holds together (couriers go straight)
            home = ((self.p_free_t[f] > 0.4 + 1.4 * ns) & (coh > 0.5) & (disp[h[f]] < 0.5)) | self.p_courier[f]
            k_home = np.where(self.p_courier[f], 9.0, 6.0) * home
            swirl_k = np.where(home, 0.25, 1.0)[:, None] * (1.0 + 2.5 * inp.energy) * (0.3 + ns)
            # homing matches the host's velocity (a moving body can still catch its machines back)
            ref = np.where(home[:, None], v[h[f]], 0.0)
            acc = flow * swirl_k + (ref - pv[f]) * np.where(home, 3.0, 0.9)[:, None] + (tgt[f] - q) * k_home[:, None]
            pv[f] += acc * dt
            back = np.linalg.norm(tgt[f] - q, axis=1) < 0.18 * cfg.size
            fi = np.nonzero(f)[0][back & home]
            b[fi] = True
            self.p_courier[fi] = False
        px += pv * dt
        com = x.mean(0)
        far = np.linalg.norm(px - com, axis=1) > 30.0          # keep the swarm around the organism
        if far.any():
            px[far] = tgt[far]
            pv[far] = 0.0
            b[far] = True
        low = px[:, 2] < 0.02
        px[low, 2] = 0.02
        pv[low, 2] = np.abs(pv[low, 2]) * 0.3

    # ------------------------------------------------------------------ state
    def state(self) -> HiveState:
        s = super().state()
        d = s.__dict__.copy()
        return HiveState(**d, particles=self.p_x.copy(), rd=self.rd_v.copy(), structures=len(self._structures()),
                         memories=len(self.memory))


__all__ = ["HiveEngine", "HiveConfig", "HiveState", "STRUCTURES", "EVENTS"]
