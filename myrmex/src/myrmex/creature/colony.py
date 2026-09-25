"""Polyalloy Colony (creature v3): the mimetic polyalloy as a colony that can become a flock.

Everything the Mimetic Polyalloy (v2) does, plus:

* **One finite material, up to four autonomous bodies.**  On a drop the organism splits into a
  flock (each body a complete smaller shape, sized by its share of the material): formation flight,
  pincer attacks, flowing past an obstacle on both sides; on a breakdown the bodies fly back and
  merge (their networks re-fuse).
* **Per-node material field.**  Every node has its own cohesion / stiffness / damping / dispersion,
  relaxing towards its body's state, diffusing through the elastic network and hardening where it is
  hit.  Snares (back-beat hits) send hardening waves through the body: armour plates rise in a
  travelling band and sink back; struts appear only where the material is locally hard.
* **Articulated mechanisms** (knob *mechanism*): wings flap with the beat, blade rotors and rings
  spin faster with the energy, tendrils carry travelling waves, the spindle pumps (peristalsis),
  a crown of spikes pulses on kicks, and when it perches it walks on six legs in a tripod gait.
* **Prey** (knob *hunt*): a lure flies around; the colony hunts it (split bodies attack from
  several sides), envelops it in a closed shell, holds it for two bars and lets it go.

Procedural and deterministic (seeded); a hand-written controller, no learning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..util.noise import Noise1D
from ..util.rng import RngStreams, stable_hash64
from .config import DEFAULT_PARAMS
from .control import CreatureControlInput, ParameterSet
from .polyalloy import MATERIAL, Obstacle, PolyalloyState, _knn_edges, attractor_shape

SHAPES = ("CORE", "SPINDLE", "RING", "SHIELD", "BLADES", "LATTICE", "WINGS", "CLOUD", "TENDRILS", "CROWN", "LEGS",
          "ENVELOP")
INTENTS = ("CRUISE", "HOVER", "EXPLORE", "DISPLAY", "EVADE", "REFORM", "HUNT", "ENVELOP", "PERCH", "FORMATION",
           "MERGE", "PATROL", "STRUCTURE")
PLAN = {   # preferred shapes, material state, cruise factor
    "CRUISE": (("SPINDLE", "WINGS", "TENDRILS", "LATTICE"), "ELASTIC", 1.0),
    "HOVER": (("CORE", "RING", "CROWN", "CLOUD"), "COHESIVE", 0.12),
    "EXPLORE": (("CLOUD", "TENDRILS", "SPINDLE", "RING"), "FLUID", 0.6),
    "DISPLAY": (("LATTICE", "BLADES", "CROWN", "RING", "SHIELD"), "STRUCTURED", 0.35),
    "EVADE": (("SPINDLE", "SHIELD", "CLOUD"), "FLUID", 1.0),
    "REFORM": (("CORE", "SPINDLE"), "COHESIVE", 0.6),
    "HUNT": (("SPINDLE", "BLADES", "WINGS"), "ELASTIC", 1.5),
    "ENVELOP": (("ENVELOP",), "COHESIVE", 0.0),
    "PERCH": (("LEGS",), "ELASTIC", 0.25),
    "FORMATION": (("SPINDLE", "WINGS", "BLADES", "TENDRILS"), "ELASTIC", 1.0),
    "MERGE": (("CORE", "CLOUD"), "FLUID", 1.3),
    "PATROL": (("SPINDLE", "WINGS", "TENDRILS"), "ELASTIC", 1.1),       # v4: around / through its structures
    "STRUCTURE": (("CORE",), "HIGH_STIFFNESS", 0.0),
}
EVENTS = ("MORPHOLOGY_SHIFT", "MASS_REBALANCE", "APPENDAGE_BURST", "COLLAPSE", "RECONSTRUCTION", "IMPULSE",
          "OBSTACLE", "PRESSURE", "TURBULENCE", "SPLIT", "MERGE", "WAVE", "HUNT", "PERCH")
KICK_MODES = ("IMPULSE", "OBSTACLE", "PRESSURE", "TURBULENCE", "MIX")
G = 9.81


@dataclass
class ColonyConfig:
    seed: int = 0
    nodes: int = 128
    size: float = 1.8
    sim_rate: float = 120.0
    k_neighbors: int = 5
    max_links: int = 640
    max_obstacles: int = 4
    max_bodies: int = 4
    stage_radius: float = 16.0
    altitude: tuple[float, float] = (1.8, 6.0)
    cruise: float = 3.4
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))


@dataclass
class ColonyState(PolyalloyState):
    plate: np.ndarray | None = None     # (N,) armour plate size 0..1
    nrm: np.ndarray | None = None       # (N, 3) outward direction of every node (plate orientation)
    owner: np.ndarray | None = None     # (N,) body index
    bodies: int = 1
    lure: np.ndarray | None = None      # x, y, z, radius (0 = no prey)


def _rot_x(P: np.ndarray, a) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    out = P.copy()
    out[:, 1] = c * P[:, 1] - s * P[:, 2]
    out[:, 2] = s * P[:, 1] + c * P[:, 2]
    return out


_SPIKES = np.array([[math.cos(2 * math.pi * k / 9) * math.cos(e), math.sin(2 * math.pi * k / 9) * math.cos(e), math.sin(e)]
                    for k, e in zip(range(9), (0.15, 0.6, 0.2, 0.75, 0.1, 0.5, 0.3, 0.9, 0.05))])


def shape3(name: str, U: np.ndarray, s: float, elong: float, ph: dict, ground: float) -> np.ndarray:
    """Local target positions (x forward, z up); the mechanisms move with the music (``ph``)."""
    u, v, w = U[:, 0], U[:, 1], U[:, 2]
    mech = ph["mech"]
    if name == "WINGS":                                     # both wings flap together
        P = attractor_shape("WINGS", U, s, elong)
        return _rot_x(P, np.sign(P[:, 1]) * ph["flap"])
    if name in ("BLADES", "RING"):                          # rotor / spinning ring: material flows around
        return _rot_x(attractor_shape(name, U, s, elong), ph["spin"])
    if name == "SPINDLE":                                   # peristaltic pumping
        P = attractor_shape("SPINDLE", U, s, elong)
        P[:, 1:] *= (1.0 + 0.35 * mech * np.sin(2 * np.pi * (3.0 * u - 0.5 * ph["beat"])))[:, None]
        return P
    if name == "TENDRILS":                                  # six trailing tendrils with travelling waves
        k = np.floor(u * 6.0)
        a0 = 2 * np.pi * k / 6.0 + 0.4
        r = 0.2 * s + 0.3 * s * v
        wave = np.sin(2 * np.pi * (1.4 * v - ph["t"] * (0.5 + 1.3 * mech)) + k) * 0.25 * s * v
        th = (w - 0.5) * 0.05 * s
        y = np.cos(a0) * (r + th) - np.sin(a0) * wave
        z = np.sin(a0) * (r + th) + np.cos(a0) * wave
        return np.stack([-0.2 * s - 1.5 * s * elong * v, y, z], 1)
    if name == "CROWN":                                     # core + nine spikes that pulse on kicks
        core = u < 0.35
        P = attractor_shape("CORE", U, s * 0.85, 1.0)
        d = _SPIKES[np.minimum(8, np.floor(v * 9.0).astype(int))]
        along = 0.22 * s + np.clip((u - 0.35) / 0.65, 0, 1) * (0.5 * s + 0.45 * s * ph["pulse"])
        spike = d * along[:, None] + (w - 0.5)[:, None] * 0.03 * s
        return np.where(core[:, None], P, spike)
    if name == "LEGS":                                      # perched: body + six legs, tripod gait
        body = u < 0.4
        P = attractor_shape("CORE", U, s * 0.75, 1.1)
        k = np.minimum(5, np.floor(v * 6.0))
        side = np.where(k < 3, 1.0, -1.0)
        ax = (k % 3 - 1.0) * 0.55 * s
        phase = ph["gait"] + np.where(k % 2 == 0, 0.0, np.pi)
        lift = np.maximum(0.0, np.sin(phase)) * 0.18 * s
        swing = np.cos(phase) * 0.18 * s
        f = np.clip((u - 0.4) / 0.6, 0, 1)
        hip = np.stack([ax, side * 0.22 * s, np.zeros_like(u)], 1)
        knee = np.stack([ax + 0.5 * swing, side * 0.75 * s, 0.35 * s + lift], 1)
        foot = np.stack([1.3 * ax + swing, side * 1.05 * s, ground + lift], 1)
        t1 = np.clip(f / 0.5, 0, 1)[:, None]
        t2 = np.clip((f - 0.5) / 0.5, 0, 1)[:, None]
        leg = np.where((f < 0.5)[:, None], hip + (knee - hip) * t1, knee + (foot - knee) * t2)
        return np.where(body[:, None], P, leg)
    return attractor_shape(name if name in ("CORE", "SHIELD", "LATTICE", "CLOUD") else "CORE", U, s, elong)


_SITUATIONAL = [SHAPES.index("ENVELOP"), SHAPES.index("LEGS")]


class Body:
    """One autonomous part of the colony (the lead is body 0)."""

    def __init__(self, heading: float = 0.0):
        self.alive = False
        self.P = np.array([0.0, 0.0, 3.0])
        self.vel = np.zeros(3)
        self.heading, self.pitch, self.yaw_rate = heading, 0.0, 0.0
        self.intent, self.intent_t, self.dwell = "CRUISE", 0.0, 6.0
        self.z = np.zeros(len(SHAPES))
        self.z_goal = np.zeros(len(SHAPES))
        self.mat = np.array(MATERIAL["COHESIVE"], float)
        self.mat_goal = self.mat.copy()
        self.wander = 0.0
        self.slot = np.zeros(3)
        self.hunt_offset = np.zeros(3)
        self.response = ""

    def R(self) -> np.ndarray:
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1.0]]) @ np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])

    def weights(self) -> np.ndarray:
        e = np.exp(3.0 * (self.z - self.z.max()))
        return e / e.sum()

    def goal_shape(self, name: str, strength: float = 2.5) -> None:
        self.z_goal = np.zeros(len(SHAPES))
        self.z_goal[SHAPES.index(name)] = strength


class ColonyEngine:
    EVENTS = EVENTS

    def __init__(self, cfg: ColonyConfig | None = None):
        self.cfg = cfg = cfg or ColonyConfig()
        self.rs = RngStreams(cfg.seed)
        self.rng = self.rs.stream("colony")
        self.ev_rng = self.rs.stream("colony-events")
        self.nrng = np.random.default_rng(stable_hash64("colony-noise", cfg.seed) & 0xFFFFFFFF)
        self.params = ParameterSet(cfg.params)
        self.inp = CreatureControlInput()
        n = self.n = cfg.nodes
        r = self.rs.stream("colony-shape")
        self.U0 = np.array([[r.random(), r.random(), r.random()] for _ in range(n)])
        self.U = self.U0.copy()
        self.mass = np.full(n, 1.0 / n)                       # finite material: fixed forever
        self.t = 0.0
        self.own = np.zeros(n, int)
        self.bodies = [Body(r.uniform(-math.pi, math.pi)) for _ in range(cfg.max_bodies)]
        lead = self.bodies[0]
        lead.alive = True
        lead.goal_shape("SPINDLE", 2.0)
        ph = {"t": 0.0, "beat": 0.0, "spin": 0.0, "flap": 0.0, "pulse": 0.0, "gait": 0.0, "mech": 0.5}
        self.x = lead.P + shape3("CLOUD", self.U, cfg.size, 1.0, ph, -3.0) @ lead.R().T
        self.v = np.zeros((n, 3))
        self.edges, self.rest = _knn_edges(self.x, cfg.k_neighbors)
        self.alive = np.ones(len(self.edges), bool)
        self.m = np.tile(np.array(MATERIAL["COHESIVE"], float), (n, 1))   # per-node material field
        self.local = np.zeros(n)
        self.waves: list[tuple[np.ndarray, float, float]] = []
        self.obstacles: list[Obstacle | None] = [None] * cfg.max_obstacles
        self.lure: dict | None = None
        self.lure_noise = [Noise1D(cfg.seed + 41 + i, 0.11, 2) for i in range(3)]
        self.turb_until = -1.0
        self.last_kick = self.last_obstacle = self.last_wave = self.last_split = self.last_rebuild = -1e9
        self.split_t = 0.0
        self.hi_t = self.lo_t = 0.0
        self.spin = self.flap = self.pulse = self.gait = 0.0
        self.arousal, self.instab, self.glow, self.surface = 0.2, 0.0, 0.0, 0.3
        self.response_hist: list[str] = []
        self.events: list[tuple[float, str, object]] = []
        self._pending: list[tuple[str, object]] = []

    # ------------------------------------------------------------------ API
    def set_input(self, inp: CreatureControlInput) -> None:
        self.inp = inp.sanitized()

    def set_parameter(self, name: str, value) -> bool:
        return self.params.set(name, value)

    def trigger_event(self, name: str, arg=None) -> bool:
        if name.upper() not in EVENTS:
            return False
        self._pending.append((name.upper(), arg))
        return True

    # ------------------------------------------------------------------ helpers
    def _log(self, name, arg=None):
        self.events.append((self.t, name, arg))
        self.events = self.events[-60:]

    def _alive(self) -> list[int]:
        return [k for k, b in enumerate(self.bodies) if b.alive]

    def _physics_bodies(self) -> list[int]:
        """Bodies that get a step (subclasses add static structures)."""
        return self._alive()

    def _is_static(self, k: int) -> bool:
        return False

    def _post_targets(self, T: np.ndarray, dt: float) -> np.ndarray:
        return T

    def _set_intent(self, b: Body, intent: str, shape: str | None = None) -> None:
        prefs, mstate, _ = PLAN[intent]
        b.intent, b.intent_t = intent, 0.0
        b.dwell = self.rng.uniform(4.0, 10.0) * (0.6 + 0.8 * self.params["coherence"])
        b.goal_shape(shape or prefs[self.rng.randint(0, len(prefs) - 1)])
        b.mat_goal = np.array(MATERIAL[mstate])

    def _renormalize(self, k: int) -> None:
        """A body built from part of the material still forms complete shapes (rank-mapped coordinates)."""
        idx = np.nonzero(self.own == k)[0]
        if len(self._alive()) == 1:
            self.U = self.U0.copy()
            return
        for c in range(3):
            rank = np.argsort(np.argsort(self.U0[idx, c]))
            self.U[idx, c] = (rank + 0.5) / len(idx)

    def _rebuild(self) -> None:
        """Re-form the elastic network inside every body (recombination)."""
        pairs, rests = [], []
        for k in self._physics_bodies():
            idx = np.nonzero(self.own == k)[0]
            if len(idx) < 3:
                continue
            e, r = _knn_edges(self.x[idx], min(self.cfg.k_neighbors, len(idx) - 1))
            pairs.append(idx[e])
            rests.append(r)
        self.edges = np.concatenate(pairs)[:self.cfg.max_links] if pairs else np.zeros((0, 2), int)
        self.rest = np.concatenate(rests)[:self.cfg.max_links] if rests else np.zeros(0)
        self.alive = np.ones(len(self.edges), bool)
        self.last_rebuild = self.t

    def _split(self, K: int, axis: np.ndarray | None = None) -> None:
        if len(self._alive()) > 1:
            return
        K = max(2, min(int(K), self.cfg.max_bodies))
        lead = self.bodies[0]
        R = lead.R()
        rel = (self.x - lead.P) @ R                       # local: x forward, y left
        if axis is not None:                                # flow past an obstacle on both sides
            K = 2
            side = (self.x - lead.P) @ axis
            order = np.argsort(side)
        else:
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            order = np.argsort((ang + math.pi / K) % (2 * math.pi))   # the forward sector stays together
        groups = np.array_split(order, K)
        fwd = [float(np.cos(np.arctan2(rel[g, 1], rel[g, 0])).mean()) for g in groups]
        lead_g = int(np.argmax(fwd)) if axis is None else 0
        groups = [groups[lead_g]] + [g for i, g in enumerate(groups) if i != lead_g]
        for k, g in enumerate(groups):
            self.own[g] = k
        for k, g in enumerate(groups[1:], start=1):
            b = self.bodies[k]
            d = rel[g].mean(0)
            d[2] = 0.0
            d = d / max(np.linalg.norm(d), 1e-6)
            b.__init__(lead.heading + 0.35 * float(np.sign(d[1] or 1.0)))
            b.alive = True
            b.P, b.vel = self.x[g].mean(0), lead.vel.copy()
            b.slot = d * (2.8 + 1.2 * k) + np.array([0.0, 0.0, 0.4 * k])
            b.z, b.mat = lead.z.copy(), lead.mat.copy()
            self._set_intent(b, "FORMATION")
            self.v[g] += (R @ d) * 4.0                       # the parts push apart
            self._renormalize(k)
        self._renormalize(0)
        e = self.edges
        self.alive &= self.own[e[:, 0]] == self.own[e[:, 1]]  # separation
        self.last_split, self.split_t = self.t, 0.0
        self._log("SPLIT", K)

    def _absorb(self, k: int) -> None:
        self.own[self.own == k] = 0
        self.bodies[k].alive = False
        self._renormalize(0)
        self._rebuild()                                      # the networks fuse
        if len(self._alive()) == 1:
            self._log("MERGE", None)

    def _start_merge(self) -> None:
        for b in self.bodies[1:]:
            if b.alive and b.intent != "MERGE":
                self._set_intent(b, "MERGE")
        lead = self.bodies[0]
        if lead.intent in ("FORMATION", "CRUISE", "EXPLORE", "DISPLAY"):
            self._set_intent(lead, "HOVER")

    def _spawn_obstacle(self, strength: float) -> None:
        cfg, r = self.cfg, self.ev_rng
        if None not in self.obstacles:
            return
        slot = self.obstacles.index(None)
        alive = self._alive()
        b = self.bodies[alive[r.randint(0, len(alive) - 1)]]
        fwd = b.R()[:, 0]
        side = np.cross([0, 0, 1.0], fwd)
        dirs = [fwd, fwd + side, fwd - side, side, -side, fwd + np.array([0, 0, 0.8])]
        d = dirs[int(r.randint(0, len(dirs) - 1))]
        d = d / np.linalg.norm(d)
        speed = r.uniform(6.0, 12.0) * (0.7 + 0.5 * strength)
        dist = r.uniform(9.0, 13.0)
        start = b.P + d * dist
        tt = dist / (speed + max(0.0, float(b.vel @ d)))
        aim = b.P + b.vel * tt + np.array([r.normal(0, 0.25), r.normal(0, 0.25), r.normal(0, 0.2)])
        self.obstacles[slot] = Obstacle(start, (aim - start) / max(tt, 0.2), r.uniform(0.35, 1.0) * cfg.size * 0.6, self.t)
        self._log("OBSTACLE", None)

    def _spawn_lure(self) -> None:
        lead = self.bodies[0]
        a = self.ev_rng.uniform(-math.pi, math.pi)
        pos = lead.P + np.array([math.cos(a) * 10.0, math.sin(a) * 10.0, 0.0])
        pos[2] = self.ev_rng.uniform(2.0, 5.0)
        self.lure = {"pos": pos, "vel": np.zeros(3), "state": "FREE", "born": self.t, "until": -1.0, "r": 0.22}
        self._log("PREY", None)

    def _graph_dist(self, seed: int) -> np.ndarray:
        adj = [[] for _ in range(self.n)]
        for i, j in self.edges[self.alive]:
            adj[int(i)].append(int(j))
            adj[int(j)].append(int(i))
        d = np.full(self.n, np.inf)
        d[seed] = 0.0
        q, h = [seed], 0
        while h < len(q):
            a = q[h]
            h += 1
            for b in adj[a]:
                if d[b] == np.inf:
                    d[b] = d[a] + 1.0
                    q.append(b)
        return d

    def _wave(self, strength: float = 1.0, at: np.ndarray | None = None) -> None:
        """A hardening wave travels through the network from one point (armour rises in a band)."""
        if at is None:
            seed = int(self.ev_rng.randint(0, self.n - 1))
        else:
            seed = int(np.argmin(((self.x - at) ** 2).sum(1)))
        self.waves.append((self._graph_dist(seed), self.t, 9.0 + 5.0 * strength))
        self.waves = self.waves[-4:]
        self.last_wave = self.t
        self._log("WAVE", None)

    def _kick(self, strength: float) -> None:
        pr = self.params
        backbeat = abs((self.inp.beat % 2.0) - 1.0) < 0.2
        if backbeat and pr["armor"] > 0.15 and self.t - self.last_wave > 0.8:
            self._pending.append(("WAVE", strength))
            return
        mode = KICK_MODES[min(4, int(pr["kick_mode"] * 5))]
        if mode == "MIX":
            mode = KICK_MODES[self.ev_rng.weighted_index([1.0, 1.4 + pr["obstacle_rate"], 0.7, 0.6])]
        if mode == "OBSTACLE":
            if self.t - self.last_obstacle < 2.0 + 4.0 * (1.0 - pr["obstacle_rate"]):
                mode = "IMPULSE"
            else:
                self.last_obstacle = self.t
        self._pending.append((mode, strength))

    # ------------------------------------------------------------------ events
    def _apply_event(self, name: str, arg) -> None:
        r = self.ev_rng
        s = float(arg) if isinstance(arg, (int, float)) else 1.0
        lead = self.bodies[0]
        alive = self._alive()
        if name == "IMPULSE":
            b = self.bodies[alive[r.randint(0, len(alive) - 1)]]
            c = b.P + np.array([r.normal(0, 1), r.normal(0, 1), r.normal(0, 0.6)]) * self.cfg.size * 0.7
            d = self.x - c
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            self.v += 4.0 * s * np.clip(1.5 - dist / self.cfg.size, 0, 1) * d / np.maximum(dist, 1e-3)
        elif name == "OBSTACLE":
            self._spawn_obstacle(s)
            return
        elif name == "PRESSURE":
            for k in alive:
                idx = self.own == k
                self.v[idx] -= 2.2 * s * (self.x[idx] - self.x[idx].mean(0))
        elif name == "TURBULENCE":
            self.turb_until = self.t + 0.7
        elif name == "WAVE":
            self._wave(s, arg if isinstance(arg, np.ndarray) else None)
            return
        elif name == "SPLIT":
            self._split(2 + int(self.params["swarm"] * 2.99) if not isinstance(arg, int) else arg)
            return
        elif name == "MERGE":
            self._start_merge()
        elif name == "HUNT":
            if self.lure is None:
                self._spawn_lure()
            for k in alive:
                self._set_intent(self.bodies[k], "HUNT")
        elif name == "PERCH":
            if len(alive) > 1:
                self._start_merge()
            self._set_intent(lead, "PERCH", "LEGS")
        elif name == "MORPHOLOGY_SHIFT":                  # within the vocabulary of what each body is doing
            for k in alive:
                b = self.bodies[k]
                prefs = PLAN[b.intent][0]
                if arg in SHAPES and arg not in ("ENVELOP", "LEGS") and b.intent not in ("PERCH", "ENVELOP"):
                    b.goal_shape(arg)
                elif b.intent in ("PERCH", "ENVELOP") or r.chance(0.7):
                    b.goal_shape(prefs[r.randint(0, len(prefs) - 1)])
                else:
                    b.goal_shape(SHAPES[r.randint(0, len(SHAPES) - 3)])
        elif name == "COLLAPSE":
            for k in alive:
                self.bodies[k].mat_goal = np.array(MATERIAL["DISPERSED"])
                self._set_intent(self.bodies[k], "REFORM") if False else None
            self.alive[:] = False
            lead.intent, lead.intent_t = "REFORM", -2.0
        elif name in ("RECONSTRUCTION", "MASS_REBALANCE"):
            for k in alive:
                self.bodies[k].mat_goal = np.array(MATERIAL["COHESIVE"])
            lead.intent, lead.intent_t = "REFORM", 0.0
        elif name == "APPENDAGE_BURST":
            for k in alive:
                self.bodies[k].goal_shape("BLADES")
                self.bodies[k].mat_goal = np.array(MATERIAL["STRUCTURED"])
        self._log(name, arg if isinstance(arg, str) else None)

    # ------------------------------------------------------------------ behaviour
    def _think(self, dt: float) -> None:
        pr, inp, r = self.params.values(), self.inp, self.rng
        self.arousal += ((0.75 * inp.energy + 0.4 * inp.transient + 0.2 * inp.spectral_flux) * pr["reactivity"] +
                         0.1 - self.arousal) * min(1.0, dt / 2.0)
        self.instab = min(1.5, self.instab + dt * (0.3 * inp.spectral_flux + 0.1 * pr["instability"]))
        if inp.transient > 0.5 and self.t - self.last_kick > 0.25:
            self.last_kick = self.t
            self._kick(inp.transient)
        if self.ev_rng.chance(dt * pr["obstacle_rate"] * 0.2 * (0.3 + inp.energy)):
            self._spawn_obstacle(0.6)
        alive = self._alive()
        lead = self.bodies[0]
        # Drops split the colony into a flock, breakdowns bring it back together.
        self.hi_t = self.hi_t + dt if inp.energy > 0.78 else 0.0
        self.lo_t = self.lo_t + dt if inp.energy < 0.3 else 0.0
        if len(alive) == 1:
            if (pr["swarm"] > 0.2 and self.hi_t > 1.5 and self.t - self.last_split > 14.0 and
                    lead.intent not in ("ENVELOP", "PERCH", "EVADE")):
                self._split(2 + int(pr["swarm"] * 2.99))
        else:
            self.split_t += dt
            if (self.lo_t > 2.5 or self.split_t > 16.0 + 12.0 * pr["swarm"]) and \
                    any(self.bodies[k].intent not in ("MERGE",) for k in alive[1:]):
                self._start_merge()
        # Prey.
        if self.lure is None and self.ev_rng.chance(dt * pr["hunt"] * 0.05):
            self._spawn_lure()
        # Threats: the same obstacle is never answered the same way three times in a row.
        for k in alive:
            b = self.bodies[k]
            b.intent_t += dt
            for ob in self.obstacles:
                if ob is None or ob.handled:
                    continue
                rel, rv = ob.pos - b.P, ob.vel - b.vel
                tca = -float(rel @ rv) / max(float(rv @ rv), 1e-6)
                dca = float(np.linalg.norm(rel + rv * max(tca, 0.0)))
                if 0.0 < tca < 1.3 and dca < ob.radius + 0.8 * self.cfg.size and b.intent not in ("ENVELOP",):
                    ob.handled = True
                    opts = ["SPLIT", "SHIELD", "DISPERSE", "DODGE", "PARTITION"]
                    w = [1.4 * pr["fluidity"] + 0.3, 0.6 + pr["rigidity"] + 0.5 * pr["armor"], 0.4 + 0.8 * pr["instability"],
                         0.6 + pr["speed"], (1.2 * pr["swarm"] + 0.2) if (len(alive) == 1 and k == 0) else 0.0]
                    for i, o in enumerate(opts):
                        if o in self.response_hist[-2:]:
                            w[i] *= 0.3
                    b.response = opts[r.weighted_index(w)]
                    self.response_hist.append(b.response)
                    self._log("RESPONSE", b.response)
                    b.intent, b.intent_t = "EVADE", 0.0
                    if b.response == "SHIELD":
                        b.goal_shape("SHIELD", 3.0)
                        b.mat_goal = np.array(MATERIAL["HIGH_STIFFNESS"])
                        self._wave(0.8, b.P + rel / max(np.linalg.norm(rel), 1e-6))
                    elif b.response == "DISPERSE":
                        b.mat_goal = np.array(MATERIAL["DISPERSED"])
                    elif b.response == "DODGE":
                        side = np.cross([0, 0, 1.0], ob.vel)
                        side = side / max(np.linalg.norm(side), 1e-6) * (1 if r.chance(0.5) else -1)
                        self.v[self.own == k] += side * 3.0
                        b.goal_shape("SPINDLE", 3.0)
                    elif b.response == "PARTITION":            # part, let it through, close again
                        ax = np.cross([0, 0, 1.0], ob.vel)
                        self._split(2, ax / max(np.linalg.norm(ax), 1e-6))
                        for kk in self._alive()[1:]:
                            self.bodies[kk].intent, self.bodies[kk].intent_t = "EVADE", 0.0
                    else:
                        b.mat_goal = np.array(MATERIAL["FLUID"])
                    b.threat = ob
        # Intent machine.
        for k in list(self._alive()):
            b = self.bodies[k]
            if b.intent == "EVADE" and b.intent_t > 1.6:
                if b.response == "PARTITION" and len(self._alive()) > 1:
                    self._start_merge()
                    if k == 0:
                        self._set_intent(b, "CRUISE")
                    continue
                self._set_intent(b, "REFORM" if k == 0 else "FORMATION")
                b.mat_goal = np.array(MATERIAL["COHESIVE"])
                self._log("REASSEMBLY", None)
            elif b.intent == "MERGE" and k > 0 and np.linalg.norm(b.P - lead.P) < 1.3 * self.cfg.size:
                self._absorb(k)
            elif b.intent == "HUNT":
                L = self.lure
                if L is None or b.intent_t > 16.0:
                    self._set_intent(b, "CRUISE" if k == 0 else "FORMATION")
                    if k == 0 and L is not None and L["state"] == "FREE":
                        L["state"], L["until"] = "GONE", self.t + 1.0
                elif np.linalg.norm(L["pos"] - b.P) < 1.8 * self.cfg.size ** 0.5 and L["state"] == "FREE":
                    for kk in self._alive()[1:]:              # close the trap: the flock fuses around it
                        self._absorb(kk)
                    self._set_intent(lead, "ENVELOP", "ENVELOP")
                    self._log("ENVELOP", None)
                    break
            elif b.intent == "ENVELOP":
                L = self.lure
                if L is None:
                    self._set_intent(b, "CRUISE")
                elif L["state"] == "FREE" and b.intent_t > 0.8 and np.linalg.norm(L["pos"] - b.P) < 0.9:
                    L["state"], L["until"] = "CAUGHT", self.t + 8.0 * 60.0 / max(self.inp.tempo, 60.0)
                    self._log("CAUGHT", None)
                elif L["state"] == "CAUGHT" and self.t > L["until"]:
                    L["state"], L["until"] = "ESCAPE", self.t + 3.0
                    a = r.uniform(-math.pi, math.pi)
                    L["vel"] = np.array([math.cos(a) * 9.0, math.sin(a) * 9.0, 1.5])
                    b.mat_goal = np.array(MATERIAL["DISPERSED"])
                    idx = self.own == 0                          # burst open (no net push: the mass stays put)
                    self.v[idx] += (self.x[idx] - self.x[idx].mean(0)) * 3.0
                    b.intent, b.intent_t = "REFORM", -0.8
                    self._log("RELEASE", None)
                elif b.intent_t > 12.0:
                    self._set_intent(b, "CRUISE")
            elif b.intent == "REFORM" and b.intent_t > 1.5 and k == 0:
                self._set_intent(b, "CRUISE")
            elif k == 0 and b.intent_t > b.dwell and b.intent not in ("EVADE", "HUNT", "ENVELOP"):
                pool = ["CRUISE", "HOVER", "EXPLORE", "DISPLAY", "PERCH", "HUNT"]
                w = [1.2 + self.inp.energy, 0.8 * (1 - self.inp.energy), 0.5 + pr["mutation"] + self.instab,
                     0.4 + pr["rigidity"] + pr["tendril_activity"], (0.2 + 0.6 * (1 - self.inp.energy)) *
                     (1.0 if len(self._alive()) == 1 else 0.0), 1.5 * pr["hunt"] if self.lure is not None else 0.0]
                if b.intent in pool:
                    w[pool.index(b.intent)] *= 0.4
                nxt = pool[r.weighted_index(w)]
                self._set_intent(b, nxt, "LEGS" if nxt == "PERCH" else None)
                self._log("MORPHOLOGY_SHIFT", nxt)
                if nxt == "HUNT":
                    for kk in self._alive()[1:]:
                        self._set_intent(self.bodies[kk], "HUNT")
            elif k > 0 and b.intent == "FORMATION" and b.intent_t > b.dwell:
                b.intent_t = 0.0
                b.goal_shape(PLAN["FORMATION"][0][r.randint(0, 3)])
        if self.instab > 1.0:
            self.instab = 0.0
            self._pending.append(("MORPHOLOGY_SHIFT", None))

    # ------------------------------------------------------------------ flight
    def _fly(self, k: int, idx: np.ndarray, dt: float, pr: dict) -> tuple[np.ndarray, float]:
        cfg, b, lead = self.cfg, self.bodies[k], self.bodies[0]
        xb = self.x[idx]
        b.P, b.vel = xb.mean(0), self.v[idx].mean(0)
        frac = len(idx) / self.n
        sb = cfg.size * frac ** (1.0 / 3.0)
        base = cfg.cruise * (0.4 + 1.2 * pr["speed"]) * (0.6 + self.arousal)
        lo, hi = cfg.altitude
        alt_goal = lo + (hi - lo) * pr["altitude"]
        b.wander = b.wander * math.exp(-dt * 0.3) + self.nrng.standard_normal() * math.sqrt(dt) * (0.4 + pr["noise"])
        v_des = self._v_des(k, b, lead, dt, pr, sb, base, alt_goal)
        if b.intent != "PERCH":
            v_des[2] += max(0.0, 1.0 - b.P[2]) * 2.0
        a_des = (v_des - b.vel) / 0.8 + np.array([0.0, 0.0, G])
        I = float(((xb - b.P) ** 2).sum(1).mean()) / max(sb * sb, 1e-6)
        sp = float(np.linalg.norm(b.vel[:2]))
        if sp > 0.3:
            dy = (math.atan2(b.vel[1], b.vel[0]) - b.heading + math.pi) % (2 * math.pi) - math.pi
            b.yaw_rate += dt * (3.0 * dy - 2.2 * b.yaw_rate) / (0.3 + 3.0 * I)
        b.heading += dt * b.yaw_rate
        pitch_goal = 0.0 if b.intent in ("PERCH", "ENVELOP") else math.atan2(b.vel[2], max(sp, 0.5)) * 0.6
        b.pitch += (pitch_goal - b.pitch) * min(1.0, dt * 2.0)
        return a_des, sb

    def _v_des(self, k: int, b: Body, lead: Body, dt: float, pr: dict, sb: float, base: float,
               alt_goal: float) -> np.ndarray:
        """Where this body wants to go (by its intent)."""
        cfg = self.cfg
        L = self.lure
        if b.intent == "FORMATION" and k > 0:
            goal = lead.P + lead.R() @ b.slot
            v_des = lead.vel + np.clip(1.4 * (goal - b.P), -6.0, 6.0)
        elif b.intent == "MERGE" and k > 0:
            d = lead.P - b.P
            dn = float(np.linalg.norm(d))
            v_des = lead.vel + d / max(dn, 1e-6) * min(7.0, 2.0 + 1.5 * dn)
        elif b.intent == "HUNT" and L is not None:
            n_alive = len(self._alive())
            ang = 2 * math.pi * k / max(n_alive, 1) + 0.5
            off = np.array([math.cos(ang), math.sin(ang), 0.3]) * (2.2 if n_alive > 1 else 0.0)   # pincer
            aim = L["pos"] + L["vel"] * 0.6 + off
            d = aim - b.P
            dn = float(np.linalg.norm(d))
            v_des = d / max(dn, 1e-6) * min(1.6 * base + 2.0, 1.5 * dn + 1.0)
        elif b.intent == "ENVELOP" and L is not None:
            v_des = L["vel"] + 2.5 * (L["pos"] - b.P)
        else:
            cruise = base * PLAN[b.intent][2]
            want = b.heading + 1.5 * b.wander * dt
            home = -b.P[:2]
            if float(np.linalg.norm(home)) > 0.7 * cfg.stage_radius:
                wh = math.atan2(home[1], home[0])
                want += ((wh - b.heading + math.pi) % (2 * math.pi) - math.pi) * min(1.0, dt * 1.2)
            if b.intent == "PERCH":
                vz = float(np.clip((0.62 * sb - b.P[2]) * 1.2, -2.0, 1.0))
            else:
                vz = float(np.clip((alt_goal - b.P[2]) * 0.8 + 0.3 * math.sin(0.4 * self.t + k), -1.5, 1.5))
            v_des = np.array([math.cos(want) * cruise, math.sin(want) * cruise, vz])
        return v_des

    def _targets(self, k: int, idx: np.ndarray, sb: float, ph: dict) -> np.ndarray:
        pr, b = self.params, self.bodies[k]
        s = sb * (0.75 + 0.5 * pr["expansion"] - 0.3 * pr["contraction"])
        elong = 0.8 + 0.6 * pr["density"]
        w = b.weights()
        R = b.R()
        U = self.U[idx]
        loc = np.zeros((len(idx), 3))
        extra = np.zeros((len(idx), 3))
        for a, wa in zip(SHAPES, w):
            if wa < 0.01:
                continue
            # Situational structures exist only in their situation (legs on the ground, the shell on prey).
            if (a == "ENVELOP" and (b.intent != "ENVELOP" or self.lure is None)) or (a == "LEGS" and b.intent != "PERCH"):
                a = "CORE"
            if a == "ENVELOP":                                  # a closed shell around the prey (world space)
                th, pz = 2 * np.pi * U[:, 0], np.arccos(1 - 2 * U[:, 1])
                rr = (0.35 + 0.4 * s) * (1.0 + 0.08 * (U[:, 2] - 0.5))
                shell = np.stack([np.sin(pz) * np.cos(th), np.sin(pz) * np.sin(th), np.cos(pz)], 1) * rr[:, None]
                extra += wa * (self.lure["pos"] - b.P + shell)
                continue
            shp = shape3(a, U, s, elong, ph, -b.P[2])
            if a != "LEGS":                                     # thrust places the body, not the shape
                shp -= shp.mean(0)
            loc += wa * shp
        loc[:, 1] *= 1.0 + 0.35 * pr["asymmetry"] * np.sign(loc[:, 1]) * math.sin(0.11 * self.t + k)
        pressure = 0.25 * self.inp.bass * (0.5 + pr["expansion"]) + 0.03 * math.sin(2 * math.pi * self.inp.beat / 4.0)
        return b.P + (loc @ R.T) * (1.0 + pressure) + extra

    # ------------------------------------------------------------------ simulation
    def update(self, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            return self.state()
        steps = max(1, min(12, int(round(dt * self.cfg.sim_rate))))
        for _ in range(steps):
            self._step(dt / steps)
        return self.state()

    def _lure_step(self, dt: float) -> None:
        L = self.lure
        if L is None:
            return
        cfg = self.cfg
        if L["state"] == "FREE":
            want = np.array([self.lure_noise[0].sample(self.t) * 5.0, self.lure_noise[1].sample(self.t) * 5.0,
                             self.lure_noise[2].sample(self.t) * 1.2])
            want[:2] -= L["pos"][:2] * max(0.0, float(np.linalg.norm(L["pos"][:2])) - 0.6 * cfg.stage_radius) * 0.05
            want[2] += (3.5 - L["pos"][2]) * 0.6
            L["vel"] += (want - L["vel"]) * min(1.0, dt * 1.5)
            if self.t - L["born"] > 40.0:
                L["state"], L["until"] = "GONE", self.t + 1.0
        elif L["state"] == "CAUGHT":                        # carried inside the shell
            L["vel"] *= math.exp(-dt * 4.0)
            L["pos"] = L["pos"] + (self.bodies[0].P - L["pos"]) * min(1.0, dt * 2.0)
        elif L["state"] in ("ESCAPE", "GONE"):
            L["vel"] *= math.exp(-dt * 0.5)
            if self.t > L["until"]:
                if L["state"] == "ESCAPE":
                    L["state"], L["until"] = "GONE", self.t + 1.0
                else:
                    self.lure = None
                    return
        L["pos"] = L["pos"] + L["vel"] * dt
        L["pos"][2] = max(0.4, L["pos"][2])

    def _step(self, dt: float) -> None:
        self.t += dt
        cfg, pr, inp = self.cfg, self.params.values(), self.inp
        self._think(dt)
        while self._pending:
            self._apply_event(*self._pending.pop(0))
        self._lure_step(dt)
        mech = pr["mechanism"]
        self.spin += dt * mech * (1.0 + 5.0 * inp.energy)
        self.flap = mech * 0.55 * math.sin(math.pi * inp.beat)
        self.pulse = max(self.pulse * math.exp(-dt / 0.18), inp.transient)
        self.gait = math.pi * inp.beat
        ph = {"t": self.t, "beat": inp.beat, "spin": self.spin, "flap": self.flap, "pulse": self.pulse,
              "gait": self.gait, "mech": mech}
        n = self.n
        x, v = self.x, self.v
        T = np.empty_like(x)
        a_des = np.empty_like(x)
        vcom = np.empty_like(x)
        goal = np.empty((n, 6))
        tau_m = 0.8 + 3.0 * pr["rigidity"] * (1.2 - pr["fluidity"])
        for k in self._physics_bodies():
            idx = np.nonzero(self.own == k)[0]
            if len(idx) == 0:
                self.bodies[k].alive = False
                continue
            b = self.bodies[k]
            if self._is_static(k):                          # structures hold their place and shape
                ad, tgt = self._static_step(k, idx, dt)
                a_des[idx], vcom[idx], T[idx], goal[idx] = ad, b.vel, tgt, b.mat
                b.mat += (b.mat_goal - b.mat) * min(1.0, dt / 0.6)
                continue
            noise = self.nrng.standard_normal(len(SHAPES)) * pr["mutation"] * 0.6 * math.sqrt(dt)
            noise[_SITUATIONAL] = 0.0
            b.z += (b.z_goal - b.z) * min(1.0, dt / tau_m) + noise
            b.mat += (b.mat_goal - b.mat) * min(1.0, dt / (0.4 + 0.8 * pr["coherence"]))
            ad, sb = self._fly(k, idx, dt, pr)
            a_des[idx] = ad
            vcom[idx] = b.vel
            T[idx] = self._targets(k, idx, sb, ph)
            goal[idx] = b.mat
        T = self._post_targets(T, dt)
        # Per-node material field: relax to the body's state, diffuse along the network, waves, hits.
        m = self.m
        m += (goal - m) * min(1.0, dt / 0.35)
        e, al = self.edges, self.alive
        if al.any():
            i, j = e[al, 0], e[al, 1]
            lap = np.zeros(n)
            diff = m[j, 1] - m[i, 1]
            np.add.at(lap, i, diff)
            np.add.at(lap, j, -diff)
            m[:, 1] += 0.6 * lap * dt
        armor = pr["armor"]
        live = []
        for dist, t0, speed in self.waves:
            front = (self.t - t0) * speed
            band = np.exp(-(((dist - front) / 1.3) ** 2))
            band[~np.isfinite(dist)] = 0.0
            m[:, 1] = np.maximum(m[:, 1], (1.0 + 0.8 * armor) * band + m[:, 1] * (1 - band))
            if front < np.nanmax(np.where(np.isfinite(dist), dist, 0.0)) + 3.0:
                live.append((dist, t0, speed))
        self.waves = live
        coh = m[:, 0] * (0.5 + pr["coherence"])
        stiff = m[:, 1] * (0.4 + 1.2 * pr["rigidity"])
        damp = m[:, 2]
        rep = m[:, 3]
        disp = np.minimum(1.0, m[:, 5] + 0.3 * pr["fluidity"] * pr["noise"])
        local = self.local
        f_t = (2 * np.pi * (0.5 + 1.2 * stiff)) ** 2 * 0.25
        acc = (coh * (1.0 - local) * f_t)[:, None] * (T - x) - (1.5 + 3.0 * damp)[:, None] * (v - vcom)
        acc += a_des - np.array([0.0, 0.0, G])
        # Elastic network: local stiffness, links across bodies are cut (separation).
        al &= self.own[e[:, 0]] == self.own[e[:, 1]]
        if al.any():
            i, j = e[al, 0], e[al, 1]
            d = x[j] - x[i]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            rest = self.rest[al]
            lk = np.maximum(local[i], local[j])
            ks = 60.0 * np.minimum(stiff[i], stiff[j]) * (1.0 - 0.8 * lk)
            f = (ks * (L - rest) / L)[:, None] * d + (4.0 * np.minimum(damp[i], damp[j]))[:, None] * (v[j] - v[i])
            np.add.at(acc, i, f)
            np.add.at(acc, j, -f)
            # Plasticity: the network slowly accepts the shape it is held in (morphological inertia).
            tau_p = 0.6 + 2.5 * np.clip(np.minimum(stiff[i], stiff[j]), 0, 1.5)
            self.rest[al] = rest + (L - rest) * np.minimum(1.0, dt / tau_p)
            brk = np.minimum(m[i, 4], m[j, 4]) * (1.0 - 0.4 * lk)
            broken = L > rest * brk
            if broken.any():
                self.alive[np.nonzero(al)[0][broken]] = False
        D = x[:, None, :] - x[None, :, :]
        dist2 = (D * D).sum(2) + np.eye(n)
        rmin = 0.11 * cfg.size
        close = dist2 < rmin * rmin
        if close.any():
            push = np.where(close, rmin / np.sqrt(dist2) - 1.0, 0.0)
            acc += 8.0 * rep[:, None] * (push[:, :, None] * D).sum(1)
        acc += self.nrng.standard_normal((n, 3)) * (2.5 * disp + 3.0 * local + 1.2 * inp.high * pr["surface_activity"])[:, None]
        if self.t < self.turb_until:
            acc += 6.0 * np.stack([np.sin(3.1 * x[:, 1] + 5 * self.t), np.sin(2.7 * x[:, 2] + 4 * self.t),
                                   np.sin(2.3 * x[:, 0] + 6 * self.t)], 1)
        # Colliders: obstacles (material hardens where it is hit) and the prey.
        colliders = [(ob.pos, ob.vel, ob.radius, slot) for slot, ob in enumerate(self.obstacles) if ob is not None]
        if self.lure is not None and self.lure["state"] != "GONE":
            colliders.append((self.lure["pos"], self.lure["vel"], self.lure["r"] + 0.08, -1))
        for pos, vel, rad, slot in colliders:
            if slot >= 0:
                ob = self.obstacles[slot]
                ob.pos = ob.pos + ob.vel * dt
                pos = ob.pos
            rel = x - pos
            dd = np.linalg.norm(rel, axis=1)
            inside = dd < rad
            if inside.any():
                nrm = rel[inside] / np.maximum(dd[inside], 1e-6)[:, None]
                x[inside] = pos + nrm * rad
                vn = (v[inside] * nrm).sum(1, keepdims=True)
                v[inside] -= np.minimum(vn, 0) * nrm * 1.6
                if slot >= 0:
                    m[inside, 1] = np.maximum(m[inside, 1], 1.5)
                    self.glow = max(self.glow, 0.8)
            if slot >= 0:
                b = next((self.bodies[k] for k in self._alive() if getattr(self.bodies[k], "threat", None) is ob), None)
                if b is not None and b.intent == "EVADE" and b.response in ("SPLIT", "DISPERSE"):
                    ahead = rel - vel * (rel @ vel)[:, None] / max(float(vel @ vel), 1e-6)
                    corridor = np.linalg.norm(ahead, axis=1) < rad + 0.35 * cfg.size
                    local[corridor] = np.minimum(1.0, local[corridor] + dt * 4.0)
                    side = ahead / np.maximum(np.linalg.norm(ahead, axis=1), 1e-6)[:, None]
                    acc[corridor] += 9.0 * side[corridor]
                if float(np.linalg.norm(ob.pos - x.mean(0))) > 25.0 or self.t - ob.born > 8.0:
                    self.obstacles[slot] = None
        self.local = local * math.exp(-dt / 0.9)
        v += acc * dt
        spd = np.linalg.norm(v, axis=1)
        fast = spd > 30.0
        if fast.any():
            v[fast] *= (30.0 / spd[fast])[:, None]
        x += v * dt
        low = x[:, 2] < 0.05
        x[low, 2] = 0.05
        v[low, 2] = np.abs(v[low, 2]) * 0.2
        v[low, :2] *= 0.85                                   # feet grip the floor
        if (coh.mean() > 0.7 and self.t - self.last_rebuild > 0.6 and len(self.alive) and
                (~self.alive).mean() > 0.25):
            self._rebuild()
        self.glow *= math.exp(-dt / 0.4)
        self.surface += (min(1.0, pr["surface_activity"] + 0.5 * inp.high + 0.5 * float(disp.mean())) - self.surface) * \
            min(1.0, dt * 3)

    # ------------------------------------------------------------------ state
    def fragments(self) -> int:
        parent = list(range(self.n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i, j in self.edges[self.alive]:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
        return len({find(a) for a in range(self.n)})

    def state(self) -> ColonyState:
        cfg, n = self.cfg, self.n
        P = np.array([b.P for b in self.bodies])
        own_P = P[self.own]
        sizes = np.array([cfg.size * (max(1, int((self.own == k).sum())) / n) ** (1 / 3) for k in range(len(self.bodies))])
        rel = self.x - own_P
        rn = np.linalg.norm(rel, axis=1)
        disp = np.clip(rn / (1.2 * sizes[self.own]) - 0.5, 0, 1) * 0.5 + 0.3 * self.local + 0.4 * self.m[:, 5]
        disp = np.clip(disp, 0, 1)
        stiff = self.m[:, 1]
        r0 = 0.62 * cfg.size * (1.0 / n) ** (1 / 3) * 1.9
        radius = r0 * (1.0 - 0.45 * disp) * (1.0 - 0.3 * np.clip((stiff - 0.5) / 1.1, 0, 1))
        links = np.full((cfg.max_links, 3), -1.0)
        links[:, 2] = 0.0
        mm = min(cfg.max_links, len(self.edges))
        if mm:
            e = self.edges[:mm]
            links[:mm, 0:2] = e
            links[:mm, 2] = np.clip((np.minimum(stiff[e[:, 0]], stiff[e[:, 1]]) - 0.45) / 0.9, 0, 1) * self.alive[:mm]
        obs = np.zeros((cfg.max_obstacles, 4))
        for k, ob in enumerate(self.obstacles):
            if ob is not None:
                obs[k] = (*ob.pos, ob.radius)
        plate = np.clip((stiff - 0.85) / 0.65, 0, 1) * (0.25 + 0.75 * self.params["armor"])
        nrm = rel / np.maximum(rn, 1e-6)[:, None]
        lead = self.bodies[0]
        mstate = min(MATERIAL, key=lambda k: float(np.abs(np.array(MATERIAL[k]) - lead.mat).sum()))
        lure = np.zeros(4)
        if self.lure is not None:
            L = self.lure
            grow = 1.0 if L["state"] != "GONE" else max(0.0, L["until"] - self.t)
            lure = np.array([*L["pos"], L["r"] * grow])
        return ColonyState(self.t, lead.intent, SHAPES[int(np.argmax(lead.weights()))], self.x.copy(), radius,
                           np.ones((n, 3)), np.ones(n, np.int8), np.full(n, -1, np.int16), self.x.mean(0),
                           lead.heading, self.params.values(),
                           {"total": float(self.mass.sum()), "bodies": len(self._alive())}, self.surface, self.glow,
                           self.arousal, self.instab, list(self.events[-6:]), links, obs, disp, mstate,
                           self.fragments(), plate, nrm, self.own.copy(), len(self._alive()), lure)


__all__ = ["ColonyEngine", "ColonyConfig", "ColonyState", "SHAPES", "INTENTS", "EVENTS"]
