"""Mimetic Polyalloy (creature v2): an airborne, finite, self-reconfiguring material.

Not a robot that transforms: a material that can temporarily become robot-like structures.

* 96 structural nodes with fixed mass (finite material - nothing is ever created);
* an adaptive elastic network: k-nearest-neighbour springs that break when over-stretched and
  re-form when the material becomes cohesive again (separation and recombination);
* material states (FLUID .. HIGH_STIFFNESS .. DISPERSED) are blends of cohesion, stiffness,
  damping, repulsion, persistence and dispersion - never visual modes;
* morphological attractors (CORE, SPINDLE, RING, SHIELD, BLADES, LATTICE, WINGS, CLOUD): every
  node owns a fixed material coordinate, each attractor maps it to a place, so the body *flows*
  between configurations; a latent vector with inertia chooses the blend (structures persist);
* flight: distributed thrust against gravity, orientation turns with an inertia computed from
  the actual mass distribution (a spread-out body turns slower);
* music makes physical events: a kick is an impulse, an approaching obstacle, a pressure wave or
  turbulence; a reactive controller answers with a (varied) strategy - local split and flow
  around, shield + stiffen, full dispersion, or a dodge - then reassembles and flies on.

Procedural prototype of the idea (no learning): the controller is hand-written, labelled as such.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..util.rng import RngStreams, stable_hash64
from .config import DEFAULT_PARAMS
from .control import CreatureControlInput, ParameterSet
from .puppet import GloveControl, GloveDriver
from .skeleton import Skeleton

ATTRACTORS = ("CORE", "SPINDLE", "RING", "SHIELD", "BLADES", "LATTICE", "WINGS", "CLOUD",
              # bony, aggressive forms (the style of bionic vertebrae, not their anatomy)
              "SPINE", "CLAW", "SCYTHE", "THORN", "CARAPACE")
AGGRESSIVE = ("SPINE", "CLAW", "SCYTHE", "THORN", "CARAPACE", "BLADES", "SHIELD")
CLASSIC = ATTRACTORS[:8]                  # the Mimetic Polyalloy's own vocabulary (the bony forms are v5+)
MATERIAL = {   # cohesion, stiffness, damping, repulsion, break ratio (persistence), dispersion
    "FLUID":          (0.55, 0.15, 0.35, 0.6, 1.6, 0.15),
    "ELASTIC":        (0.80, 0.50, 0.45, 0.7, 2.2, 0.05),
    "COHESIVE":       (1.00, 0.35, 0.60, 0.8, 1.9, 0.00),
    "STRUCTURED":     (0.90, 1.00, 0.70, 0.9, 3.0, 0.00),
    "HIGH_STIFFNESS": (0.90, 1.60, 0.80, 1.0, 4.0, 0.00),
    "DISPERSED":      (0.10, 0.05, 0.25, 1.3, 1.2, 0.90),
}
INTENTS = ("CRUISE", "HOVER", "EVADE", "REFORM", "EXPLORE", "DISPLAY", "STRIKE")
INTENT_PLAN = {  # preferred attractors, material state, cruise factor
    "CRUISE": (("SPINDLE", "WINGS", "LATTICE"), "ELASTIC", 1.0),
    "HOVER": (("CORE", "RING", "CLOUD"), "COHESIVE", 0.15),
    "EVADE": (("SPINDLE", "SHIELD", "CLOUD"), "FLUID", 1.0),
    "REFORM": (("CORE", "SPINDLE"), "COHESIVE", 0.7),
    "EXPLORE": (CLASSIC, "FLUID", 0.6),
    "DISPLAY": (("LATTICE", "BLADES", "RING", "SHIELD"), "STRUCTURED", 0.5),
}
INTENT_PLAN_BONE = {  # Osseous Polyalloy (v5): bony, aggressive vocabulary + the strike
    "CRUISE": (("SPINDLE", "SPINE", "WINGS", "SCYTHE", "LATTICE"), "ELASTIC", 1.0),
    "HOVER": (("CORE", "RING", "CARAPACE", "THORN", "CLOUD"), "COHESIVE", 0.15),
    "EVADE": (("SPINDLE", "SHIELD", "CARAPACE", "CLOUD"), "FLUID", 1.0),
    "REFORM": (("CORE", "SPINDLE", "SPINE"), "COHESIVE", 0.7),
    "STRIKE": (("CLAW", "SCYTHE", "SPINE"), "HIGH_STIFFNESS", 1.0),
    "EXPLORE": (ATTRACTORS, "FLUID", 0.6),
    "DISPLAY": (("LATTICE", "THORN", "SCYTHE", "CLAW", "CARAPACE", "BLADES"), "STRUCTURED", 0.5),
}
KICK_MODES = ("IMPULSE", "OBSTACLE", "PRESSURE", "TURBULENCE", "MIX")
G = 9.81


@dataclass
class PolyalloyConfig:
    seed: int = 0
    nodes: int = 96
    size: float = 1.6
    sim_rate: float = 120.0
    k_neighbors: int = 5
    max_links: int = 480
    max_obstacles: int = 4
    stage_radius: float = 14.0
    altitude: tuple[float, float] = (1.8, 5.5)
    cruise: float = 3.2
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))


_QUILLS = np.array([[math.cos(math.pi * (1 + 5 ** 0.5) * k) * math.sin(math.acos(1 - 2 * (k + 0.5) / 14)),
                      math.sin(math.pi * (1 + 5 ** 0.5) * k) * math.sin(math.acos(1 - 2 * (k + 0.5) / 14)),
                      1 - 2 * (k + 0.5) / 14] for k in range(14)])


def attractor_shape(name: str, U: np.ndarray, s: float, elong: float, pulse: float = 0.0,
                    snap: float = 0.0, swing: float = 0.0) -> np.ndarray:
    """Local target positions (x forward, z up) of every material coordinate for one attractor.

    ``pulse`` (kick envelope), ``snap`` (0 open .. 1 closed) and ``swing`` (-1..1) animate the bony forms.
    """
    u, v, w = U[:, 0], U[:, 1], U[:, 2]
    if name == "SPINE":                                    # a segmented ridge with swept-back dorsal thorns
        core = u < 0.62
        x = (u / 0.62 - 0.5) * 2.0 * s * elong
        a = 2 * np.pi * v
        rr = 0.1 * s * (1.0 + 0.5 * np.abs(np.sin(x / (0.18 * s) * np.pi)))          # knuckled, not smooth
        k = np.floor(v * 8.0)
        along = np.clip((u - 0.62) / 0.38, 0, 1)
        tx = (k / 7.0 - 0.5) * 1.7 * s * elong - along * 0.25 * s
        tz = 0.12 * s + along * (0.42 * s + 0.25 * s * pulse) * (1.0 - 0.4 * np.abs(k / 7.0 - 0.5))
        thorn = np.stack([tx, (w - 0.5) * 0.04 * s, tz], 1)
        return np.where(core[:, None], np.stack([x, rr * np.cos(a), rr * np.sin(a)], 1), thorn)
    if name == "CLAW":                                     # two hooked claws ahead of a compact core
        core = u < 0.34
        P = attractor_shape("CORE", U, s * 0.7, 1.0)
        side = np.where(v < 0.5, -1.0, 1.0)
        f = np.clip((u - 0.34) / 0.66, 0, 1)
        open_ = 1.0 - 0.75 * snap
        y = side * (0.28 * s + 0.42 * s * np.sin(np.pi * np.minimum(f, 0.8) / 0.8 * 0.5) * open_)
        hook = np.clip((f - 0.72) / 0.28, 0, 1)
        y = y - side * hook * 0.38 * s * open_
        x = 0.15 * s + f * 1.05 * s - hook ** 2 * 0.15 * s
        z = 0.18 * s * np.sin(np.pi * f) + (w - 0.5) * 0.05 * s
        return np.where(core[:, None], P - np.array([0.25 * s, 0, 0]), np.stack([x, y, z], 1))
    if name == "SCYTHE":                                   # curved blades sweeping back from the body
        k = np.floor(u * 4.0)
        phi = np.array([0.7, 2.44, 3.84, 5.58])[np.minimum(3, k).astype(int)] + 0.45 * swing
        f = v
        bx = -f * 1.05 * s * np.cos(1.25 * f) + 0.1 * s
        bz = 0.18 * s + f * 0.95 * s * np.sin(1.25 * f)
        th = (w - 0.5) * 0.05 * s * (1.0 - f)
        return np.stack([bx + th, -np.sin(phi) * bz, np.cos(phi) * bz], 1)
    if name == "THORN":                                    # a core bristling with quills that jump on kicks
        core = u < 0.38
        P = attractor_shape("CORE", U, s * 0.85, 1.0)
        d = _QUILLS[np.minimum(13, np.floor(v * 14.0)).astype(int)]
        along = 0.22 * s + np.clip((u - 0.38) / 0.62, 0, 1) * (0.62 * s + 0.55 * s * pulse)
        return np.where(core[:, None], P, d * along[:, None] + (w - 0.5)[:, None] * 0.02 * s)
    if name == "CARAPACE":                                 # an angular, faceted shell with a keel
        th, ph = 2 * np.pi * u, np.arccos(1 - 2 * v)
        d = np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], 1)
        d = d / np.maximum(np.abs(d).sum(1, keepdims=True), 1e-6)          # octahedral facets
        r = 0.62 * s * (0.85 + 0.15 * w)
        P = d * r[:, None] * np.array([1.5 * elong, 1.0, 0.8])
        P[:, 2] += np.maximum(0.0, 0.25 * s - np.abs(P[:, 1]) * 0.9) * (P[:, 2] > 0)      # dorsal keel
        return P
    if name == "CORE":
        r, th, ph = 0.34 * s * np.cbrt(w), 2 * np.pi * u, np.arccos(1 - 2 * v)
        return np.stack([r * np.sin(ph) * np.cos(th) * elong, r * np.sin(ph) * np.sin(th), r * np.cos(ph)], 1)
    if name == "SPINDLE":
        prof = np.sin(np.pi * u) ** 0.8 * 0.2 * s * np.sqrt(w)
        return np.stack([(u - 0.5) * 1.7 * s * elong, prof * np.cos(2 * np.pi * v), prof * np.sin(2 * np.pi * v)], 1)
    if name == "RING":
        R, tube, a, b = 0.55 * s, 0.09 * s * np.sqrt(w), 2 * np.pi * u, 2 * np.pi * v
        return np.stack([tube * np.sin(b), (R + tube * np.cos(b)) * np.cos(a), (R + tube * np.cos(b)) * np.sin(a)], 1)
    if name == "SHIELD":
        rr, a = 0.62 * s * np.sqrt(u), 2 * np.pi * v
        return np.stack([0.26 * s * (1 - (rr / (0.62 * s)) ** 2) + (w - 0.5) * 0.05 * s, rr * np.cos(a), rr * np.sin(a)], 1)
    if name == "BLADES":
        k = np.floor(u * 5.0)
        ang = 2 * np.pi * k / 5.0 + 0.3
        along = 0.12 * s + v * 0.85 * s
        thick = (w - 0.5) * 0.05 * s
        return np.stack([-v * 0.45 * s + thick, along * np.cos(ang), along * np.sin(ang)], 1)
    if name == "LATTICE":                                  # a spine with rib arcs: skeletal, mechanical-looking
        spine = u < 0.3
        x_sp = (u / 0.3 - 0.5) * 1.8 * s * elong
        rib = np.floor(v * 6.0)
        x_rib = (rib / 5.0 - 0.5) * 1.5 * s * elong
        side = np.where(w > 0.5, 1.0, -1.0)
        a = np.clip((u - 0.3) / 0.7, 0, 1) * np.pi * 0.9
        y = np.where(spine, 0.0, side * np.sin(a) * 0.45 * s)
        z = np.where(spine, 0.03 * s * np.sin(40 * u), 0.3 * s - np.cos(a) * 0.3 * s)
        return np.stack([np.where(spine, x_sp, x_rib), y, z], 1)
    if name == "WINGS":
        span = (u - 0.5) * 2.0
        x = (v - 0.5) * 0.4 * s * (1 - np.abs(span) * 0.6) - np.abs(span) * 0.3 * s
        return np.stack([x, span * 1.15 * s, (w - 0.5) * 0.04 * s + 0.14 * s * span ** 2], 1)
    # CLOUD
    r, th, ph = 1.15 * s * np.cbrt(w), 2 * np.pi * u, np.arccos(1 - 2 * v)
    return np.stack([r * np.sin(ph) * np.cos(th), r * np.sin(ph) * np.sin(th), 0.7 * r * np.cos(ph)], 1)


def _knn_edges(P: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    i = np.repeat(np.arange(len(P)), k)
    j = nn.ravel()
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    pairs = np.unique(np.stack([lo, hi], 1), axis=0)
    return pairs, d[pairs[:, 0], pairs[:, 1]]


@dataclass
class Obstacle:
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    born: float
    handled: bool = False


@dataclass
class PolyalloyState:
    t: float
    behavior: str
    morphology: str
    pos: np.ndarray
    radius: np.ndarray
    stretch: np.ndarray
    kind: np.ndarray
    anchor: np.ndarray
    com: np.ndarray
    heading: float
    params: dict
    volumes: dict
    surface: float
    glow: float
    arousal: float
    instability: float
    events: list
    links: np.ndarray            # (max_links, 3) i, j, strength (i = -1 unused)
    obstacles: np.ndarray        # (max_obstacles, 4) x, y, z, radius (radius 0 unused)
    dispersion: np.ndarray       # (N,)
    material: str = "COHESIVE"
    fragments: int = 1
    style: int = 0               # 0 classic (struts, plates) · 1 osseous (bone links, scutes)


class PolyalloyEngine:
    EVENTS = ("MORPHOLOGY_SHIFT", "MASS_REBALANCE", "APPENDAGE_BURST", "COLLAPSE", "RECONSTRUCTION", "IMPULSE",
              "OBSTACLE", "PRESSURE", "TURBULENCE")
    BONY = False                 # the Osseous subclass (v5) switches on bone links, scutes, strikes
    PLAN = INTENT_PLAN
    VOCAB = CLASSIC

    def __init__(self, cfg: PolyalloyConfig | None = None):
        self.cfg = cfg = cfg or PolyalloyConfig()
        self.rs = RngStreams(cfg.seed)
        self.rng = self.rs.stream("poly")
        self.nrng = np.random.default_rng(stable_hash64("poly-noise", cfg.seed) & 0xFFFFFFFF)
        self.ev_rng = self.rs.stream("poly-events")
        self.params = ParameterSet(cfg.params)
        self.inp = CreatureControlInput()
        self.glove = GloveDriver()                     # a hand (Hand Glove) holding the body
        n = self.n = cfg.nodes
        r = self.rs.stream("poly-shape")
        self.U = np.array([[r.random(), r.random(), r.random()] for _ in range(n)])   # material coordinates
        self.mass = np.full(n, 1.0 / n)                        # finite material: fixed forever
        self.t = 0.0
        self.P = np.array([0.0, 0.0, 3.0])
        self.heading, self.pitch, self.yaw_rate = r.uniform(-math.pi, math.pi), 0.0, 0.0
        self.z = np.zeros(len(ATTRACTORS))                     # morphology latent (logits)
        self.z_goal = np.zeros(len(ATTRACTORS))
        self.z_goal[ATTRACTORS.index("SPINDLE")] = 2.0
        self.mat = np.array(MATERIAL["COHESIVE"], float)
        self.mat_goal = self.mat.copy()
        self.intent, self.intent_t, self.intent_dwell = "CRUISE", 0.0, 6.0
        R = self._R()
        w0 = np.array([1.0 if a in self.VOCAB else 0.0 for a in ATTRACTORS])
        self.x = self.P + self._targets(R, w0 / w0.sum()) @ np.eye(3)
        self.v = np.zeros((n, 3))
        self.edges, self.rest = _knn_edges(self.x, cfg.k_neighbors)
        self.alive = np.ones(len(self.edges), bool)
        self.local = np.zeros(n)                               # local dispersion (threat corridors)
        self.obstacles: list[Obstacle | None] = [None] * cfg.max_obstacles   # stable slots (render objects)
        self.turb_until = -1.0
        self.last_kick = -1e9
        self.last_obstacle = -1e9
        self.last_rebuild = 0.0
        self.arousal, self.instab, self.glow, self.surface = 0.2, 0.0, 0.0, 0.3
        self.response_hist: list[str] = []
        self.events: list[tuple[float, str, object]] = []
        self._pending: list[tuple[str, object]] = []
        self.wander = r.uniform(-1, 1)
        # Bone: ossification (0 liquid .. 1 bone), the skeleton's growing links, strikes.
        self.oss = 0.3
        self.pulse = 0.0
        self.hn = np.array([(k * 0.7548776662) % 1.0 for k in range(n)])
        self.skel = Skeleton(cfg.max_links)
        self._state_t = 0.0
        self.strike_target = None
        self.sculpt: tuple | None = None

    # ------------------------------------------------------------------ API (same as CreatureEngine)
    def set_input(self, inp: CreatureControlInput) -> None:
        self.inp = inp.sanitized()

    def set_parameter(self, name: str, value) -> bool:
        return self.params.set(name, value)

    def set_glove(self, ctrl: GloveControl | None, shapes: tuple | None = None) -> None:
        """Direct control by a hand: rotation, fingers, position (see creature/puppet.py)."""
        self.glove.set(ctrl)
        self.sculpt = tuple(x for x in (shapes or ()) if x in ATTRACTORS) or None

    def trigger_event(self, name: str, arg=None) -> bool:
        if name.upper() not in self.EVENTS:
            return False
        self._pending.append((name.upper(), arg))
        return True

    # ------------------------------------------------------------------ helpers
    def _R(self) -> np.ndarray:
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Rz = np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1.0]])
        Ry = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
        return Rz @ Ry

    def _weights(self) -> np.ndarray:
        e = np.exp(3.0 * (self.z - self.z.max()))
        return e / e.sum()

    def _targets(self, R: np.ndarray, w: np.ndarray) -> np.ndarray:
        pr = self.params
        s = self.cfg.size * (0.75 + 0.5 * pr["expansion"] - 0.3 * pr["contraction"])
        elong = 0.8 + 0.6 * pr["density"]
        loc = np.zeros((self.n, 3))
        for a, wa in zip(ATTRACTORS, w):
            if wa > 0.01:
                shp = attractor_shape(a, self.U, s, elong, getattr(self, "pulse", 0.0))
                loc += wa * (shp - shp.mean(0))            # thrust places the body, not the shape
        asym = pr["asymmetry"]
        loc[:, 1] *= 1.0 + 0.35 * asym * np.sign(loc[:, 1]) * math.sin(0.11 * self.t + 1.0)
        loc = self.glove.local(loc)                    # fingers / scale of a hand, if one is holding it
        return loc @ R.T

    def _log(self, name, arg=None):
        self.events.append((self.t, name, arg))
        self.events = self.events[-60:]

    def _spawn_obstacle(self, strength: float) -> None:
        cfg, r = self.cfg, self.ev_rng
        if None not in self.obstacles:
            return
        slot = self.obstacles.index(None)
        fwd = self._R()[:, 0]
        side = np.cross([0, 0, 1.0], fwd)
        dirs = [fwd, fwd + side, fwd - side, side, -side, fwd + np.array([0, 0, 0.8]), fwd - np.array([0, 0, 0.6])]
        d = dirs[int(r.randint(0, len(dirs) - 1))]
        d = d / np.linalg.norm(d)
        speed = r.uniform(6.0, 12.0) * (0.7 + 0.5 * strength)
        vcom = self.v.mean(axis=0)
        dist = r.uniform(9.0, 13.0)
        start = self.P + d * dist
        tt = dist / (speed + max(0.0, float(vcom @ d)))
        aim = self.P + vcom * tt + np.array([r.normal(0, 0.25), r.normal(0, 0.25), r.normal(0, 0.2)])
        vel = (aim - start) / max(tt, 0.2)
        self.obstacles[slot] = Obstacle(start, vel, r.uniform(0.35, 1.0) * cfg.size * 0.7, self.t)
        self._log("OBSTACLE", round(float(np.linalg.norm(vel)), 1))

    def _kick(self, strength: float) -> None:
        mode = KICK_MODES[min(4, int(self.params["kick_mode"] * 5))]
        if mode == "MIX":
            mode = KICK_MODES[self.ev_rng.weighted_index([1.0, 1.4 + self.params["obstacle_rate"], 0.7, 0.6])]
        # An obstacle needs time to be met, answered and recovered from: meanwhile kicks are impulses.
        if mode == "OBSTACLE":
            if self.t - self.last_obstacle < 2.0 + 4.0 * (1.0 - self.params["obstacle_rate"]):
                mode = "IMPULSE"
            else:
                self.last_obstacle = self.t
        self._pending.append((mode, strength))

    # ------------------------------------------------------------------ events
    def _apply_event(self, name: str, arg) -> None:
        r = self.ev_rng
        s = float(arg) if isinstance(arg, (int, float)) else 1.0
        if name == "IMPULSE":
            c = self.P + np.array([r.normal(0, 1), r.normal(0, 1), r.normal(0, 0.6)]) * self.cfg.size * 0.8
            d = self.x - c
            dist = np.linalg.norm(d, axis=1, keepdims=True)
            self.v += 4.0 * s * np.clip(1.5 - dist / self.cfg.size, 0, 1) * d / np.maximum(dist, 1e-3)
        elif name == "OBSTACLE":
            self._spawn_obstacle(s)
            return
        elif name == "PRESSURE":
            self.v -= 2.2 * s * (self.x - self.x.mean(axis=0))
        elif name == "TURBULENCE":
            self.turb_until = self.t + 0.7
        elif name == "MORPHOLOGY_SHIFT":
            self.z_goal = np.zeros(len(ATTRACTORS))
            self.z_goal[ATTRACTORS.index(arg) if arg in self.VOCAB else
                        ATTRACTORS.index(self.VOCAB[int(r.randint(0, len(self.VOCAB) - 1))])] = 2.5
        elif name in ("COLLAPSE",):
            self.mat_goal = np.array(MATERIAL["DISPERSED"])
            self.alive[:] = False
            self.intent, self.intent_t = "REFORM", -2.0
        elif name in ("RECONSTRUCTION", "MASS_REBALANCE"):
            self.mat_goal = np.array(MATERIAL["COHESIVE"])
            self.intent, self.intent_t = "REFORM", 0.0
        elif name == "STRIKE":                              # a lunge: harden, claws out, forward
            self._strike(None)
        elif name == "OSSIFY":
            self.oss = 1.0
            self.mat_goal = np.array(MATERIAL["STRUCTURED"])
        elif name == "APPENDAGE_BURST":
            self.z_goal = np.zeros(len(ATTRACTORS))
            self.z_goal[ATTRACTORS.index(self._pick(("SCYTHE", "BLADES", "THORN")) if self.BONY else "BLADES")] = 2.5
            self.mat_goal = np.array(MATERIAL["STRUCTURED"])
        self._log(name, arg if isinstance(arg, str) else None)

    def _pick(self, prefs) -> str:
        """A shape from the intent's vocabulary; aggression favours the bony, aggressive ones."""
        if not self.BONY:
            return prefs[self.rng.randint(0, len(prefs) - 1)]
        a = self.params["aggression"]
        w = [(0.5 + 1.8 * a) if p in AGGRESSIVE else (1.3 - 0.8 * a) for p in prefs]
        return prefs[self.rng.weighted_index(w)]

    def _strike(self, target) -> None:
        self.intent, self.intent_t, self.strike_target = "STRIKE", 0.0, target
        self.z_goal = np.zeros(len(ATTRACTORS))
        self.z_goal[ATTRACTORS.index(self._pick(self.PLAN["STRIKE"][0]))] = 3.0
        self.mat_goal = np.array(MATERIAL["HIGH_STIFFNESS"])
        self.oss = 1.0
        fwd = self._R()[:, 0]
        self.v += fwd * 5.0 if target is None else 0.0
        self._log("STRIKE", None)

    # ------------------------------------------------------------------ behaviour (hand-written controller)
    def _think(self, dt: float) -> None:
        pr, inp, r = self.params.values(), self.inp, self.rng
        self.arousal += ((0.75 * inp.energy + 0.4 * inp.transient + 0.2 * inp.spectral_flux) * pr["reactivity"] +
                         0.1 - self.arousal) * min(1.0, dt / 2.0)
        self.instab = min(1.5, self.instab + dt * (0.3 * inp.spectral_flux + 0.1 * pr["instability"]))
        if inp.transient > 0.5 and self.t - self.last_kick > 0.25:
            self.last_kick = self.t
            self._kick(inp.transient)
        if self.ev_rng.chance(dt * pr["obstacle_rate"] * 0.25 * (0.3 + inp.energy)):
            self._spawn_obstacle(0.6)
        # Threat assessment: time and distance of closest approach of every obstacle.
        threat = None
        vcom = self.v.mean(axis=0)
        for ob in self.obstacles:
            if ob is None:
                continue
            rel, rv = ob.pos - self.P, ob.vel - vcom
            tca = -float(rel @ rv) / max(float(rv @ rv), 1e-6)
            dca = float(np.linalg.norm(rel + rv * max(tca, 0.0)))
            if 0.0 < tca < 1.3 and dca < ob.radius + 0.8 * self.cfg.size and not ob.handled:
                threat = (ob, tca)
        self.intent_t += dt
        if threat is not None and self.intent not in ("EVADE", "STRIKE"):
            ob, tca = threat
            ob.handled = True
            # The same event must not always produce the same answer: memory + state choose.
            opts = ["SPLIT", "SHIELD", "DISPERSE", "DODGE"] + (["STRIKE"] if self.BONY else [])
            w = [1.4 * pr["fluidity"] + 0.3, 0.6 + pr["rigidity"] + 0.5 * pr["aggression"], 0.4 + 0.8 * pr["instability"],
                 0.6 + pr["speed"]] + ([0.3 + 1.6 * pr["aggression"]] if self.BONY else [])
            for k, o in enumerate(opts):
                if o in self.response_hist[-2:]:
                    w[k] *= 0.3
            self.response = opts[r.weighted_index(w)]
            self.response_hist.append(self.response)
            self.threat, self.intent, self.intent_t = ob, "EVADE", 0.0
            self._log("RESPONSE", self.response)
            if self.response == "STRIKE":                    # meet it: harden into blades and hit it away
                self._strike(ob)
            elif self.response == "SHIELD":
                self.z_goal = np.zeros(len(ATTRACTORS))
                self.z_goal[ATTRACTORS.index("SHIELD")] = 3.0
                self.mat_goal = np.array(MATERIAL["HIGH_STIFFNESS"])
            elif self.response == "DISPERSE":
                self.mat_goal = np.array(MATERIAL["DISPERSED"])
            elif self.response == "DODGE":
                side = np.cross([0, 0, 1.0], ob.vel)
                side = side / max(np.linalg.norm(side), 1e-6) * (1 if r.chance(0.5) else -1)
                self.v += side * 3.0
                self.z_goal = np.zeros(len(ATTRACTORS))
                self.z_goal[ATTRACTORS.index("SPINDLE")] = 3.0
            else:
                self.mat_goal = np.array(MATERIAL["FLUID"])
        elif self.intent in ("EVADE", "STRIKE") and self.intent_t > (1.3 if self.intent == "STRIKE" else 1.6):
            self.strike_target = None
            self.intent, self.intent_t = "REFORM", 0.0
            self.mat_goal = np.array(MATERIAL["COHESIVE"])
            self._log("REASSEMBLY", None)
        elif self.intent_t > self.intent_dwell and self.intent not in ("EVADE", "STRIKE"):
            pool = ["CRUISE", "HOVER", "EXPLORE", "DISPLAY"]
            w = [1.2 + inp.energy, 0.8 * (1 - inp.energy), 0.5 + pr["mutation"] + self.instab,
                 0.4 + pr["rigidity"] + pr["tendril_activity"]]
            if self.intent in pool:
                w[pool.index(self.intent)] *= 0.4
            self.intent = pool[r.weighted_index(w)]
            self.intent_t, self.intent_dwell = 0.0, r.uniform(4.0, 10.0) * (0.6 + 0.8 * pr["coherence"])
            prefs, mstate, _ = self.PLAN[self.intent]
            self.z_goal = np.zeros(len(ATTRACTORS))
            self.z_goal[ATTRACTORS.index(self._pick(prefs))] = 2.5
            self.mat_goal = np.array(MATERIAL[mstate])
            self._log("MORPHOLOGY_SHIFT", self.intent)
        if self.instab > 1.0:
            self.instab = 0.0
            self._pending.append(("MORPHOLOGY_SHIFT", None))

    # ------------------------------------------------------------------ simulation
    def update(self, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            return self.state()
        steps = max(1, min(12, int(round(dt * self.cfg.sim_rate))))
        for _ in range(steps):
            self._step(dt / steps)
        return self.state()

    def _step(self, dt: float) -> None:
        self.t += dt
        cfg, pr, inp = self.cfg, self.params.values(), self.inp
        self._think(dt)
        while self._pending:
            self._apply_event(*self._pending.pop(0))
        # Morphological inertia: latent and material state move with time constants, never jump.
        gc = self.glove.ctrl
        if gc.active:                                  # the hand: sculpting, material, energy
            if gc.finger_mode == "morph" and self.sculpt:
                self.z_goal = np.zeros(len(ATTRACTORS))
                for shp, ex in zip(self.sculpt, gc.fingers):
                    self.z_goal[ATTRACTORS.index(shp)] = max(self.z_goal[ATTRACTORS.index(shp)], 0.4 + 2.4 * ex)
            if gc.material is not None:
                a, b = ("ELASTIC", "HIGH_STIFFNESS") if gc.material >= 0 else ("ELASTIC", "FLUID")
                f = abs(gc.material)
                self.mat_goal = (1 - f) * np.array(MATERIAL[a]) + f * np.array(MATERIAL[b])
                if self.BONY and gc.material > 0.5:
                    self.oss = max(self.oss, gc.material)
            if gc.energy is not None:
                self.arousal = max(self.arousal, gc.energy)
        tau_m = 0.8 + 3.0 * pr["rigidity"] * (1.2 - pr["fluidity"])
        if gc.active and gc.finger_mode == "morph":
            tau_m = 0.25                               # the form follows the fingers at once
        noise = self.nrng.standard_normal(len(ATTRACTORS)) * pr["mutation"] * 0.6
        self.z += (self.z_goal - self.z) * min(1.0, dt / tau_m) + noise * math.sqrt(dt)
        self.mat += (self.mat_goal - self.mat) * min(1.0, dt / (0.4 + 0.8 * pr["coherence"]))
        self.pulse = max(self.pulse * math.exp(-dt / 0.18), inp.transient)
        oss_goal = float(np.clip(0.2 + 0.55 * max(0.0, self.mat[1] - 0.3) + 0.35 * pr["aggression"] +
                                 0.25 * pr["rigidity"] - 0.9 * self.mat[5], 0.0, 1.0))
        self.oss += (oss_goal - self.oss) * min(1.0, dt / 1.5)
        if inp.transient > 0.5:
            self.oss = min(1.0, self.oss + 0.4 * dt * pr["aggression"])
        coh, stiff, damp, rep, brk, disp = self.mat
        coh *= (0.5 + pr["coherence"]) * (1.0 + 0.8 * gc.grip if gc.active else 1.0)
        stiff *= 0.4 + 1.2 * pr["rigidity"]
        disp = min(1.0, disp + 0.3 * pr["fluidity"] * pr["noise"])
        # Flight: desired velocity (cruise along a wandering heading, altitude band), distributed thrust.
        vcom = self.v.mean(axis=0)
        self.P = (self.x * self.mass[:, None]).sum(0) / self.mass.sum()
        cruise = cfg.cruise * self.PLAN.get(self.intent, (0, 0, 1.0))[2] * (0.4 + 1.2 * pr["speed"]) * (0.6 + self.arousal)
        self.wander += self.rng.normal(0, 1) * math.sqrt(dt) * (0.4 + pr["noise"])
        self.wander *= math.exp(-dt * 0.3)
        home = -self.P[:2]
        dist = float(np.linalg.norm(home))
        want = self.heading + 0.5 * self.wander * dt * 3 + (gc.offset[0] * 1.6 * dt if gc.active else 0.0)
        if dist > 0.7 * cfg.stage_radius:
            want_home = math.atan2(home[1], home[0])
            want += ((want_home - self.heading + math.pi) % (2 * math.pi) - math.pi) * min(1.0, dt * 1.2)
        lo, hi = cfg.altitude
        alt_goal = lo + (hi - lo) * pr["altitude"]
        if gc.active:
            alt_goal = float(np.clip(alt_goal + 2.2 * gc.offset[1], 0.8, 12.0))
        vz = np.clip((alt_goal - self.P[2]) * 0.8 + 0.3 * math.sin(0.4 * self.t), -1.5, 1.5)
        v_des = np.array([math.cos(want) * cruise, math.sin(want) * cruise, vz])
        if self.intent == "STRIKE":
            ob = self.strike_target
            if ob is not None:                              # lunge where it will be
                aim = ob.pos + ob.vel * 0.25 - self.P
                v_des = aim / max(float(np.linalg.norm(aim)), 1e-6) * (cfg.size * 7.0)
            else:
                v_des = self._R()[:, 0] * max(cruise, 1.0) * 2.5
        a_des = (v_des - vcom) / 0.8 + np.array([0.0, 0.0, G])
        # Moment of inertia of the current body limits the turn rate (morphology changes flight).
        I = float((self.mass * ((self.x - self.P) ** 2).sum(1)).sum()) / max(cfg.size ** 2, 1e-6)
        sp = float(np.linalg.norm(vcom[:2]))
        if sp > 0.3:
            yaw_goal = math.atan2(vcom[1], vcom[0])
            dy = (yaw_goal - self.heading + math.pi) % (2 * math.pi) - math.pi
            self.yaw_rate += dt * (3.0 * dy - 2.2 * self.yaw_rate) / (0.3 + 3.0 * I)
        self.heading += dt * self.yaw_rate
        self.pitch += (math.atan2(vcom[2], max(sp, 0.5)) * 0.6 - self.pitch) * min(1.0, dt * 2.0)
        R = self._R()
        # The hand turns the body: every node rigidly with it, and the shape targets too (no lag).
        dR = self.glove.begin(dt)
        self.glove.rigid(self.x, self.v, self.P, R @ dR @ R.T)
        R = R @ self.glove.G
        # Targets from the blended attractors + bass pressure + breathing.
        w = self._weights()
        pressure = 0.25 * inp.bass * (0.5 + pr["expansion"]) + 0.03 * math.sin(2 * math.pi * inp.beat / 4.0)
        T = self.P + self._targets(R, w) * (1.0 + pressure)
        x, v = self.x, self.v
        f_t = (2 * math.pi * (0.5 + 1.2 * stiff)) ** 2 * 0.25
        local = self.local
        acc = (coh * (1.0 - local))[:, None] * f_t * (T - x) - (1.5 + 3.0 * damp) * (v - vcom)
        acc += a_des - np.array([0.0, 0.0, G])                 # thrust balances gravity at the CoM
        # Network springs (break when over-stretched: separation; re-form later: recombination).
        e, al = self.edges, self.alive
        if al.any():
            i, j = e[al, 0], e[al, 1]
            d = x[j] - x[i]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            rest = self.rest[al]
            ks = 60.0 * stiff * (1.0 - 0.8 * np.maximum(local[i], local[j]))
            f = (ks * (L - rest) / L)[:, None] * d + 4.0 * damp * (v[j] - v[i])
            np.add.at(acc, i, f)
            np.add.at(acc, j, -f)
            broken = L > rest * brk * (1.0 - 0.4 * np.maximum(local[i], local[j]))
            if broken.any():
                idx = np.nonzero(al)[0][broken]
                self.alive[idx] = False
        # Short-range repulsion keeps the material from collapsing into a point.
        D = x[:, None, :] - x[None, :, :]
        dist2 = (D * D).sum(2) + np.eye(self.n)
        rmin = 0.12 * cfg.size
        close = dist2 < rmin * rmin
        if close.any():
            invd = 1.0 / np.sqrt(dist2)
            push = np.where(close, (rmin * invd - 1.0), 0.0)
            acc += 8.0 * rep * (push[:, :, None] * D).sum(1)
        # Dispersion noise, turbulence, high-frequency shimmer.
        jitter = self.nrng.standard_normal((self.n, 3))
        acc += jitter * (2.5 * disp + 3.0 * local[:, None] + 1.2 * inp.high * pr["surface_activity"])
        if self.t < self.turb_until:
            acc += 6.0 * np.stack([np.sin(3.1 * x[:, 1] + 5 * self.t), np.sin(2.7 * x[:, 2] + 4 * self.t),
                                   np.sin(2.3 * x[:, 0] + 6 * self.t)], 1)
        # Obstacles: kinematic spheres; the material is pushed out and flows around.
        for slot, ob in enumerate(self.obstacles):
            if ob is None:
                continue
            ob.pos = ob.pos + ob.vel * dt
            rel = x - ob.pos
            dd = np.linalg.norm(rel, axis=1)
            inside = dd < ob.radius
            if inside.any():
                nrm = rel[inside] / np.maximum(dd[inside], 1e-6)[:, None]
                x[inside] = ob.pos + nrm * ob.radius
                vn = (v[inside] * nrm).sum(1, keepdims=True)
                v[inside] -= np.minimum(vn, 0) * nrm * 1.6
                self.glow = max(self.glow, 0.8)
                if self.intent == "STRIKE" and not getattr(ob, "hit", False):    # the blow knocks it away
                    away = ob.pos - self.P
                    away /= max(float(np.linalg.norm(away)), 1e-6)
                    ob.vel = away * (10.0 + 8.0 * self.params["aggression"]) + ob.vel * 0.15
                    ob.hit = True
                    self.glow = 1.0
                    self._log("HIT", None)
            # Threat corridor: material in the obstacle's path loosens (local dispersion) to let it through.
            if self.intent == "EVADE" and getattr(self, "response", "") in ("SPLIT", "DISPERSE"):
                ahead = rel - ob.vel * (rel @ ob.vel)[:, None] / max(float(ob.vel @ ob.vel), 1e-6)
                corridor = np.linalg.norm(ahead, axis=1) < ob.radius + 0.35 * cfg.size
                local[corridor] = np.minimum(1.0, local[corridor] + dt * 4.0)
                side = ahead / np.maximum(np.linalg.norm(ahead, axis=1), 1e-6)[:, None]
                acc[corridor] += 9.0 * side[corridor]
            if float(np.linalg.norm(ob.pos - self.P)) > 25.0 or self.t - ob.born > 8.0:
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
        # Recombination: cohesive material re-forms its network (the structure it has *now* is remembered).
        if coh > 0.7 and self.t - self.last_rebuild > 0.6 and (~self.alive).mean() > 0.25:
            self.edges, self.rest = _knn_edges(x, cfg.k_neighbors)
            self.alive = np.ones(len(self.edges), bool)
            self.last_rebuild = self.t
        self.glow *= math.exp(-dt / 0.4)
        self.surface += (min(1.0, pr["surface_activity"] + 0.5 * inp.high + 0.5 * disp) - self.surface) * min(1.0, dt * 3)

    # ------------------------------------------------------------------ state
    def fragments(self) -> int:
        parent = list(range(self.n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for (i, j) in self.edges[self.alive]:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
        return len({find(a) for a in range(self.n)})

    def state(self) -> PolyalloyState:
        cfg = self.cfg
        n = self.n
        disp = np.clip(np.linalg.norm(self.x - self.P, axis=1) / (1.2 * cfg.size) - 0.5, 0, 1) * 0.6 + 0.4 * self.local
        r0 = 0.62 * cfg.size * (1.0 / n) ** (1 / 3) * 1.9
        if self.BONY:
            # Ossified material thins into beads along its skeleton: the bone links show.
            oss_n = np.clip(self.oss * (0.65 + 0.35 * self.hn) * (1.0 - disp), 0.0, 1.0)
            radius = r0 * (1.0 - 0.45 * disp) * (1.0 - 0.38 * oss_n)
            dt_s, self._state_t = max(0.0, self.t - self._state_t), self.t
            links = self.skel.update(self.x, np.zeros(n, int), oss_n, dt_s, 0.5 * cfg.size)
        else:
            stiff = float(self.mat[1])
            # Hardened material thins into beads along its internal frame: the struts show.
            radius = r0 * (1.0 - 0.45 * disp) * (1.0 - 0.3 * min(1.0, max(0.0, (stiff - 0.5) / 1.1)))
            links = np.full((cfg.max_links, 3), -1.0)
            links[:, 2] = 0.0
            m = min(cfg.max_links, len(self.edges))
            links[:m, 0:2] = self.edges[:m]
            links[:m, 2] = min(1.0, max(0.0, (stiff - 0.4) / 1.0)) * self.alive[:m]
        obs = np.zeros((cfg.max_obstacles, 4))
        for k, ob in enumerate(self.obstacles):
            if ob is not None:
                obs[k] = (*ob.pos, ob.radius)
        mstate = min(MATERIAL, key=lambda k: float(np.abs(np.array(MATERIAL[k]) - self.mat).sum()))
        w = self._weights()
        return PolyalloyState(self.t, self.intent, ATTRACTORS[int(np.argmax(w))], self.x.copy(), radius,
                              np.ones((n, 3)), np.ones(n, np.int8), np.full(n, -1, np.int16), self.P.copy(),
                              self.heading, self.params.values(), {"total": float(self.mass.sum())},
                              self.surface, self.glow, self.arousal, self.instab, list(self.events[-6:]),
                              links, obs, disp, mstate, self.fragments(), int(self.BONY))


__all__ = ["PolyalloyEngine", "PolyalloyConfig", "PolyalloyState", "ATTRACTORS", "MATERIAL", "INTENTS"]
