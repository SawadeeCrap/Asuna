"""The Mimetic line (creatures v9-v13): carbon-based mimetic polyalloy in five regimes.

One finite quantity of dense black artificial material (nodes + an adaptive graph + a local material
field, as in every Myrmex organism) that never switches bodies - it reorganises itself.  The five
organisms share the colony / hive machinery (flock split / merge, hardening waves, strikes, obstacles,
prey, the nanomachine swarm) and differ in the regions of morphology space they live in and in one
mechanism of their own:

* **Swarm** (``swarm``, v9) - distributed fluid-state flight: a dense head and streaming filaments,
  splits into several streams on the music and weaves back; SURGE bursts forward shedding a wake.
* **Spear** (``spear``, v10) - elongated, high-speed, directional: a needle with swept blades;
  DASH lunges along the axis on the kick, the material stiffens into the lance.
* **Cloud** (``cloud``, v11) - dispersion, camouflage, reassembly: a drifting cloud of clumps and
  flakes that gathers into a solid form on the drop and scatters again (SCATTER / GATHER).
* **Blade** (``blade``, v12) - high-velocity cutting precision: flame-like swept blades; zigzag
  slashes on the beat, SLASH lunges and flares the blades.
* **Crawler** (``crawler``, v13) - multi-contact terrain adaptation: a segmented body on 4-8 clawed
  legs walking on rough ground (feet on the terrain, body tilted with the slope); RECONFIGURE
  regrows it with another number of legs, POUNCE leaps.

The look in Blender: glossy black liquid metal with very dark red internal glints, thin tendons along
the skeleton and fins / filaments / flakes / claws that follow the motion (per organism).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .colony import G, MIMETIC_SET, ColonyConfig
from .hive import HiveConfig
from .osseous import OsseousColonyEngine, OsseousHiveEngine
from .polyalloy import MATERIAL

# ---------------------------------------------------------------------------- terrain (Crawler)
_TR = np.random.default_rng(20260925)
_RX, _RY = _TR.uniform(-40.0, 40.0, (2, 70))
_RR = _TR.uniform(0.45, 2.2, 70)
_RH = _TR.uniform(0.12, 0.85, 70) * np.where(_TR.random(70) < 0.8, 1.0, -0.5)     # rocks, a few hollows


def _terrain_raw(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    h = 0.35 * np.sin(0.21 * x + 0.7) * np.cos(0.17 * y - 0.4) + 0.18 * np.sin(0.43 * x - 0.31 * y + 1.3)
    d2 = (x[..., None] - _RX) ** 2 + (y[..., None] - _RY) ** 2
    return h + (_RH * np.exp(-d2 / _RR ** 2)).sum(-1)


_H0 = float(_terrain_raw(0.0, 0.0))


def terrain_height(x, y):
    """Rough ground of the Crawler: rolling hills + rocks (m); the same function builds the mesh in Blender."""
    return _terrain_raw(x, y) - _H0


# ---------------------------------------------------------------------------- behaviour plans
def _plan(**kw) -> dict:
    base = {"ENVELOP": (("ENVELOP",), "COHESIVE", 0.0), "PERCH": (("LEGS",), "ELASTIC", 0.25),
            "MERGE": (("CORE", "CLOUD"), "FLUID", 1.3), "STRUCTURE": (("CORE",), "HIGH_STIFFNESS", 0.0)}
    base.update(kw)
    return base


PLAN_SWARM = _plan(
    CRUISE=(("STREAM", "STREAM", "TENDRILS", "SPINDLE"), "FLUID", 1.6),
    HOVER=(("CLOUD", "STREAM", "CROWN"), "FLUID", 0.25),
    EXPLORE=(("STREAM", "CLOUD", "TENDRILS"), "FLUID", 1.0),
    DISPLAY=(("STREAM", "CROWN", "TENDRILS", "SPINE"), "ELASTIC", 0.6),
    EVADE=(("CLOUD", "STREAM"), "DISPERSED", 1.5),
    REFORM=(("CORE", "STREAM"), "COHESIVE", 0.8),
    HUNT=(("STREAM", "SPINE"), "ELASTIC", 1.9),
    STRIKE=(("STREAM", "SPINE", "CLAW"), "HIGH_STIFFNESS", 1.0),
    FORMATION=(("STREAM", "TENDRILS", "STREAM", "SPINDLE"), "FLUID", 1.6),
    PATROL=(("STREAM", "TENDRILS"), "ELASTIC", 1.2))
PLAN_SPEAR = _plan(
    CRUISE=(("LANCE", "LANCE", "SPINDLE", "SPINE"), "STRUCTURED", 1.5),
    HOVER=(("LANCE", "THORN", "CORE"), "ELASTIC", 0.1),
    EXPLORE=(("LANCE", "SPINDLE", "SCYTHE"), "ELASTIC", 0.9),
    DISPLAY=(("LANCE", "THORN", "SCYTHE", "CROWN"), "STRUCTURED", 0.3),
    EVADE=(("SPINDLE", "LANCE", "CLOUD"), "FLUID", 1.4),
    REFORM=(("CORE", "LANCE"), "COHESIVE", 0.6),
    HUNT=(("LANCE", "SPINE"), "STRUCTURED", 1.9),
    STRIKE=(("LANCE", "THORN"), "HIGH_STIFFNESS", 1.0),
    FORMATION=(("LANCE", "SPINDLE", "LANCE", "SPINE"), "STRUCTURED", 1.5),
    PATROL=(("LANCE", "SPINDLE"), "STRUCTURED", 1.2))
PLAN_CLOUD = _plan(
    CRUISE=(("SHARDS", "SHARDS", "CLOUD", "STREAM"), "DISPERSED", 0.7),
    HOVER=(("SHARDS", "CLOUD"), "DISPERSED", 0.1),
    EXPLORE=(("SHARDS", "CLOUD", "TENDRILS"), "FLUID", 0.5),
    DISPLAY=(("SHARDS", "CROWN", "RING", "SHIELD"), "FLUID", 0.3),
    EVADE=(("CLOUD", "SHARDS"), "DISPERSED", 1.0),
    REFORM=(("CORE", "SHIELD", "SPINDLE", "RING"), "COHESIVE", 0.6),
    HUNT=(("STREAM", "SHARDS"), "FLUID", 1.3),
    STRIKE=(("SPINE", "CLAW"), "HIGH_STIFFNESS", 1.0),
    FORMATION=(("SHARDS", "CLOUD", "SHARDS", "STREAM"), "DISPERSED", 0.8),
    PATROL=(("SHARDS", "CLOUD"), "FLUID", 0.8))
PLAN_BLADE = _plan(
    CRUISE=(("SWEEP", "SWEEP", "SCYTHE", "WINGS"), "ELASTIC", 1.9),
    HOVER=(("SWEEP", "CROWN", "SCYTHE"), "STRUCTURED", 0.15),
    EXPLORE=(("SWEEP", "SPINDLE"), "ELASTIC", 1.2),
    DISPLAY=(("SWEEP", "SCYTHE", "BLADES"), "STRUCTURED", 0.4),
    EVADE=(("SWEEP", "SPINDLE"), "FLUID", 1.8),
    REFORM=(("CORE", "SWEEP"), "COHESIVE", 0.7),
    HUNT=(("SWEEP", "SCYTHE"), "ELASTIC", 2.2),
    STRIKE=(("SWEEP", "SCYTHE", "CLAW"), "HIGH_STIFFNESS", 1.0),
    FORMATION=(("SWEEP", "WINGS", "SWEEP", "SCYTHE"), "ELASTIC", 1.9),
    PATROL=(("SWEEP", "SCYTHE"), "ELASTIC", 1.4))
_C = ("CRAWL",)
PLAN_CRAWL = _plan(
    CRUISE=(_C, "ELASTIC", 0.45), HOVER=(_C, "STRUCTURED", 0.0), EXPLORE=(_C, "ELASTIC", 0.3),
    DISPLAY=(_C, "STRUCTURED", 0.1), EVADE=(_C, "FLUID", 0.9), REFORM=(_C, "COHESIVE", 0.2),
    HUNT=(_C, "ELASTIC", 0.7), STRIKE=(_C, "HIGH_STIFFNESS", 1.0), FORMATION=(_C * 4, "ELASTIC", 0.45),
    PATROL=(_C, "ELASTIC", 0.4), PERCH=(_C, "ELASTIC", 0.0), MERGE=(_C, "FLUID", 0.6), STRUCTURE=(_C, "HIGH_STIFFNESS", 0.0))


class _Mimetic:
    """Shared by the five: their forms, visible tendons, a sudden forward burst."""
    SHAPE_SET = MIMETIC_SET
    OSS_BASE = 0.3                    # how readily the material shows its tendons

    def _oss_extra(self):
        return super()._oss_extra() + self.OSS_BASE

    def _bodies_forward(self, speed: float, shape: str | None = None, strength: float = 3.2) -> None:
        for k in self._alive():
            b = self.bodies[k]
            idx = self.own == k
            self.v[idx] += b.R()[:, 0] * speed
            if shape is not None and b.intent not in ("ENVELOP", "PERCH"):
                b.goal_shape(shape, strength)

    def _edge(self, key: str, value: float, level: float) -> bool:
        """Rising edge of an input over ``level``."""
        prev = getattr(self, "_prev_" + key, 0.0)
        setattr(self, "_prev_" + key, value)
        return value > level >= prev


# ---------------------------------------------------------------------------- v9 Swarm
@dataclass
class SwarmConfig(HiveConfig):
    max_links: int = 192
    cruise: float = 4.2


class SwarmEngine(_Mimetic, OsseousHiveEngine):
    STYLE = 3
    PLAN = PLAN_SWARM
    VOCAB = ("STREAM", "TENDRILS", "CLOUD", "SPINDLE", "SPINE", "CROWN", "CORE")
    SPLIT_ENERGY = 0.55               # the music splits it into streams easily
    SWARM_BIAS = 0.45
    EVENTS = OsseousHiveEngine.EVENTS + ("SURGE",)

    def __init__(self, cfg: HiveConfig | None = None):
        super().__init__(cfg or SwarmConfig())
        self.last_surge = -1e9

    def _apply_event(self, name: str, arg) -> None:
        if name == "SURGE":
            self._surge()
            return
        super()._apply_event(name, arg)

    def _surge(self) -> None:
        """A burst forward: every stream stretches into filaments, the machines are shed as a wake."""
        self._bodies_forward(7.0, "STREAM", 3.4)
        self.pulse = 1.0
        b = self.p_bound
        shed = b & (self.nrng.random(len(b)) < 0.45)
        back = np.array([-math.cos(self.bodies[0].heading), -math.sin(self.bodies[0].heading), 0.0])
        self.p_v[shed] += back * 4.0 + self.nrng.standard_normal((int(shed.sum()), 3)) * 1.2
        b[shed] = False
        self.p_free_t[shed] = 0.0
        self.last_surge = self.t
        self._log("SURGE", None)

    def _think(self, dt: float) -> None:
        super()._think(dt)
        inp = self.inp
        if self._edge("tr", inp.transient, 0.8) and inp.energy > 0.5 and self.t - self.last_surge > 5.0 and \
                self.ev_rng.chance(0.5):
            self._surge()


# ---------------------------------------------------------------------------- v10 Spear
@dataclass
class SpearConfig(ColonyConfig):
    max_links: int = 192
    cruise: float = 4.6


class SpearEngine(_Mimetic, OsseousColonyEngine):
    STYLE = 4
    PLAN = PLAN_SPEAR
    VOCAB = ("LANCE", "SPINDLE", "SPINE", "SCYTHE", "THORN", "CLAW", "CORE")
    SPLIT_ENERGY = 0.95               # one spear (a flock only on the hardest drops)
    EVENTS = OsseousColonyEngine.EVENTS + ("DASH",)

    def __init__(self, cfg: ColonyConfig | None = None):
        super().__init__(cfg or SpearConfig())
        self.last_dash = -1e9

    def _apply_event(self, name: str, arg) -> None:
        if name == "DASH":
            self._dash(float(arg) if isinstance(arg, (int, float)) else 1.0)
            return
        super()._apply_event(name, arg)

    def _dash(self, strength: float = 1.0) -> None:
        """A lunge along the axis: the material hardens into the lance and the body shoots forward."""
        self._bodies_forward(8.0 + 6.0 * strength, "LANCE", 3.6)
        for k in self._alive():
            self.bodies[k].mat_goal = np.array(MATERIAL["HIGH_STIFFNESS"])
        self.m[:, 1] = np.maximum(self.m[:, 1], 1.2)
        self.glow = 1.0
        self.last_dash = self.t
        self._log("DASH", None)

    def _think(self, dt: float) -> None:
        super()._think(dt)
        inp = self.inp
        if self._edge("tr", inp.transient, 0.6) and inp.energy > 0.4 and self.t - self.last_dash > 1.8:
            self._dash(inp.transient)


# ---------------------------------------------------------------------------- v11 Cloud
@dataclass
class CloudConfig(HiveConfig):
    max_links: int = 192
    cruise: float = 2.4


class CloudEngine(_Mimetic, OsseousHiveEngine):
    STYLE = 5
    PLAN = PLAN_CLOUD
    VOCAB = ("SHARDS", "CLOUD", "STREAM", "SHIELD", "CORE", "TENDRILS", "RING")
    OSS_BASE = 0.1
    SWARM_BIAS = 0.2
    EVENTS = OsseousHiveEngine.EVENTS + ("SCATTER", "GATHER")

    def __init__(self, cfg: HiveConfig | None = None):
        super().__init__(cfg or CloudConfig())
        self.gathered = False
        self.last_turn = 0.0
        self.last_phrase = None

    def _apply_event(self, name: str, arg) -> None:
        if name == "SCATTER":
            self._scatter()
            return
        if name == "GATHER":
            self._gather(arg if isinstance(arg, str) else None)
            return
        super()._apply_event(name, arg)

    def _scatter(self) -> None:
        """Dissolve into the cloud: cohesion drops, clumps drift apart, the machines fly out (camouflage)."""
        for k in self._alive():
            b = self.bodies[k]
            if b.intent not in ("ENVELOP",):
                self._set_intent(b, "EXPLORE", "SHARDS")
                b.mat_goal = np.array(MATERIAL["DISPERSED"])
        self.m[:, 0] *= 0.5
        b = self.p_bound
        out = b & (self.nrng.random(len(b)) < 0.8)
        self.p_v[out] += self._nrm[self.p_host[out]] * 2.5 + self.nrng.standard_normal((int(out.sum()), 3)) * 1.5
        b[out] = False
        self.p_free_t[out] = 0.0
        self.gathered, self.last_turn = False, self.t
        self._log("SCATTER", None)

    def _gather(self, shape: str | None = None) -> None:
        """Reassembly: the material pulls together into one solid form (the machines come home)."""
        forms = ("CORE", "SHIELD", "SPINDLE", "RING", "STREAM")
        form = shape or forms[self.rng.randint(0, len(forms) - 1)]
        for k in self._alive():
            b = self.bodies[k]
            if b.intent not in ("ENVELOP",):
                self._set_intent(b, "REFORM", form)
                b.mat_goal = np.array(MATERIAL["COHESIVE"])
        self.m[:, 0] = np.maximum(self.m[:, 0], 0.9)
        self.gathered, self.last_turn = True, self.t
        self._log("GATHER", form)

    def _think(self, dt: float) -> None:
        super()._think(dt)
        inp = self.inp
        phrase = int(inp.beat // 16.0) if inp.playing else int(self.t // 12.0)
        if phrase != self.last_phrase:                       # every phrase: gather on energy, scatter in calm
            first = self.last_phrase is None
            self.last_phrase = phrase
            if not first and self.t - self.last_turn > 4.0:
                if not self.gathered and (inp.energy > 0.55 or self.ev_rng.chance(0.35)):
                    self._gather()
                elif self.gathered and (inp.energy < 0.5 or self.ev_rng.chance(0.5)):
                    self._scatter()


# ---------------------------------------------------------------------------- v12 Blade
@dataclass
class BladeConfig(ColonyConfig):
    max_links: int = 192
    cruise: float = 4.8


class BladeEngine(_Mimetic, OsseousColonyEngine):
    STYLE = 6
    PLAN = PLAN_BLADE
    VOCAB = ("SWEEP", "SCYTHE", "WINGS", "BLADES", "CLAW", "SPINE", "CORE")
    SPLIT_ENERGY = 0.9
    EVENTS = OsseousColonyEngine.EVENTS + ("SLASH",)

    def __init__(self, cfg: ColonyConfig | None = None):
        super().__init__(cfg or BladeConfig())
        self.zig = 1.0
        self.last_beat = None
        self.last_slash = -1e9

    def _apply_event(self, name: str, arg) -> None:
        if name == "SLASH":
            self._slash()
            return
        super()._apply_event(name, arg)

    def _turn(self, angle: float) -> None:
        for k in self._alive():
            b = self.bodies[k]
            if b.intent not in ("ENVELOP", "PERCH", "MERGE"):
                b.heading += angle
                idx = self.own == k
                self.v[idx] += b.R()[:, 0] * 3.0

    def _slash(self) -> None:
        """A cut: lunge, the blades flare and harden, a hard turn."""
        self._bodies_forward(10.0, "SWEEP", 3.6)
        self.m[:, 1] = np.maximum(self.m[:, 1], 1.3)
        self.pulse = 1.0
        self.glow = 1.0
        self.zig = -self.zig
        self._turn(0.6 * self.zig)
        self.last_slash = self.t
        self._log("SLASH", None)

    def _think(self, dt: float) -> None:
        super()._think(dt)
        inp = self.inp
        beat = int(math.floor(inp.beat)) if inp.playing else None
        if beat is not None and beat != self.last_beat:        # zigzag slashes on the beat
            self.last_beat = beat
            if inp.energy > 0.55:
                self.zig = -self.zig
                self._turn(0.42 * self.zig * (0.6 + 0.6 * inp.energy))
        if self._edge("tr", inp.transient, 0.85) and self.t - self.last_slash > 3.0 and self.ev_rng.chance(0.4):
            self._slash()


# ---------------------------------------------------------------------------- v13 Crawler
@dataclass
class CrawlerConfig(ColonyConfig):
    max_links: int = 192
    cruise: float = 1.4


class CrawlerEngine(_Mimetic, OsseousColonyEngine):
    STYLE = 7
    PLAN = PLAN_CRAWL
    VOCAB = ("CRAWL",)
    OSS_BASE = 0.35
    SPLIT_ENERGY = 0.97
    EVENTS = OsseousColonyEngine.EVENTS + ("RECONFIGURE", "POUNCE")
    LEGS = (4, 6, 8)

    def __init__(self, cfg: ColonyConfig | None = None):
        super().__init__(cfg or CrawlerConfig())
        self.legs = 6
        self.last_phrase = None
        self.bodies[0].goal_shape("CRAWL", 2.6)
        lift = self._ground_at(self.bodies[0].P) + 0.8 - self.bodies[0].P[2]     # start standing on the ground
        self.x[:, 2] += lift
        for b in self.bodies:
            b.P[2] = self._ground_at(b.P) + 0.8

    # the ground
    def _ground_at(self, P: np.ndarray) -> float:
        return float(terrain_height(P[0], P[1]))

    def _floor(self, x: np.ndarray):
        return terrain_height(x[:, 0], x[:, 1]) + 0.04

    def _min_alt(self, b) -> float:
        return self._ground_at(b.P) + 0.3

    def _target_gain(self, T: np.ndarray):
        stance = T[:, 2] - terrain_height(T[:, 0], T[:, 1]) < 0.08          # a foot planted on the ground grips
        return np.where(stance, 6.0, 1.0)

    def _shape_ph(self, k, b, R, ph):
        P = b.P.copy()

        def ground_fn(xl, yl):                                  # local ground height under a local point
            wx = P[0] + R[0, 0] * xl + R[0, 1] * yl
            wy = P[1] + R[1, 0] * xl + R[1, 1] * yl
            return terrain_height(wx, wy) - P[2]
        return dict(ph, legs=self.legs, ground_fn=ground_fn)

    def _set_intent(self, b, intent: str, shape: str | None = None) -> None:
        super()._set_intent(b, intent, "CRAWL" if shape in (None, "LEGS") and intent != "ENVELOP" else shape)

    def _pick(self, prefs) -> str:
        return "CRAWL"

    def _v_des(self, k, b, lead, dt, pr, sb, base, alt_goal):
        v = super()._v_des(k, b, lead, dt, pr, sb, base, alt_goal)
        rear = 0.3 * sb if b.intent == "DISPLAY" else 0.0     # displaying: it rears up
        h = self._ground_at(b.P) + 0.55 * sb + rear
        v[2] = float(np.clip((h - b.P[2]) * 3.0, -6.0, 2.5))
        return v

    def _fly(self, k, idx, dt, pr):
        a_des, sb = super()._fly(k, idx, dt, pr)
        b = self.bodies[k]
        f = np.array([math.cos(b.heading), math.sin(b.heading)]) * 0.6 * sb
        slope = (self._ground_at(b.P[:2] + f) - self._ground_at(b.P[:2] - f)) / (1.2 * sb)
        b.pitch += (math.atan(slope) - b.pitch) * min(1.0, dt * 3.0)      # the body follows the slope
        over = b.P[2] - (self._ground_at(b.P) + 0.55 * sb)
        a_des = np.array(a_des, float)
        if over > 0.05:                                   # airborne (a leap, a blow): it falls, it is no flyer
            a_des[2] -= G * min(1.0, over / 0.4)
        elif over < 0.0:                                  # the legs carry it: a stiff spring and damper
            a_des[2] += -over * 30.0 - min(0.0, float(b.vel[2])) * 6.0
        return a_des, sb

    # reconfiguration
    def _apply_event(self, name: str, arg) -> None:
        if name == "RECONFIGURE":
            self._reconfigure(arg if isinstance(arg, int) else None)
            return
        if name == "POUNCE":
            self._pounce()
            return
        if name == "MORPHOLOGY_SHIFT":
            self._reconfigure()
        super()._apply_event(name, arg)

    def _reconfigure(self, legs: int | None = None) -> None:
        """Another number of legs: the material of the old ones flows into the new ones."""
        opts = [n for n in self.LEGS if n != self.legs]
        self.legs = legs if legs in self.LEGS else opts[self.rng.randint(0, len(opts) - 1)]
        self.pulse = 1.0
        self._log("RECONFIGURE", self.legs)

    def _pounce(self) -> None:
        self.last_pounce = self.t
        for k in self._alive():
            b = self.bodies[k]
            idx = self.own == k
            self.v[idx] += b.R()[:, 0] * 6.0 + np.array([0.0, 0.0, 3.5])
        self.m[:, 1] = np.maximum(self.m[:, 1], 1.2)
        self._log("POUNCE", None)

    def _think(self, dt: float) -> None:
        super()._think(dt)
        inp = self.inp
        phrase = int(inp.beat // 16.0) if inp.playing else int(self.t // 15.0)
        if phrase != self.last_phrase:
            first = self.last_phrase is None
            self.last_phrase = phrase
            if not first and self.ev_rng.chance(0.35 + 0.3 * inp.energy):
                self._reconfigure()
        if self._edge("tr", inp.transient, 0.9) and inp.energy > 0.6 and \
                self.t - getattr(self, "last_pounce", -1e9) > 4.0 and self.ev_rng.chance(0.12):
            self._pounce()


VARIANTS = {"swarm": (SwarmEngine, SwarmConfig), "spear": (SpearEngine, SpearConfig),
            "cloud": (CloudEngine, CloudConfig), "blade": (BladeEngine, BladeConfig),
            "crawler": (CrawlerEngine, CrawlerConfig)}
MIMETIC_EVENTS = ("SURGE", "DASH", "SCATTER", "GATHER", "SLASH", "RECONFIGURE", "POUNCE")

__all__ = ["SwarmEngine", "SpearEngine", "CloudEngine", "BladeEngine", "CrawlerEngine", "VARIANTS", "MIMETIC_EVENTS",
           "terrain_height", "PLAN_SWARM", "PLAN_SPEAR", "PLAN_CLOUD", "PLAN_BLADE", "PLAN_CRAWL"]
