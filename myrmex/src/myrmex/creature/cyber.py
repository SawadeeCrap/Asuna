"""Cyber Hive (creature v8): the Osseous Hive (v7) rebuilt as a hi-tech machine organism.

Same physics, behaviour and features as v7 (flock, structures, nanomachine swarm, pattern, memory,
strikes, quills), but the material is a white nanomaterial and the organism is a machine rather than
an animal:

* **Rails instead of bones**: the skeleton through each body is drawn as hexagonal white modules with
  two light lines along them; hard material thins into beads strung along the rails.
* **Hex panels instead of scutes**: flat white tiles aligned with the direction of travel, each with a
  light ring inset.
* **Light**: every node carries a light level (soft acid green in Blender, low contrast): a scan front
  sweeps the body tail to head on the bar, hardening waves and hits light up, the pattern glows, pulses
  of light run along the rails.  Events SCAN and GLITCH (a digital glitch: parts of the body jump in
  quantized steps and flicker).
* **Machine forms**: HALO (a core in two gyroscope rings), ARRAY (two panel arrays that fold notch by
  notch), PRISM (a hexagonal crystal whose rings turn against each other like a lock) - their
  mechanisms move in robotic steps on the beat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .colony import ALL_SHAPES, CYBER_SHAPES, FREE_SHAPES
from .hive import HiveConfig, HiveState
from .osseous import OsseousHiveEngine

PLAN_CYBER = {   # preferred forms, material state, cruise factor
    "CRUISE": (("SPINDLE", "ARRAY", "WINGS", "SPINE", "PRISM"), "ELASTIC", 1.0),
    "HOVER": (("HALO", "CORE", "RING", "PRISM", "CROWN"), "COHESIVE", 0.12),
    "EXPLORE": (("CLOUD", "TENDRILS", "HALO", "SPINDLE", "ARRAY"), "FLUID", 0.6),
    "DISPLAY": (("HALO", "ARRAY", "PRISM", "LATTICE", "CROWN", "SCYTHE"), "STRUCTURED", 0.35),
    "EVADE": (("SPINDLE", "SHIELD", "PRISM", "CLOUD"), "FLUID", 1.0),
    "REFORM": (("CORE", "PRISM", "SPINDLE"), "COHESIVE", 0.6),
    "HUNT": (("SPINE", "SCYTHE", "ARRAY", "MANDIBLE"), "ELASTIC", 1.5),
    "STRIKE": (("MANDIBLE", "CLAW", "SCYTHE", "PRISM"), "HIGH_STIFFNESS", 1.0),
    "ENVELOP": (("ENVELOP",), "COHESIVE", 0.0),
    "PERCH": (("LEGS",), "ELASTIC", 0.25),
    "FORMATION": (("SPINDLE", "ARRAY", "WINGS", "SCYTHE", "HALO"), "ELASTIC", 1.0),
    "MERGE": (("CORE", "CLOUD"), "FLUID", 1.3),
    "PATROL": (("SPINDLE", "ARRAY", "HALO"), "ELASTIC", 1.1),
    "STRUCTURE": (("CORE",), "HIGH_STIFFNESS", 0.0),
}
CYBER_VOCAB = FREE_SHAPES + CYBER_SHAPES


@dataclass
class CyberHiveConfig(HiveConfig):
    max_links: int = 192                  # rail slots


@dataclass
class CyberHiveState(HiveState):
    light: np.ndarray | None = None       # per node 0..1
    scan: float = float("nan")            # scan front along the lead body (m from its centre)


class CyberHiveEngine(OsseousHiveEngine):
    STYLE = 2
    PLAN = PLAN_CYBER
    VOCAB = CYBER_VOCAB
    SHAPE_SET = ALL_SHAPES
    EVENTS = OsseousHiveEngine.EVENTS + ("SCAN", "GLITCH")

    def __init__(self, cfg: HiveConfig | None = None):
        super().__init__(cfg or CyberHiveConfig())
        self.light = np.full(self.n, 0.25)
        self.scan_t: float | None = None
        self.scan_x = float("nan")
        self.last_scan_beat = -1
        self.glitch_until = -1.0
        self.glitch_next = 0.0
        self.glitch_idx = np.zeros(0, int)
        self.glitch_off = np.zeros((0, 3))
        self.last_glitch = -1e9
        self._prev_tr = 0.0

    # ------------------------------------------------------------------ events
    def _apply_event(self, name: str, arg) -> None:
        if name == "SCAN":
            self._scan()
            return
        if name == "GLITCH":
            self._glitch(arg if isinstance(arg, (int, float)) else 0.7)
            return
        if name in ("OSSIFY", "STRIKE", "QUILLS"):            # locking / striking lights every line
            self.light[:] = 1.0
        super()._apply_event(name, arg)

    def _scan(self) -> None:
        self.scan_t = self.t
        self._log("SCAN", None)

    def _glitch(self, length: float = 0.7) -> None:
        self.glitch_until = self.t + float(length)
        self.glitch_next = self.t
        self.last_glitch = self.t
        self._log("GLITCH", None)

    def _think(self, dt: float) -> None:
        super()._think(dt)
        inp = self.inp
        # A scan sweeps the body on every second bar (or every few seconds without a song).
        if inp.playing:
            bar = int(inp.beat // 8.0)
            if bar != self.last_scan_beat:
                if self.last_scan_beat >= 0 and inp.energy > 0.2:
                    self._scan()
                self.last_scan_beat = bar
        elif self.scan_t is None and self.ev_rng.chance(dt / 7.0):
            self._scan()
        # A glitch now and then on a hard hit (its rising edge, not every kick).
        hit = inp.transient > 0.85 and self._prev_tr <= 0.85
        self._prev_tr = inp.transient
        if hit and self.t - self.last_glitch > 8.0 and self.ev_rng.chance(0.25):
            self._glitch(0.35 + 0.5 * inp.energy)

    # ------------------------------------------------------------------ glitch: parts jump in quantized steps
    def _post_targets(self, T: np.ndarray, dt: float) -> np.ndarray:
        T = super()._post_targets(T, dt)
        if self.t < self.glitch_until:
            if self.t >= self.glitch_next:                    # a new glitch frame (~15 per second)
                self.glitch_next = self.t + 1.0 / 15.0
                fly = np.nonzero(self.own < self.n_fly)[0]
                k = max(1, int(0.18 * len(fly)))
                self.glitch_idx = self.nrng.choice(fly, k, replace=False) if len(fly) else np.zeros(0, int)
                q = 0.09 * self.cfg.size
                self.glitch_off = np.round(self.nrng.standard_normal((len(self.glitch_idx), 3)) * 1.2) * q
                self.x[self.glitch_idx] += 0.6 * self.glitch_off   # a jump, not a flow
                self.light[self.glitch_idx] = 1.0
            T[self.glitch_idx] += self.glitch_off
        return T

    # ------------------------------------------------------------------ light
    def _step(self, dt: float) -> None:
        super()._step(dt)
        self._light_step(dt)

    def _light_step(self, dt: float) -> None:
        pr, inp, cfg = self.params.values(), self.inp, self.cfg
        base = 0.22 + 0.25 * self.arousal + 0.15 * inp.energy
        L = self.light * math.exp(-dt / 0.35)
        L = np.maximum(L, np.clip((self.m[:, 1] - 1.05) / 0.5, 0, 1) * 0.9)          # hardening waves, hits
        L = np.maximum(L, np.clip((self.rd_v - 0.15) / 0.3, 0, 1) * 0.55 * pr["pattern"])   # the pattern
        if self.scan_t is not None:                           # the scan front: tail -> head
            lead = self.bodies[0]
            ax = lead.R()[:, 0]
            xs = (self.x - lead.P) @ ax
            span = max(float(np.ptp(xs)), 0.3 * cfg.size)
            front = float(xs.min()) + (self.t - self.scan_t) * 2.2 * span - 0.1 * span
            L = np.maximum(L, np.exp(-((xs - front) / (0.07 * span)) ** 2))
            self.scan_x = front
            if front > float(xs.max()) + 0.2 * span:
                self.scan_t, self.scan_x = None, float("nan")
        gc = self.glove.ctrl
        if gc.active and gc.lines is not None:                # the hand plays the light
            base = max(base, 0.6 * gc.lines)
        self.light = np.clip(np.maximum(L, base), 0.0, 1.0)

    # ------------------------------------------------------------------ state
    def state(self) -> CyberHiveState:
        s = super().state()
        return CyberHiveState(**s.__dict__.copy(), light=self.light.copy(), scan=self.scan_x)


VARIANTS = {"cyber_hive": (CyberHiveEngine, CyberHiveConfig)}

__all__ = ["CyberHiveEngine", "CyberHiveConfig", "CyberHiveState", "PLAN_CYBER", "CYBER_VOCAB", "VARIANTS"]
