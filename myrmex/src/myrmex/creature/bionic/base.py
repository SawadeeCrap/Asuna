"""The Bionic line (creatures v14-v18): bodies whose form comes from structural physics, not from blobs.

Every earlier organism is drawn as an implicit surface around its nodes - one metaball per node - so
whatever the underlying form, hard material reads as beads on a string.  The Bionic organisms have no
blob at all: what you see *is* the structure, and each one lives by a different law of form:

* **Tensor** (v14) - a tensegrity spine: rigid struts floating in a net of tension-only cables that
  contract like muscles; waves of contraction make it swim, curl, coil and spring.
* **Fold** (v15) - a rigid-foldable Miura-ori sheet: panels on hinges folding between wing, pleat,
  tube, shell, fan and ribbon; flapping and travelling fold waves.
* **Arbor** (v16) - a vascular tree grown by space colonisation, thickened by Murray's law from a
  finite amount of material, pruned in silence; light pulses run from the root to the tips.
* **Ferro** (v17) - a ferrofluid under an invisible magnet: Rosensweig spikes above the critical field,
  tongues and droplet chains towards the pole, iron filings tracing the field.
* **Truss** (v18) - a lattice that remodels by Wolff's law: loaded members thicken, idle ones vanish,
  the total volume stays constant, blows and beats leave bone where they land.

This module: what they share - inputs, parameters, intents, a latent space of form *regimes* with
morphological inertia, flight, obstacles, the Hand Glove and the state on the wire.  Structure is sent
as members (i, j, kind, radius, stress, phase) + a few creature-specific floats (``extra``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...util.rng import RngStreams, stable_hash64
from ..control import CreatureControlInput, ParameterSet
from ..polyalloy import Obstacle
from ..puppet import GloveControl, GloveDriver

KINDS = ("tensor", "fold", "arbor", "ferro", "truss")
STYLE0 = 8                                    # style on the wire: 8 + kind
G = 9.81
INTENTS = ("CRUISE", "HOVER", "EXPLORE", "DISPLAY", "EVADE", "REFORM", "STRIKE")


@dataclass
class BionicConfig:
    seed: int = 7
    params: dict = field(default_factory=dict)
    size: float = 1.8
    sim_rate: float = 120.0
    stage_radius: float = 14.0
    altitude: tuple = (2.0, 5.5)
    cruise: float = 2.6
    max_obstacles: int = 4


@dataclass
class BionicState:
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
    members: np.ndarray                       # (M, 6) i, j, kind, radius, stress, phase (i < 0: empty slot)
    extra: np.ndarray                         # creature-specific floats
    obstacles: np.ndarray                     # (O, 4) x, y, z, radius
    bkind: int = 0
    style: int = STYLE0
    material: str = "STRUCTURED"
    fragments: int = 1
    links: None = None                        # (none of the blob-line blocks)
    plate: None = None
    particles: None = None
    rms: float = 0.0                          # how far the body reaches from its centre (m, rms) - camera framing


class BionicEngine:
    VARIANT = ""
    KIND = 0
    EVENTS = ("MORPHOLOGY_SHIFT", "IMPULSE", "OBSTACLE", "COLLAPSE", "RECONSTRUCTION")
    REGIMES: tuple = ("REST",)
    PLAN: dict = {}                           # intent -> (preferred regimes, cruise factor)
    GROUND = False                            # walks / rolls instead of flying

    def __init__(self, cfg: BionicConfig | None = None):
        self.cfg = cfg = cfg or BionicConfig()
        self.rs = RngStreams(cfg.seed)
        self.rng = self.rs.stream(self.VARIANT or "bionic")
        self.ev_rng = self.rs.stream((self.VARIANT or "bionic") + "-events")
        self.nrng = np.random.default_rng(stable_hash64(self.VARIANT + "-noise", cfg.seed) & 0xFFFFFFFF)
        self.params = ParameterSet(cfg.params)
        self.inp = CreatureControlInput()
        self.t = 0.0
        self.glove = GloveDriver()
        self.sculpt: tuple | None = None
        self.events: list[tuple[float, str, object]] = []
        self._pending: list[tuple[str, object]] = []
        self.obstacles: list[Obstacle | None] = [None] * cfg.max_obstacles
        self.P = np.array([0.0, 0.0, 3.2])
        self.vel = np.zeros(3)
        self.heading = self.rng.uniform(-math.pi, math.pi)
        self.pitch = self.yaw_rate = self.wander = 0.0
        self.intent, self.intent_t, self.dwell = "CRUISE", 0.0, 8.0
        n = len(self.REGIMES)
        self.z_goal = np.zeros(n)
        self.z_goal[0] = 2.5
        self.z = self.z_goal.copy()                   # born in its first regime
        self.arousal, self.glow, self.surface, self.instab = 0.2, 0.0, 0.3, 0.0
        self.last_kick = self.last_obstacle = -1e9
        self.kick = 0.0                               # a decaying kick envelope (0..1)
        self.turb_until = -1.0

    # ------------------------------------------------------------------ API
    def set_input(self, inp: CreatureControlInput) -> None:
        self.inp = inp.sanitized()

    def set_parameter(self, name: str, value) -> bool:
        return self.params.set(name, value)

    def set_glove(self, ctrl: GloveControl | None, shapes: tuple | None = None) -> None:
        self.glove.set(ctrl)
        self.sculpt = tuple(x for x in (shapes or ()) if x in self.REGIMES) or None

    def trigger_event(self, name: str, arg=None) -> bool:
        if name.upper() not in self.EVENTS:
            return False
        self._pending.append((name.upper(), arg))
        return True

    def update(self, dt: float):
        if not math.isfinite(dt) or dt <= 0:
            return self.state()
        steps = max(1, min(12, int(round(dt * self.cfg.sim_rate))))
        for _ in range(steps):
            self._step(dt / steps)
        return self.state()

    # ------------------------------------------------------------------ helpers
    def _log(self, name: str, arg=None) -> None:
        self.events.append((self.t, name, arg))
        self.events = self.events[-60:]

    def weights(self) -> np.ndarray:
        e = np.exp(3.0 * (self.z - self.z.max()))
        return e / e.sum()

    def regime(self) -> str:
        return self.REGIMES[int(np.argmax(self.z))]

    def _R(self) -> np.ndarray:
        ch, sh = math.cos(self.heading), math.sin(self.heading)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        return np.array([[ch, -sh, 0], [sh, ch, 0], [0, 0, 1.0]]) @ np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])

    def goal(self, name: str, strength: float = 2.5) -> None:
        self.z_goal = np.zeros(len(self.REGIMES))
        self.z_goal[self.REGIMES.index(name)] = strength

    def _set_intent(self, intent: str, regime: str | None = None) -> None:
        prefs, _ = self.PLAN.get(intent, (self.REGIMES, 1.0))
        self.intent, self.intent_t = intent, 0.0
        self.dwell = self.rng.uniform(5.0, 11.0) * (0.6 + 0.8 * self.params["coherence"])
        self.goal(regime or prefs[self.rng.randint(0, len(prefs) - 1)])

    # ------------------------------------------------------------------ behaviour
    def _step(self, dt: float) -> None:
        self.t += dt
        self._think(dt)
        while self._pending:
            self._apply_event(*self._pending.pop(0))
        self._latent(dt)
        self._simulate(dt)
        self._obstacle_step(dt)
        self.glow *= math.exp(-dt / 0.4)
        gc = self.glove.ctrl
        if gc.active and gc.lines is not None:
            self.glow = max(self.glow, gc.lines)
        self.kick *= math.exp(-dt / 0.18)

    def _think(self, dt: float) -> None:
        pr, inp = self.params.values(), self.inp
        self.arousal += ((0.75 * inp.energy + 0.4 * inp.transient + 0.2 * inp.spectral_flux) * pr["reactivity"] + 0.1
                         - self.arousal) * min(1.0, dt / 2.0)
        self.instab = min(1.5, self.instab + dt * (0.3 * inp.spectral_flux + 0.1 * pr["instability"]))
        if inp.transient > 0.5 and self.t - self.last_kick > 0.25:
            self.last_kick = self.t
            self.kick = max(self.kick, inp.transient)
            self._on_kick(inp.transient)
        if self.ev_rng.chance(dt * pr["obstacle_rate"] * 0.15 * (0.3 + inp.energy)):
            self._spawn_obstacle(0.6)
        self.intent_t += dt
        if self.intent_t > self.dwell and self.intent not in ("EVADE", "STRIKE"):
            w = [1.0 + inp.energy, 0.4 + 0.8 * (1.0 - inp.energy), 0.6 + pr["mutation"], 0.3 + pr["mechanism"] +
                 inp.spectral_flux]
            self._set_intent(("CRUISE", "HOVER", "EXPLORE", "DISPLAY")[self.ev_rng.weighted_index(w)])
        if self.intent in ("EVADE", "STRIKE", "REFORM") and self.intent_t > 2.0:
            self._set_intent("CRUISE")
        if self.instab > 1.0:
            self.instab = 0.0
            self._pending.append(("MORPHOLOGY_SHIFT", None))

    def _on_kick(self, strength: float) -> None:
        """A hit in the music (subclasses make it physical)."""

    def _latent(self, dt: float) -> None:
        pr, gc = self.params.values(), self.glove.ctrl
        tau = 0.8 + 3.0 * pr["rigidity"] * (1.2 - pr["fluidity"])
        if gc.active and gc.finger_mode == "morph" and self.sculpt:
            self.z_goal = np.zeros(len(self.REGIMES))
            for name, ex in zip(self.sculpt, gc.fingers):
                q = self.REGIMES.index(name)
                self.z_goal[q] = max(self.z_goal[q], 0.4 + 2.4 * ex)
            tau = 0.25
        held = 1.0 - (gc.freeze if gc.active else 0.0)
        noise = self.nrng.standard_normal(len(self.REGIMES)) * pr["mutation"] * 0.5 * math.sqrt(dt)
        self.z += ((self.z_goal - self.z) * min(1.0, dt / tau) + noise) * held
        if gc.active and gc.energy is not None:
            self.arousal = max(self.arousal, gc.energy)

    def _apply_event(self, name: str, arg) -> None:
        if name == "MORPHOLOGY_SHIFT":
            prefs, _ = self.PLAN.get(self.intent, (self.REGIMES, 1.0))
            pick = arg if arg in self.REGIMES else (prefs[self.rng.randint(0, len(prefs) - 1)]
                                                    if self.rng.chance(0.7) else
                                                    self.REGIMES[self.rng.randint(0, len(self.REGIMES) - 1)])
            self.goal(pick)
        elif name == "OBSTACLE":
            self._spawn_obstacle(1.0)
        elif name == "IMPULSE":
            self.kick = 1.0
            self._on_kick(1.0)
        elif name == "COLLAPSE":
            self._set_intent("REFORM", self.REGIMES[0])
        elif name == "RECONSTRUCTION":
            self._set_intent("DISPLAY")
        self._log(name, arg)

    # ------------------------------------------------------------------ flight
    def _flight(self, dt: float, P: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Desired acceleration of the body (distributed thrust), heading follows the velocity."""
        cfg, pr, gc = self.cfg, self.params.values(), self.glove.ctrl
        self.P, self.vel = P, vel
        cruise = cfg.cruise * self.PLAN.get(self.intent, ((), 1.0))[1] * (0.4 + 1.2 * pr["speed"]) * (0.6 + self.arousal)
        self.wander = self.wander * math.exp(-dt * 0.3) + self.nrng.standard_normal() * math.sqrt(dt) * \
            (0.4 + pr["noise"])
        want = self.heading + 1.2 * self.wander * dt + (gc.offset[0] * 1.6 * dt if gc.active else 0.0)
        home = -P[:2]
        if float(np.linalg.norm(home)) > 0.6 * cfg.stage_radius:
            wh = math.atan2(home[1], home[0])
            want += ((wh - self.heading + math.pi) % (2 * math.pi) - math.pi) * min(1.0, dt * 1.5)
        lo, hi = cfg.altitude
        alt = lo + (hi - lo) * pr["altitude"] + (2.2 * gc.offset[1] if gc.active else 0.0)
        vz = float(np.clip((alt - P[2]) * 0.8 + 0.3 * math.sin(0.4 * self.t), -1.5, 1.5))
        v_des = np.array([math.cos(want) * cruise, math.sin(want) * cruise, vz])
        v_des = self.glove.flight(v_des, P, cruise)
        a = (v_des - vel) / 0.8
        sp = float(np.linalg.norm(vel[:2]))
        if sp > 0.2:
            dy = (math.atan2(vel[1], vel[0]) - self.heading + math.pi) % (2 * math.pi) - math.pi
            self.yaw_rate += dt * (3.0 * dy - 2.2 * self.yaw_rate)
        self.heading += dt * self.yaw_rate
        self.pitch += (math.atan2(vel[2], max(sp, 0.5)) * 0.5 - self.pitch) * min(1.0, dt * 2.0)
        return a

    # ------------------------------------------------------------------ obstacles
    def _spawn_obstacle(self, strength: float) -> None:
        if None not in self.obstacles:
            return
        r = self.ev_rng
        slot = self.obstacles.index(None)
        fwd = self._R()[:, 0]
        side = np.cross([0, 0, 1.0], fwd)
        dirs = [fwd, fwd + side, fwd - side, side, -side, fwd + np.array([0, 0, 0.8])]
        d = dirs[r.randint(0, len(dirs) - 1)]
        d = d / np.linalg.norm(d)
        speed = r.uniform(6.0, 11.0) * (0.7 + 0.5 * strength)
        dist = r.uniform(9.0, 13.0)
        start = self.P + d * dist
        tt = dist / (speed + max(0.0, float(self.vel @ d)))
        aim = self.P + self.vel * tt
        self.obstacles[slot] = Obstacle(start, (aim - start) / max(tt, 0.2), r.uniform(0.35, 0.9) * self.cfg.size * 0.5,
                                        self.t)
        self.last_obstacle = self.t
        self._log("OBSTACLE", None)

    def _obstacle_step(self, dt: float) -> None:
        for k, ob in enumerate(self.obstacles):
            if ob is None:
                continue
            ob.pos = ob.pos + ob.vel * dt
            if not ob.handled and float(np.linalg.norm(ob.pos - self.P)) < 4.0 + ob.radius:
                ob.handled = True
                self._threat(ob)
            if float(np.linalg.norm(ob.pos - self.P)) > 30.0 or self.t - ob.born > 9.0:
                self.obstacles[k] = None

    def _threat(self, ob: Obstacle) -> None:
        self._set_intent("EVADE")
        self._log("RESPONSE", None)

    @staticmethod
    def project_lengths(x: np.ndarray, pairs: np.ndarray, rest: np.ndarray, iters: int, omega: float = 1.5,
                        deg: np.ndarray | None = None) -> None:
        """Hard distance constraints, Jacobi-style: each node moves by the average of its corrections."""
        i, j = pairs[:, 0], pairs[:, 1]
        if deg is None:
            deg = np.bincount(np.concatenate([i, j]), minlength=len(x)).astype(float)
        inv = (omega / np.maximum(deg, 1.0))[:, None]
        for _ in range(iters):
            d = x[j] - x[i]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            corr = ((L - rest) / L * 0.5)[:, None] * d
            acc = np.zeros_like(x)
            np.add.at(acc, i, corr)
            np.add.at(acc, j, -corr)
            x += acc * inv

    @staticmethod
    def _push_out(x: np.ndarray, v: np.ndarray, ob: Obstacle) -> np.ndarray:
        """Nodes inside an obstacle are pushed to its surface; returns the hit mask."""
        rel = x - ob.pos
        d = np.linalg.norm(rel, axis=1)
        hit = d < ob.radius
        if hit.any():
            nrm = rel[hit] / np.maximum(d[hit], 1e-6)[:, None]
            x[hit] = ob.pos + nrm * ob.radius
            vn = (v[hit] * nrm).sum(1, keepdims=True)
            v[hit] -= np.minimum(vn, 0.0) * nrm * 1.6
            v[hit] += ob.vel * 0.3
        return hit

    def _obstacle_array(self) -> np.ndarray:
        out = np.zeros((self.cfg.max_obstacles, 4))
        for k, ob in enumerate(self.obstacles):
            if ob is not None:
                out[k] = (*ob.pos, ob.radius)
        return out

    # ------------------------------------------------------------------ to implement
    def _simulate(self, dt: float) -> None:
        raise NotImplementedError

    def state(self) -> BionicState:
        raise NotImplementedError

    def _state(self, pos: np.ndarray, radius: np.ndarray, members: np.ndarray, extra, volumes: dict | None = None,
               material: str = "STRUCTURED", fragments: int = 1, com: np.ndarray | None = None,
               rms: float | None = None) -> BionicState:
        n = len(pos)
        c = (pos.mean(0) if n else self.P.copy()) if com is None else np.asarray(com, float)
        if rms is None:
            rms = float(np.sqrt(((pos - c) ** 2).sum(1).mean())) if n else 0.0
        return BionicState(self.t, self.intent, self.regime(), pos.copy(), radius.copy(), np.ones((n, 3)),
                           np.ones(n, np.int8), np.full(n, -1, np.int16), c, self.heading, self.params.values(),
                           volumes or {}, self.surface, self.glow, self.arousal, self.instab, list(self.events[-6:]),
                           members, np.asarray(extra, float), self._obstacle_array(), self.KIND, STYLE0 + self.KIND,
                           material, fragments, rms=rms)


__all__ = ["BionicEngine", "BionicConfig", "BionicState", "KINDS", "STYLE0", "INTENTS", "G"]
