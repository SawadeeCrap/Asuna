"""The brain without Blender: scripted music and a scripted hand, an organism, a long run, the metrics.

Two organisms to run it on:

* ``AbstractColony`` - the colony-family engine reduced to what decides its morphology: the lead body's
  latent blend ``z`` (inertia, mutation noise), its material, the engine's own intent state machine (dwell
  times, weighted intents, forms picked from the organism's plans), its reactions (kick bursts into the
  signature form, evasions, instability shifts, splits and merges) - with the real forms' geometry from the
  engine's own shape functions.  An hour of performance runs in seconds, so modes can be compared on long
  runs and several seeds.
* ``engine`` - the real organism engine (``SpearEngine`` ...) at 120 Hz for shorter validation runs: the
  same adapter the live session uses, the same brain.

Nothing here talks to Blender, MIDI or the network (except a Kev server, when a Kev mode is asked for).
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..creature.config import DEFAULT_PARAMS
from ..creature.control import CreatureControlInput, ParameterSet
from ..creature.puppet import GloveDriver
from .adapter import make_adapter, softmax3
from .candidates import UserCue
from .core import BrainConfig, BrainControls, BrainCore
from .fingerprint import blend_cloud, embed
from .metrics import evaluate
from .vocab import SITUATIONAL


def organism_class(name: str):
    from ..creature.colony import ColonyEngine
    from ..creature.cyber import CyberHiveEngine
    from ..creature.hive import HiveEngine
    from ..creature.mimetic import BladeEngine, CloudEngine, CrawlerEngine, SpearEngine, SwarmEngine
    from ..creature.osseous import OsseousColonyEngine, OsseousHiveEngine
    return {"colony": ColonyEngine, "hive": HiveEngine, "osseous_colony": OsseousColonyEngine,
            "osseous_hive": OsseousHiveEngine, "cyber_hive": CyberHiveEngine, "swarm": SwarmEngine,
            "spear": SpearEngine, "cloud": CloudEngine, "blade": BladeEngine, "crawler": CrawlerEngine}[name]


KICK_FORM = {"spear": "LANCE", "swarm": "STREAM", "blade": "SWEEP"}      # the Mimetic line's kick reactions


# ---------------------------------------------------------------------------- scripts
class Music:
    """calm -> build -> drop -> break, again and again (lengths vary), kicks on the beat when it plays."""
    SECTIONS = (("calm", 0.18, 0.1, 36.0), ("build", 0.5, 0.55, 28.0), ("drop", 0.92, 0.9, 40.0),
                ("break", 0.32, 0.2, 26.0))

    def __init__(self, seed: int = 0, bpm: float = 124.0):
        self.rng = np.random.default_rng(seed + 101)
        self.bpm = bpm
        self.plan: list[tuple[float, str, float, float]] = []
        t, k = 0.0, 0
        while t < 6 * 3600:
            name, e, kick, L = self.SECTIONS[k % 4]
            L *= float(self.rng.uniform(0.75, 1.3))
            self.plan.append((t, name, e, kick))
            t += L
            k += 1
        self._i = 0

    def at(self, t: float) -> tuple[str, float, float, float, bool]:
        while self._i + 1 < len(self.plan) and self.plan[self._i + 1][0] <= t:
            self._i += 1
        _, name, e, kick = self.plan[self._i]
        beat = t * self.bpm / 60.0
        energy = min(1.0, max(0.0, e + 0.08 * math.sin(0.37 * t) + 0.05 * math.sin(1.3 * t)))
        transient = kick * math.exp(-(beat % 1.0) / 0.08)
        flux = 0.1 + 0.5 * abs(math.sin(0.21 * t)) * e + (0.6 if t - self.plan[self._i][0] < 1.0 else 0.0)
        return name, energy, transient, min(1.0, flux), True


class Hand:
    """A performer's hand: present in episodes, opening / closing slowly, turning, gesturing, now and then
    sculpting the form itself (then the brain must yield)."""

    def __init__(self, seed: int = 0, presence: float = 0.45):
        self.rng = np.random.default_rng(seed + 202)
        self.episodes: list[tuple[float, float, bool]] = []
        t = 0.0
        while t < 6 * 3600:
            gap = float(self.rng.exponential(40.0 * (1 - presence) / max(presence, 0.05)))
            length = float(self.rng.uniform(15.0, 60.0))
            sculpt = bool(self.rng.random() < 0.15)
            self.episodes.append((t + gap, t + gap + length, sculpt))
            t += gap + length
        self.gest_t = {}
        self._i = 0

    def at(self, t: float) -> UserCue:
        while self._i + 1 < len(self.episodes) and self.episodes[self._i][1] < t:
            self._i += 1
        a, b, sculpt = self.episodes[self._i]
        if not (a <= t <= b):
            return UserCue()
        u = (t - a) / max(b - a, 1e-6)
        open_ = 0.5 + 0.45 * math.sin(2 * math.pi * (0.07 * t) + 3.0 * a)
        spin = max(0.0, math.sin(0.23 * t + a)) ** 4
        motion = max(0.0, math.sin(0.5 * t + 2 * a)) ** 6
        g = ""
        k = int(t // 9.0)
        if k not in self.gest_t:
            self.gest_t[k] = str(self.rng.choice(["", "", "SPREAD", "CLENCH", "FLICK", "PINCH", "PUSH"]))
        if (t % 9.0) < 3.0:
            g = self.gest_t[k]
        return UserCue(True, float(np.clip(open_, 0, 1)), float(spin), float(motion), g, sculpt and 0.2 < u < 0.8)


# ---------------------------------------------------------------------------- the abstract organism
class _Body:
    def __init__(self, n: int):
        self.z = np.zeros(n)
        self.z_goal = np.zeros(n)
        self.mat = np.array([1.0, 0.35, 0.6, 0.8, 1.9, 0.0])
        self.mat_goal = self.mat.copy()
        self.intent, self.intent_t, self.dwell = "CRUISE", 0.0, 6.0
        self.shapes = ()

    def weights(self) -> np.ndarray:
        return softmax3(self.z)

    def goal_shape(self, name: str, strength: float = 2.5) -> None:
        self.z_goal = np.zeros(len(self.shapes))
        self.z_goal[self.shapes.index(name)] = strength


class AbstractColony:
    """What decides a colony-family organism's morphology, nothing else (see the module docstring)."""
    FREE = ("CRUISE", "HOVER", "EXPLORE", "DISPLAY", "PATROL", "FORMATION")

    @classmethod
    def of(cls, organism: str, seed: int = 0) -> "AbstractColony":
        eng = organism_class(organism)
        sub = type(f"Abstract{eng.__name__}", (cls,), {"SHAPE_SET": eng.SHAPE_SET, "VOCAB": eng.VOCAB,
                                                         "PLAN": eng.PLAN, "EVENTS": eng.EVENTS,
                                                         "BONY": getattr(eng, "BONY", False)})
        return sub(organism, seed)

    def __init__(self, organism: str, seed: int = 0):
        from ..realtime.glove import SCULPT_SHAPES
        from .vocab import vocabulary
        self.organism = organism
        self.vocab = vocabulary(type(self), organism)
        self.geo = self.vocab.geometry
        n = len(self.SHAPE_SET)
        self.bodies = [_Body(n)]
        self.bodies[0].shapes = tuple(self.SHAPE_SET)
        self.bodies[0].goal_shape(self.VOCAB[0], 2.5)
        self.bodies[0].z = self.bodies[0].z_goal.copy()
        self.nbodies, self.merge_at = 1, 0.0
        self.params = ParameterSet(dict(DEFAULT_PARAMS))
        self.inp = CreatureControlInput()
        self.glove = GloveDriver()
        self.sculpt = tuple(SCULPT_SHAPES.get(organism, ()))
        self.rng = np.random.default_rng(seed)
        self.t = self.instab = 0.0
        self.last_kick, self._prev_tr = -1e9, 0.0
        self.sit = [self.SHAPE_SET.index(s) for s in SITUATIONAL if s in self.SHAPE_SET]
        self.kick_form = KICK_FORM.get(organism)
        self.own = None
        self.log: list[tuple[float, str, str]] = []

    # --- the engine API the adapter uses
    def _alive(self) -> list:
        return list(range(self.nbodies))

    @property
    def x(self) -> np.ndarray:
        X = blend_cloud(self.bodies[0].weights(), self.geo)
        if self.nbodies > 1:                                   # parts fly apart
            k = np.arange(len(X)) % self.nbodies
            X = X + np.stack([np.cos(2.1 * k), np.sin(2.1 * k), 0.2 * k], 1) * 1.4 * (k > 0)[:, None]
        return X

    def trigger_event(self, name: str, arg=None) -> bool:
        name = name.upper()
        if name not in self.EVENTS:
            return False
        b = self.bodies[0]
        from ..creature.polyalloy import MATERIAL
        if name == "SPLIT" and self.nbodies == 1:
            self.nbodies, self.merge_at = int(self.rng.integers(2, 4)), self.t + float(self.rng.uniform(8, 20))
        elif name == "MERGE":
            self.nbodies = 1
        elif name == "APPENDAGE_BURST":
            opts = [s for s in ("SCYTHE", "BLADES", "THORN") if s in self.SHAPE_SET]
            if opts:
                b.goal_shape(opts[int(self.rng.integers(len(opts)))], 2.8)
            b.mat_goal = np.array(MATERIAL["STRUCTURED"])
        elif name == "COLLAPSE":
            b.mat_goal = np.array(MATERIAL["DISPERSED"])
            b.intent, b.intent_t = "REFORM", -2.0
        elif name in ("WAVE", "OSSIFY", "QUILLS"):
            b.mat_goal = np.array(MATERIAL["STRUCTURED"])
        self.log.append((self.t, "event", name))
        return True

    def _set_intent(self, intent: str, shape: str | None = None) -> None:
        from ..creature.polyalloy import MATERIAL
        b = self.bodies[0]
        prefs, mstate, _ = self.PLAN.get(intent, ((self.VOCAB[0],), "COHESIVE", 1.0))
        b.intent, b.intent_t = intent, 0.0
        b.dwell = float(self.rng.uniform(4.0, 10.0)) * (0.6 + 0.8 * self.params["coherence"])
        pick = shape or self._pick(prefs)
        b.goal_shape(pick)
        b.mat_goal = np.array(MATERIAL[mstate])
        self.log.append((self.t, "engine", f"{intent}:{pick}"))

    def _pick(self, prefs) -> str:
        if not self.BONY:
            return prefs[int(self.rng.integers(len(prefs)))]
        a = self.params["aggression"]
        from ..creature.polyalloy import AGGRESSIVE
        w = np.array([(0.5 + 1.8 * a) if p in AGGRESSIVE + ("MANDIBLE",) else (1.3 - 0.8 * a) for p in prefs])
        return prefs[int(self.rng.choice(len(prefs), p=w / w.sum()))]

    def step(self, dt: float, energy: float, transient: float, flux: float, playing: bool, hand: UserCue) -> None:
        from ..creature.polyalloy import MATERIAL
        self.inp = CreatureControlInput(energy=energy, transient=transient, spectral_flux=flux, playing=playing)
        pr, b = self.params.values(), self.bodies[0]
        self.t += dt
        b.intent_t += dt
        self.instab = min(1.5, self.instab + dt * (0.25 * flux + 0.1 * pr["instability"]))
        if b.intent == "EVADE" and b.intent_t > 1.6:
            self._set_intent("REFORM")
        elif b.intent == "REFORM" and b.intent_t > 1.5:
            self._set_intent("CRUISE")
        elif b.intent in self.FREE and b.intent_t > b.dwell:                # the engine's own choice
            pool = ["CRUISE", "HOVER", "EXPLORE", "DISPLAY", "PERCH"]
            w = np.array([1.2 + energy, 0.8 * (1 - energy), 0.5 + pr["mutation"] + self.instab,
                          0.4 + pr["rigidity"] + pr["tendril_activity"],
                          (0.2 + 0.6 * (1 - energy)) * (self.nbodies == 1) * ("LEGS" in self.SHAPE_SET)])
            if b.intent in pool:
                w[pool.index(b.intent)] *= 0.4
            nxt = pool[int(self.rng.choice(len(pool), p=w / w.sum()))]
            self._set_intent(nxt, "LEGS" if nxt == "PERCH" and "PERCH" in self.PLAN else None)
        elif b.intent == "PERCH" and b.intent_t > b.dwell:
            self._set_intent("CRUISE")
        if b.intent in self.FREE and self.rng.random() < dt * pr["obstacle_rate"] * 0.15 * (0.3 + energy):
            self._set_intent("EVADE")
        if self.instab > 1.0:
            self.instab = 0.0
            b.goal_shape(self.VOCAB[int(self.rng.integers(len(self.VOCAB)))])
        kick = transient > 0.6 >= self._prev_tr
        if self.kick_form and kick and energy > 0.4 and self.t - self.last_kick > 1.8:
            self.last_kick = self.t                                          # e.g. the Spear's dash
            b.goal_shape(self.kick_form, 3.6)
            b.mat_goal = np.array(MATERIAL["HIGH_STIFFNESS"])
        elif kick and b.intent in self.FREE and self.rng.random() < 0.12 * energy:
            self._set_intent("EVADE")                                        # a kick as an obstacle
        self._prev_tr = transient
        if self.nbodies > 1 and self.t > self.merge_at:
            self.nbodies = 1
        if energy > 0.85 and self.nbodies == 1 and "SPLIT" in self.EVENTS and self.rng.random() < dt * 0.01:
            self.trigger_event("SPLIT")
        gc = self.glove.ctrl                                                 # the hand sculpting the form itself
        gc.active = hand.sculpt
        if hand.sculpt and self.sculpt:
            gc.finger_mode, gc.freeze = "morph", 0.0
            fingers = np.clip(hand.open + 0.3 * np.sin(np.arange(5) + 0.4 * self.t), 0, 1)
            b.z_goal = np.zeros(len(self.SHAPE_SET))
            for shp, ex in zip(self.sculpt, fingers):
                if shp in self.SHAPE_SET:
                    i = self.SHAPE_SET.index(shp)
                    b.z_goal[i] = max(b.z_goal[i], 0.4 + 2.4 * ex)
        else:
            gc.finger_mode = "limbs"
        tau_m = 0.8 + 3.0 * pr["rigidity"] * (1.2 - pr["fluidity"])
        noise = self.rng.standard_normal(len(b.z)) * pr["mutation"] * 0.6 * math.sqrt(dt)
        noise[self.sit] = 0.0
        b.z += (b.z_goal - b.z) * min(1.0, dt / tau_m) + noise
        b.mat += (b.mat_goal - b.mat) * min(1.0, dt / (0.4 + 0.8 * pr["coherence"]))


# ---------------------------------------------------------------------------- the real engine
class EngineRunner:
    """A real organism engine at 120 Hz, fed by the same scripts (shorter runs: it is the real cost)."""

    def __init__(self, organism: str, seed: int = 0):
        from ..creature.colony import ColonyConfig
        cls = organism_class(organism)
        cfg_cls = {"spear": "SpearConfig", "swarm": "SwarmConfig", "cloud": "CloudConfig", "blade": "BladeConfig",
                   "crawler": "CrawlerConfig"}.get(organism)
        if cfg_cls:
            from ..creature import mimetic
            cfg = getattr(mimetic, cfg_cls)(seed=seed)
        else:
            cfg = ColonyConfig(seed=seed)
        self.engine = cls(cfg)
        self.organism = organism

    def step(self, dt: float, energy: float, transient: float, flux: float, playing: bool, hand: UserCue) -> None:
        self.engine.set_input(CreatureControlInput(energy=energy, transient=transient, spectral_flux=flux,
                                                   playing=playing, bass=0.5 * energy, amplitude=energy))
        self.engine.update(dt)


# ---------------------------------------------------------------------------- one run
def _hand(u: UserCue) -> str:
    if not u.present:
        return "-"
    return ("sculpting" if u.sculpt else f"open {u.open:.1f}") + (f" {u.gesture}" if u.gesture else "")


def run(organism: str = "spear", mode: str = "deterministic", minutes: float = 5.0, seed: int = 0,
        controls: dict | None = None, kev_url: str = "", hand: bool = True, engine: str = "abstract",
        dt: float | None = None, sample_dt: float = 0.5, trace: list | None = None,
        checkpoints: list | None = None, on_decision=None, kev_timeout: float = 0.8, kev=None,
        kev_candidates: int = 6) -> dict:
    """Run ``minutes`` of a performance; -> {"metrics", "stats", "decisions", "by_minutes"}.

    ``checkpoints`` (minutes): the metrics of the run's first 1, 5, 30 ... minutes too - the simulation is
    deterministic for a seed, so a prefix of a long run is exactly the shorter run.  ``on_decision(entry)``
    sees every decision-log line as it happens (the demo prints them)."""
    ctl = BrainControls(**(controls or {}))
    cfg = BrainConfig(enabled=True, mode=mode, controls=ctl, kev_url=kev_url, kev_timeout=kev_timeout, seed=seed,
                      kev_candidates=kev_candidates)
    if engine == "abstract":
        org = AbstractColony.of(organism, seed)
        eng = org
        dt = dt or 0.05
    else:
        org = EngineRunner(organism, seed)
        eng = org.engine
        dt = dt or 1.0 / 120.0
    adapter = make_adapter(eng, organism)
    core = BrainCore(adapter.vocab, cfg, kev=kev)
    music, handscript = Music(seed), Hand(seed)
    scale = core.scale
    free_idx = [adapter.vocab.index(f) for f in adapter.vocab.free]
    samples = {"t": [], "emb": [], "w": [], "geo": [], "size": [], "hand_open": []}
    commits: list[float] = []
    t, next_dec, next_smp = 0.0, 0.0, 0.0
    T = minutes * 60.0
    period = 1.0 / max(0.1, ctl.rate_hz)
    seen = 0
    wall = time.perf_counter()
    while t < T:
        label, energy, transient, flux, playing = music.at(t)
        user = handscript.at(t) if hand else UserCue()
        org.step(dt, energy, transient, flux, playing, user)
        t += dt
        adapter.maintain(t)
        if t >= next_dec:
            next_dec += period
            snap = adapter.snapshot(t, None, user, {}, f"{label}, energy {energy:.1f}", lookahead=period)
            d = core.step(snap)
            if d is not None:
                adapter.apply(d, t)
                commits.append(t)
                if trace is not None:
                    trace.append({"t": round(t, 2), "music": label, "hand": user.present, **{k: v for k, v in
                                  d.to_dict().items() if k in ("op", "label", "source", "novelty", "utility",
                                                                 "confidence", "strength", "hold")}})
            if on_decision is not None and core.logged != seen:
                seen = core.logged
                on_decision({**core.log[-1], "music": label, "hand": _hand(user),
                             "memory": core.memory.summary(t, adapter.vocab.forms)})
        if t >= next_smp:
            next_smp += sample_dt
            b = adapter._body()
            w = softmax3(np.asarray(b.z, float))
            snap = adapter.snapshot(t, None, user, {}, "")
            fp = snap.fp
            samples["t"].append(t)
            samples["emb"].append(embed(fp, scale))
            samples["w"].append(w)
            samples["geo"].append(fp.geo / scale.sd)
            samples["size"].append(float(fp.geo[0]))
            samples["hand_open"].append(user.open if user.present else float("nan"))
    wall = time.perf_counter() - wall

    def metrics_until(minutes_: float) -> dict:
        n = sum(1 for x in samples["t"] if x <= minutes_ * 60.0 + 1e-6)
        part = {k: v[:n] for k, v in samples.items()}
        cm = [c for c in commits if c <= minutes_ * 60.0]
        return evaluate(part, free_idx, adapter.vocab.identity, sample_dt, cm if mode != "current" else None)

    st = core.stats
    return {"organism": organism, "mode": mode, "seed": seed, "engine": engine, "minutes": minutes,
            "metrics": metrics_until(minutes),
            "by_minutes": {float(m): metrics_until(m) for m in (checkpoints or []) if m <= minutes},
            "stats": {"commits": st["commits"], "decisions": st["decisions"], "holds": st["holds"],
                      "yields": st["yields"], "kev_calls": st["kev_calls"], "kev_fail": st["kev_fail"],
                      "kev_lowconf": st["kev_lowconf"],
                      "kev_ms_median": round(float(np.median(st["kev_ms"])), 1) if st["kev_ms"] else None,
                      "kev_ms_p95": round(float(np.percentile(st["kev_ms"], 95)), 1) if st["kev_ms"] else None,
                      "step_ms_median": round(float(np.median(st["step_ms"])), 2) if st["step_ms"] else None,
                      "step_ms_p95": round(float(np.percentile(st["step_ms"], 95)), 2) if st["step_ms"] else None,
                      "steps": st["steps"], "cpu_ms_per_min": round(st["step_ms_total"] / max(minutes, 1e-9), 1),
                      "restored": adapter.restored, "wall_s": round(wall, 1)},
            "decisions": core.log[-40:], "memory": core.memory.summary(t, adapter.vocab.forms)}


__all__ = ["run", "Music", "Hand", "AbstractColony", "EngineRunner", "organism_class"]
