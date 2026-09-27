"""Candidate mutations of the current form - only what the engines can actually do.

Operations (the action vocabulary, mapped to real capabilities):

    HOLD        stay (persistence)
    SHIFT       flow into another form of the organism's vocabulary        (engine: z_goal)
    HYBRID      a blend of two forms - "unexpected combinations"            (engine: z_goal, several peaks)
    INTENSIFY   push the current form further: harder material, stronger   (engine: z_goal strength, mat_goal)
    DISSOLVE    soften / disperse the material of the current form          (engine: mat_goal)
    EVENT       a structural event it knows: split, merge, burst, collapse  (engine: trigger_event)
    RETURN      come back to a remembered form                              (memory -> z_goal, mat_goal)
    MUTATE      come back to a remembered form, changed                     (memory -> z_goal with a new share)

Every candidate carries its predicted fingerprint (the blend's geometry from the organism's own shape
functions) and its features: novelty, continuity, identity, fit with the hand, repetition, oscillation,
the pull of memory, and whether it is a radical change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .fingerprint import Fingerprint, Scale, blend_cloud, embed, geometry
from .memory import MorphMemory
from .novelty import novelty
from .vocab import Vocabulary

RADICAL = 0.75                     # embedding distance of a radical change


# What an event does to the material (the engines' own trigger_event), so its prediction is not "nothing".
EVENT_MATERIAL = {"COLLAPSE": "DISPERSED", "SCATTER": "DISPERSED", "APPENDAGE_BURST": "STRUCTURED",
                  "WAVE": "STRUCTURED", "OSSIFY": "STRUCTURED", "QUILLS": "STRUCTURED", "GATHER": "COHESIVE",
                  "MASS_REBALANCE": "ELASTIC"}


def depth(ctl) -> float:
    """Memory depth (the ``memory`` control, 0..1) as a time factor: 1 at the default 0.6, 0.18 at 0 (forms are
    "new" again after ~20 s), 3.2 at 1 (they stay familiar for ~6 min).  Scales novelty's half-life, the
    repetition window and the return refractory period together."""
    return 18.0 ** (float(getattr(ctl, "memory", 0.6)) - 0.6)


def material_vector(name: str | None) -> np.ndarray | None:
    if not name:
        return None
    from ..creature.polyalloy import MATERIAL
    return np.array(MATERIAL[name], float)


@dataclass
class Candidate:
    op: str
    blend: dict
    material: str | None = None
    event: str | None = None
    strength: float = 2.6
    proto: int | None = None
    label: str = ""
    fp: Fingerprint | None = None
    emb: np.ndarray | None = None
    feats: dict = field(default_factory=dict)
    utility: float = 0.0
    direction: str = ""                # what the change does to the shape (from its predicted geometry)

    def describe(self) -> str:
        return (self.label or self.op) + (f" ({self.direction})" if self.direction else "")


@dataclass
class UserCue:
    """What the hand says, as tendencies (from the glove the app already reads - no new gestures)."""
    present: bool = False
    open: float = 0.5                  # finger extension: 0 fist .. 1 open hand
    spin: float = 0.0                  # 0..1 how fast the hand turns
    motion: float = 0.0                # 0..1 how fast it moves
    gesture: str = ""                  # the last gesture (FLICK, CLENCH, SPREAD, PUSH, PINCH) within 3 s
    sculpt: bool = False               # the hand is shaping the form itself (sculpt preset / freeze): yield

    def bias(self) -> np.ndarray:
        """A direction in GEO space (size, elong, flat, asym, clump, reach, lumpy, twist)."""
        b = np.zeros(8)
        if not self.present:
            return b
        o = (self.open - 0.5) * 2.0
        g = self.gesture
        if g == "SPREAD":
            o = max(o, 0.8)
        elif g == "CLENCH":
            o = min(o, -0.8)
        b += o * np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
        b[7] += 1.2 * self.spin
        if g == "PUSH":
            b[1] += 0.8
        if g == "PINCH":
            b[6] += 0.8
        return b


# The requested operation words, where the organisms actually have them: a change of form moves the shape's
# descriptors (fingerprint.GEO); the strongest move (in the forms' own spread) names it.
DIRECTIONS = {"size": ("expand", "contract"), "elong": ("elongate", "compress"), "flat": ("flatten", "thicken"),
              "asym": ("asymmetrize", "symmetrize"), "clump": ("gather", "scatter"), "reach": ("branch out", "retract"),
              "lumpy": ("ripple", "smooth"), "twist": ("twist", "untwist")}


def direction(dgeo: np.ndarray, floor: float = 0.6) -> str:
    """The one or two strongest shape moves of a change (``dgeo``: descriptor change / the forms' spread)."""
    from .fingerprint import GEO
    order = np.argsort(-np.abs(dgeo))
    words = [DIRECTIONS[GEO[i]][0 if dgeo[i] > 0 else 1] for i in order[:2] if abs(dgeo[i]) >= floor]
    return " + ".join(words)


def _blend_name(blend: dict) -> str:
    items = sorted(blend.items(), key=lambda kv: -kv[1])
    if len(items) == 1 or items[1][1] < 0.12:
        return items[0][0]
    return " + ".join(f"{k} {v:.0%}" for k, v in items[:2])


def predict(vocab: Vocabulary, blend: dict, material: str | None, event: str | None, cur: Fingerprint) -> Fingerprint:
    w = vocab.weights(blend)
    w = 0.95 * w + 0.05 * cur.w                              # the body never arrives exactly
    w = w / w.sum()
    geo = geometry(blend_cloud(w, vocab.geometry)) if vocab.geometry is not None else cur.geo.copy()
    mat = material_vector(material)
    mat = cur.mat.copy() if mat is None or not len(cur.mat) else mat
    bodies = 2 if event == "SPLIT" else (1 if event == "MERGE" else cur.bodies)
    return Fingerprint(w, geo, mat, bodies)


def generate(vocab: Vocabulary, cur_blend: dict, cur: Fingerprint, memory: MorphMemory, t: float, energy: float,
             ctl, rng: np.random.Generator) -> list[Candidate]:
    """10-24 candidates round the current form (``ctl``: BrainControls)."""
    free = [f for f in vocab.free if f in vocab.forms]
    dom = max(cur_blend, key=cur_blend.get) if cur_blend else vocab.forms[int(np.argmax(cur.w))]
    out = [Candidate("HOLD", dict(cur_blend) or {dom: 1.0}, label=f"stay {_blend_name(cur_blend or {dom: 1.0})}")]
    others = [f for f in free if f != dom]
    if others:
        idn = np.array([vocab.identity[vocab.index(f)] for f in others])
        p = (0.25 + idn) ** (1.0 - 0.7 * ctl.autonomy)
        p = p / p.sum()
        k = min(len(others), 6)
        for f in rng.choice(others, size=k, replace=False, p=p):
            out.append(Candidate("SHIFT", {str(f): 1.0}, label=f"become {f}"))
        for f in rng.choice(others, size=min(len(others), 3), replace=False, p=p):
            m = float(rng.uniform(0.3, min(0.75, 0.45 + 0.4 * ctl.mutation)))
            out.append(Candidate("HYBRID", {dom: 1.0 - m, str(f): m}, label=f"hybrid {dom} {1 - m:.0%} + {f} {m:.0%}"))
        if len(others) >= 2 and (ctl.mutation > 0.3 or ctl.autonomy > 0.6):
            a, b = rng.choice(others, size=2, replace=False, p=p)
            out.append(Candidate("HYBRID", {str(a): 0.5, str(b): 0.5}, label=f"hybrid {a} + {b}"))
    if vocab.materials:
        hard = "HIGH_STIFFNESS" if ctl.mutation > 0.5 else "STRUCTURED"
        out.append(Candidate("INTENSIFY", dict(cur_blend or {dom: 1.0}), material=hard, strength=3.4,
                             label=f"intensify {dom}: harden"))
        soft = "DISPERSED" if ctl.mutation > 0.6 else "FLUID"
        out.append(Candidate("DISSOLVE", dict(cur_blend or {dom: 1.0}), material=soft, strength=2.2,
                             label=f"dissolve {dom}: {soft.lower()}"))
    evs = [e for e in vocab.events if e != "COLLAPSE" or ctl.autonomy > 0.6]
    for e in rng.choice(evs, size=min(2, len(evs)), replace=False) if evs else ():
        out.append(Candidate("EVENT", dict(cur_blend or {dom: 1.0}), event=str(e), label=f"{str(e).lower()}",
                             material=EVENT_MATERIAL.get(str(e)) if vocab.materials else None))
    k = depth(ctl)
    rets = [(p, pull) for p, pull in memory.return_scores(t, energy, 45.0 * k, 60.0 * k) if pull > 0.12][:3]
    for p, pull in rets[:2]:
        blend = p.goal.get("blend") or {vocab.forms[i]: float(p.fp.w[i]) for i in np.argsort(p.fp.w)[::-1][:2]}
        out.append(Candidate("RETURN", dict(blend), material=p.goal.get("material"), proto=p.id,
                             label=f"return to {p.summary(vocab.forms)} (seen {int(t - p.last)} s ago)"))
    if rets and free:
        p, pull = rets[int(rng.integers(len(rets)))]
        blend = dict(p.goal.get("blend") or {vocab.forms[int(np.argmax(p.fp.w))]: 1.0})
        f = str(rng.choice(free))
        m = float(rng.uniform(0.25, 0.45))
        blend = {k: v * (1.0 - m) for k, v in blend.items()}
        blend[f] = blend.get(f, 0.0) + m
        out.append(Candidate("MUTATE", blend, material=p.goal.get("material"), proto=p.id,
                             label=f"mutate {p.summary(vocab.forms)} with {f} {m:.0%}"))
    return out


def evaluate(cands: list[Candidate], vocab: Vocabulary, scale: Scale, cur: Fingerprint, memory: MorphMemory,
             t: float, energy: float, user: UserCue, ctl) -> None:
    """Features of every candidate (in place)."""
    e_cur = embed(cur, scale)
    k = depth(ctl)
    pulls = {p.id: pull for p, pull in memory.return_scores(t, energy, 45.0 * k, 60.0 * k)}
    bias = user.bias()
    for c in cands:
        c.fp = predict(vocab, c.blend, c.material, c.event, cur)
        c.emb = embed(c.fp, scale)
        d = float(np.linalg.norm(c.emb - e_cur))
        pid, dp = memory.nearest(c.emb)
        same = pid if dp <= memory.radius else None
        dgeo = (c.fp.geo - cur.geo) / scale.sd
        c.direction = direction(dgeo) if c.op in ("SHIFT", "HYBRID", "RETURN", "MUTATE") else ""
        c.feats = {
            "novelty": novelty(c.emb, memory, t, half_life=120.0 * k),
            "distance": d,
            "continuity": math.exp(-d / (0.35 + 0.5 * ctl.mutation)),
            "identity": float(np.dot(c.fp.w, vocab.identity)),
            "user": float(math.tanh(np.dot(bias, dgeo) / 2.0)) if user.present else 0.0,
            "repetition": memory.repetition(same, t, 120.0 * k),
            "oscillation": 1.0 if memory.oscillating(same, t) else 0.0,
            "pull": (pulls.get(c.proto, 0.0) * (0.6 if c.op == "MUTATE" else 1.0)) if c.proto is not None else 0.0,
            "radical": 1.0 if d > RADICAL else 0.0,
        }


__all__ = ["Candidate", "UserCue", "generate", "evaluate", "predict", "material_vector", "direction", "DIRECTIONS",
           "RADICAL"]
