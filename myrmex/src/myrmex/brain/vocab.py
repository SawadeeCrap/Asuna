"""What an organism can become - read from its engine class, never hard-coded.

* ``forms``: the latent space the engine blends (colony family: ``SHAPE_SET``; Mimetic Polyalloy:
  ``ATTRACTORS``; Bionic line: ``REGIMES``; Black Nanomaterial: ``MORPHS``) - the order of the engine's ``z``;
* ``free``: the forms the brain may ask for (situational ones - legs only when perched, the shell only
  round prey - excluded);
* ``identity``: how much each form is *this* organism, from the author's own plans (``VOCAB``, every
  intent's preferred forms, the glove's sculpt forms) - the brain explores around it, not away from it;
* ``events``: the morphological events the organism knows (split, merge, bursts ...);
* ``geometry``: each form's point cloud from the engine's own shape functions (colony / polyalloy
  families), so a blend's shape can be predicted before it is asked for.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Events that change the body's structure (the brain may ask for them); reactive ones stay the engine's.
MORPH_EVENTS = ("SPLIT", "MERGE", "APPENDAGE_BURST", "COLLAPSE", "MASS_REBALANCE", "WAVE", "OSSIFY", "QUILLS",
                "SCATTER", "GATHER", "RECONFIGURE", "BUILD", "RECALL", "SPROUT", "SHED", "PULSE", "COIL", "UNFURL",
                "CLAP", "FURL", "BLOOM", "CALM", "ANNEAL")
SITUATIONAL = ("ENVELOP", "LEGS")
MATERIALS = ("FLUID", "ELASTIC", "COHESIVE", "STRUCTURED", "HIGH_STIFFNESS", "DISPERSED")
_PH0 = {"t": 0.0, "beat": 0.0, "spin": 0.0, "flap": 0.0, "pulse": 0.0, "gait": 0.0, "mech": 0.5, "snap": 0.3}


@dataclass
class Vocabulary:
    organism: str
    family: str                       # colony | polyalloy | bionic | creature
    forms: tuple                      # the engine's latent order
    free: tuple                       # forms the brain may ask for
    identity: np.ndarray              # per form, 0..1 (1 = the organism's signature)
    events: tuple                     # morphological events this organism knows
    materials: tuple                  # material states it blends (colony / polyalloy families), else ()
    geometry: np.ndarray | None       # (forms, points, 3) target clouds, or None (Bionic: structure physics)

    def index(self, name: str) -> int:
        return self.forms.index(name)

    def weights(self, blend: dict) -> np.ndarray:
        """{form: share} -> weights over ``forms`` (normalised)."""
        w = np.zeros(len(self.forms))
        for k, v in blend.items():
            if k in self.forms:
                w[self.forms.index(k)] += max(0.0, float(v))
        s = w.sum()
        return w / s if s > 0 else w

    def signature(self) -> str:
        return self.forms[int(np.argmax(self.identity))]


def _identity(forms: tuple, vocab: tuple, plan: dict, sculpt: tuple) -> np.ndarray:
    score = np.zeros(len(forms))
    for prefs in plan.values():
        shapes = prefs[0] if isinstance(prefs, tuple) and prefs and isinstance(prefs[0], tuple) else prefs
        for s in shapes if isinstance(shapes, tuple) else ():
            if s in forms:
                score[forms.index(s)] += 1.0
    for s in vocab:
        if s in forms:
            score[forms.index(s)] += 2.0
    for s in sculpt:
        if s in forms:
            score[forms.index(s)] += 1.0
    return score / score.max() if score.max() > 0 else np.ones(len(forms))


def _colony_geometry(forms: tuple, n: int = 128, seed: int = 7) -> np.ndarray:
    from ..creature.colony import shape3
    U = np.random.default_rng(seed).random((n, 3))
    out = []
    for f in forms:
        P = shape3(f if f not in SITUATIONAL else "CORE", U, 1.0, 1.1, _PH0, -3.0)
        out.append(P - P.mean(0))
    return np.array(out)


def _polyalloy_geometry(forms: tuple, n: int = 96, seed: int = 7) -> np.ndarray:
    from ..creature.polyalloy import attractor_shape
    U = np.random.default_rng(seed).random((n, 3))
    out = []
    for f in forms:
        P = attractor_shape(f, U, 1.0, 1.1)
        out.append(P - P.mean(0))
    return np.array(out)


def vocabulary(engine_or_class, organism: str = "") -> Vocabulary:
    """The vocabulary of an engine (instance or class)."""
    cls = engine_or_class if isinstance(engine_or_class, type) else type(engine_or_class)
    from ..realtime.glove import SCULPT_SHAPES
    sculpt = tuple(SCULPT_SHAPES.get(organism, ()))
    events = tuple(e for e in getattr(cls, "EVENTS", ()) if e in MORPH_EVENTS)
    if hasattr(cls, "SHAPE_SET"):                                    # the colony family (v3 .. v13)
        forms = tuple(cls.SHAPE_SET)
        vocab, plan = tuple(cls.VOCAB), dict(cls.PLAN)
        free = tuple(f for f in forms if f not in SITUATIONAL and (f in vocab or f in sculpt or any(
            f in (p[0] if isinstance(p, tuple) else ()) for p in plan.values())))
        return Vocabulary(organism or cls.__name__, "colony", forms, free, _identity(forms, vocab, plan, sculpt),
                          events, MATERIALS, _colony_geometry(forms))
    if hasattr(cls, "REGIMES") and len(getattr(cls, "REGIMES", ())) > 1:   # the Bionic line (v14 .. v18)
        forms = tuple(cls.REGIMES)
        plan = {k: (v[0],) if isinstance(v, tuple) else v for k, v in getattr(cls, "PLAN", {}).items()}
        return Vocabulary(organism or cls.__name__, "bionic", forms, forms,
                          _identity(forms, (), {k: v for k, v in plan.items()}, sculpt), events, (), None)
    if hasattr(cls, "VOCAB") and hasattr(cls, "PLAN"):               # Mimetic Polyalloy (v2), Osseous (v5)
        from ..creature.polyalloy import ATTRACTORS
        forms = tuple(ATTRACTORS)
        vocab, plan = tuple(cls.VOCAB), dict(cls.PLAN)
        free = tuple(f for f in forms if f in vocab or f in sculpt)
        return Vocabulary(organism or cls.__name__, "polyalloy", forms, free, _identity(forms, vocab, plan, sculpt),
                          events, MATERIALS, _polyalloy_geometry(forms))
    from ..creature.morphology import MORPHS                          # Black Nanomaterial (v1)
    forms = tuple(MORPHS)
    return Vocabulary(organism or "creature", "creature", forms, forms, np.ones(len(forms)), events, (), None)


__all__ = ["Vocabulary", "vocabulary", "MORPH_EVENTS", "MATERIALS", "SITUATIONAL"]
