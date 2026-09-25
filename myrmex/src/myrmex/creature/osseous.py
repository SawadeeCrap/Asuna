"""The Osseous line (creatures v5-v7): bony, aggressive versions of the three polyalloy organisms.

Same physics, behaviour and features as their liquid originals - Mimetic Polyalloy (v2),
Polyalloy Colony (v3), Polyalloy Hive (v4) - which stay as they are, plus:

* **Bone-link skeleton** instead of struts: a minimum spanning tree through each body; every link is
  an articulated bone segment (knuckled ends, thin waist, a dorsal crest, twist, hooked tip) that
  grows where the material ossifies, dissolves where it liquefies and keeps mutating its shape.
* **Ossification**: hardness + aggression turn the material to bone; hard nodes thin into beads
  strung along the bones; hardening waves ossify in bands.
* **Bony, aggressive forms**: spine with swept-back thorns, hooked claws, snapping mandibles,
  swinging scythes, quills that jump on kicks, a faceted carapace (the style of bionic vertebrae,
  not their anatomy); the *aggression* knob favours them.
* **Strike**: instead of evading, it can harden into blades and lunge - obstacles are knocked away,
  prey is struck before it is enveloped.  Events STRIKE and OSSIFY.
* Colony / Hive: **bony scutes** swept back into spikes instead of flat plates.  Hive: **quill volley**
  on drops (bristling + nanomachines fired outwards), ring gates grow **fangs**, pattern spines are bone.
"""
from __future__ import annotations

from dataclasses import dataclass

from .colony import EVENTS as COLONY_EVENTS
from .colony import FREE_SHAPES, PLAN_BONE, ColonyConfig, ColonyEngine
from .hive import EVENTS as HIVE_EVENTS
from .hive import HiveConfig, HiveEngine
from .polyalloy import ATTRACTORS, INTENT_PLAN_BONE, PolyalloyConfig, PolyalloyEngine


@dataclass
class OsseousPolyalloyConfig(PolyalloyConfig):
    max_links: int = 128                  # bone-link slots


class OsseousPolyalloyEngine(PolyalloyEngine):
    BONY = True
    PLAN = INTENT_PLAN_BONE
    VOCAB = ATTRACTORS
    EVENTS = PolyalloyEngine.EVENTS + ("STRIKE", "OSSIFY")

    def __init__(self, cfg: PolyalloyConfig | None = None):
        super().__init__(cfg or OsseousPolyalloyConfig())


@dataclass
class OsseousColonyConfig(ColonyConfig):
    max_links: int = 192


class OsseousColonyEngine(ColonyEngine):
    BONY = True
    PLAN = PLAN_BONE
    VOCAB = FREE_SHAPES
    EVENTS = COLONY_EVENTS + ("STRIKE", "OSSIFY")

    def __init__(self, cfg: ColonyConfig | None = None):
        super().__init__(cfg or OsseousColonyConfig())


@dataclass
class OsseousHiveConfig(HiveConfig):
    max_links: int = 192


class OsseousHiveEngine(HiveEngine):
    BONY = True
    PLAN = PLAN_BONE
    VOCAB = FREE_SHAPES
    EVENTS = HIVE_EVENTS + ("STRIKE", "OSSIFY", "QUILLS")

    def __init__(self, cfg: HiveConfig | None = None):
        super().__init__(cfg or OsseousHiveConfig())


VARIANTS = {"osseous": (OsseousPolyalloyEngine, OsseousPolyalloyConfig),
            "osseous_colony": (OsseousColonyEngine, OsseousColonyConfig),
            "osseous_hive": (OsseousHiveEngine, OsseousHiveConfig)}

__all__ = ["OsseousPolyalloyEngine", "OsseousColonyEngine", "OsseousHiveEngine", "OsseousPolyalloyConfig",
           "OsseousColonyConfig", "OsseousHiveConfig", "VARIANTS"]
