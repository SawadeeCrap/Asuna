"""The Bionic line (creatures v14-v18): bodies whose form is structural physics - no blobs, no beads.

    tensor  v14  a tensegrity spine: rigid struts in a net of muscle cables
    fold    v15  a rigid-foldable Miura-ori sheet
    arbor   v16  a vascular tree grown by space colonisation, pruned in silence
    ferro   v17  ferrofluid led by an invisible magnet (Rosensweig spikes, labyrinths, droplets)
    truss   v18  a flying variable-geometry truss that remodels itself like bone (Wolff's law)
"""
from __future__ import annotations

from .arbor import ArborConfig, ArborEngine
from .base import KINDS, STYLE0, BionicConfig, BionicEngine, BionicState
from .ferro import FerroConfig, FerroEngine
from .fold import FoldConfig, FoldEngine
from .tensor import TensorConfig, TensorEngine
from .truss import TrussConfig, TrussEngine

VARIANTS = {"tensor": (TensorEngine, TensorConfig), "fold": (FoldEngine, FoldConfig),
            "arbor": (ArborEngine, ArborConfig), "ferro": (FerroEngine, FerroConfig),
            "truss": (TrussEngine, TrussConfig)}
BIONIC = tuple(VARIANTS)                      # == KINDS
REGIMES = tuple(dict.fromkeys(r for e, _ in VARIANTS.values() for r in e.REGIMES))
BIONIC_EVENTS = tuple(dict.fromkeys(ev for e, _ in VARIANTS.values() for ev in e.EVENTS
                                    if ev not in BionicEngine.EVENTS))

__all__ = ["VARIANTS", "BIONIC", "REGIMES", "BIONIC_EVENTS", "KINDS", "STYLE0", "BionicConfig", "BionicEngine",
           "BionicState", "TensorEngine", "FoldEngine", "ArborEngine", "FerroEngine", "TrussEngine"]
