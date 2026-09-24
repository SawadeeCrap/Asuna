"""Black Nanomaterial Creature: a finite quantity of self-organising material, simulated procedurally.

Layers (each its own module): control input -> behaviour -> dynamics (nodes) -> morphology
(mass field, appendages) -> state for presentation (Blender builds the surface locally).
"""
from .config import PARAMS, CreatureConfig
from .control import CreatureControlInput
from .engine import CreatureEngine, CreatureState

__all__ = ["CreatureConfig", "CreatureControlInput", "CreatureEngine", "CreatureState", "PARAMS"]
