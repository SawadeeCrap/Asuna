"""The morphology brain: an optional, isolated layer that makes an organism explore its own morphology.

Memory (what it has been), novelty (what would be new), candidates (what it can actually become), an
arbiter (deterministic, or a local Kev decision model), a low decision rate - and the engine doing the
continuous transformation.  See docs/MORPHOLOGY_BRAIN.md.
"""
from .core import MODES, BrainConfig, BrainControls, BrainCore, Decision, Snapshot

__all__ = ["BrainCore", "BrainConfig", "BrainControls", "Decision", "Snapshot", "MODES"]
