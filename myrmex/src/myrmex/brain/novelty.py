"""Novelty pressure - plain arithmetic, no model: how new would this form be *now*?

Novelty search style: the distance to the k nearest remembered forms, where a form seen long ago counts as
further away than one seen a moment ago (``half_life``) - so an old form can be new again, and returning
to it is not repetition.  Repetition (the same form arriving again and again) and oscillation (A -> B -> A)
are separate penalties.
"""
from __future__ import annotations

import numpy as np

from .memory import MorphMemory


def novelty(emb: np.ndarray, memory: MorphMemory, t: float, k: int = 4, half_life: float = 120.0,
            fade: float = 0.6) -> float:
    """0 = exactly what it just was .. ~1+ = far from everything it remembers."""
    if not memory.protos:
        return 1.0
    ps = list(memory.protos.values())
    E = np.stack([p.emb for p in ps])
    d = np.linalg.norm(E - emb, axis=1)
    age = np.array([t - p.last for p in ps])
    recency = 0.5 ** (age / half_life)                     # 1 = just now .. 0 = long forgotten
    d_eff = d + (1.0 - recency) * fade
    k = min(k, len(d_eff))
    return float(np.sort(d_eff)[:k].mean())


def novelty_of_trace(embs: list, k: int = 4) -> list:
    """Offline (metrics): each point's distance to the k nearest points *before* it."""
    out = []
    for i, e in enumerate(embs):
        if i == 0:
            out.append(1.0)
            continue
        d = np.linalg.norm(np.stack(embs[:i]) - e, axis=1)
        out.append(float(np.sort(d)[:min(k, i)].mean()))
    return out


__all__ = ["novelty", "novelty_of_trace"]
