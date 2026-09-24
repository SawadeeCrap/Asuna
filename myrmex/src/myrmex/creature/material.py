"""MassField: the creature is a finite amount of material.  Volumes always sum to the total."""
from __future__ import annotations

import numpy as np


class MassField:
    def __init__(self, total: float, min_core_fraction: float, size: float):
        self.total = float(total)
        self.min_core = float(min_core_fraction)
        self.size = float(size)

    def allocate(self, w_primary: np.ndarray, w_secondary: np.ndarray, app_volumes: np.ndarray,
                 primary_frac: float, secondary_frac: float) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Core takes what the structures leave.  Returns (core, primary, secondary, appendage) volumes."""
        T = self.total
        wp = np.maximum(w_primary, 1e-6)
        ws = np.maximum(w_secondary, 1e-6)
        vp = T * primary_frac * wp / wp.sum()
        vs = T * secondary_frac * ws / ws.sum()
        va = np.maximum(app_volumes, 0.0)
        others = vp.sum() + vs.sum() + va.sum()
        cap = T * (1.0 - self.min_core)
        if others > cap:                                  # not enough material: everything shrinks
            s = cap / others
            vp, vs, va = vp * s, vs * s, va * s
            others = cap
        return T - others, vp, vs, va

    def radius(self, volume: np.ndarray | float) -> np.ndarray:
        """Visual (metaball influence) radius of a volume share, in metres."""
        v = np.maximum(np.asarray(volume, float), 0.0) / self.total
        return 0.62 * self.size * np.cbrt(v)
