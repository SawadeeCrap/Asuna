"""CreatureControlInput: the generic music/control layer, normalised for the creature."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

from .config import DEFAULT_PARAMS, PARAMS


def _clean(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return lo
    if not math.isfinite(v):
        return lo
    return min(hi, max(lo, v))


@dataclass
class CreatureControlInput:
    bass: float = 0.0
    mid: float = 0.0
    high: float = 0.0
    energy: float = 0.0
    transient: float = 0.0          # onset strength this tick (kick-like hits)
    spectral_flux: float = 0.0
    amplitude: float = 0.0
    tempo: float = 120.0
    beat: float = 0.0
    beat_phase: float = 0.0
    playing: bool = False

    def sanitized(self) -> "CreatureControlInput":
        d = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "tempo":
                d[f.name] = _clean(v, 30.0, 300.0) if _clean(v, 0.0, 1e9) > 0 else 120.0
            elif f.name == "beat":
                d[f.name] = float(v) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
            elif f.name == "playing":
                d[f.name] = bool(v)
            else:
                d[f.name] = _clean(v)
        return CreatureControlInput(**d)

    @classmethod
    def from_frame(cls, fr, notes=(), playing: bool = True) -> "CreatureControlInput":
        """From the existing FeatureExtractor ControlFrame (+ this tick's onsets)."""
        g = getattr(fr, "groups", {}) or {}
        hit = 0.0
        for n in notes:
            if (n.group or "") in ("kick", "snare", "perc", "bass", "hit"):
                hit = max(hit, float(n.velocity))
        return cls(bass=fr.low_activity, mid=max(g.get("snare", 0.0), g.get("melody", 0.0), g.get("harmony", 0.0)),
                   high=fr.high_activity, energy=fr.energy, transient=max(hit, 0.6 * fr.impulse if hit else 0.0),
                   spectral_flux=min(1.0, 0.7 * fr.novelty + 0.3 * abs(fr.trend)), amplitude=max(fr.energy, fr.impulse),
                   tempo=fr.tempo, beat=fr.beat, beat_phase=fr.beat_phase, playing=playing).sanitized()


class ParameterSet:
    """Automatic values (behaviour) + manual overrides (MIDI / CV / app); manual None = automatic."""

    def __init__(self, defaults: dict | None = None):
        self.auto = dict(DEFAULT_PARAMS)
        self.auto.update(defaults or {})
        self.manual: dict[str, float] = {}

    def set(self, name: str, value: float | None) -> bool:
        if name not in PARAMS:
            return False
        if value is None or not math.isfinite(float(value)) or float(value) < 0.0:
            self.manual.pop(name, None)
        else:
            self.manual[name] = _clean(value)
        return True

    def __getitem__(self, name: str) -> float:
        return self.manual.get(name, self.auto[name])

    def values(self) -> dict:
        return {p: self[p] for p in PARAMS}
