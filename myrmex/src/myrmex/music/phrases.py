"""Musical structure: sections, phrase boundaries and salient events.

Offline (whole timeline known): bar-level feature vectors -> self-similarity
matrix -> Foote checkerboard novelty -> boundaries (with a preference for
4/8-bar phrase grids and for Ableton locators) -> labelled sections
(intro / build / peak / drop / break / return / groove / outro).

Online: :class:`EventDetector` turns the feature stream into discrete events
(phrase boundary, drop, break, silence start/end, accent, pitch jump) using
only the past.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .timeline import MusicTimeline

BAR_KEYS = ("energy", "density", "low_activity", "high_activity", "pitch", "silence",
            "kick_activity", "snare_activity", "hats_activity", "perc_activity", "bass_activity",
            "melody_activity", "harmony_activity", "texture_activity", "fx_activity")


@dataclass
class Section:
    start: float
    end: float
    start_bar: int
    end_bar: int
    label: str
    energy: float
    density: float
    trend: float
    novelty: float = 0.0
    name: str = ""

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Structure:
    sections: list[Section]
    bar_times: np.ndarray
    novelty: np.ndarray
    bar_features: np.ndarray

    def section_at(self, t: float) -> Section | None:
        for s in self.sections:
            if s.start <= t < s.end:
                return s
        return self.sections[-1] if self.sections else None

    def next_boundary(self, t: float) -> float | None:
        for s in self.sections:
            if s.start > t + 1e-6:
                return s.start
        return None


def bar_matrix(tl: MusicTimeline, F: dict[str, np.ndarray], rate: float) -> tuple[np.ndarray, np.ndarray]:
    total_beats = tl.tempo.beats(tl.duration)
    bpb = tl.tempo.beats_per_bar(0.0)
    n_bars = max(1, int(math.ceil(total_beats / bpb - 1e-6)))
    bar_times = np.array([tl.tempo.seconds(i * bpb) for i in range(n_bars + 1)])
    X = np.zeros((n_bars, len(BAR_KEYS)))
    for i in range(n_bars):
        a = int(bar_times[i] * rate)
        b = max(a + 1, int(bar_times[i + 1] * rate))
        for j, k in enumerate(BAR_KEYS):
            arr = F.get(k)
            if arr is not None and a < len(arr):
                X[i, j] = float(np.mean(arr[a:b]))
    return X, bar_times


def foote_novelty(X: np.ndarray, half: int = 4) -> np.ndarray:
    n = len(X)
    if n < 2:
        return np.zeros(n)
    Z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    S = (Z / norms) @ (Z / norms).T
    L = half
    k = np.zeros((2 * L, 2 * L))
    k[:L, :L] = 1
    k[L:, L:] = 1
    k[:L, L:] = -1
    k[L:, :L] = -1
    g = np.exp(-((np.arange(2 * L) - L + 0.5) ** 2) / (2 * (L * 0.6) ** 2))
    k *= np.outer(g, g)
    Sp = np.pad(S, L, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        nov[i] = float((Sp[i:i + 2 * L, i:i + 2 * L] * k).sum())
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-9)


def analyze_structure(tl: MusicTimeline, F: dict[str, np.ndarray], rate: float,
                      min_bars: int = 4) -> Structure:
    X, bar_times = bar_matrix(tl, F, rate)
    n = len(X)
    nov = np.maximum(foote_novelty(X, 4), 0.7 * foote_novelty(X, 2))
    # Energy jumps are strong boundary cues on their own.
    e = X[:, 0]
    jump = np.zeros(n)
    jump[1:] = np.abs(np.diff(e)) / (np.ptp(e) + 1e-6)
    score = 0.7 * nov + 0.5 * jump
    grid_bonus = np.array([1.25 if i % 8 == 0 else 1.12 if i % 4 == 0 else 1.0 for i in range(n)])
    score = score * grid_bonus
    marker_bars = set()
    for m in tl.markers:
        beat = tl.tempo.beats(m.time)
        marker_bars.add(int(round(beat / tl.tempo.beats_per_bar(beat))))
    thr = float(np.median(score) + 0.6 * np.std(score))
    cands = sorted(range(1, n), key=lambda i: -score[i])
    bounds = [0]
    for i in cands:
        if i in marker_bars or score[i] >= thr:
            if all(abs(i - b) >= min_bars for b in bounds):
                bounds.append(i)
    for mb in marker_bars:
        if 0 < mb < n and all(abs(mb - b) >= 2 for b in bounds):
            bounds.append(mb)
    bounds = sorted(bounds) + [n]
    sections: list[Section] = []
    energies = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = X[a:b]
        en = float(seg[:, 0].mean())
        dn = float(seg[:, 1].mean())
        tr = float(np.polyfit(np.arange(len(seg)), seg[:, 0], 1)[0] * len(seg)) if len(seg) > 2 else 0.0
        energies.append(en)
        sections.append(Section(float(bar_times[a]), float(bar_times[b]), a, b, "groove", en, dn, tr,
                                float(score[a]) if a < n else 0.0))
    _label_sections(sections)
    for m in tl.markers:
        for s in sections:
            if abs(s.start - m.time) < 0.5 * (bar_times[1] - bar_times[0] if len(bar_times) > 1 else 2.0):
                s.name = m.name
    return Structure(sections, bar_times, nov, X)


def _label_sections(sections: list[Section]) -> None:
    if not sections:
        return
    en = np.array([s.energy for s in sections])
    lo, hi = np.percentile(en, 33), np.percentile(en, 67)
    span = max(en.max() - en.min(), 1e-6)
    if span < 0.12:
        for s in sections:
            s.label = "silence" if (s.energy < 0.08 and s.density < 0.1) else "groove"
        return
    for i, s in enumerate(sections):
        prev = sections[i - 1] if i else None
        rel = (s.energy - en.min()) / span
        if s.energy < 0.08 and s.density < 0.1:
            s.label = "silence"
        elif i == 0 and rel < 0.5:
            s.label = "intro"
        elif i == len(sections) - 1 and prev is not None and s.energy < prev.energy - 0.1 * span:
            s.label = "outro"
        elif s.trend > 0.15 * span and rel < 0.9:
            s.label = "build"
        elif s.energy >= hi:
            if prev is not None and (prev.label in ("build", "break", "intro", "silence") or prev.energy < lo):
                s.label = "drop"
            else:
                s.label = "peak"
        elif s.energy <= lo and prev is not None and prev.energy >= hi - 0.05 * span:
            s.label = "break"
        elif prev is not None and prev.label in ("break", "silence") and s.energy > prev.energy:
            s.label = "return"
        else:
            s.label = "groove"


@dataclass
class MusicEvent:
    time: float
    type: str                      # onset|accent|surprise|omission|phrase|drop|break|silence_start|silence_end|pitch_jump
    strength: float = 1.0
    group: str | None = None
    data: dict = field(default_factory=dict)


class EventDetector:
    """Online (causal) discrete events from the feature stream."""

    def __init__(self, beats_per_phrase: int = 16):
        self.bpp = beats_per_phrase
        self.prev_energy = 0.0
        self.e_slow = 0.0
        self.e_fast = 0.0
        self.in_silence = True
        self.last_bar = -1
        self.bar_rows: list[np.ndarray] = []
        self.last_phrase_bar = -99
        self.pitch_prev = None

    def update(self, fr, dt: float) -> list[MusicEvent]:
        ev: list[MusicEvent] = []
        t = fr.time
        a_fast = 1.0 - math.exp(-dt / 0.6)
        a_slow = 1.0 - math.exp(-dt / 6.0)
        self.e_fast += (fr.energy - self.e_fast) * a_fast
        self.e_slow += (fr.energy - self.e_slow) * a_slow
        # Silence transitions.
        if not self.in_silence and fr.silence > 0.6:
            self.in_silence = True
            ev.append(MusicEvent(t, "silence_start", float(max(fr.silence_contrast, 0.2))))
        elif self.in_silence and fr.silence < 0.05 and fr.impulse > 0.2:
            self.in_silence = False
            ev.append(MusicEvent(t, "silence_end", float(fr.impulse)))
        # Drops and breaks: fast energy departs strongly from slow energy.
        if self.e_fast - self.e_slow > 0.28 and fr.impulse > 0.5:
            ev.append(MusicEvent(t, "drop", float(min(1.0, (self.e_fast - self.e_slow) * 2.0))))
            self.e_slow = self.e_fast * 0.85
        elif self.e_slow - self.e_fast > 0.3:
            ev.append(MusicEvent(t, "break", float(min(1.0, (self.e_slow - self.e_fast) * 2.0))))
            self.e_slow = self.e_fast * 1.1
        # Phrase boundaries: bar-level change detection on 4-bar grids.
        if fr.bar != self.last_bar:
            if self.last_bar >= 0:
                row = np.array([fr.energy, fr.density, fr.low_activity, fr.high_activity, fr.pitch])
                self.bar_rows.append(row)
                if len(self.bar_rows) >= 5 and fr.bar % 4 == 0 and fr.bar - self.last_phrase_bar >= 4:
                    prev = np.array(self.bar_rows[-5:-1]).mean(axis=0)
                    d = float(np.linalg.norm(row - prev))
                    if d > 0.22 or fr.bar % self.bpp == 0:
                        ev.append(MusicEvent(t, "phrase", min(1.0, d * 2.0 + 0.3), data={"bar": fr.bar}))
                        self.last_phrase_bar = fr.bar
            self.last_bar = fr.bar
        # Pitch jumps.
        if self.pitch_prev is not None and abs(fr.pitch - self.pitch_prev) > 0.08:
            ev.append(MusicEvent(t, "pitch_jump", float(min(1.0, abs(fr.pitch - self.pitch_prev) * 6.0)),
                                 data={"direction": 1.0 if fr.pitch > self.pitch_prev else -1.0}))
        self.pitch_prev = fr.pitch
        return ev
