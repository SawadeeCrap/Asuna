"""Streaming, causal feature extraction: the *fast* musical layer.

The same :class:`FeatureExtractor` runs in the offline generator and in the
realtime engine, so a baked performance and a live one see identical inputs.

Each tick it consumes the note/hit onsets that happened since the previous
tick (from any source) plus sounding notes and control curves, and produces a
:class:`ControlFrame` of normalised features.  Normalisation is *adaptive*
(slow automatic gain control), so a quiet ambient piece and a loud techno
track both span the useful range without manual calibration.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .timeline import PERCUSSIVE, MusicTimeline, NoteEvent

GROUP_ENERGY_WEIGHT = {"kick": 1.0, "snare": 0.85, "hats": 0.35, "perc": 0.5, "bass": 0.8, "melody": 0.6,
                       "harmony": 0.45, "texture": 0.3, "fx": 0.35, "clock": 0.0, "cv": 0.3, "other": 0.4}
LOW_GROUPS = ("kick", "bass")
HIGH_GROUPS = ("hats", "perc")
FEATURE_GROUPS = ("kick", "snare", "hats", "perc", "bass", "melody", "harmony", "texture", "fx")


@dataclass
class ControlFrame:
    time: float
    energy: float = 0.0
    density: float = 0.0
    pitch: float = 0.5
    pitch_range: float = 0.0
    pitch_velocity: float = 0.0
    velocity: float = 0.0
    impulse: float = 0.0
    rhythm: float = 0.0            # regularity 0..1
    activity: float = 0.0
    novelty: float = 0.0
    low_activity: float = 0.0
    high_activity: float = 0.0
    silence: float = 1.0
    silence_contrast: float = 0.0  # silence right after strong activity
    trend: float = 0.0             # energy slope, roughly -1..1 per 8 s
    acceleration: float = 0.0
    contrast: float = 0.0
    repetition: float = 0.0
    beat: float = 0.0              # absolute beat position
    beat_phase: float = 0.0
    bar_phase: float = 0.0
    bar: int = 0
    tempo: float = 120.0
    groups: dict[str, float] = field(default_factory=dict)     # activity per group
    impulses: dict[str, float] = field(default_factory=dict)   # transient envelope per group
    rates: dict[str, float] = field(default_factory=dict)      # onsets / s per group

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not isinstance(v, dict)}
        for g, v in self.groups.items():
            d[f"{g}_activity"] = v
        for g, v in self.impulses.items():
            d[f"{g}_impulse"] = v
        for g, v in self.rates.items():
            d[f"{g}_density"] = min(1.0, math.log1p(v) / math.log1p(12.0))
        return d


class _AGC:
    """Slow automatic gain: tracks a decaying running maximum."""

    def __init__(self, decay_s: float = 25.0, floor: float = 1e-3):
        self.peak = floor
        self.decay_s = decay_s
        self.floor = floor

    def __call__(self, x: float, dt: float) -> float:
        self.peak = max(x, self.peak * math.exp(-dt / self.decay_s), self.floor)
        return x / self.peak


class History:
    """Fixed-rate ring buffer of feature snapshots (the musical memory)."""

    def __init__(self, rate: float = 10.0, seconds: float = 64.0):
        self.rate = rate
        self.maxlen = int(rate * seconds)
        self.times: deque[float] = deque(maxlen=self.maxlen)
        self.rows: deque[np.ndarray] = deque(maxlen=self.maxlen)
        self._next = 0.0

    def push(self, t: float, row: np.ndarray) -> bool:
        if t + 1e-9 < self._next:
            return False
        self._next = t + 1.0 / self.rate
        self.times.append(t)
        self.rows.append(row)
        return True

    def window(self, seconds: float) -> np.ndarray:
        n = min(len(self.rows), int(seconds * self.rate))
        if n <= 0:
            return np.zeros((0, 0))
        return np.array(list(self.rows)[-n:])


class FeatureExtractor:
    ROW = ("energy", "density", "pitch", "low", "high", "velocity", "rhythm")

    def __init__(self, timeline: MusicTimeline | None = None, energy_ref: float | None = None):
        self.tl = timeline
        self.energy_ref = energy_ref       # offline: global normalisation reference (keeps true dynamics)
        self.raw_energy = 0.0
        self.t = 0.0
        self.imp = {g: 0.0 for g in FEATURE_GROUPS}
        self.act = {g: 0.0 for g in FEATURE_GROUPS}
        self.rate = {g: 0.0 for g in FEATURE_GROUPS}
        self.sounding: list[NoteEvent] = []
        self.agc_energy = _AGC(25.0, 0.05)
        self.energy_s = 0.0
        self.vel_ema = 0.0
        self.pitch_s = 0.5
        self.pitch_prev = 0.5
        self.pitch_range_s = 0.0
        self.last_onset = -1e9
        self.silence_start_energy = 0.0
        self.onsets: deque[tuple[float, float]] = deque(maxlen=256)   # (beat or time, velocity)
        self.hist = History(10.0, 64.0)
        self.bar_patterns: deque[np.ndarray] = deque(maxlen=16)
        self._cur_bar = -1
        self._cur_pattern = np.zeros(16)
        self.frame = ControlFrame(0.0)

    # ------------------------------------------------------------------ feed
    def group_of(self, n: NoteEvent) -> str:
        if n.group:
            return n.group
        if self.tl is not None and n.track in self.tl.tracks:
            return self.tl.tracks[n.track].group
        return "other"

    def update(self, t: float, onsets: list[NoteEvent], curves: dict[str, float] | None = None, *,
               beat: float | None = None, tempo: float | None = None,
               beats_per_bar: float | None = None) -> ControlFrame:
        """``beat``/``tempo`` override the timeline's tempo map (live clock input)."""
        dt = max(1e-4, t - self.t) if self.t > 0 else 1.0 / 120.0
        self.t = t
        tl = self.tl
        if beat is None:
            tempo = tl.tempo.bpm_at(t) if tl else (tempo or 120.0)
            beat = tl.tempo.beats(t) if tl else t * tempo / 60.0
            bpb = tl.tempo.beats_per_bar(beat) if tl else 4.0
            bar, in_bar = (tl.tempo.bar_position(beat) if tl else (int(beat // bpb), beat % bpb))
        else:
            tempo = tempo or 120.0
            bpb = beats_per_bar or 4.0
            bar, in_bar = int(beat // bpb), beat % bpb
        # ---- decay envelopes
        for g in FEATURE_GROUPS:
            self.imp[g] *= math.exp(-dt / 0.09)
            self.act[g] *= math.exp(-dt / 0.45)
            self.rate[g] *= math.exp(-dt / 2.0)
        # ---- new onsets
        vel_sum = 0.0
        for n in onsets:
            g = self.group_of(n)
            if g not in self.imp:
                g = "texture" if g in ("other", "cv") else g
                if g not in self.imp:
                    continue
            v = float(n.velocity)
            self.imp[g] = max(self.imp[g], v * (0.6 + 0.4 * n.sharpness))
            self.act[g] = min(1.5, self.act[g] + 0.55 * v)
            self.rate[g] += 1.0 / 2.0
            vel_sum += v
            self.vel_ema += (v - self.vel_ema) * 0.15
            self.last_onset = t
            if g in PERCUSSIVE or n.duration < 0.2:
                self.onsets.append((beat, v))
            step = int((in_bar / bpb) * 16) % 16
            self._cur_pattern[step] = max(self._cur_pattern[step], v)
            if n.duration > 0.12 and g not in ("kick", "snare", "hats", "perc"):
                self.sounding.append(n)
        # ---- sounding notes (sustain)
        self.sounding = [n for n in self.sounding if n.time + n.duration > t]
        sus = {g: 0.0 for g in FEATURE_GROUPS}
        pitches, weights = [], []
        for n in self.sounding:
            g = self.group_of(n)
            if g in sus:
                sus[g] = max(sus[g], 0.5 * n.velocity)
            if g in ("bass", "melody", "harmony", "texture"):
                pitches.append(n.pitch)
                weights.append(n.velocity * (1.4 if g == "melody" else 1.0))
        for g in FEATURE_GROUPS:
            self.act[g] = max(self.act[g], sus[g])
        # ---- bar patterns (for repetition)
        if bar != self._cur_bar:
            if self._cur_bar >= 0:
                self.bar_patterns.append(self._cur_pattern.copy())
            self._cur_pattern = np.zeros(16)
            self._cur_bar = bar
        # ---- aggregate features
        raw_energy = sum(GROUP_ENERGY_WEIGHT[g] * min(1.0, self.act[g]) for g in FEATURE_GROUPS)
        raw_energy += 0.6 * sum(GROUP_ENERGY_WEIGHT[g] * self.imp[g] for g in FEATURE_GROUPS)
        if curves:
            raw_energy += 0.3 * float(np.mean(list(curves.values())))
        self.raw_energy = raw_energy
        e = min(1.25, raw_energy / self.energy_ref) if self.energy_ref else self.agc_energy(raw_energy, dt)
        self.energy_s += (e - self.energy_s) * (1.0 - math.exp(-dt / (0.25 if e > self.energy_s else 1.2)))
        total_rate = sum(self.rate.values())
        density = min(1.0, math.log1p(total_rate) / math.log1p(14.0))
        if pitches:
            w = np.asarray(weights)
            p = np.asarray(pitches)
            pc = float((p * w).sum() / w.sum())
            self.pitch_s += (min(1.0, max(0.0, (pc - 24.0) / 72.0)) - self.pitch_s) * (1.0 - math.exp(-dt / 0.35))
            rng = float(p.max() - p.min()) / 36.0
            self.pitch_range_s += (min(1.0, rng) - self.pitch_range_s) * (1.0 - math.exp(-dt / 0.6))
        pv = (self.pitch_s - self.pitch_prev) / dt
        self.pitch_prev = self.pitch_s
        low = min(1.0, max(self.act["kick"], self.act["bass"]))
        high = min(1.0, max(self.act["hats"], self.act["perc"], 0.7 * self.act["melody"] if self.pitch_s > 0.6 else 0.0))
        impulse = min(1.0, max(self.imp.values()))
        silent_for = t - self.last_onset if not self.sounding else 0.0
        silence = min(1.0, max(0.0, (silent_for - 0.25) / 1.25))
        if silent_for < 0.05:
            self.silence_start_energy = self.energy_s
        silence_contrast = silence * self.silence_start_energy
        rhythm = self._regularity(beat)
        activity = sum(1 for g in FEATURE_GROUPS if self.act[g] > 0.08) / len(FEATURE_GROUPS)
        row = np.array([self.energy_s, density, self.pitch_s, low, high, self.vel_ema, rhythm])
        self.hist.push(t, row)
        trend, accel, contrast, novelty = self._memory_features(row)
        repetition = self._repetition()
        fr = ControlFrame(
            time=t, energy=float(min(1.0, self.energy_s)), density=density, pitch=self.pitch_s,
            pitch_range=self.pitch_range_s, pitch_velocity=float(max(-1.0, min(1.0, pv))),
            velocity=self.vel_ema, impulse=impulse, rhythm=rhythm, activity=activity, novelty=novelty,
            low_activity=low, high_activity=high, silence=silence, silence_contrast=silence_contrast,
            trend=trend, acceleration=accel, contrast=contrast, repetition=repetition, beat=beat,
            beat_phase=beat % 1.0, bar_phase=in_bar / bpb, bar=bar, tempo=tempo,
            groups={g: min(1.0, self.act[g]) for g in FEATURE_GROUPS},
            impulses=dict(self.imp), rates=dict(self.rate))
        self.frame = fr
        return fr

    # ------------------------------------------------------------------ helpers
    def _regularity(self, beat: float) -> float:
        recent = [(b, v) for b, v in self.onsets if beat - b < 8.0]
        if len(recent) < 4:
            return 0.0
        # Fraction of onset weight close to the 1/16 grid, and stability of inter-onset intervals.
        dev = [abs(((b * 4.0) % 1.0) - 0.5) * 2.0 for b, _ in recent]    # 1 = on grid
        on_grid = float(np.mean([d > 0.7 for d in dev]))
        bs = np.array([b for b, _ in recent])
        ioi = np.diff(np.unique(np.round(bs * 4.0) / 4.0))
        if len(ioi) < 3:
            return on_grid * 0.5
        vals, counts = np.unique(np.round(ioi, 2), return_counts=True)
        concentration = float(counts.max() / counts.sum())
        return float(min(1.0, 0.55 * on_grid + 0.45 * (0.3 + 0.7 * concentration)))

    def _memory_features(self, row: np.ndarray) -> tuple[float, float, float, float]:
        w8 = self.hist.window(8.0)
        if len(w8) < 10:
            return 0.0, 0.0, 0.0, 0.0
        e = w8[:, 0]
        x = np.arange(len(e)) / self.hist.rate
        slope = float(np.polyfit(x, e, 1)[0]) * 8.0            # change per 8 s
        half = len(e) // 2
        s1 = float(np.polyfit(x[:half], e[:half], 1)[0]) if half > 3 else 0.0
        s2 = float(np.polyfit(x[half:], e[half:], 1)[0]) if len(e) - half > 3 else 0.0
        accel = (s2 - s1) * 8.0
        contrast = float(abs(row[0] - e.mean()))
        w1 = self.hist.window(1.0)
        novelty = 0.0
        if len(w1) >= 3:
            a = w1.mean(axis=0)
            b = w8.mean(axis=0)
            sd = w8.std(axis=0) + 0.05
            novelty = float(min(1.0, np.sqrt(np.mean(((a - b) / sd) ** 2)) / 2.5))
        return (float(max(-1.0, min(1.0, slope))), float(max(-1.0, min(1.0, accel))),
                min(1.0, contrast), novelty)

    def _repetition(self) -> float:
        if len(self.bar_patterns) < 2:
            return 0.0
        last = self.bar_patterns[-1]
        sims = []
        for p in list(self.bar_patterns)[-5:-1]:
            na, nb = np.linalg.norm(last), np.linalg.norm(p)
            if na < 1e-6 or nb < 1e-6:
                continue
            sims.append(float(np.dot(last, p) / (na * nb)))
        return float(np.mean(sims)) if sims else 0.0


def energy_reference(tl: MusicTimeline, rate: float = 30.0, percentile: float = 95.0) -> float:
    """Global loudness reference for offline normalisation (pre-pass)."""
    fx = FeatureExtractor(tl)
    n = int(math.ceil(tl.duration * rate)) + 1
    notes = tl.notes
    k = 0
    vals = []
    for i in range(n):
        t = i / rate
        batch = []
        while k < len(notes) and notes[k].time <= t:
            batch.append(notes[k])
            k += 1
        fx.update(t, batch)
        vals.append(fx.raw_energy)
    ref = float(np.percentile(vals, percentile)) if vals else 1.0
    return max(ref, 0.05)


def analyze_features(tl: MusicTimeline, rate: float = 60.0, normalize: str = "global") -> dict[str, np.ndarray]:
    """Run the streaming extractor over a whole timeline; returns feature arrays."""
    ref = energy_reference(tl) if normalize == "global" else None
    fx = FeatureExtractor(tl, energy_ref=ref)
    n = int(math.ceil(tl.duration * rate)) + 1
    notes = tl.notes
    k = 0
    cols: dict[str, list] = {}
    for i in range(n):
        t = i / rate
        batch = []
        while k < len(notes) and notes[k].time <= t:
            batch.append(notes[k])
            k += 1
        fr = fx.update(t, batch)
        for key, v in fr.as_dict().items():
            cols.setdefault(key, []).append(v)
    return {key: np.asarray(v, dtype=float) for key, v in cols.items()}
