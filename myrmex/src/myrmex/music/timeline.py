"""Source-independent musical data: the *unified* representation.

Ableton sets, MIDI files, audio stems, VCV Rack streams and synthetic test
patterns all become a :class:`MusicTimeline`:

* a :class:`TempoMap` (beats <-> seconds, tempo changes, time signatures),
* tracks with a semantic group (kick, snare, hats, perc, bass, melody,
  harmony, texture, fx, clock, cv) and a confidence,
* note / hit events in **seconds** (and beats when a tempo is known),
* continuous control curves (automation, CV, audio envelopes),
* optional markers (Ableton locators, VCV labels).

Nothing downstream knows where the data came from.
"""
from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass, field

import numpy as np

GROUPS = ("kick", "snare", "hats", "perc", "bass", "melody", "harmony", "texture", "fx", "clock", "cv", "other")
PERCUSSIVE = ("kick", "snare", "hats", "perc", "clock")


@dataclass
class TempoMap:
    """Piecewise-constant tempo: list of (beat, bpm) change points."""
    points: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 120.0)])
    time_signatures: list[tuple[float, int, int]] = field(default_factory=lambda: [(0.0, 4, 4)])

    def __post_init__(self) -> None:
        self.points = sorted((float(b), float(t)) for b, t in self.points) or [(0.0, 120.0)]
        if self.points[0][0] > 0.0:
            self.points.insert(0, (0.0, self.points[0][1]))
        self._rebuild()

    def _rebuild(self) -> None:
        self._beats = [b for b, _ in self.points]
        self._secs = [0.0]
        for i in range(1, len(self.points)):
            b0, t0 = self.points[i - 1]
            b1, _ = self.points[i]
            self._secs.append(self._secs[-1] + (b1 - b0) * 60.0 / t0)

    def seconds(self, beat: float) -> float:
        i = max(0, bisect.bisect_right(self._beats, beat) - 1)
        b0, bpm = self.points[i]
        return self._secs[i] + (beat - b0) * 60.0 / bpm

    def beats(self, sec: float) -> float:
        i = max(0, bisect.bisect_right(self._secs, sec) - 1)
        b0, bpm = self.points[i]
        return b0 + (sec - self._secs[i]) * bpm / 60.0

    def bpm_at(self, sec: float) -> float:
        i = max(0, bisect.bisect_right(self._secs, sec) - 1)
        return self.points[i][1]

    def signature_at(self, beat: float) -> tuple[int, int]:
        cur = self.time_signatures[0]
        for ts in self.time_signatures:
            if ts[0] <= beat:
                cur = ts
        return int(cur[1]), int(cur[2])

    def beats_per_bar(self, beat: float = 0.0) -> float:
        num, den = self.signature_at(beat)
        return num * 4.0 / den

    def bar_position(self, beat: float) -> tuple[int, float]:
        """(bar index, beat within bar) – assumes signature changes land on bar lines."""
        bar = 0
        acc = 0.0
        sigs = sorted(self.time_signatures)
        for i, (b0, num, den) in enumerate(sigs):
            b1 = sigs[i + 1][0] if i + 1 < len(sigs) else float("inf")
            bpb = num * 4.0 / den
            if beat < b1:
                n = int((beat - b0) // bpb)
                return bar + n, beat - b0 - n * bpb
            bar += int(round((b1 - b0) / bpb))
            acc = b1
        return bar, beat - acc


@dataclass
class Track:
    id: str
    name: str
    kind: str = "midi"            # midi | audio | cv | group
    group: str = "other"          # semantic group
    confidence: float = 0.0
    override: bool = False        # group set by the user
    color: int | None = None
    devices: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class NoteEvent:
    time: float                   # seconds
    duration: float               # seconds
    pitch: float                  # MIDI pitch (float allows CV pitch)
    velocity: float               # 0..1
    track: str
    beat: float | None = None
    group: str | None = None      # per-note override (drum-rack pads)
    sharpness: float = 0.5        # transient sharpness 0..1 (audio / derived)


@dataclass
class ControlCurve:
    track: str
    name: str
    times: np.ndarray             # seconds
    values: np.ndarray            # normalised 0..1

    def sample(self, t: float) -> float:
        if len(self.times) == 0:
            return 0.0
        return float(np.interp(t, self.times, self.values))


@dataclass
class Marker:
    time: float
    name: str


@dataclass
class MusicTimeline:
    tempo: TempoMap = field(default_factory=TempoMap)
    tracks: dict[str, Track] = field(default_factory=dict)
    notes: list[NoteEvent] = field(default_factory=list)
    curves: list[ControlCurve] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    duration: float = 0.0
    source: str = "unknown"
    meta: dict = field(default_factory=dict)
    audio_path: str | None = None

    def finalize(self) -> "MusicTimeline":
        self.notes.sort(key=lambda n: (n.time, n.pitch))
        for n in self.notes:
            if n.beat is None:
                n.beat = self.tempo.beats(n.time)
        end = max((n.time + n.duration for n in self.notes), default=0.0)
        for c in self.curves:
            if len(c.times):
                end = max(end, float(c.times[-1]))
        self.duration = max(self.duration, end)
        return self

    def group_of(self, note: NoteEvent) -> str:
        if note.group:
            return note.group
        tr = self.tracks.get(note.track)
        return tr.group if tr else "other"

    def notes_in(self, t0: float, t1: float) -> list[NoteEvent]:
        """Notes whose onset lies in [t0, t1)."""
        times = [n.time for n in self.notes]
        i0 = bisect.bisect_left(times, t0)
        i1 = bisect.bisect_left(times, t1)
        return self.notes[i0:i1]

    def summary(self) -> dict:
        per_track: dict[str, int] = {}
        for n in self.notes:
            per_track[n.track] = per_track.get(n.track, 0) + 1
        return {
            "source": self.source,
            "duration": round(self.duration, 3),
            "tempo": self.tempo.points[:8],
            "time_signatures": self.tempo.time_signatures[:4],
            "tracks": [{"id": t.id, "name": t.name, "kind": t.kind, "group": t.group,
                        "confidence": round(t.confidence, 2), "notes": per_track.get(t.id, 0),
                        "devices": t.devices[:6]} for t in self.tracks.values()],
            "curves": [{"track": c.track, "name": c.name, "points": int(len(c.times))} for c in self.curves[:32]],
            "markers": [{"time": round(m.time, 3), "name": m.name} for m in self.markers],
            "notes": len(self.notes),
        }

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "format": "myrmex.timeline/1",
            "source": self.source,
            "duration": self.duration,
            "tempo": {"points": self.tempo.points, "time_signatures": self.tempo.time_signatures},
            "tracks": [asdict(t) for t in self.tracks.values()],
            "notes": [[n.time, n.duration, n.pitch, n.velocity, n.track, n.group, n.sharpness] for n in self.notes],
            "curves": [{"track": c.track, "name": c.name, "times": c.times.tolist(), "values": c.values.tolist()}
                       for c in self.curves],
            "markers": [[m.time, m.name] for m in self.markers],
            "meta": self.meta,
            "audio_path": self.audio_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MusicTimeline":
        tm = TempoMap([tuple(p) for p in d["tempo"]["points"]],
                      [tuple(p) for p in d["tempo"].get("time_signatures", [(0.0, 4, 4)])])
        tl = cls(tm, source=d.get("source", "file"), duration=float(d.get("duration", 0.0)),
                 meta=d.get("meta", {}), audio_path=d.get("audio_path"))
        for t in d.get("tracks", []):
            tl.tracks[t["id"]] = Track(**t)
        for row in d.get("notes", []):
            tl.notes.append(NoteEvent(row[0], row[1], row[2], row[3], row[4], group=row[5] if len(row) > 5 else None,
                                      sharpness=row[6] if len(row) > 6 else 0.5))
        for c in d.get("curves", []):
            tl.curves.append(ControlCurve(c["track"], c["name"], np.asarray(c["times"]), np.asarray(c["values"])))
        tl.markers = [Marker(float(m[0]), str(m[1])) for m in d.get("markers", [])]
        return tl.finalize()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> "MusicTimeline":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
