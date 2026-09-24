"""Semantic classification of tracks / pads / stems (never requires perfect naming).

Evidence is combined with weights:

* names (track, clip, Drum Rack pad, sample file, device) matched against a
  multilingual keyword table,
* General-MIDI drum map for notes on drum tracks,
* note statistics (pitch centre, range, polyphony, duration, density),
* device hints (Drum Rack / drum machines, bass synth presets).

Every result carries a confidence; user overrides always win.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np

from .timeline import MusicTimeline, NoteEvent, Track

KEYWORDS: dict[str, tuple[str, ...]] = {
    "kick": ("kick", "kik", "bd", "bassdrum", "bass drum", "808 kick", "бочка", "кик", "kck"),
    "snare": ("snare", "snr", "sd", "clap", "clp", "rim", "rimshot", "малый", "снейр", "хлоп"),
    "hats": ("hat", "hh", "hihat", "hi-hat", "hi hat", "cymbal", "ride", "crash", "open hat", "oh", "ch",
             "шейкер", "shaker", "тарел", "хэт", "хет"),
    "perc": ("perc", "tom", "conga", "bongo", "clave", "cowbell", "tamb", "tambourine", "block", "click",
             "перк", "том"),
    "bass": ("bass", "sub", "808", "reese", "бас", "саб"),
    "melody": ("lead", "melody", "arp", "pluck", "synth lead", "vox", "vocal", "voice", "piano", "keys", "bell",
               "guitar", "flute", "мелод", "лид", "вокал", "голос"),
    "harmony": ("pad", "chord", "string", "choir", "organ", "аккорд", "пэд", "пад", "струн"),
    "texture": ("texture", "atmo", "ambient", "drone", "noise", "field", "атмо", "шум"),
    "fx": ("fx", "sfx", "riser", "sweep", "impact", "uplifter", "downlifter", "whoosh", "crash fx", "reverse",
           "эффект", "свип"),
    "clock": ("clock", "clk", "sync"),
}

GM_DRUMS = {35: "kick", 36: "kick", 37: "snare", 38: "snare", 39: "snare", 40: "snare", 41: "perc", 42: "hats",
            43: "perc", 44: "hats", 45: "perc", 46: "hats", 47: "perc", 48: "perc", 49: "hats", 50: "perc",
            51: "hats", 52: "hats", 53: "hats", 54: "perc", 55: "hats", 56: "perc", 57: "hats", 58: "perc",
            59: "hats", 60: "perc", 61: "perc", 62: "perc", 63: "perc", 64: "perc", 65: "perc", 66: "perc",
            67: "perc", 68: "perc", 69: "perc", 70: "perc", 75: "perc", 76: "perc", 77: "perc"}


def keyword_scores(text: str) -> dict[str, float]:
    t = " " + re.sub(r"[_\-\.\d]+", " ", (text or "").lower()) + " "
    scores: dict[str, float] = defaultdict(float)
    for group, words in KEYWORDS.items():
        for w in words:
            if len(w) <= 3:
                if re.search(rf"(^|\s){re.escape(w)}(\s|$)", t):
                    scores[group] += 0.8
            elif w in t:
                scores[group] += 1.0
    return dict(scores)


def note_stats(notes: list[NoteEvent], duration: float) -> dict[str, float]:
    if not notes:
        return {"count": 0}
    p = np.array([n.pitch for n in notes])
    d = np.array([n.duration for n in notes])
    times = np.array([n.time for n in notes])
    uniq, counts = np.unique(np.round(times, 3), return_counts=True)
    return {"count": len(notes), "pitch_mean": float(p.mean()), "pitch_std": float(p.std()),
            "pitch_min": float(p.min()), "pitch_max": float(p.max()), "dur_mean": float(d.mean()),
            "polyphony": float(counts.mean()), "rate": len(notes) / max(duration, 1e-3),
            "distinct_pitches": int(len(np.unique(np.round(p))))}


def classify_track(track: Track, notes: list[NoteEvent], duration: float,
                   drum_pads: dict[int, str] | None = None) -> tuple[str, float]:
    """Returns (group, confidence)."""
    scores: dict[str, float] = defaultdict(float)
    for text in [track.name] + track.devices + [track.meta.get("clip_names", "")]:
        for g, s in keyword_scores(str(text)).items():
            scores[g] += s
    st = note_stats(notes, duration)
    is_drum_device = any(d.lower() in ("drumgroupdevice", "drum rack", "drumrack", "impulse") or "drum" in d.lower()
                         for d in track.devices)
    if st["count"]:
        if is_drum_device or drum_pads:
            scores["perc"] += 0.6
        if st["dur_mean"] > 0.8 and st["polyphony"] >= 2.5:
            scores["harmony"] += 1.2
        elif st["polyphony"] >= 2.0 and st["dur_mean"] > 0.3:
            scores["harmony"] += 0.7
        if st["pitch_mean"] < 48 and st["polyphony"] < 1.6:
            scores["bass"] += 1.1
        if 55 <= st["pitch_mean"] <= 90 and st["polyphony"] < 1.6 and st["distinct_pitches"] >= 4:
            scores["melody"] += 0.8
        if st["distinct_pitches"] <= 2 and st["dur_mean"] < 0.25:
            scores["perc"] += 0.5
        if st["dur_mean"] > 3.0:
            scores["texture"] += 0.6
    elif track.kind == "audio":
        scores["texture"] += 0.2
    if not scores:
        return "other", 0.0
    best = max(scores, key=scores.get)
    total = sum(scores.values())
    return best, float(min(1.0, scores[best] / max(total, 1e-6) * min(1.0, scores[best] / 1.2)))


def classify_drum_note(pitch: int, pad_name: str | None) -> str:
    if pad_name:
        ks = keyword_scores(pad_name)
        if ks:
            return max(ks, key=ks.get)
    return GM_DRUMS.get(int(round(pitch)), "perc")


def apply_classification(tl: MusicTimeline, overrides: dict[str, str] | None = None) -> MusicTimeline:
    """Assign groups to tracks (and per-note groups on drum tracks) unless overridden."""
    overrides = overrides or {}
    by_track: dict[str, list[NoteEvent]] = defaultdict(list)
    for n in tl.notes:
        by_track[n.track].append(n)
    for tid, tr in tl.tracks.items():
        ov = overrides.get(tid) or overrides.get(tr.name)
        if ov:
            tr.group, tr.confidence, tr.override = ov, 1.0, True
            continue
        if tr.override:
            continue
        pads = tr.meta.get("drum_pads") or {}
        g, c = classify_track(tr, by_track.get(tid, []), tl.duration, pads)
        tr.group, tr.confidence = g, c
        if pads or g in ("perc",) and any(d.lower().startswith("drum") for d in tr.devices):
            # Split drum tracks per pad / GM note.
            for n in by_track.get(tid, []):
                if n.group is None:
                    n.group = classify_drum_note(n.pitch, pads.get(int(round(n.pitch))))
    return tl


def group_histogram(tl: MusicTimeline) -> dict[str, int]:
    c: Counter = Counter()
    for n in tl.notes:
        c[tl.group_of(n)] += 1
    return dict(c)
