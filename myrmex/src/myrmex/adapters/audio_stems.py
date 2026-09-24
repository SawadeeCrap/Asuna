"""Audio analysis (numpy only): onsets, envelopes, brightness -> timeline events.

Used for exported stems (one WAV per group, e.g. ``kick.wav``, ``bass.wav``)
and for audio clips referenced by an Ableton set.  Onsets become hit events
(velocity from onset strength, sharpness from high-frequency content,
"pitch" from spectral centroid), the RMS envelope becomes a control curve.
"""
from __future__ import annotations

import os

import numpy as np

from ..music.classify import keyword_scores
from ..music.timeline import ControlCurve, MusicTimeline, NoteEvent, TempoMap, Track
from .audio_io import read_audio

AUDIO_EXT = (".wav", ".wave", ".aif", ".aiff", ".aifc")


def analyze_signal(x: np.ndarray, sr: int, hop: int = 512, frame: int = 2048) -> dict:
    """Mono float signal -> onset times/strengths, envelope, centroid, hfc."""
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    win = np.hanning(frame).astype(np.float32)
    frames = x[idx] * win
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    logspec = np.log1p(10.0 * spec)
    flux = np.maximum(np.diff(logspec, axis=0, prepend=logspec[:1]), 0.0).sum(axis=1)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    power = spec.sum(axis=1) + 1e-9
    centroid = (spec * freqs[None, :]).sum(axis=1) / power
    hfc = (spec[:, freqs > 4000.0].sum(axis=1)) / power
    times = (np.arange(n) * hop + frame / 2) / sr
    # Adaptive threshold peak picking.
    fr = sr / hop
    w = max(3, int(0.15 * fr))
    pad = np.pad(flux, (w, w), mode="edge")
    med = np.array([np.median(pad[i:i + 2 * w + 1]) for i in range(n)])
    thr = med + 0.35 * (flux.max() - med.mean()) * 0.25 + 1e-6
    min_gap = max(1, int(0.045 * fr))
    peaks = []
    last = -min_gap
    for i in range(1, n - 1):
        if flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] and i - last >= min_gap:
            peaks.append(i)
            last = i
    peaks = np.array(peaks, dtype=int)
    strength = flux[peaks] / (np.percentile(flux[peaks], 95) + 1e-9) if len(peaks) else np.zeros(0)
    return {"times": times, "rms": rms, "centroid": centroid, "hfc": hfc, "flux": flux,
            "onset_times": times[peaks] if len(peaks) else np.zeros(0),
            "onset_strength": np.clip(strength, 0.05, 1.0), "onset_hfc": hfc[peaks] if len(peaks) else np.zeros(0),
            "onset_centroid": centroid[peaks] if len(peaks) else np.zeros(0), "sr": sr, "hop": hop}


def centroid_to_pitch(c: np.ndarray) -> np.ndarray:
    return np.clip(69.0 + 12.0 * np.log2(np.maximum(c, 20.0) / 440.0), 12.0, 120.0)


def analysis_to_events(an: dict, track_id: str, time_map=None) -> tuple[list[NoteEvent], ControlCurve]:
    """Turn an analysis into hit events and an envelope curve (optionally remapping times)."""
    tm = time_map or (lambda t: t)
    notes = []
    pitches = centroid_to_pitch(an["onset_centroid"])
    for t, s, h, p in zip(an["onset_times"], an["onset_strength"], an["onset_hfc"], pitches):
        tt = tm(float(t))
        if tt is None:
            continue
        notes.append(NoteEvent(tt, 0.08, float(p), float(s), track_id, sharpness=float(np.clip(h * 3.0, 0.0, 1.0))))
    env_t = an["times"][::2]
    env = an["rms"][::2]
    env = env / (np.percentile(env, 98) + 1e-9)
    mapped = [tm(float(t)) for t in env_t]
    keep = [i for i, m in enumerate(mapped) if m is not None]
    curve = ControlCurve(track_id, "envelope", np.array([mapped[i] for i in keep]),
                         np.clip(env[keep], 0.0, 1.0))
    return notes, curve


def load_stems(folder: str, bpm: float = 120.0, overrides: dict[str, str] | None = None) -> MusicTimeline:
    """Analyse every WAV/AIFF in ``folder`` as one track (group from the file name)."""
    tl = MusicTimeline(TempoMap([(0.0, bpm)]), source=f"stems:{folder}")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(AUDIO_EXT))
    for f in files:
        path = os.path.join(folder, f)
        x, sr = read_audio(path)
        an = analyze_signal(x, sr)
        tid = os.path.splitext(f)[0]
        ks = keyword_scores(tid)
        group = (overrides or {}).get(tid) or (max(ks, key=ks.get) if ks else "other")
        tl.tracks[tid] = Track(tid, tid, "audio", group, 0.8 if ks else 0.1, override=bool(overrides and tid in overrides))
        notes, curve = analysis_to_events(an, tid)
        tl.notes.extend(notes)
        tl.curves.append(curve)
        tl.duration = max(tl.duration, len(x) / sr)
    return tl.finalize()


def estimate_tempo(onset_times: np.ndarray, lo: float = 70.0, hi: float = 180.0) -> tuple[float, float]:
    """Tempo (bpm) and confidence from onset times via IOI autocorrelation."""
    if len(onset_times) < 8:
        return 120.0, 0.0
    fr = 200.0
    n = int(onset_times.max() * fr) + 2
    sig = np.zeros(n)
    sig[(onset_times * fr).astype(int)] = 1.0
    sig = np.convolve(sig, np.hanning(9), mode="same")
    ac = np.correlate(sig, sig, mode="full")[n - 1:]
    lags = np.arange(len(ac)) / fr
    bpms = 60.0 / np.maximum(lags, 1e-9)
    m = (bpms >= lo) & (bpms <= hi)
    if not m.any():
        return 120.0, 0.0
    i = np.argmax(np.where(m, ac, -1))
    conf = float(ac[i] / (ac[0] + 1e-9))
    return float(bpms[i]), conf
