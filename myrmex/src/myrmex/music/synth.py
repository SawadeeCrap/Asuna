"""Tiny numpy synthesiser: renders a MusicTimeline to WAV for preview videos.

Not meant to sound great – meant to make synthetic test material audible so
the sync and the character's responses can be judged by ear.
"""
from __future__ import annotations

import math
import wave

import numpy as np

from .timeline import MusicTimeline

SR = 44100


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0.0, 1.0)
    return a * np.exp(-t / max(decay, 1e-4))


def _saw(freq: float, n: int, harmonics: int = 10, detune: float = 0.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    out = np.zeros(n)
    f = freq * (1.0 + detune)
    for k in range(1, harmonics + 1):
        if f * k > SR * 0.45:
            break
        out += np.sin(2 * np.pi * f * k * t + phase * k) / k
    return out * 0.6


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, n)


def _voice(group: str, pitch: float, vel: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    f = 440.0 * 2 ** ((pitch - 69.0) / 12.0)
    if group == "kick":
        n = int(0.45 * SR)
        t = np.arange(n) / SR
        fr = 45.0 + 110.0 * np.exp(-t / 0.035)
        ph = 2 * np.pi * np.cumsum(fr) / SR
        return np.sin(ph) * _env(n, 0.001, 0.28) * 1.1 + _noise(n, rng) * _env(n, 0.0005, 0.004) * 0.3
    if group == "snare":
        n = int(0.3 * SR)
        body = np.sin(2 * np.pi * 185.0 * np.arange(n) / SR) * _env(n, 0.001, 0.07) * 0.5
        return body + np.diff(_noise(n + 1, rng)) * 0.45 * _env(n, 0.001, 0.12)
    if group == "hats":
        n = int((0.25 if pitch >= 46 else 0.07) * SR)
        x = np.diff(np.diff(_noise(n + 2, rng)))
        return x * 0.22 * _env(n, 0.0005, 0.12 if pitch >= 46 else 0.03)
    if group == "perc":
        n = int(0.2 * SR)
        return np.sin(2 * np.pi * f * np.arange(n) / SR) * _env(n, 0.001, 0.06) * 0.6
    if group == "bass":
        n = int(max(dur, 0.1) * SR + 0.08 * SR)
        e = _env(n, 0.004, 0.9) * np.clip((dur * SR + 0.06 * SR - np.arange(n)) / (0.06 * SR), 0, 1)
        return (_saw(f, n, 6) * 0.7 + np.sin(2 * np.pi * f * np.arange(n) / SR) * 0.6) * e
    if group in ("harmony", "texture"):
        n = int((dur + 0.6) * SR)
        t = np.arange(n) / SR
        e = np.clip(t / 0.35, 0, 1) * np.clip((dur + 0.6 - t) / 0.6, 0, 1)
        return (_saw(f, n, 5, -0.004) + _saw(f, n, 5, 0.005, 1.3) + _saw(f, n, 4, 0.0, 2.1)) * e * 0.18
    if group == "melody":
        n = int((dur + 0.4) * SR)
        t = np.arange(n) / SR
        vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.2 * t)
        ph = 2 * np.pi * np.cumsum(f * vib) / SR
        tri = 2.0 / np.pi * np.arcsin(np.sin(ph))
        return tri * _env(n, 0.01, dur * 0.8 + 0.2) * 0.35
    if group == "fx":
        n = int(max(dur, 0.3) * SR)
        t = np.arange(n) / SR
        sweep = np.sin(2 * np.pi * np.cumsum(f * (1 + 2 * t / max(dur, 0.3))) / SR)
        return (sweep * 0.3 + _noise(n, rng) * 0.2) * np.clip(t / max(dur * 0.8, 0.01), 0, 1) * np.exp(-np.maximum(0, t - dur * 0.8) / 0.1)
    n = int(0.2 * SR)
    return np.sin(2 * np.pi * f * np.arange(n) / SR) * _env(n, 0.002, 0.1) * 0.4


def render_audio(tl: MusicTimeline, path: str, tail: float = 1.0, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    n = int((tl.duration + tail) * SR)
    left = np.zeros(n)
    right = np.zeros(n)
    gains = {"kick": 0.9, "snare": 0.6, "hats": 0.5, "perc": 0.45, "bass": 0.55, "harmony": 0.5,
             "melody": 0.5, "texture": 0.4, "fx": 0.45}
    pans = {"hats": 0.25, "perc": -0.3, "melody": 0.1, "harmony": 0.0}
    for note in tl.notes:
        g = tl.group_of(note)
        v = _voice(g, note.pitch, note.velocity, note.duration, rng) * note.velocity * gains.get(g, 0.4)
        i0 = int(note.time * SR)
        i1 = min(n, i0 + len(v))
        if i1 <= i0:
            continue
        p = pans.get(g, 0.0)
        left[i0:i1] += v[: i1 - i0] * (1.0 - max(0.0, p))
        right[i0:i1] += v[: i1 - i0] * (1.0 + min(0.0, p))
    mix = np.stack([left, right], axis=1)
    peak = float(np.abs(mix).max()) or 1.0
    mix = np.tanh(mix / peak * 1.2) / math.tanh(1.2) * 0.89
    data = (mix * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    return path
