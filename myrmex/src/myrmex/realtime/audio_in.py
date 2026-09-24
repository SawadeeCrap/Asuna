"""Listen to the music itself: loopback audio -> kick / snare / hats onsets + level.

For sets built from audio clips (no MIDI to read), route Ableton's master into a
loopback device (macOS: BlackHole 2ch, or Loopback) and start the engine with
``--audio "BlackHole 2ch"``.  Analysis per 256-sample hop (5 ms at 48 kHz):

* log-magnitude spectrum (1024-point Hann), spectral flux in three bands
  (kick 40-120 Hz, snare 180-2.5 kHz, hats 6-16 kHz);
* adaptive threshold = running median + k x running MAD, refractory period per band;
* velocity from the flux relative to its recent peak (automatic gain);
* RMS level for the energy curve.

``BandOnsetDetector`` is pure numpy (testable offline); ``AudioInput`` wraps it
in a ``sounddevice`` stream whose callback pushes ``hit`` events to the hub.
"""
from __future__ import annotations

import queue
import time
from collections import deque

import numpy as np

from ..bus.transport import LiveEvent

BANDS = {"snare": (1000.0, 4500.0, 0.12), "hats": (6000.0, 16000.0, 0.06)}
KICK_LP = 150.0          # Hz: the kick is found on the low-passed envelope (FFT bins are too coarse there)
KICK_REFRACTORY = 0.1
KICK_RISE = 1.35         # log ratio of the low-band envelope over its 30 ms minimum (x3.9): kicks ~1.6-2.0, bass ~1.1
KICK_FLOOR = 0.03        # low-band RMS (~ -30 dBFS): mastered kicks sit around -15..-6, pads well below
GATE = 0.008             # overall RMS gate (~ -42 dBFS): no onsets in near-silence
FLUX_FLOOR = {"snare": 0.12, "hats": 0.08}   # minimum flux peak (log-magnitude units)


class BandOnsetDetector:
    def __init__(self, sr: float = 48000.0, n_fft: int = 1024, hop: int = 256, k: float = 3.5,
                 history: float = 0.8, rel: float = 0.3):
        self.sr, self.n, self.hop = sr, n_fft, hop
        self.win = np.hanning(n_fft).astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        self.bins = {g: (freqs >= lo) & (freqs < hi) for g, (lo, hi, _) in BANDS.items()}
        self.refractory = {g: r for g, (_, _, r) in BANDS.items()}
        self.buf = np.zeros(n_fft, np.float32)
        self.prev = None
        H = max(8, int(history * sr / hop))
        self.hist = {g: deque(maxlen=H) for g in BANDS}
        self.peak = {g: FLUX_FLOOR.get(g, 0.05) for g in BANDS}
        self.last = {g: -1e9 for g in BANDS}
        self.k = k
        self.rel = rel
        # Kick: two cascaded one-pole low-passes, RMS envelope per hop, sharp rise over ~30 ms.
        self.lp_a = float(np.exp(-2.0 * np.pi * KICK_LP / sr))
        self.lp1 = self.lp2 = 0.0
        self.env_hist: deque[float] = deque(maxlen=max(3, int(0.03 * sr / hop)))
        self.env_peak = KICK_FLOOR
        self.env_decay = 0.5 ** (hop / sr / 8.0)          # long-term peak, half-life 8 s
        self.env_prev_rise = 0.0
        self.env_prev = 0.0
        self.last_kick = -1e9
        self.f1 = {g: 0.0 for g in BANDS}      # flux one hop ago
        self.f2 = {g: 0.0 for g in BANDS}      # two hops ago
        self.thr1 = {g: 1e9 for g in BANDS}
        self.level = 0.0
        self.t = 0.0
        self._pending = np.zeros(0, np.float32)

    def _kick(self, h: np.ndarray, th: float):
        # Two cascaded one-pole low-passes, block-wise and exact: y[i] = a*y[i-1] + (1-a)*x[i]
        # unrolled as a weighted cumulative sum (a**-n stays small for 256-sample hops).
        a = self.lp_a
        z1, z2 = self.lp1, self.lp2
        n = len(h)
        p = a ** np.arange(1, n + 1, dtype=np.float64)
        c1 = (1.0 - a) * np.cumsum(h / p) * p + z1 * p
        c2 = (1.0 - a) * np.cumsum(c1 / p) * p + z2 * p
        self.lp1, self.lp2 = float(c1[-1]), float(c2[-1])
        env = float(np.sqrt(np.mean(c2 * c2) + 1e-12))
        base = min(self.env_hist) if self.env_hist else env
        self.env_hist.append(env)
        rise = np.log((env + 1e-5) / (base + 1e-5))
        self.env_peak = max(self.env_peak * self.env_decay, env, KICK_FLOOR)
        hit = None
        prev = self.env_prev
        if self.env_prev_rise > KICK_RISE and rise < self.env_prev_rise and prev > KICK_FLOOR and \
                prev > 0.35 * self.env_peak and th - self.last_kick > KICK_REFRACTORY:
            self.last_kick = th
            hit = (th - self.hop / self.sr, "kick", float(np.clip(prev / self.env_peak, 0.2, 1.0)))
        self.env_prev_rise = rise
        self.env_prev = env
        return hit

    def process(self, mono: np.ndarray, t_end: float | None = None) -> list[tuple[float, str, float]]:
        """Feed samples; returns [(time, group, velocity)] (time at the hop, seconds)."""
        x = np.concatenate([self._pending, np.asarray(mono, np.float32)])
        out = []
        n_hops = len(x) // self.hop
        base = (t_end - len(x) / self.sr) if t_end is not None else self.t
        for i in range(n_hops):
            h = x[i * self.hop:(i + 1) * self.hop]
            self.buf = np.roll(self.buf, -self.hop)
            self.buf[-self.hop:] = h
            th = base + (i + 1) * self.hop / self.sr
            rms = float(np.sqrt(np.mean(h * h) + 1e-12))
            self.level += (rms - self.level) * (0.3 if rms > self.level else 0.02)
            k = self._kick(h, th)
            if k is not None and self.level > GATE:
                out.append(k)
            mag = np.log1p(40.0 * np.abs(np.fft.rfft(self.buf * self.win)))
            if self.prev is not None:
                d = np.maximum(0.0, mag - self.prev)
                for g, sel in self.bins.items():
                    f = float(d[sel].mean())
                    hist = self.hist[g]
                    f1, f2 = self.f1[g], self.f2[g]
                    # Peak picking with one hop of look-ahead (5 ms): the previous hop fires if it was a
                    # local maximum above an adaptive threshold and a fraction of the recent peak.
                    if f2 < f1 >= f and f1 > self.thr1[g] and f1 > self.rel * self.peak[g] and \
                            f1 > FLUX_FLOOR.get(g, 0.05) and self.level > GATE and \
                            th - self.last[g] > self.refractory[g]:
                        self.peak[g] = max(self.peak[g], f1)
                        vel = float(np.clip(f1 / self.peak[g], 0.2, 1.0))
                        self.last[g] = th
                        out.append((th - self.hop / self.sr, g, vel))
                    if len(hist) >= 8:
                        arr = np.fromiter(hist, float)
                        med = float(np.median(arr))
                        mad = float(np.median(np.abs(arr - med))) + 1e-4
                        self.thr1[g] = med + self.k * mad * 1.4826 + 0.02
                    self.peak[g] = max(self.peak[g] * 0.9993, f, FLUX_FLOOR.get(g, 0.05))
                    self.f2[g], self.f1[g] = f1, f
                    hist.append(f)
            self.prev = mag
        self._pending = x[n_hops * self.hop:]
        self.t = base + n_hops * self.hop / self.sr
        return out


class AudioInput:
    """sounddevice input stream -> BandOnsetDetector -> hub queue (``hit`` + ``cv`` level events)."""

    def __init__(self, device: str | int | None, out: queue.Queue, sr: float = 48000.0, block: int = 256):
        import sounddevice as sd  # optional dependency: pip install sounddevice
        self.q = out
        self.det = BandOnsetDetector(sr, hop=block)
        self._last_level = 0.0

        def cb(indata, frames, tinfo, status):
            mono = indata.mean(axis=1) if indata.ndim > 1 else indata
            now = time.perf_counter()
            for (_t, g, v) in self.det.process(mono, t_end=now):
                self.q.put(LiveEvent("hit", now, {"group": g, "velocity": v, "sharpness": 0.9, "source": "audio"}))
            if now - self._last_level > 0.05:
                self._last_level = now
                self.q.put(LiveEvent("cv", now, {"name": "audio_level", "value": min(1.0, 4.0 * self.det.level)}))

        self.stream = sd.InputStream(device=device, channels=2, samplerate=sr, blocksize=block, callback=cb,
                                     latency="low")
        self.stream.start()

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


__all__ = ["BandOnsetDetector", "AudioInput", "BANDS"]
