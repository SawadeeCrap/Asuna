"""Myrmex FX: the picture effects Blender draws itself (no TouchDesigner needed).

The engine streams, with every frame it sends to Blender, one small block of floats (``CHANNELS``) and
records the same block into takes, so a rendered take shows exactly what was seen live:

* the drives - the music (beat, kick, snare, energy ...), the organism (speed, size, glow, impacts, shape
  changes) and the camera (cuts), as computed by :mod:`myrmex.realtime.touch` (``DRIVES``);
* the rack - the effect amounts set in the app (a preset, sliders, MIDI knobs ``fx_<name>``) (``RACK``);
* the output: effects on / off, horizontal 1920x1080 or vertical 1080x1920, live preview scale.

What Blender makes of it (blender/myrmex_blender/fx*.py):

* afterimages - the "Sandevistan" look: copies of the organism left behind as it moves, in shifting neon
  colours, fading and breaking up; dense copies become a continuous echo smear;
* light ribbons trailing from its extremities; screen-space light trails (feedback of what glows);
* bloom, chromatic aberration, glitch, anime impact frames, speed lines, shockwaves, grain, vignette,
  scanlines and a grade (exposure, contrast, saturation, hue drift).
"""
from __future__ import annotations

import math
import struct

import numpy as np

DRIVES = ("beat", "phase", "bpm", "playing", "energy", "bass", "high", "flux", "kick", "snare", "hats", "speed",
          "size", "glow", "arousal", "tension", "impact", "morph", "event", "cut", "shot", "fx_flash", "fx_shake",
          "fx_glitch", "fx_chroma", "fx_bloom", "fx_hue", "fx_strobe")
RACK = ("ghosts", "ghost_life", "ghost_density", "palette", "ribbons", "trails", "bloom", "chroma", "glitch",
        "impact_frames", "speed_lines", "shock", "grain", "vignette", "scanlines", "react", "exposure", "contrast",
        "saturation", "hue")
STATE = ("on", "vertical", "preview")
CHANNELS = DRIVES + tuple("r_" + k for k in RACK) + STATE

LABELS = {"ghosts": "Afterimages", "ghost_life": "Afterimage life", "ghost_density": "Afterimage density (echo)",
          "palette": "Afterimage colours", "ribbons": "Light ribbons", "trails": "Light trails (feedback)",
          "bloom": "Bloom", "chroma": "Chromatic aberration", "glitch": "Glitch",
          "impact_frames": "Impact frames (anime)", "speed_lines": "Speed lines", "shock": "Shockwaves",
          "grain": "Grain", "vignette": "Vignette", "scanlines": "Scanlines", "react": "Music moves the effects",
          "exposure": "Exposure", "contrast": "Contrast", "saturation": "Saturation", "hue": "Colour drift"}
PALETTES = ("Sandevistan", "Neon", "Ice", "Blood", "Gold", "Toxic")

DEFAULT_RACK = {"ghosts": 0.75, "ghost_life": 0.45, "ghost_density": 0.35, "palette": 0.0, "ribbons": 0.3,
                "trails": 0.3, "bloom": 0.6, "chroma": 0.3, "glitch": 0.15, "impact_frames": 0.35,
                "speed_lines": 0.35, "shock": 0.45, "grain": 0.2, "vignette": 0.4, "scanlines": 0.08, "react": 0.8,
                "exposure": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.0}


def _palette(name: str) -> float:
    return PALETTES.index(name) / (len(PALETTES) - 1)


PRESETS = {
    # Cyberpunk: Edgerunners - the fast one leaves a string of neon copies of itself behind.
    "Sandevistan": {"ghosts": 0.85, "ghost_life": 0.5, "ghost_density": 0.4, "palette": _palette("Sandevistan"),
                    "ribbons": 0.25, "trails": 0.25, "bloom": 0.7, "chroma": 0.35, "glitch": 0.15,
                    "impact_frames": 0.3, "speed_lines": 0.3, "shock": 0.4, "grain": 0.2, "vignette": 0.4,
                    "scanlines": 0.1, "saturation": 0.6, "hue": 0.0},
    # Dense copies: a continuous smear of the motion.
    "Echo": {"ghosts": 0.7, "ghost_life": 0.3, "ghost_density": 0.95, "palette": _palette("Ice"), "ribbons": 0.0,
             "trails": 0.5, "bloom": 0.6, "chroma": 0.2, "glitch": 0.0, "impact_frames": 0.0, "speed_lines": 0.1,
             "shock": 0.2, "grain": 0.15, "vignette": 0.4, "scanlines": 0.0, "saturation": 0.45, "hue": 0.0},
    # Manga: inverted impact frames, focus lines, shockwaves on every hit.
    "Anime Impact": {"ghosts": 0.5, "ghost_life": 0.35, "ghost_density": 0.3, "palette": _palette("Blood"),
                     "ribbons": 0.2, "trails": 0.15, "bloom": 0.55, "chroma": 0.4, "glitch": 0.2,
                     "impact_frames": 1.0, "speed_lines": 0.85, "shock": 0.8, "grain": 0.3, "vignette": 0.5,
                     "scanlines": 0.0, "contrast": 0.65, "saturation": 0.5, "hue": 0.0},
    # Light writing: ribbons from the extremities, glowing trails.
    "Neon Ribbons": {"ghosts": 0.3, "ghost_life": 0.4, "ghost_density": 0.3, "palette": _palette("Neon"),
                     "ribbons": 0.9, "trails": 0.65, "bloom": 0.9, "chroma": 0.3, "glitch": 0.05,
                     "impact_frames": 0.1, "speed_lines": 0.15, "shock": 0.4, "grain": 0.2, "vignette": 0.45,
                     "scanlines": 0.05, "saturation": 0.65, "hue": 0.25},
    "Glitch": {"ghosts": 0.45, "ghost_life": 0.25, "ghost_density": 0.5, "palette": _palette("Toxic"),
               "ribbons": 0.0, "trails": 0.2, "bloom": 0.5, "chroma": 0.7, "glitch": 0.9, "impact_frames": 0.5,
               "speed_lines": 0.2, "shock": 0.6, "grain": 0.45, "vignette": 0.45, "scanlines": 0.6,
               "saturation": 0.45, "hue": 0.1},
    "Dream": {"ghosts": 0.6, "ghost_life": 0.95, "ghost_density": 0.6, "palette": _palette("Gold"), "ribbons": 0.4,
              "trails": 0.85, "bloom": 0.95, "chroma": 0.2, "glitch": 0.0, "impact_frames": 0.0,
              "speed_lines": 0.0, "shock": 0.3, "grain": 0.2, "vignette": 0.5, "scanlines": 0.0,
              "saturation": 0.7, "hue": 0.45},
    "Clean": {"ghosts": 0.0, "ghost_life": 0.45, "ghost_density": 0.35, "palette": 0.0, "ribbons": 0.0,
              "trails": 0.0, "bloom": 0.45, "chroma": 0.1, "glitch": 0.0, "impact_frames": 0.0, "speed_lines": 0.0,
              "shock": 0.15, "grain": 0.15, "vignette": 0.35, "scanlines": 0.0, "saturation": 0.5, "hue": 0.0},
}
PRESET_NAMES = tuple(PRESETS)
FORMATS = {False: (1920, 1080), True: (1080, 1920)}
DEFAULTS = {"enabled": True, "preset": "Sandevistan", "rack": dict(DEFAULT_RACK), "vertical": False, "preview": 1.0}
TRAILER = b"MFX1"
_TAIL = struct.Struct("<H4s")


def rack_from_preset(name: str) -> dict:
    out = dict(DEFAULT_RACK)
    out.update(PRESETS.get(name, {}))
    return out


def size_of(vertical: bool) -> tuple[int, int]:
    return FORMATS[bool(vertical)]


class FxRack:
    """The app's effect settings + MIDI knobs -> the per-frame block for Blender."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = {"enabled": True, "preset": "Sandevistan", "rack": rack_from_preset("Sandevistan"),
                    "vertical": False, "preview": 1.0}
        self.configure(**(cfg or {}))
        from .touch import CHANNELS as TD
        self._src = np.array([TD.index(n) for n in DRIVES])
        self.values = np.zeros(len(CHANNELS))

    def configure(self, **kw) -> None:
        rack = kw.pop("rack", None)
        preset = kw.get("preset")
        for k in ("enabled", "preset", "vertical", "preview"):
            if k in kw:
                self.cfg[k] = kw[k]
        if preset in PRESETS and rack is None:                  # a preset sets the whole rack
            self.cfg["rack"] = rack_from_preset(preset)
        if isinstance(rack, dict):
            for k, v in rack.items():
                if k in RACK:
                    self.cfg["rack"][k] = float(min(1.0, max(0.0, float(v))))

    def frame(self, drives, controls: dict | None = None) -> np.ndarray:
        """``drives``: the TouchBridge values (touch.CHANNELS order).  Returns CHANNELS order."""
        v = self.values
        n = len(DRIVES)
        d = np.asarray(drives, float)
        v[:n] = d[self._src] if len(d) > int(self._src.max()) else 0.0
        c = controls or {}
        rack = self.cfg["rack"]
        for i, name in enumerate(RACK):
            knob = c.get("fx_" + name)
            v[n + i] = float(knob) if knob is not None else float(rack.get(name, DEFAULT_RACK[name]))
        m = n + len(RACK)
        v[m] = 1.0 if self.cfg.get("enabled", True) else 0.0
        v[m + 1] = 1.0 if self.cfg.get("vertical") else 0.0
        v[m + 2] = float(self.cfg.get("preview", 1.0))
        v[~np.isfinite(v)] = 0.0
        return v.copy()

    def blender_settings(self) -> dict:
        """What a Blender without a running engine needs (take preview / render): MYRMEX_FX."""
        return {"on": bool(self.cfg.get("enabled", True)), "rack": dict(self.cfg["rack"]),
                "vertical": bool(self.cfg.get("vertical")), "preview": float(self.cfg.get("preview", 1.0)),
                "preset": self.cfg.get("preset", "")}


def trailer(values) -> bytes:
    """The block appended to a frame: floats, their count, the magic (found from the end of the packet)."""
    a = np.ascontiguousarray(np.asarray(values, float), "<f4")
    return a.tobytes() + _TAIL.pack(len(a), TRAILER)


def decode_trailer(data: bytes) -> np.ndarray | None:
    if len(data) < _TAIL.size or data[-4:] != TRAILER:
        return None
    n, _ = _TAIL.unpack_from(data, len(data) - _TAIL.size)
    start = len(data) - _TAIL.size - 4 * n
    if n <= 0 or start < 0:
        return None
    return np.frombuffer(data, "<f4", count=n, offset=start).astype(float)


def as_dict(values, names=CHANNELS) -> dict:
    """Channel values -> {name: value}; a shorter (older) block leaves the rest out."""
    return {k: float(x) for k, x in zip(names, values) if math.isfinite(float(x))}


__all__ = ["FxRack", "CHANNELS", "DRIVES", "RACK", "STATE", "LABELS", "PALETTES", "PRESETS", "PRESET_NAMES",
           "DEFAULT_RACK", "DEFAULTS", "FORMATS", "rack_from_preset", "size_of", "trailer", "decode_trailer",
           "as_dict"]
