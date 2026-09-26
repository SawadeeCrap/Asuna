"""Myrmex FX: the effects Blender draws itself (no TouchDesigner needed) - only on the organism, in its own tones.

The engine streams, with every frame it sends to Blender, one small block of floats (``CHANNELS``) and
records the same block into takes, so a rendered take shows exactly what was seen live:

* the drives - the music (beat, kick, energy ...), the organism (speed, size, glow, impacts, shape changes)
  and the camera (cuts), as computed by :mod:`myrmex.realtime.touch` (``DRIVES``);
* the rack - the effect amounts set in the app (a preset, sliders, MIDI knobs ``fx_<name>``) (``RACK``);
* the output: effects on / off, horizontal 1920x1080 or vertical 1080x1920, live preview scale.

What Blender makes of it (blender/myrmex_blender/fx*.py), always on a black stage:

* afterimages - copies of the organism left behind as it moves, made of its *own* materials, fading and
  dissolving; dense copies become a continuous echo of the motion;
* light traces from its extremities, in its own accent colour (its glow, its light lines, its cables);
* motion echo (what the organism is leaves a fading trace on screen) and a soft glow of its highlights;
* exposure / contrast / saturation that never lift the black.
Nothing moves, tears, tints or dirties the frame itself.
"""
from __future__ import annotations

import math
import struct

import numpy as np

DRIVES = ("beat", "phase", "bpm", "playing", "energy", "bass", "high", "flux", "kick", "snare", "hats", "speed",
          "size", "glow", "arousal", "tension", "impact", "morph", "event", "cut", "shot", "fx_flash", "fx_shake",
          "fx_glitch", "fx_chroma", "fx_bloom", "fx_hue", "fx_strobe")
RACK = ("ghosts", "ghost_life", "ghost_density", "ribbons", "trails", "bloom", "react", "exposure", "contrast",
        "saturation")
STATE = ("on", "vertical", "preview")
CHANNELS = DRIVES + tuple("r_" + k for k in RACK) + STATE

LABELS = {"ghosts": "Afterimages", "ghost_life": "Afterimage life", "ghost_density": "Afterimage density (echo)",
          "ribbons": "Light traces", "trails": "Motion echo", "bloom": "Glow", "react": "Music moves the effects",
          "exposure": "Exposure", "contrast": "Contrast", "saturation": "Saturation"}

DEFAULT_RACK = {"ghosts": 0.7, "ghost_life": 0.45, "ghost_density": 0.35, "ribbons": 0.0, "trails": 0.25,
                "bloom": 0.25, "react": 0.5, "exposure": 0.5, "contrast": 0.5, "saturation": 0.5}

PRESETS = {
    # the organism leaves copies of itself behind as it moves - made of what it is made of
    "Afterimage": {"ghosts": 0.75, "ghost_life": 0.45, "ghost_density": 0.35, "ribbons": 0.0, "trails": 0.2,
                   "bloom": 0.25},
    # dense copies: one continuous echo of the motion
    "Echo": {"ghosts": 0.65, "ghost_life": 0.3, "ghost_density": 0.95, "ribbons": 0.0, "trails": 0.4, "bloom": 0.2},
    # long copies dissolving slowly, a deep motion echo
    "Phantom": {"ghosts": 0.6, "ghost_life": 0.95, "ghost_density": 0.55, "ribbons": 0.0, "trails": 0.55,
                "bloom": 0.3},
    # thin traces of light from its extremities, in its own accent colour, a few copies
    "Trace": {"ghosts": 0.35, "ghost_life": 0.4, "ghost_density": 0.3, "ribbons": 0.7, "trails": 0.3, "bloom": 0.3},
    # the organism alone
    "Clean": {"ghosts": 0.0, "ghost_life": 0.45, "ghost_density": 0.35, "ribbons": 0.0, "trails": 0.0, "bloom": 0.15},
}
PRESET_NAMES = tuple(PRESETS)
FORMATS = {False: (1920, 1080), True: (1080, 1920)}
DEFAULTS = {"enabled": True, "preset": "Afterimage", "rack": dict(DEFAULT_RACK), "vertical": False, "preview": 1.0}
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
        self.cfg = {"enabled": True, "preset": "Afterimage", "rack": rack_from_preset("Afterimage"),
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


__all__ = ["FxRack", "CHANNELS", "DRIVES", "RACK", "STATE", "LABELS", "PRESETS", "PRESET_NAMES",
           "DEFAULT_RACK", "DEFAULTS", "FORMATS", "rack_from_preset", "size_of", "trailer", "decode_trailer",
           "as_dict"]
