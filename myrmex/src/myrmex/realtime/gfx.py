"""Myrmex graphics: how much the live Blender viewport draws - presets and fine knobs (no Blender needed here;
blender/myrmex_blender/gfx.py applies them).

What costs what (measured on Blender 5.2, docs/GRAPHICS.md):

* the organism's metaball body is re-polygonised on the CPU every frame, on Blender's main thread: 17 ms (Spear)
  to 31 ms (Hive) per frame at the viewport resolution 0.03, about 60 % of that at 0.04, 40 % at 0.05 - the
  largest single cost of a live frame;
* afterimages: every copy is a translucent double of the whole organism (+50 000 - 175 000 blended triangles);
* picture effects (motion echo, glow): a second render of the scene at the live-picture size;
* the viewport render itself: its pixel size and shadows.

The body's detail follows the shot: a metaball cell is kept at most ``body_px`` pixels wide on the 1920-pixel
picture - never finer than ``body_res`` (the close-up detail) and never coarser than ``body_res`` x
``body_max``.  At 0.03 a cell is ~8 px in the director's median shot, 4 px in wide shots and up to 20 px in close
ups: coarsening the wide shots to what the median shot already shows costs nothing visible.  Renders of takes
keep the metaball's render resolution (0.02) - this is only the live viewport.
"""
from __future__ import annotations

import json
import math
import os

LEVELS = ("quality", "balanced", "performance", "max_fps")
PRESETS = {
    # as before these settings existed: fixed 0.03 body, every afterimage part, full live picture
    "quality": {"body_res": 0.03, "body_auto": False, "body_px": 8.0, "body_max": 2.0, "ghost_max": 16,
                "ghost_parts": "all", "ghost_coarse": 1.6, "picture": 1.0, "pixel_size": 1, "shadows": True,
                "shadow_scale": 0.5},
    # the close-ups as before, wide shots only as detailed as they are seen
    "balanced": {"body_res": 0.03, "body_auto": True, "body_px": 8.0, "body_max": 2.0, "ghost_max": 10,
                 "ghost_parts": "all", "ghost_coarse": 1.9, "picture": 0.75, "pixel_size": 1, "shadows": True,
                 "shadow_scale": 0.5},
    "performance": {"body_res": 0.04, "body_auto": True, "body_px": 10.0, "body_max": 2.0, "ghost_max": 6,
                    "ghost_parts": "body", "ghost_coarse": 2.2, "picture": 0.5, "pixel_size": 1, "shadows": True,
                    "shadow_scale": 0.25},
    "max_fps": {"body_res": 0.05, "body_auto": True, "body_px": 12.0, "body_max": 2.0, "ghost_max": 4,
                "ghost_parts": "body", "ghost_coarse": 2.5, "picture": 0.5, "pixel_size": 2, "shadows": False,
                "shadow_scale": 0.25},
}
KNOBS = tuple(PRESETS["quality"])
# (min, max) of the numeric knobs
RANGES = {"body_res": (0.015, 0.12), "body_px": (2.0, 30.0), "body_max": (1.0, 4.0), "ghost_max": (1, 16),
          "ghost_coarse": (1.0, 4.0), "picture": (0.25, 1.0), "pixel_size": (1, 8), "shadow_scale": (0.1, 1.0)}
DEFAULTS = {"preset": "balanced", **PRESETS["balanced"],
            "eevee": False,        # also set the viewport's EEVEE options (pixel size, shadows); off = your file's
            "auto": False,         # step the preset down / up to keep ``target_fps``
            "target_fps": 50.0}
REF_WIDTH = 1920.0                 # body_px is measured on this picture width (the live format)


def _num(key: str, v):
    lo, hi = RANGES[key]
    x = min(hi, max(lo, float(v)))
    return int(round(x)) if isinstance(lo, int) else x


def normalize(d: dict | None, base: dict | None = None) -> dict:
    """A full, valid settings dict: ``base`` (or the defaults) with ``d`` on top.  A preset name sets every
    knob of that preset; knobs given with it are fine-tuning on top ("custom" is kept as the name then)."""
    out = dict(DEFAULTS if base is None else base)
    d = dict(d or {})
    p = d.get("preset")
    if p in PRESETS:
        out.update(PRESETS[p])
        out["preset"] = p
    changed = False
    for k, v in d.items():
        if k not in DEFAULTS or k == "preset":
            continue
        try:
            if k in RANGES:
                v = _num(k, v)
            elif k == "ghost_parts":
                v = "body" if str(v) == "body" else "all"
            elif k in ("body_auto", "eevee", "auto"):
                v = bool(v)
            elif k == "target_fps":
                v = min(120.0, max(15.0, float(v)))
        except (TypeError, ValueError):
            continue
        if out.get(k) != v:
            out[k] = v
            changed = changed or k in KNOBS
    if changed and (p is None or p not in PRESETS) and out.get("preset") in PRESETS:
        if any(out[k] != PRESETS[out["preset"]][k] for k in KNOBS):
            out["preset"] = "custom"
    if out.get("preset") not in PRESETS and out.get("preset") != "custom":
        out["preset"] = "balanced"
    return out


def world_per_pixel(distance: float, fov: float, width: float = REF_WIDTH) -> float:
    """Metres per pixel at ``distance`` in front of a camera whose wider side sees ``fov`` (radians)."""
    return max(1e-6, 2.0 * max(distance, 1e-3) * math.tan(0.5 * fov) / max(1.0, width))


def body_resolution(cfg: dict, distance: float | None, fov: float | None, current: float | None = None) -> float:
    """The metaball's viewport resolution for this shot (see the module docstring); quantised to 5 % steps
    and with a 12 % dead band so it changes on cuts and zooms, not with every breath of the camera."""
    base = float(cfg.get("body_res", 0.03))
    if not cfg.get("body_auto") or distance is None or fov is None or not math.isfinite(distance):
        return base
    want = float(cfg.get("body_px", 8.0)) * world_per_pixel(distance, fov)
    res = min(base * float(cfg.get("body_max", 2.0)), max(base, want))
    res = base * 1.05 ** round(math.log(res / base) / math.log(1.05))
    if current is not None and current > 0 and abs(res - current) / current < 0.12:
        return float(current)
    return float(res)


def step(level: str, down: bool) -> str:
    """The next preset toward speed (``down``) or toward quality."""
    i = LEVELS.index(level) if level in LEVELS else 1
    return LEVELS[min(len(LEVELS) - 1, i + 1) if down else max(0, i - 1)]


def to_env(cfg: dict) -> dict:
    return {"MYRMEX_GFX": json.dumps(normalize(cfg))}


def from_env() -> dict | None:
    raw = os.environ.get("MYRMEX_GFX")
    if not raw:
        return None
    try:
        return normalize(json.loads(raw))
    except (ValueError, TypeError):
        return None


__all__ = ["LEVELS", "PRESETS", "DEFAULTS", "KNOBS", "RANGES", "normalize", "world_per_pixel", "body_resolution",
           "step", "to_env", "from_env"]
