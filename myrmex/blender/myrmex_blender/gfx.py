"""Myrmex graphics in Blender: what the live viewport draws, and what it costs - measured, per frame.

The knobs and presets are in ``myrmex.realtime.gfx`` (no Blender needed); here they act:

* the organism's body - the metaball is re-polygonised on the main thread every frame, the largest single
  cost of a live frame: its viewport resolution follows the shot (``body_res``) and is relative to the look's
  own (a look saved at 0.02 stays twice as fine as one at 0.04 under every preset);
* the afterimages - how many copies at most, of every part or of the body only, how much coarser their
  metaballs are (fx_ghosts);
* the live picture of the picture effects - its size (fx_post);
* EEVEE's viewport - pixel size and shadows - only with ``eevee`` on (else your file's own settings stay).

Measured while live: the frames Blender shows a second, the milliseconds of the organism's update and of the
depsgraph evaluation (the body), the body's resolution.  ``auto`` steps the preset down when the view shows
fewer than ``target_fps`` frames a second for 3 s, and back up after 15 s with room to spare.  The app sets it
all (``MYRMEX_GFX`` at start, the ``gfx`` command after) and hears the numbers back (``gfx_stats``).
"""
from __future__ import annotations

import contextlib
import time

import bpy

from myrmex.realtime import gfx as G

S: dict = {"cfg": G.normalize(None), "level": None, "res": None, "set": None, "handler": None, "frame": 0, "drawn": 0,
           "assume_view": False,        # (tests, profiling: a headless Blender counts as the live view)
           "draws": 0, "t0": 0.0, "slow_since": None, "fast_since": None, "last_send": 0.0,
           "stats": {"fps": 0.0, "poll_ms": 0.0, "eval_ms": 0.0, "res": 0.0, "level": "", "frames": 0}}
SYNC = {"busy": False}              # the Graphics panel is being set from here (not by you)
LOOK_RES = "myrmex_look_res"        # (metaball) the look's own viewport resolution: the presets scale it
PIXEL_SIZES = (1, 2, 4, 8)


# ---------------------------------------------------------------------- settings
def configure(d: dict | None) -> dict:
    """Change the graphics (any subset; a preset name sets its knobs) -> the whole settings + the numbers."""
    S["cfg"] = G.normalize(d, S["cfg"])
    S["level"] = None
    S["slow_since"] = S["fast_since"] = None
    with contextlib.suppress(Exception):
        apply_scene()
    with contextlib.suppress(Exception):
        sync_props(bpy.context.scene)
    return status()


def from_env() -> dict | None:
    d = G.from_env()
    if d is not None:
        configure(d)
    return d


def effective() -> dict:
    """The knobs in force: auto's current level, or what was chosen."""
    c = S["cfg"]
    if c.get("auto") and S["level"] in G.PRESETS:
        return {**c, **G.PRESETS[S["level"]]}
    return c


# ---------------------------------------------------------------------- the body
def body_res(scene, mb, com) -> float:
    """The metaball's viewport resolution for this frame (the look's own x the preset's, by the shot)."""
    c = effective()
    look = mb.get(LOOK_RES)
    if look is None:
        look = float(mb.resolution)
        mb[LOOK_RES] = look
    base = max(0.01, float(look) * float(c["body_res"]) / 0.03)
    dist = fov = None
    cam = scene.camera if scene is not None else None
    if cam is not None and cam.type == "CAMERA" and com is not None:
        m = cam.matrix_world
        fwd = -m.col[2].xyz
        try:
            p = (float(com[0]), float(com[1]), float(com[2]))
        except (TypeError, IndexError):
            p = None
        if p is not None:
            dist = sum((p[i] - m.translation[i]) * fwd[i] for i in range(3))
            fov = float(cam.data.angle)
    res = G.body_resolution({**c, "body_res": base}, dist if dist and dist > 0 else None, fov, S["res"])
    S["res"] = res
    return res


def set_body(scene, mb, com) -> None:
    """Put this frame's resolution on the body's metaball (only when it changes).  A resolution changed by
    hand since (Properties > Resolution Viewport) becomes the look's own - it is not overwritten."""
    cur = float(mb.resolution)
    last = S["set"]
    if last is not None and last[0] == mb.as_pointer() and abs(cur - last[1]) > 1e-6:
        mb[LOOK_RES] = cur
        S["res"] = None
    res = body_res(scene, mb, com)
    if abs(cur - res) > 1e-6:
        mb.resolution = res
    S["set"] = (mb.as_pointer(), float(mb.resolution))


@contextlib.contextmanager
def saving():
    """(looks.py) A look is being saved: the file gets the body's own resolution, not this shot's."""
    from .creature import META
    mb = bpy.data.metaballs.get(META)
    keep = None
    if mb is not None and mb.get(LOOK_RES) is not None:
        keep = float(mb.resolution)
        mb.resolution = float(mb[LOOK_RES])
    try:
        yield
    finally:
        if keep is not None:
            mb.resolution = keep


# ---------------------------------------------------------------------- afterimages, the live picture
def rendering() -> bool:
    """A take is being rendered (headless or F12): renders keep every detail - these knobs are the viewport's."""
    if bpy.app.background:
        return not S["assume_view"]
    try:
        return bool(bpy.app.is_job_running("RENDER"))
    except (TypeError, ValueError, AttributeError):
        return False


def ghost_limits() -> tuple[int, bool, float]:
    """(copies at most, the body only, how much coarser a copy's metaball is than the body)."""
    if rendering():
        q = G.PRESETS["quality"]
        return int(q["ghost_max"]), False, float(q["ghost_coarse"])
    c = effective()
    return int(c["ghost_max"]), c["ghost_parts"] == "body", float(c["ghost_coarse"])


def picture_scale() -> float:
    return float(effective()["picture"])


# ---------------------------------------------------------------------- EEVEE's viewport
def apply_scene(scene=None) -> None:
    """Pixel size and shadows of the viewport - only when Myrmex may set EEVEE (``eevee``)."""
    c = effective()
    if not c.get("eevee"):
        return
    sc = scene or bpy.context.scene
    if sc is None:
        return
    ps = min(PIXEL_SIZES, key=lambda v: abs(v - int(c["pixel_size"])))
    with contextlib.suppress(Exception):
        if sc.render.preview_pixel_size != str(ps):
            sc.render.preview_pixel_size = str(ps)
    ee = sc.eevee
    for attr, val in (("use_shadows", bool(c["shadows"])), ("shadow_resolution_scale", float(c["shadow_scale"]))):
        if hasattr(ee, attr):
            with contextlib.suppress(Exception):
                if getattr(ee, attr) != val:
                    setattr(ee, attr, val)


# ---------------------------------------------------------------------- measuring (live)
def _counted() -> None:
    """(3D view draw) a new frame was shown."""
    if S["frame"] != S["drawn"]:
        S["drawn"] = S["frame"]
        S["draws"] += 1


def start() -> None:
    if S["handler"] is None and not bpy.app.background:
        S["handler"] = bpy.types.SpaceView3D.draw_handler_add(_counted, (), "WINDOW", "POST_PIXEL")
    S.update(t0=time.perf_counter(), draws=0, slow_since=None, fast_since=None)
    with contextlib.suppress(Exception):
        apply_scene()


def stop() -> None:
    if S["handler"] is not None:
        with contextlib.suppress(ValueError):
            bpy.types.SpaceView3D.draw_handler_remove(S["handler"], "WINDOW")
        S["handler"] = None


def after_frame(poll_s: float) -> None:
    """A live frame was just applied (live.LiveLink): evaluate it now - the evaluation the next redraw would
    do anyway (the body's polygons), so its cost is measured - then keep the numbers and pace auto."""
    t0 = time.perf_counter()
    with contextlib.suppress(Exception):
        wm = bpy.context.window_manager
        vl = bpy.context.view_layer
        win = wm.windows[0] if wm is not None and len(wm.windows) else None
        if vl is not None and (win is None or win.view_layer == vl):
            bpy.context.evaluated_depsgraph_get()
    ev = time.perf_counter() - t0
    st = S["stats"]
    st["poll_ms"] = 1000.0 * poll_s if not st["frames"] else 0.9 * st["poll_ms"] + 100.0 * poll_s
    st["eval_ms"] = 1000.0 * ev if not st["frames"] else 0.9 * st["eval_ms"] + 100.0 * ev
    st["res"] = float(S["res"] or 0.0)
    st["frames"] += 1
    S["frame"] += 1
    now = time.perf_counter()
    if now - S["t0"] >= 1.0:
        st["fps"] = S["draws"] / (now - S["t0"]) if S["handler"] is not None else 0.0
        S["t0"], S["draws"] = now, 0
        _auto(now, st["fps"])
        st["level"] = S["level"] or S["cfg"]["preset"]
        if now - S["last_send"] >= 2.0:
            S["last_send"] = now
            _send()


def _auto(now: float, fps: float) -> None:
    """Step toward speed after 3 s under ``target_fps``, back after 15 s with 8 fps to spare - never above what
    was chosen (``S["level"]`` None = the chosen settings, your Custom knobs included; below Custom come
    Performance and Max FPS)."""
    c = S["cfg"]
    if not c.get("auto") or S["handler"] is None or fps <= 0.0:
        return
    chosen = c["preset"]
    level = S["level"] or chosen
    target = float(c["target_fps"])
    nxt = None
    if fps < target:
        S["fast_since"] = None
        S["slow_since"] = S["slow_since"] or now
        if now - S["slow_since"] >= 3.0 and level != G.LEVELS[-1]:
            nxt, S["slow_since"] = ("performance" if level not in G.LEVELS else G.step(level, True)), None
    elif fps > target + 8.0:
        S["slow_since"] = None
        S["fast_since"] = S["fast_since"] or now
        if now - S["fast_since"] >= 15.0 and level != chosen:
            nxt, S["fast_since"] = G.step(level, False), None
            if chosen not in G.LEVELS:
                nxt = chosen if level == "performance" else nxt
            elif G.LEVELS.index(nxt) <= G.LEVELS.index(chosen):
                nxt = chosen
    else:
        S["slow_since"] = S["fast_since"] = None
    if nxt is not None:
        S["level"] = None if nxt == chosen else nxt
        apply_scene()


def status() -> dict:
    return {**S["cfg"], "stats": dict(S["stats"]), "level": S["level"] or S["cfg"]["preset"]}


def _send() -> None:
    from . import control
    if control.enabled():
        st = S["stats"]
        control.reply({"cmd": "gfx_stats", "fps": round(st["fps"], 1), "poll_ms": round(st["poll_ms"], 2),
                       "eval_ms": round(st["eval_ms"], 2), "res": round(st["res"], 4),
                       "level": st["level"], "auto": bool(S["cfg"].get("auto"))})


# ---------------------------------------------------------------------- the panel
def sync_props(scene) -> None:
    """The Graphics panel (Myrmex sidebar) shows the settings as they are now."""
    p = getattr(scene, "myrmex_gfx", None)
    if p is None:
        return
    c = S["cfg"]
    SYNC["busy"] = True
    try:
        for k in ("preset", "body_res", "body_auto", "body_px", "ghost_max", "ghost_parts", "ghost_coarse", "picture",
                  "shadows", "shadow_scale", "eevee", "auto", "target_fps"):
            v = c[k]
            if k == "preset":
                v = v if v in G.PRESETS else "custom"
            with contextlib.suppress(Exception):
                if getattr(p, k) != v:
                    setattr(p, k, v)
        with contextlib.suppress(Exception):
            ps = str(min(PIXEL_SIZES, key=lambda v: abs(v - int(c["pixel_size"]))))
            if p.pixel_size != ps:
                p.pixel_size = ps
    finally:
        SYNC["busy"] = False


@bpy.app.handlers.persistent
def _on_load(*_args):
    """Another file: its Graphics panel shows the settings in force; EEVEE follows them (when allowed)."""
    S["res"] = S["set"] = None
    with contextlib.suppress(Exception):
        sync_props(bpy.context.scene)
        apply_scene()


def register() -> None:
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)


def unregister() -> None:
    stop()
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)


__all__ = ["register", "unregister", "configure", "from_env", "effective", "body_res", "set_body", "saving",
           "rendering", "ghost_limits", "picture_scale", "apply_scene", "start", "stop", "after_frame", "status",
           "sync_props", "S"]
