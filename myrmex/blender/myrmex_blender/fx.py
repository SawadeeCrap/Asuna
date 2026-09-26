"""Myrmex FX in Blender: afterimages, light ribbons and the picture effects - no TouchDesigner.

Where the numbers come from:

* live - the engine appends its FX block (myrmex.realtime.fx: music + organism drives, the app's effect
  rack with its MIDI knobs, on / format / preview) to every frame; live.LiveLink calls ``live_frame``;
* an opened take - the take player calls ``take_frame`` every frame with the recorded block (a take made
  before the effects existed gets one rebuilt from what it has);
* Blender on its own - ``configure`` (the app's control channel, MYRMEX_FX, the Myrmex panel) sets the
  rack; the drives stay at rest.

What runs where: fx_ghosts / fx_ribbons are real geometry in the scene (viewport, Syphon, renders);
fx_post draws the picture effects on the GPU (the 3D view looking through the camera shows the result,
Syphon sends it, fx_render applies it to rendered takes).
"""
from __future__ import annotations

import json
import math
import os

import bpy
import numpy as np

from myrmex.realtime.fx import (CHANNELS, DEFAULT_RACK, PRESETS, RACK, as_dict, decode_trailer, rack_from_preset,
                                size_of)

S: dict = {
    "cfg": {"on": False, "rack": dict(DEFAULT_RACK), "vertical": False, "preview": 1.0, "monitor": True,
            "preset": "", "replay": True},
    "vals": {},                    # the latest block: drives (+ r_<rack> and on / vertical / preview when live)
    "rack_src": "cfg",             # "block": the frame's rack (live knobs, a take's recorded moves) | "cfg"
    "live": False,                 # the engine is streaming (its block says on / off)
    "frame": 0,                    # +1 per new frame: the picture effects run once per frame
    "t": 0.0,
    "org": {"com": np.zeros(3), "size": 0.8, "pos": None, "radius": None},
    "ghosts": None, "ribbons": None,
    "post_by_frame": None,         # take renders: the picture parameters of every frame
    "error": "",
}


# ---------------------------------------------------------------------- settings
def configure(d: dict) -> dict:
    """{"on", "preset", "rack": {...}, "vertical", "preview", "monitor", "replay"} (any subset)."""
    cfg = S["cfg"]
    rack = d.get("rack")
    if d.get("preset") in PRESETS and not isinstance(rack, dict):
        cfg["rack"] = rack_from_preset(d["preset"])
    if isinstance(rack, dict):
        for k, v in rack.items():
            if k in RACK:
                cfg["rack"][k] = float(min(1.0, max(0.0, float(v))))
    for k in ("on", "vertical", "monitor", "replay"):
        if k in d:
            cfg[k] = bool(d[k])
    if "preview" in d:
        cfg["preview"] = float(min(1.0, max(0.25, float(d["preview"]))))
    if "preset" in d:
        cfg["preset"] = str(d["preset"] or "")
    if not bpy.app.background:
        if "vertical" in d:
            apply_format(bpy.context.scene, cfg["vertical"])
        _sync_monitor()
    _sync_props()
    S["frame"] += 1
    return status()


def render_settings() -> dict:
    """What a background Blender needs to render a take with these effects (MYRMEX_FX)."""
    c = S["cfg"]
    return {"on": active(), "preset": c.get("preset", ""), "rack": dict(rack()), "vertical": c.get("vertical"),
            "replay": c.get("replay", True)}


def _sync_props() -> None:
    """The Myrmex FX panel shows what the app set (item assignment: no update callbacks)."""
    try:
        sc = bpy.context.scene
        p = getattr(sc, "myrmex_fx", None) if sc is not None else None
    except (AttributeError, RuntimeError):
        return
    if p is None:
        return
    c = S["cfg"]
    try:
        p["on"] = bool(c["on"])
        p["vertical"] = 1 if c.get("vertical") else 0
        p["monitor"] = bool(c.get("monitor", True))
        for k in RACK:
            p[k] = float(c["rack"].get(k, DEFAULT_RACK[k]))
        from myrmex.realtime.fx import PRESET_NAMES
        name = c.get("preset", "")
        p["preset"] = PRESET_NAMES.index(name) if name in PRESET_NAMES else len(PRESET_NAMES)
    except (TypeError, KeyError, AttributeError):
        pass


def from_env() -> dict | None:
    """MYRMEX_FX='{"on": true, "preset": ..., "rack": {...}, "vertical": false}' (set by the app)."""
    raw = os.environ.get("MYRMEX_FX")
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    return configure(d) if isinstance(d, dict) else None


def rack() -> dict:
    v = S["vals"]
    if S["rack_src"] == "block" and ("r_" + RACK[0]) in v:
        return {k: v.get("r_" + k, S["cfg"]["rack"].get(k, DEFAULT_RACK[k])) for k in RACK}
    return S["cfg"]["rack"]


def drives() -> dict:
    return S["vals"]


def active() -> bool:
    v = S["vals"]
    if S["live"] and "on" in v:                      # live: the app's switch travels with every frame
        return v["on"] > 0.5
    return bool(S["cfg"]["on"])


def picture_on(r: dict | None = None) -> bool:
    """Any effect of the picture (the GPU passes: motion echo, glow, colour) away from neutral."""
    r = r or rack()
    return r.get("trails", 0.0) > 0.01 or r.get("bloom", 0.0) > 0.01 or \
        any(abs(r.get(k, 0.5) - 0.5) > 0.01 for k in ("exposure", "contrast", "saturation"))


def apply_format(scene, vertical: bool) -> None:
    """Horizontal 1920x1080 / vertical 1080x1920 - only when the scene has the other orientation."""
    if scene is None:
        return
    r = scene.render
    portrait = r.resolution_y > r.resolution_x
    if bool(vertical) != portrait:
        r.resolution_x, r.resolution_y = size_of(bool(vertical))
        r.resolution_percentage = 100


WARMUP = "MyrmexFXWarmup"


def _sync_monitor(views: bool = False) -> None:
    """The monitor on / off (the 3D view keeps its own shading: the monitor never switches it)."""
    if bpy.app.background:
        return
    from . import fx_post
    want = active() and bool(S["cfg"].get("monitor", True))
    if want and not fx_post.enabled():
        fx_post.enable()
    elif not want and fx_post.enabled():
        fx_post.disable()


def warmup(scene=None) -> None:
    """A speck carrying the ghost versions of the organism's materials and the trace material: the 3D view
    compiles them in the background before the first copy appears (an offscreen picture would compile
    them on the spot, stopping Blender)."""
    from .fx_ghosts import fx_collection, ghost_materials_for, sources
    from .fx_ribbons import ribbon_material
    scene = scene or bpy.context.scene
    mats = ghost_materials_for(sources(scene)) + [ribbon_material()]
    ob = bpy.data.objects.get(WARMUP)
    if ob is not None and ob.type == "MESH" and list(ob.data.materials) == mats:
        return
    if ob is not None:
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    me = bpy.data.meshes.new(WARMUP)
    k = len(mats)
    verts = [(0.001 * i, 0.0, 0.0) for i in range(k + 2)] + [(0.0, 0.001, 0.0)]
    me.from_pydata(verts, [], [(i, i + 1, k + 2) for i in range(k)])
    for m in mats:
        me.materials.append(m)
    me.polygons.foreach_set("material_index", list(range(k)))
    ob = bpy.data.objects.new(WARMUP, me)
    fx_collection(scene).objects.link(ob)
    ob.hide_render = True
    ob.visible_shadow = False
    ob["myrmex_birth"] = -1e9                              # (a copy long gone: nothing shows)


def _parts():
    from .fx_ghosts import Ghosts
    from .fx_ribbons import Ribbons
    if S["ghosts"] is None:
        S["ghosts"] = Ghosts()
    if S["ribbons"] is None:
        S["ribbons"] = Ribbons()
    return S["ghosts"], S["ribbons"]


def reset(remove: bool = False) -> None:
    """Forget the copies / ribbons / trails (a new file, a new take)."""
    g, r = S.get("ghosts"), S.get("ribbons")
    try:
        if g is not None:
            g.clear(remove=remove)
        if r is not None:
            r.clear()
    except ReferenceError:
        pass
    if remove:
        S["ghosts"] = S["ribbons"] = None
    try:
        from . import fx_post
        pipe = fx_post._M.get("pipe")
        if pipe is not None:
            pipe.reset()
    except Exception:
        pass


# ---------------------------------------------------------------------- one frame
def _organism(t: float, com, pos, radius, size=None) -> None:
    org = S["org"]
    com = np.asarray(com, float) if com is not None else org["com"]
    if size is None or not size > 0:
        if pos is not None and len(pos):
            p = np.asarray(pos, float)
            vis = np.asarray(radius, float) > 0.01 if radius is not None else np.ones(len(p), bool)
            q = p[vis] if vis.any() else p
            size = float(np.sqrt(((q - com) ** 2).sum(1).mean()))
        else:
            size = org["size"]
    org.update(com=com, pos=pos, radius=radius, size=max(0.05, float(size)))
    S["t"] = float(t)


def _step(scene, prepare=None, cam_pos=None) -> None:
    """The geometry effects of this frame (after the organism of the frame is in the scene)."""
    S["frame"] += 1
    if not active():
        g, r = S.get("ghosts"), S.get("ribbons")
        if g is not None and any(b > -1e8 for b in g.birth):
            g.clear()
        if r is not None and r.hist is not None:
            r.clear()
        return
    g, rb = _parts()
    rk, dv, org = rack(), drives(), S["org"]
    try:
        if rk.get("ghosts", 0.0) > 0.01 and prepare is not None:
            prepare()
        g.update(scene, S["t"], org["com"], org["size"], rk, dv)
        if cam_pos is None or not np.all(np.isfinite(cam_pos)):
            cam = scene.camera
            cam_pos = np.array(cam.matrix_world.translation) if cam is not None else \
                org["com"] + np.array([0, -5.0, 1.5])
        rb.update(scene, S["t"], org["pos"], org["radius"], org["com"], org["size"], rk, cam_pos)
        S["error"] = ""
    except ReferenceError:                                  # data re-created (undo, file load)
        reset(remove=True)
    except Exception as e:
        S["error"] = f"{type(e).__name__}: {e}"
        print("Myrmex FX:", S["error"])


def live_frame(fr, block, scene=None) -> None:
    """A creature frame has just been applied (live.LiveLink)."""
    scene = scene or bpy.context.scene
    if block is not None:
        S["vals"] = as_dict(block)
        S["rack_src"] = "block"
        S["live"] = True
        v = S["vals"]
        if "vertical" in v:
            vert = v["vertical"] > 0.5
            if vert != S["cfg"].get("vertical"):
                S["cfg"]["vertical"] = vert
            apply_format(scene, vert)
        if "preview" in v:
            S["cfg"]["preview"] = min(1.0, max(0.25, v["preview"]))
    _sync_monitor()
    _organism(fr.t, fr.com, fr.pos, fr.radius, S["vals"].get("size"))
    _step(scene)


def live_pose(fr, block, scene=None) -> None:
    """A humanoid pose frame has just been applied: copies of the skinned character."""
    scene = scene or bpy.context.scene
    if block is not None:
        S["vals"] = as_dict(block)
        S["rack_src"] = "block"
        S["live"] = True
        v = S["vals"]
        if "vertical" in v:
            S["cfg"]["vertical"] = v["vertical"] > 0.5
            apply_format(scene, S["cfg"]["vertical"])
    _sync_monitor()
    _organism(fr.t, fr.subject_pos, None, None, 0.6)
    _step(scene)


def take_frame(scene, f: int, d: dict) -> None:
    """The take player has just built frame ``f`` (index into the take arrays ``d``)."""
    vals = {}
    if "fx" in d:
        vals = as_dict(d["fx"][f], d.get("fx_names") or CHANNELS)
    elif "td" in d:
        vals = as_dict(d["td"][f], d.get("td_names") or ())
    else:
        vals = _rebuilt(d, f)
    S["vals"] = vals
    S["rack_src"] = "block" if ("fx" in d and S["cfg"].get("replay", True)) else "cfg"
    S["live"] = False
    if not bpy.app.background:
        _sync_monitor()
    pos = d["pos"][f] if "pos" in d else None
    com = d["com"][f] if "com" in d else (np.asarray(pos, float).mean(0) if pos is not None else None)
    t = float(d["t"][f]) if "t" in d else f / float(d.get("fps", 30.0))
    _organism(t, com, pos, d["radius"][f] if "radius" in d else None, vals.get("size"))

    def prepare():                                       # the body as it is in this frame (renders: the
        _meta_from_take(d, f)                            # originals are not animated)
    _step(scene, prepare, np.asarray(d["cam_p"][f], float) if "cam_p" in d else None)


def _rebuilt(d: dict, f: int) -> dict:
    """Drives for a take recorded before the effects existed: speed, glow, arousal from what it has."""
    fps = float(d.get("fps", 30.0))
    v = {}
    if "com" in d and f > 0:
        v["speed"] = float(np.linalg.norm(np.asarray(d["com"][f], float) - np.asarray(d["com"][f - 1], float))) * fps
    for k in ("glow", "arousal"):
        if k in d:
            v[k] = float(d[k][f])
    ev = d.get("ev_frames")
    if ev is not None and len(ev):
        k = np.searchsorted(ev, f, side="right")
        if k > 0:
            v["impact"] = float(math.exp(-(f - ev[k - 1]) / fps / 0.35))
    v["energy"] = float(d["glow"][f]) if "glow" in d else 0.3
    return v


def _meta_from_take(d: dict, f: int) -> None:
    """Write the recorded metaball body of frame ``f`` into the metaball (what a copy is made from)."""
    mb = bpy.data.metaballs.get("CreatureBody")
    if mb is None or "pos" not in d or "radius" not in d:
        return
    pos, rad = np.asarray(d["pos"][f], np.float32), np.asarray(d["radius"][f], np.float32)
    n = len(rad)
    if len(mb.elements) != n or n == 0:
        return
    hide = rad < 0.01
    mb.elements.foreach_set("co", pos.ravel())
    mb.elements.foreach_set("radius", np.where(hide, 1e-4, rad).astype(np.float32))
    mb.elements.foreach_set("hide", hide)
    if "stretch" in d and d.get("variant") != "polyalloy":      # (as the take import keys them)
        st = 0.55 * np.asarray(d["stretch"][f], np.float32)
        for a, ax in enumerate("xyz"):
            mb.elements.foreach_set("size_" + ax, np.ascontiguousarray(st[:, a]))


# ---------------------------------------------------------------------- the picture effects' numbers
def post_params(scene, w: int, h: int, cam=None) -> np.ndarray:
    """The FxParams block (fx_post.UBO) for this frame: glow, motion echo, colour - nothing else."""
    from .fx_post import N_PARAMS
    r, d = rack(), drives()
    react = float(r.get("react", 0.5))
    kick, energy = float(d.get("kick", 0.0)), float(d.get("energy", 0.0))
    glow = float(r.get("bloom", 0.0))
    echo = float(r.get("trails", 0.0))
    out = np.zeros(N_PARAMS, np.float32)
    out[0:4] = (S["t"], w / max(h, 1), float(S["frame"]), 0.0)                       # frame
    out[16:20] = (0.82, 0.12, 0.9 * glow * (1.0 + react * (0.35 * kick + 0.15 * energy)), 0.0)   # bloom
    out[20:24] = ((0.72 + 0.24 * echo) if echo > 0.01 else 0.0, 0.0, 0.0, 0.0)     # trail: decay
    out[32:36] = (0.0, min(1.0, 0.9 * echo), 0.0, 0.0)                               # look3: echo mix
    out[36:40] = ((float(r.get("exposure", 0.5)) - 0.5) * 2.0,                        # grade
                  2.0 ** ((float(r.get("contrast", 0.5)) - 0.5) * 1.4), 2.0 * float(r.get("saturation", 0.5)), 0.0)
    out[44:48] = (float(w), float(h), 1.0 / max(w, 1), 1.0 / max(h, 1))              # size
    return np.nan_to_num(out)


def status() -> dict:
    out = {"on": active(), "cfg": {k: v for k, v in S["cfg"].items() if k != "rack"}, "rack": dict(rack()),
           "source": S["rack_src"], "error": S["error"]}
    g = S.get("ghosts")
    if g is not None:
        out["ghosts"] = {"spawned": g.spawned, "slots": g.count}
    if not bpy.app.background:
        from . import fx_post
        out["monitor"] = fx_post.status()
    return out


@bpy.app.handlers.persistent
def _on_load(*_args):
    reset(remove=True)
    S["vals"], S["rack_src"], S["live"] = {}, "cfg", False
    if not bpy.app.background:
        from . import fx_post
        fx_post.reapply(bpy.context.scene)


def register() -> None:
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)


def unregister() -> None:
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    if not bpy.app.background:
        from . import fx_post
        fx_post.disable()


def look_through_camera() -> None:
    """The 3D views look through the scene camera (the live / take start): the monitor follows the switch."""
    _sync_monitor(True)


__all__ = ["look_through_camera", "warmup", "configure", "render_settings", "from_env", "live_frame", "live_pose", "take_frame", "post_params", "active", "rack",
           "drives", "status", "apply_format", "picture_on", "reset", "register", "unregister", "decode_trailer",
           "S"]
