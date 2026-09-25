"""Imported takes: the geometry that is not plain animation (struts / bone links, plates / scutes) is rebuilt
every frame.

Keyframes would need hundreds of megabytes for a take (every vertex of every link, every frame), so
the importer keeps the recorded skeleton and a frame handler builds the links and scutes of the
current frame from it - while scrubbing, playing and rendering (Render Animation, Render Video).
Opening a saved .blend in a Blender started by the Myrmex app re-attaches the take automatically.
"""
from __future__ import annotations

import os

import bpy
import numpy as np

_P: dict = {"key": None}


def attach(scene: bpy.types.Scene, take, frame_start: int, fps: int) -> None:
    d = {"pos": take.resampled("pos", fps).astype(np.float32)}
    for k in ("links",):
        if k in take.d:
            d[k] = take.resampled(k, fps).astype(np.float32)
    for k in ("nrm", "plate", "radius", "heading", "t", "arousal", "com", "light"):
        if k in take.d:
            d[k] = take.resampled(k, fps).astype(np.float32)
    if "light" in d:
        d["light"] /= 255.0
    if "heading" in take.d:
        d["heading"] = np.unwrap(take.d["heading"])[take.sample_index(fps)[0]]
    from .creature import variant_style
    _P.update(key=take.path, d=d, start=int(frame_start), n=len(d["pos"]), style=variant_style(take.variant)[0])
    scene["myrmex_take_player"] = take.path
    scene["myrmex_take_start"] = int(frame_start)
    scene["myrmex_take_fps"] = int(fps)
    if _on_frame not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(_on_frame)
    apply(scene)


def detach(scene: bpy.types.Scene | None = None) -> None:
    scene = scene or bpy.context.scene
    for k in ("myrmex_take_player", "myrmex_take_start", "myrmex_take_fps"):
        if k in scene:
            del scene[k]
    _P.clear()
    _P["key"] = None


def apply(scene: bpy.types.Scene) -> None:
    d = _P.get("d")
    if d is None or scene.get("myrmex_take_player") != _P.get("key"):
        return
    from .creature import (BONES, FINS, LATTICE, PANELS, PLATES, RAILS, SCUTES, TENDON_THICK, TENDONS, _flow,
                           bone_points, fin_points, panel_light, panel_points, plate_points, rail_light, rail_points,
                           scute_points, set_cyber_mesh, strut_points, tendon_points)
    style = _P.get("style", 0)
    f = int(np.clip(scene.frame_current - _P["start"], 0, _P["n"] - 1))
    pos = d["pos"][f].astype(float)
    if "nrm" in d:
        up = d["nrm"][f].astype(float)
    else:
        c = d["com"][f] if "com" in d else pos.mean(0)
        up = pos - c
    t = float(d["t"][f]) if "t" in d else f / 30.0
    ar = float(d["arousal"][f]) if "arousal" in d else 0.5
    if style >= 3:                                   # Mimetic line: tendons + fins that trail the recorded motion
        ob = bpy.data.objects.get(TENDONS)
        if ob is not None and "links" in d:
            pts = tendon_points(pos, d["links"][f].astype(float), up, t, ar, TENDON_THICK.get(style, 1.0))
            if len(pts) == len(ob.data.vertices):
                ob.data.vertices.foreach_set("co", pts.astype(np.float32).ravel())
                ob.data.update()
        ob = bpy.data.objects.get(FINS)
        if ob is not None:
            ts = d["t"] if "t" in d else np.arange(_P["n"]) / 30.0
            hist = [(float(ts[g]), d["pos"][g].astype(float)) for g in range(f, max(-1, f - 24), -1)]
            com = d["com"][f] if "com" in d else pos.mean(0)
            pts = fin_points(style, hist, up, com, float(d["heading"][f]) if "heading" in d else 0.0, t)
            if len(pts) == len(ob.data.vertices):
                ob.data.vertices.foreach_set("co", pts.astype(np.float32).ravel())
                ob.data.update()
        return
    if style == 2:                                   # Cyber Hive: rails + panels carry their light
        com = d["com"][f] if "com" in d else pos.mean(0)
        hd = float(d["heading"][f]) if "heading" in d else 0.0
        light = d["light"][f] if "light" in d else None
        ob = bpy.data.objects.get(RAILS)
        if ob is not None and "links" in d:
            links = d["links"][f].astype(float)
            pts = rail_points(pos, links, up, t, ar)
            set_cyber_mesh(ob, pts, _flow(pts, com, hd), rail_light(links, light))
        ob = bpy.data.objects.get(PANELS)
        if ob is not None and "plate" in d:
            rad = d["radius"][f].astype(float) if "radius" in d else np.full(len(pos), 0.3)
            pts = panel_points(pos, up, d["plate"][f].astype(float), rad, hd)
            set_cyber_mesh(ob, pts, _flow(pts, com, hd), panel_light(light, len(pos)))
        return
    ob = bpy.data.objects.get(BONES if style == 1 else LATTICE)
    if ob is not None and "links" in d:
        links = d["links"][f].astype(float)
        pts = bone_points(pos, links, up, t, ar) if style == 1 else strut_points(pos, links, len(links))
        if len(pts) == len(ob.data.vertices):
            ob.data.vertices.foreach_set("co", pts.astype(np.float32).ravel())
            ob.data.update()
    ob = bpy.data.objects.get(SCUTES if style == 1 else PLATES)
    if ob is not None and "plate" in d:
        rad = d["radius"][f].astype(float) if "radius" in d else np.full(len(pos), 0.3)
        plate = d["plate"][f].astype(float)
        pts = scute_points(pos, up, plate, rad, float(d["heading"][f]) if "heading" in d else 0.0) if style == 1 \
            else plate_points(pos, up, plate, rad)
        if len(pts) == len(ob.data.vertices):
            ob.data.vertices.foreach_set("co", pts.astype(np.float32).ravel())
            ob.data.update()


@bpy.app.handlers.persistent
def _on_frame(scene, *_args):
    try:
        apply(scene)
    except Exception as e:                    # never break playback / rendering
        print("Myrmex take player:", e)


@bpy.app.handlers.persistent
def _on_load(*_args):
    scene = bpy.context.scene
    path = scene.get("myrmex_take_player") if scene is not None else None
    if not path or not os.path.exists(path):
        return
    from myrmex.creature.take import CreatureTake
    attach(scene, CreatureTake(path), int(scene.get("myrmex_take_start", 1)), int(scene.get("myrmex_take_fps", 30)))


def register() -> None:
    if _on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load)


def unregister() -> None:
    for lst, fn in ((bpy.app.handlers.load_post, _on_load), (bpy.app.handlers.frame_change_pre, _on_frame)):
        if fn in lst:
            lst.remove(fn)


__all__ = ["attach", "detach", "apply", "register", "unregister"]
