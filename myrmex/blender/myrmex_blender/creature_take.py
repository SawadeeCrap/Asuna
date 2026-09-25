"""Creature takes -> a renderable Blender animation (Black Nanomaterial v1 and Mimetic Polyalloy v2).

    from myrmex_blender import creature_take
    creature_take.import_take("~/Myrmex/takes/polyalloy_take_20260925_114209.npz", audio="~/song.wav")
    # then Render > Render Animation (or creature_take.configure_video_output() + render)

Everything the live view showed is rebuilt as ordinary Blender animation, so the .blend is
self-contained and renders anywhere (no add-on needed):
  * metaball elements  -> keyframes (co, radius, size), decimated where the motion is linear;
  * internal strut lattice (v2) -> a Point Cache 2 file + Mesh Cache modifier (links re-form over time);
  * obstacles (v2), light rig and floor (they follow the organism) -> keyframes;
  * shader activity / glow / time -> keyframes on the material's value nodes;
  * the live camera -> one camera per shot + timeline markers (as in the humanoid takes);
  * the song -> a sound strip placed so that the music lines up with the recorded song position.
"""
from __future__ import annotations

import os
import struct

import bpy
import numpy as np

from myrmex.creature.take import CreatureTake

from . import cinema, compat, preview
from .creature import CreatureView, setup_creature_scene


def _decimate(vals: np.ndarray, tol: float, stride: int) -> np.ndarray:
    """Indices worth a key: curvature above ``tol`` or every ``stride`` frames; flat holds collapse."""
    n = len(vals)
    if n <= 2:
        return np.arange(n)
    if np.all(vals == vals[0]):
        return np.array([0])
    d2 = np.abs(vals[:-2] - 2 * vals[1:-1] + vals[2:])
    flat = (vals[:-2] == vals[1:-1]) & (vals[1:-1] == vals[2:])
    idx = np.arange(1, n - 1)
    keep = (d2 > tol) | ((idx % stride == 0) & ~flat)
    return np.concatenate([[0], idx[keep], [n - 1]])


def key_channels(owner: bpy.types.ID, action_name: str, channels, frames: np.ndarray, tol: float = 2e-4,
                 stride: int = 3, interpolation: int = 1) -> int:
    """Fast F-curve keying with decimation.  Returns the number of keys written."""
    act = compat.new_action_for(owner, action_name)
    total = 0
    for path, idx, vals in channels:
        keep = _decimate(np.asarray(vals, float), tol, stride)
        fc = compat.ensure_fcurve(act, owner, path, idx, "Myrmex")
        k = len(keep)
        fc.keyframe_points.add(k)
        co = np.empty(2 * k)
        co[0::2] = frames[keep]
        co[1::2] = np.asarray(vals, float)[keep]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", [interpolation] * k)
        fc.update()
        total += k
    return total


def write_pc2(path: str, frames: np.ndarray, start: float = 0.0, rate: float = 1.0) -> None:
    """Point Cache 2 (Blender's Mesh Cache modifier reads it natively)."""
    frames = np.ascontiguousarray(frames, "<f4")
    with open(path, "wb") as f:
        f.write(struct.pack("<12siiffi", b"POINTCACHE2\0", 1, frames.shape[1], start, rate, frames.shape[0]))
        f.write(frames.tobytes())


def _clear_anim(idb) -> None:
    if idb is not None and idb.animation_data is not None:
        idb.animation_data_clear()


def import_take(path: str, audio: str | None = None, frame_start: int = 1, fps: int | None = None,
                use_camera: bool = True, keep_look: bool = True, scene: bpy.types.Scene | None = None) -> dict:
    path = os.path.expanduser(path)
    take = CreatureTake(path)
    sc = scene or bpy.context.scene
    try:                                                   # a running live link would overwrite the take
        from . import ui
        ui._stop_link()
    except Exception:
        pass
    fps = int(fps or round(take.fps))
    sc.render.fps, sc.render.fps_base = fps, 1.0
    view = setup_creature_scene(sc, take.variant, keep_look=keep_look)
    mb = view.mb
    pos = take.resampled("pos", fps)
    radius = take.resampled("radius", fps)
    stretch = take.resampled("stretch", fps)
    m, n = radius.shape
    frames = np.arange(m, dtype=float) + frame_start
    view._ensure(n)
    kind = take.d.get("kind", np.ones(n))
    for i, e in enumerate(mb.elements):
        e.type, e.hide = "ELLIPSOID", False
        e.stiffness = 2.0 if kind[i] == 0 else 1.6
    # Hidden material (radius ~0) holds its last visible place: flat curves, few keys.
    hidden = radius < 0.01
    radius = np.where(hidden, 1e-4, radius)
    if hidden.any():
        for i in np.nonzero(hidden.any(0))[0]:
            h = hidden[:, i]
            last = np.maximum.accumulate(np.where(~h, np.arange(m), 0))
            pos[:, i] = pos[last, i]
    _clear_anim(mb)
    ch = []
    for i in range(n):
        ch += [(f"elements[{i}].co", a, pos[:, i, a]) for a in range(3)]
        ch.append((f"elements[{i}].radius", 0, radius[:, i]))
        if take.variant != "polyalloy":
            ch += [(f"elements[{i}].size_{ax}", 0, 0.55 * stretch[:, i, a]) for a, ax in enumerate("xyz")]
    keys = key_channels(mb, "MyrmexTakeBody", ch, frames)
    # Shader activity.
    t = take.resampled("t", fps)
    surface, glow = take.resampled("surface", fps), take.resampled("glow", fps)
    mat = mb.materials[0] if mb.materials else None
    nt = mat.node_tree if mat is not None else None
    if nt is not None:
        _clear_anim(nt)
        sch = []
        for name, vals in (("MyrmexTime", t * (0.05 + 0.25 * surface)), ("MyrmexActivity", 0.2 + 0.8 * surface),
                           ("MyrmexGlow", 4.0 * glow)):
            if name in nt.nodes:
                sch.append((f'nodes["{name}"].outputs[0].default_value', 0, vals))
        keys += key_channels(nt, "MyrmexTakeShader", sch, frames, tol=1e-3)
    # Lights and floor follow the organism (and rise with it when it flies).
    com = take.resampled("com", fps) if "com" in take.d else pos.mean(1)
    heading = np.unwrap(take.d["heading"])[take.sample_index(fps)[0]] if "heading" in take.d else np.zeros(m)
    lift = np.maximum(0.0, com[:, 2] - 1.2) if take.variant in ("polyalloy", "colony", "hive") else np.zeros(m)
    rig, floor = bpy.data.objects.get("MyrmexLightRig"), bpy.data.objects.get("MyrmexFloor")
    if rig is not None:
        _clear_anim(rig)
        keys += key_channels(rig, "MyrmexTakeLights", [("location", 0, com[:, 0]), ("location", 1, com[:, 1]),
                                                      ("location", 2, lift), ("rotation_euler", 2, heading)], frames)
    if floor is not None:
        _clear_anim(floor)
        keys += key_channels(floor, "MyrmexTakeFloor", [("location", 0, com[:, 0]), ("location", 1, com[:, 1])], frames)
    # Mimetic Polyalloy / Colony: strut lattice (PC2 cache) and obstacles.
    if take.variant in ("polyalloy", "colony", "hive") and "links" in take.d:
        links = take.resampled("links", fps).astype(float)
        n_links = max(1, int((links[:, :, 0] >= 0).sum(1).max()))
        if len(view.lattice.data.vertices) != 2 * n_links:        # only the slots this take uses (smaller cache)
            view.make_polyalloy(n_links, len(view.obstacles) or 4)
        pts = np.stack([CreatureView.strut_points(pos[k], links[k], n_links) for k in range(m)])
        pc2 = os.path.splitext(path)[0] + "_struts.pc2"
        write_pc2(pc2, pts)
        lat = view.lattice
        mc = lat.modifiers.get("TakeCache") or lat.modifiers.new("TakeCache", "MESH_CACHE")
        mc.cache_format, mc.filepath, mc.time_mode, mc.play_mode = "PC2", pc2, "FRAME", "SCENE"
        mc.frame_start = float(frame_start)
        lat.modifiers.move(lat.modifiers.find(mc.name), 0)   # the cache must come before the struts
        obs = take.resampled("obstacles", fps)
        for k, o in enumerate(view.obstacles):
            _clear_anim(o)
            if k >= obs.shape[1]:
                continue
            r = obs[:, k, 3]
            keys += key_channels(o, f"MyrmexTakeObstacle{k}", [("location", a, obs[:, k, a]) for a in range(3)] +
                                 [("scale", a, r) for a in range(3)], frames, tol=1e-3)
    # Hive: the nanomachine swarm (PC2 cache).
    parts = take.particles(fps)
    if parts is not None:
        view.make_hive(parts.shape[1])
        pc2 = os.path.splitext(path)[0] + "_swarm.pc2"
        write_pc2(pc2, parts)
        sw = view.swarm
        mc = sw.modifiers.get("TakeCache") or sw.modifiers.new("TakeCache", "MESH_CACHE")
        mc.cache_format, mc.filepath, mc.time_mode, mc.play_mode = "PC2", pc2, "FRAME", "SCENE"
        mc.frame_start = float(frame_start)
        sw.modifiers.move(sw.modifiers.find(mc.name), 0)
    # Colony / Hive: armour plates (PC2 cache) and the prey.
    if take.variant in ("colony", "hive") and "plate" in take.d:
        plate, nrm = take.resampled("plate", fps), take.resampled("nrm", fps)
        view.make_colony(n)
        pts = np.stack([CreatureView.plate_points(pos[k], nrm[k], plate[k], radius[k]) for k in range(m)])
        pc2 = os.path.splitext(path)[0] + "_plates.pc2"
        write_pc2(pc2, pts)
        pl = view.plates
        mc = pl.modifiers.get("TakeCache") or pl.modifiers.new("TakeCache", "MESH_CACHE")
        mc.cache_format, mc.filepath, mc.time_mode, mc.play_mode = "PC2", pc2, "FRAME", "SCENE"
        mc.frame_start = float(frame_start)
        lure = take.resampled("lure", fps)
        _clear_anim(view.lure)
        keys += key_channels(view.lure, "MyrmexTakePrey", [("location", a, lure[:, a]) for a in range(3)] +
                             [("scale", a, lure[:, 3]) for a in range(3)], frames, tol=1e-3)
    # Camera and markers.
    cams = []
    if use_camera:
        track = take.camera_track(fps)
        if track is not None:
            cams = cinema.apply_camera_track(track, frame_start)
    # Music.
    off = take.audio_offset()
    if audio:
        audio = os.path.expanduser(audio)
        preview.add_audio(audio, int(round(frame_start - (off or 0.0) * fps)))
    sc.frame_start, sc.frame_end = frame_start, frame_start + m - 1
    sc.frame_set(frame_start)
    sc["myrmex_take"] = path
    return {"variant": take.variant, "frames": m, "fps": fps, "keys": keys, "cameras": len(cams),
            "audio_offset": off, "duration": m / fps}


def configure_video_output(path: str, resolution=(1920, 1080), preset: str = "eevee", samples: int | None = None,
                           motion_blur: bool = False, keep: bool = False) -> str:
    """Render straight to an .mp4 (H.264 + AAC) with the scene's sound strips.

    ``keep``: the scene's own engine / samples / colour / shadows stay as they are (a saved look);
    only the size, the frame rate and the output are set.
    """
    sc = bpy.context.scene
    r = sc.render
    if keep:
        r.resolution_x, r.resolution_y = int(resolution[0]), int(resolution[1])
        r.resolution_percentage = 100
        r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    else:
        cinema.configure_render(preset, resolution, sc.render.fps, motion_blur, samples)
    r.use_lock_interface = True
    ims = r.image_settings
    if hasattr(ims, "media_type"):
        try:
            ims.media_type = "VIDEO"
        except Exception:
            pass
    ims.file_format = "FFMPEG"
    ff = r.ffmpeg
    ff.format = "MPEG4"
    ff.codec = "H264"
    ff.constant_rate_factor = "HIGH"
    ff.audio_codec = "AAC"
    ff.audio_bitrate = 256
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    r.filepath = path
    return path


__all__ = ["import_take", "configure_video_output", "write_pc2", "key_channels"]
