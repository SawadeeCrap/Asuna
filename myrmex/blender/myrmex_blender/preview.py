"""Fast preview rendering (Workbench) with a follow camera and the music attached."""
from __future__ import annotations

import math
import os
import shutil
import subprocess

import bpy
import numpy as np
from mathutils import Vector

from myrmex.performance.performance import Performance

from . import compat


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def bone_path(perf: Performance, rest_head: np.ndarray, bone: str) -> np.ndarray:
    i = perf.bone_names.index(bone)
    D = perf.deltas[:, i].astype(float)
    return np.einsum("tij,j->ti", D[:, :3, :3], rest_head) + D[:, :3, 3]


def smooth_path(P: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return P
    k = np.ones(window) / window
    pad = window // 2
    return np.stack([np.convolve(np.pad(P[:, i], (pad, pad), mode="edge"), k, mode="valid")[: len(P)]
                     for i in range(P.shape[1])], axis=1)


def setup_workbench(scene: bpy.types.Scene, resolution=(640, 360), shadows: bool = False) -> None:
    scene.render.engine = "BLENDER_WORKBENCH"
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "SINGLE"
    sh.single_color = (0.8, 0.8, 0.82)
    sh.show_shadows = shadows
    sh.show_cavity = False
    sh.background_type = "VIEWPORT"
    sh.background_color = (0.93, 0.93, 0.94)
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"


def ensure_ground(size: float = 60.0) -> bpy.types.Object:
    ob = bpy.data.objects.get("MyrmexGround")
    if ob:
        return ob
    me = bpy.data.meshes.new("MyrmexGround")
    s = size / 2
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    ob = bpy.data.objects.new("MyrmexGround", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def follow_camera(perf: Performance, center: np.ndarray, name: str = "MyrmexPreviewCam",
                  distance: float = 4.2, height: float = 1.0, lens: float = 40.0,
                  orbit_speed: float = 0.05, start_angle: float = -0.6) -> bpy.types.Object:
    sc = bpy.context.scene
    cam = bpy.data.objects.get(name)
    if cam is None:
        cd = bpy.data.cameras.new(name)
        cam = bpy.data.objects.new(name, cd)
        sc.collection.objects.link(cam)
    cam.data.lens = lens
    sc.camera = cam
    P = smooth_path(center, int(perf.fps * 1.2))
    for k in range(perf.frames):
        ang = start_angle + orbit_speed * k / perf.fps
        tgt = Vector((P[k, 0], P[k, 1], height))
        d = Vector((math.cos(ang), math.sin(ang), 0.22)).normalized()
        cam.location = tgt + d * distance
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=k + 1)
        cam.keyframe_insert("rotation_euler", frame=k + 1)
    return cam


def add_audio(path: str, frame_start: int = 1) -> None:
    sc = bpy.context.scene
    strips = compat.sequence_strips(sc)
    for s in list(strips):
        if s.name == "MyrmexAudio":
            strips.remove(s)
    strips.new_sound("MyrmexAudio", path, 1, frame_start)


def render_frames(outdir: str, frames: range | None = None) -> list[str]:
    sc = bpy.context.scene
    os.makedirs(outdir, exist_ok=True)
    frames = frames or range(sc.frame_start, sc.frame_end + 1)
    files = []
    for f in frames:
        sc.frame_set(f)
        path = os.path.join(outdir, f"f_{f:05d}.png")
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True)
        files.append(path)
    return files


def encode_video(frames_dir: str, fps: float, out_path: str, audio: str | None = None,
                 start_number: int = 1, crf: int = 20, audio_offset: float = 0.0) -> str | None:
    exe = ffmpeg_exe()
    if exe is None:
        return None
    cmd = [exe, "-y", "-loglevel", "error", "-framerate", f"{fps}", "-start_number", str(start_number),
           "-i", os.path.join(frames_dir, "f_%05d.png")]
    if audio:
        if audio_offset > 0:
            cmd += ["-ss", f"{audio_offset:.4f}"]
        cmd += ["-i", audio, "-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out_path]
    subprocess.run(cmd, check=True)
    return out_path
