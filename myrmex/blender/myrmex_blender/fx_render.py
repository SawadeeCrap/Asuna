"""Rendering a take with Myrmex FX.

Afterimages and ribbons are geometry, so any render has them.  The picture effects (trails, bloom,
aberration, impact frames, speed lines ...) are the same GPU passes as live (fx_post), run on the rendered
frames afterwards:

    1. the take renders to PNG frames (the picture parameters of every frame are kept on the way: where
       the organism is on screen, the hits, the knobs);
    2. every frame goes through the passes, in order (the light trails need the frames before);
    3. the frames and the song become the .mp4 (Blender's own encoder, the song lined up as in the take).

Step 2 needs the gpu module in background mode (Blender 5.0+: gpu.init()); an older Blender renders the
take straight to the .mp4 with the geometry effects only.
"""
from __future__ import annotations

import os
import shutil
import time

import bpy
import numpy as np

from . import compat, fx


def post_available() -> bool:
    if not bpy.app.background:
        return True
    import gpu
    return hasattr(gpu, "init")


def _gpu_ready() -> bool:
    import gpu
    if not bpy.app.background:
        return True
    try:
        gpu.init()
    except Exception as e:                                   # already initialised is fine
        if "already" not in str(e).lower():
            print("Myrmex FX: gpu.init failed:", e, flush=True)
    try:
        gpu.types.GPUOffScreen(8, 8).free()
        return True
    except Exception as e:
        print("Myrmex FX: no GPU in this Blender:", e, flush=True)
        return False


@bpy.app.handlers.persistent
def _keep_params(scene, depsgraph=None):
    """frame_change_post while rendering: this frame's picture parameters (evaluated camera)."""
    store = fx.S.get("post_by_frame")
    if store is None or scene.camera is None:
        return
    cam = scene.camera.evaluated_get(depsgraph) if depsgraph is not None else scene.camera
    w = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    h = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    try:
        store[scene.frame_current] = fx.post_params(scene, w, h, cam)
    except Exception as e:
        print("Myrmex FX: frame", scene.frame_current, e, flush=True)


def render(out: str, size=(1920, 1080), preset: str = "eevee", keep: bool = False, keep_frames: bool = False,
           progress=print) -> str:
    """Render the open take to ``out`` (.mp4) with Myrmex FX.  Returns the file written."""
    from .creature_take import configure_video_output
    sc = bpy.context.scene
    configure_video_output(out, size, preset, keep=keep)
    post = fx.active() and fx.picture_on() and post_available()
    if not post:
        if fx.active() and fx.picture_on():
            progress("Myrmex FX: picture effects in renders need Blender 5.0+ (afterimages and ribbons are in)")
        bpy.ops.render.render(animation=True)
        return out
    w = int(sc.render.resolution_x * sc.render.resolution_percentage / 100)
    h = int(sc.render.resolution_y * sc.render.resolution_percentage / 100)
    base = os.path.splitext(os.path.abspath(os.path.expanduser(out)))[0]
    folder = base + "_fxframes"
    os.makedirs(folder, exist_ok=True)
    r = sc.render
    saved = (r.filepath,)
    if hasattr(r.image_settings, "media_type"):
        try:
            r.image_settings.media_type = "IMAGE"
        except Exception:
            pass
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.color_depth = "8"
    r.image_settings.compression = 15
    r.filepath = os.path.join(folder, "raw_")
    fx.S["post_by_frame"] = {}
    fx.reset()
    if _keep_params not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_keep_params)
    t0 = time.time()
    try:
        bpy.ops.render.render(animation=True)
    finally:
        if _keep_params in bpy.app.handlers.frame_change_post:
            bpy.app.handlers.frame_change_post.remove(_keep_params)
    params = fx.S.get("post_by_frame") or {}
    fx.S["post_by_frame"] = None
    frames = list(range(sc.frame_start, sc.frame_end + 1))
    raw = [r.frame_path(frame=f) for f in frames]
    progress(f"Myrmex FX: rendered {len(frames)} frames in {time.time() - t0:.0f} s, now the picture effects")
    done = raw
    if _gpu_ready():
        done = apply_post(raw, params, frames, folder, w, h, progress)
    else:
        progress("Myrmex FX: no GPU for the picture effects: the video gets the plain frames")
    r.filepath = saved[0]
    encode(done, sc, out, (w, h))
    if not keep_frames:
        shutil.rmtree(folder, ignore_errors=True)
    return out


def apply_post(raw: list, params: dict, frames: list, folder: str, w: int, h: int, progress=print) -> list:
    """Every rendered frame through the GPU passes, in order.  Returns the finished frames."""
    import gpu

    from .fx_post import Pipeline
    pipe = Pipeline(w, h)
    img_out = bpy.data.images.new("MyrmexFXOut", w, h, alpha=False)
    out = []
    t0 = time.time()
    last = np.zeros(pipe.params.shape, np.float32)
    try:
        for k, (f, path) in enumerate(zip(frames, raw)):
            if not os.path.isfile(path):
                continue
            img = bpy.data.images.load(path, check_existing=False)
            try:
                iw, ih = img.size
                px = np.empty(iw * ih * 4, np.float32)
                img.pixels.foreach_get(px)
            finally:
                bpy.data.images.remove(img)
            tex = gpu.types.GPUTexture((iw, ih), format="RGBA16F",
                                       data=gpu.types.Buffer("FLOAT", iw * ih * 4, px))
            p = params.get(f, last)
            last = p
            pipe.run(p, src_tex=tex)
            res = pipe.read().astype(np.float32).ravel() / 255.0
            img_out.pixels.foreach_set(res)
            dst = os.path.join(folder, f"fx_{f:05d}.png")
            img_out.filepath_raw = dst
            img_out.file_format = "PNG"
            img_out.save()
            out.append(dst)
            if k % 25 == 0:
                el = time.time() - t0
                progress(f"Myrmex FX: frame {k + 1}/{len(frames)} ({el / (k + 1):.2f} s each)")
    finally:
        pipe.free()
        bpy.data.images.remove(img_out)
    return out


def encode(files: list, take_scene, out: str, size) -> str:
    """Frames + the take's song -> .mp4 (H.264 + AAC) through a sequencer scene."""
    if not files:
        raise RuntimeError("no frames to encode")
    enc = bpy.data.scenes.get("MyrmexFXEncode")
    if enc is not None:
        bpy.data.scenes.remove(enc)
    enc = bpy.data.scenes.new("MyrmexFXEncode")
    r = enc.render
    r.resolution_x, r.resolution_y = int(size[0]), int(size[1])
    r.resolution_percentage = 100
    r.fps, r.fps_base = take_scene.render.fps, take_scene.render.fps_base
    compat.set_view_transform(enc, "Standard")              # the frames are finished pictures already
    try:
        enc.view_settings.look = "None"
    except TypeError:
        pass
    enc.view_settings.exposure, enc.view_settings.gamma = 0.0, 1.0
    enc.sequence_editor_create()
    strips = compat.sequence_strips(enc)
    st = strips.new_image("MyrmexFX", files[0], channel=1, frame_start=1)
    for p in files[1:]:
        st.elements.append(os.path.basename(p))
    enc.frame_start, enc.frame_end = 1, len(files)
    se = take_scene.sequence_editor
    src = compat.sequence_strips(take_scene) if se is not None else []
    for s in src:
        if s.name == "MyrmexAudio" and getattr(s, "sound", None) is not None:
            snd = strips.new_sound("MyrmexAudio", bpy.path.abspath(s.sound.filepath), 2,
                                   int(s.frame_start - take_scene.frame_start + 1))
            snd.volume = getattr(s, "volume", 1.0)
    r.use_sequencer = True
    r.use_compositing = False
    ims = r.image_settings
    if hasattr(ims, "media_type"):
        try:
            ims.media_type = "VIDEO"
        except Exception:
            pass
    ims.file_format = "FFMPEG"
    ff = r.ffmpeg
    ff.format, ff.codec, ff.constant_rate_factor = "MPEG4", "H264", "HIGH"
    ff.audio_codec, ff.audio_bitrate = "AAC", 256
    r.filepath = os.path.abspath(os.path.expanduser(out))
    bpy.ops.render.render(animation=True, scene=enc.name)
    bpy.data.scenes.remove(enc)
    return out


__all__ = ["render", "apply_post", "encode", "post_available"]
