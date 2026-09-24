"""Bake a performance and render it with the cinematic camera, studio and look-dev.

    blender -b rigged.blend --python blender/scripts/render_cinematic.py -- \
        --performance out/perf.npz --audio out/music.wav --out out/film.mp4 \
        [--engine eevee|eevee_preview|cycles|cycles_preview|workbench] [--material black_chrome]
        [--resolution 1920x1080] [--frames 1-600] [--proxy-faces 0] [--shot front_dolly]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, ".."))

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from myrmex.camera import cinematographer as cine  # noqa: E402
from myrmex.performance.performance import Performance  # noqa: E402
from myrmex_blender import bake, cinema, ingest, preview, skin  # noqa: E402


def parse(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default=None)
    p.add_argument("--performance", required=True)
    p.add_argument("--audio", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--engine", default="eevee")
    p.add_argument("--material", default="black_chrome")
    p.add_argument("--resolution", default="1920x1080")
    p.add_argument("--frames", default=None)
    p.add_argument("--proxy-faces", type=int, default=0)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--shot", default=None, help="force a single shot type")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-blur", action="store_true")
    p.add_argument("--shadows", default="all", choices=["all", "key"])
    p.add_argument("--smooth", type=int, default=6, help="Laplacian smoothing iterations on the rest mesh")
    return p.parse_args(argv)


def vertex_weights(obj, names):
    idx = {obj.vertex_groups[n].index: j for j, n in enumerate(names) if n in obj.vertex_groups}
    W = np.zeros((len(obj.data.vertices), len(names)))
    for v in obj.data.vertices:
        for g in v.groups:
            j = idx.get(g.group)
            if j is not None:
                W[v.index, j] = g.weight
    return W


def world_point_path(perf: Performance, rig, bone: str, along: float = 0.0) -> np.ndarray:
    """World path of a point on a bone (``along`` = 0 head .. 1 tail) through the performance."""
    i = perf.bone_names.index(bone)
    b = rig.data.bones[bone]
    head = np.array(b.head_local) + along * (np.array(b.tail_local) - np.array(b.head_local))
    D = perf.deltas[:, i].astype(float)
    return np.einsum("tij,j->ti", D[:, :3, :3], head) + D[:, :3, 3]


def main():
    a = parse(sys.argv)
    if a.blend:
        bpy.ops.wm.open_mainfile(filepath=a.blend)
    rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.parent == rig]
    perf = Performance.load(a.performance)
    t0 = time.time()
    bake.bake(rig, perf, frame_start=1)
    print(f"baked {perf.frames} frames in {time.time() - t0:.2f}s")
    for o in bpy.data.objects:
        if o.name.endswith("_proxy") or o.name in ("MyrmexGround",):
            o.hide_render = o.hide_viewport = True
    body = meshes[0]
    if a.proxy_faces:
        names = [vg.name for vg in body.vertex_groups if vg.name in rig.data.bones]
        for m in body.modifiers:
            m.show_viewport = m.show_render = False
        light = ingest.decimated_copy(body, a.proxy_faces, body.name + "_preview")
        W = vertex_weights(body, names)
        HV, _ = ingest.mesh_arrays(body, world=False)
        LV, _ = ingest.mesh_arrays(light, world=False)
        WL = skin.transfer_weights_kdtree(HV, W, LV, k=3)
        skin.assign_vertex_groups(light, names, WL)
        skin.setup_deformation(light, rig, WL)
        body.hide_render = body.hide_viewport = True
        body = light
    t0 = time.time()
    cinema.smooth_surface(body, a.smooth)
    print(f"smoothed surface ({a.smooth} it) in {time.time() - t0:.1f}s")
    cinema.apply_material(body, a.material)
    # ---- camera
    pelvis = "pelvis" if "pelvis" in perf.bone_names else perf.bone_names[0]
    path = world_point_path(perf, rig, pelvis)
    head_bone = "head" if "head" in perf.bone_names else pelvis
    head = world_point_path(perf, rig, head_bone, along=0.45 if head_bone == "head" else 0.0)  # ~eye level
    feet_bones = [b for b in ("foot_l", "foot_r") if b in perf.bone_names]
    feet = np.mean([world_point_path(perf, rig, b) for b in feet_bones], axis=0) if feet_bones else path * [1, 1, 0]
    height = float(perf.meta.get("height", 1.7))
    beats = np.asarray(perf.meta.get("beat_times", []), dtype=float)
    track = cine.compose(path, head, feet, perf.fps, perf.meta.get("sections", []), beats,
                         float(perf.meta.get("heading0", -1.5708)), height, a.seed, a.shot)
    cinema.apply_camera_track(track)
    print("shots:", [(s.kind, s.start, s.end) for s in track.shots])
    S, F = cine.subject_frames(path, perf.fps, float(perf.meta.get("heading0", -1.5708)))
    cinema.setup_studio(S, F, height=height)
    w, h = (int(x) for x in a.resolution.split("x"))
    cinema.configure_render(a.engine, (w, h), perf.fps, motion_blur=not a.no_blur, samples=a.samples,
                            shadows=a.shadows)
    audio = a.audio or perf.meta.get("audio_path")
    if audio and os.path.exists(audio):
        preview.add_audio(os.path.abspath(audio))
    frames = None
    if a.frames:
        f0, f1 = (int(x) for x in a.frames.split("-"))
        frames = range(f0, f1 + 1)
    outdir = os.path.splitext(a.out)[0] + "_frames"
    t0 = time.time()
    files = preview.render_frames(outdir, frames)
    print(f"rendered {len(files)} frames in {time.time() - t0:.1f}s")
    start = frames.start if frames else 1
    res = preview.encode_video(outdir, perf.fps, a.out, audio if (audio and os.path.exists(audio)) else None,
                               start_number=start, audio_offset=(start - 1) / perf.fps)
    print("video:", res)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.splitext(a.out)[0] + ".blend")


if __name__ == "__main__":
    main()
