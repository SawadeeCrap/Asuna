"""Bake a performance onto a rigged character and render a preview video.

Usage (headless)::

    blender -b rigged.blend --python blender/scripts/render_performance.py -- \
        --performance out/perf.npz --audio out/music.wav --out out/preview.mp4 \
        [--frames 1-300] [--resolution 640x360] [--proxy-faces 60000] [--shadows]

Works the same with the ``bpy`` wheel:  python render_performance.py -- ...
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

from myrmex.performance.performance import Performance  # noqa: E402
from myrmex_blender import bake, preview, skin, ingest  # noqa: E402


def parse(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--blend", default=None, help="rigged .blend (defaults to the open file)")
    p.add_argument("--performance", required=True)
    p.add_argument("--audio", default=None)
    p.add_argument("--out", required=True, help="output .mp4 (frames go next to it)")
    p.add_argument("--rig", default=None, help="armature object name (auto)")
    p.add_argument("--frames", default=None, help="e.g. 1-300")
    p.add_argument("--resolution", default="640x360")
    p.add_argument("--proxy-faces", type=int, default=0)
    p.add_argument("--shadows", action="store_true")
    p.add_argument("--distance", type=float, default=4.2)
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


def main():
    a = parse(sys.argv)
    if a.blend:
        bpy.ops.wm.open_mainfile(filepath=a.blend)
    rig = bpy.data.objects.get(a.rig) if a.rig else next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.parent == rig]
    perf = Performance.load(a.performance)
    t = time.time()
    bake.bake(rig, perf, frame_start=1)
    print(f"baked {perf.frames} frames in {time.time() - t:.2f}s")
    for o in bpy.data.objects:
        if o.name.endswith("_proxy"):
            o.hide_render = True
            o.hide_viewport = True
    if a.proxy_faces and meshes:
        heavy = meshes[0]
        names = [vg.name for vg in heavy.vertex_groups if vg.name in rig.data.bones]
        for m in heavy.modifiers:
            m.show_viewport = m.show_render = False
        light = ingest.decimated_copy(heavy, a.proxy_faces, heavy.name + "_preview")
        W = vertex_weights(heavy, names)
        HV, _ = ingest.mesh_arrays(heavy, world=False)
        LV, _ = ingest.mesh_arrays(light, world=False)
        WL = skin.transfer_weights_kdtree(HV, W, LV, k=3)
        skin.assign_vertex_groups(light, names, WL)
        skin.setup_deformation(light, rig, WL)
        heavy.hide_render = heavy.hide_viewport = True
    sc = bpy.context.scene
    w, h = (int(x) for x in a.resolution.split("x"))
    preview.setup_workbench(sc, (w, h), shadows=a.shadows)
    preview.ensure_ground()
    pelvis = perf.bone_names[1] if perf.bone_names[0] == "root" else perf.bone_names[0]
    rest_head = np.array(rig.data.bones[pelvis].head_local)
    center = preview.bone_path(perf, rest_head, pelvis)
    preview.follow_camera(perf, center, distance=a.distance)
    audio = a.audio or perf.meta.get("audio_path")
    if audio and os.path.exists(audio):
        preview.add_audio(os.path.abspath(audio))
    frames = None
    if a.frames:
        f0, f1 = (int(x) for x in a.frames.split("-"))
        frames = range(f0, f1 + 1)
    outdir = os.path.splitext(a.out)[0] + "_frames"
    t = time.time()
    files = preview.render_frames(outdir, frames)
    print(f"rendered {len(files)} frames in {time.time() - t:.1f}s")
    start = frames.start if frames else 1
    audio_for_video = None
    if audio and os.path.exists(audio) and start == 1:
        audio_for_video = audio
    res = preview.encode_video(outdir, perf.fps, a.out, audio_for_video, start_number=start)
    print("video:", res)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.splitext(a.out)[0] + ".blend")


if __name__ == "__main__":
    main()
