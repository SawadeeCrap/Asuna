"""Hunyuan3D GLB -> live-ready .blend in one command.

    blender -b --python blender/scripts/prepare_character.py -- \
        --glb character.glb --out character_live.blend [--height 1.70] \
        [--material black_chrome|keep|chrome|liquid_metal|...] [--smooth 6] [--no-studio]

Steps: import + clean + ground/scale/face -Y, working mesh (300k faces) and
analysis proxy (40k), morphology analysis (limbs from the shape itself),
humanoid rig fit, geodesic skin weights transferred to the working mesh,
Armature + Corrective Smooth, optional surface smoothing and look-dev, studio.
The rig description is embedded in the armature and written next to the
.blend (``<out>.rig.json``) for ``myrmex live --rig``.
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

from myrmex.rig import fit_humanoid, morphology, skinning  # noqa: E402
from myrmex_blender import armature, cinema, ingest, skin  # noqa: E402


def parse(argv):
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--height", type=float, default=1.70, help="character height in metres")
    p.add_argument("--work-faces", type=int, default=300_000)
    p.add_argument("--proxy-faces", type=int, default=40_000)
    p.add_argument("--material", default="black_chrome", help="look-dev preset, or 'keep' for the GLB's textures")
    p.add_argument("--smooth", type=int, default=6, help="Laplacian smoothing iterations (0 = off)")
    p.add_argument("--no-studio", action="store_true")
    return p.parse_args(argv)


def log(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main():
    a = parse(sys.argv)
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    res = ingest.ingest(a.glb, target_height=a.height, work_faces=a.work_faces, proxy_faces=a.proxy_faces,
                        name="Character")
    mesh, proxy = res["mesh"], res["proxy"]
    log(f"ingested: {res['meta'].get('source_faces')} faces -> work {len(mesh.data.polygons)}, "
        f"proxy {len(proxy.data.polygons)}", t0)
    PV, PF = ingest.mesh_arrays(proxy)
    m = morphology.analyze(PV, PF)
    roles = sorted(l.role for l in m.limbs)
    log(f"morphology: {m.body_plan}, limbs {roles}", t0)
    if m.body_plan != "biped":
        print(f"!! body plan '{m.body_plan}' is not supported by the live rig yet (bipeds first); "
              f"the analysis is saved in the .blend for inspection.")
    rd = fit_humanoid.fit_humanoid(m)
    names, W = skinning.compute_weights(m, rd)
    rig = armature.build_armature(rd, "Rig")
    MV, _ = ingest.mesh_arrays(mesh)
    WT = skin.transfer_weights_kdtree(m.verts, W, MV)
    skin.assign_vertex_groups(mesh, names, WT)
    skin.setup_deformation(mesh, rig, WT)
    log(f"rigged: {len(rd.bones)} bones, weights on {len(MV)} vertices", t0)
    if a.smooth:
        cinema.smooth_surface(mesh, a.smooth)
    if a.material != "keep":
        cinema.apply_material(mesh, a.material)
    if not a.no_studio:
        cinema.setup_studio(np.zeros((1, 3)), np.array([[0.0, -1.0, 0.0]]), height=a.height)
        cinema.configure_render("eevee", (1920, 1080), 30.0, motion_blur=True)
    proxy.hide_viewport = proxy.hide_render = True
    base = os.path.splitext(os.path.abspath(a.out))[0]
    rd.save(base + ".rig.json")
    bpy.context.scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(a.out))
    log(f"saved {a.out} and {base}.rig.json", t0)


if __name__ == "__main__":
    main()
