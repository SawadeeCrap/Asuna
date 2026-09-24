"""Import and normalise a generated asset (Hunyuan3D GLB/OBJ/FBX) for rigging.

Steps (all context-free, so they work headless and inside the UI):

1. import the file and collect its mesh objects,
2. bake object transforms into mesh data and join into one object,
3. put the lowest point on the ground (z = 0) and centre the ground contacts
   at the origin, optionally scale to a real-world height,
4. build a decimated *working* mesh (for animation) and a coarse *proxy*
   (for topology analysis), both via the Decimate modifier evaluated through
   the depsgraph.
"""
from __future__ import annotations

import os

import bpy
import numpy as np
from mathutils import Matrix


def import_asset(filepath: str) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=filepath)
    else:
        raise ValueError(f"Unsupported asset format: {ext}")
    return [o for o in bpy.data.objects if o not in before]


def mesh_arrays(obj: bpy.types.Object, world: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (V,3) and triangle indices (F,3) of an object's mesh (triangulated)."""
    me = obj.data
    me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    if world:
        M = np.array(obj.matrix_world)
        co = co @ M[:3, :3].T + M[:3, 3]
    return co, tris


def _bake_and_join(objs: list[bpy.types.Object], name: str) -> bpy.types.Object:
    meshes = [o for o in objs if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("The imported asset contains no mesh")
    verts_all, faces_all, offset = [], [], 0
    mats = []
    for o in meshes:
        v, f = mesh_arrays(o, world=True)
        verts_all.append(v)
        faces_all.append(f + offset)
        offset += v.shape[0]
        for slot in o.material_slots:
            if slot.material and slot.material not in mats:
                mats.append(slot.material)
    if len(meshes) == 1:
        # Keep UVs / materials of single-object assets by transforming in place.
        o = meshes[0]
        o.data.transform(o.matrix_world)
        o.parent = None
        o.matrix_world = Matrix.Identity(4)
        o.name = name
        for other in objs:
            if other is not o and other.name in bpy.data.objects:
                bpy.data.objects.remove(other, do_unlink=True)
        return o
    V = np.concatenate(verts_all)
    F = np.concatenate(faces_all)
    me = bpy.data.meshes.new(name)
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    for m in mats:
        me.materials.append(m)
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    for o in objs:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    return obj


def normalize(obj: bpy.types.Object, target_height: float | None = None,
              ground_fraction: float = 0.02) -> dict:
    """Ground the mesh, centre its ground contacts and scale it. Returns metadata."""
    co, _ = mesh_arrays(obj, world=True)
    zmin, zmax = float(co[:, 2].min()), float(co[:, 2].max())
    height = zmax - zmin
    scale = 1.0 if not target_height else float(target_height) / max(height, 1e-9)
    low = co[co[:, 2] <= zmin + ground_fraction * height]
    cx, cy = float(low[:, 0].mean()), float(low[:, 1].mean())
    T = Matrix.Translation((-cx, -cy, -zmin))
    S = Matrix.Scale(scale, 4)
    obj.data.transform(S @ T)
    obj.data.update()
    return {
        "source_height": height,
        "scale": scale,
        "height": height * scale,
        "offset": [-cx, -cy, -zmin],
        "forward": "-Y",
    }


def decimated_copy(obj: bpy.types.Object, target_faces: int, name: str,
                   link: bool = True) -> bpy.types.Object:
    """Evaluated copy of ``obj`` with a collapse-decimate to roughly ``target_faces``."""
    nf = len(obj.data.polygons)
    ratio = min(1.0, max(1e-4, target_faces / max(nf, 1)))
    mod = obj.modifiers.new("myrmex_decimate_tmp", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    obj.modifiers.remove(mod)
    me.name = name
    new = bpy.data.objects.new(name, me)
    if link:
        bpy.context.scene.collection.objects.link(new)
    return new


def ingest(filepath: str, target_height: float | None = None,
           work_faces: int | None = 400_000, proxy_faces: int = 40_000,
           name: str = "Creature") -> dict:
    """Full ingestion. Returns dict with 'mesh', 'proxy' objects and metadata."""
    objs = import_asset(filepath)
    obj = _bake_and_join(objs, name)
    meta = normalize(obj, target_height)
    meta["source_file"] = os.path.abspath(filepath)
    meta["source_faces"] = len(obj.data.polygons)
    if work_faces and len(obj.data.polygons) > work_faces * 1.2:
        work = decimated_copy(obj, work_faces, name)
        old = obj
        # Preserve materials on the working copy.
        for m in old.data.materials:
            if m.name not in [x.name for x in work.data.materials if x]:
                work.data.materials.append(m)
        bpy.data.objects.remove(old, do_unlink=True)
        obj = work
        obj.name = name
    proxy = decimated_copy(obj, proxy_faces, name + "_proxy")
    proxy.hide_render = True
    proxy.hide_set(True)
    meta["work_faces"] = len(obj.data.polygons)
    meta["proxy_faces"] = len(proxy.data.polygons)
    obj["myrmex_meta"] = str(meta)
    return {"mesh": obj, "proxy": proxy, "meta": meta}
