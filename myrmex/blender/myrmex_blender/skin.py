"""Apply skin weights to (dense) meshes and set up high-quality deformation."""
from __future__ import annotations

import bpy
import numpy as np
from mathutils import kdtree

from myrmex.rig.skinning import joint_mask, prune_weights


def transfer_weights_kdtree(src_verts: np.ndarray, W: np.ndarray, dst_verts: np.ndarray,
                            k: int = 4) -> np.ndarray:
    tree = kdtree.KDTree(len(src_verts))
    for i, p in enumerate(src_verts):
        tree.insert(p, i)
    tree.balance()
    out = np.zeros((len(dst_verts), W.shape[1]))
    for i, p in enumerate(dst_verts):
        hits = tree.find_n(p, k)
        idx = np.fromiter((h[1] for h in hits), dtype=np.int64, count=len(hits))
        dd = np.fromiter((h[2] for h in hits), dtype=float, count=len(hits))
        w = 1.0 / (dd + 1e-5) ** 2
        out[i] = (W[idx] * w[:, None]).sum(axis=0) / w.sum()
    return prune_weights(out)


def assign_vertex_groups(obj: bpy.types.Object, names: list[str], W: np.ndarray,
                         levels: int = 1024) -> None:
    """Write weights (V, B) into vertex groups using quantised batched adds (fast)."""
    for vg in list(obj.vertex_groups):
        if vg.name in names:
            obj.vertex_groups.remove(vg)
    q = np.rint(W * levels).astype(np.int64)
    for j, name in enumerate(names):
        col = q[:, j]
        nz = np.nonzero(col)[0]
        if nz.size == 0:
            continue
        vg = obj.vertex_groups.new(name=name)
        vals = col[nz]
        order = np.argsort(vals, kind="stable")
        nz, vals = nz[order], vals[order]
        cuts = np.nonzero(np.diff(vals))[0] + 1
        for grp_idx, grp_val in zip(np.split(nz, cuts), np.split(vals, cuts)):
            vg.add(grp_idx.tolist(), float(grp_val[0]) / levels, "REPLACE")


def setup_deformation(mesh_obj: bpy.types.Object, arm_obj: bpy.types.Object, W: np.ndarray,
                      preserve_volume: bool = True, corrective_smooth: bool = True,
                      smooth_radius: float = 0.02) -> None:
    for m in list(mesh_obj.modifiers):
        if m.type in ("ARMATURE", "CORRECTIVE_SMOOTH"):
            mesh_obj.modifiers.remove(m)
    mod = mesh_obj.modifiers.new("MyrmexArmature", "ARMATURE")
    mod.object = arm_obj
    mod.use_deform_preserve_volume = preserve_volume
    mod.use_vertex_groups = True
    # Move the armature modifier to the top of the stack.
    while mesh_obj.modifiers.find(mod.name) > 0:
        with bpy.context.temp_override(object=mesh_obj, active_object=mesh_obj):
            bpy.ops.object.modifier_move_up(modifier=mod.name)
    mesh_obj.parent = arm_obj
    mesh_obj.matrix_parent_inverse = arm_obj.matrix_world.inverted()
    if corrective_smooth:
        mask = joint_mask(W)
        vg = mesh_obj.vertex_groups.get("myrmex_joint_smooth") or mesh_obj.vertex_groups.new(name="myrmex_joint_smooth")
        q = np.rint(mask * 64).astype(np.int64)
        nz = np.nonzero(q)[0]
        vals = q[nz]
        for v in np.unique(vals):
            vg.add(nz[vals == v].tolist(), float(v) / 64.0, "REPLACE")
        me = mesh_obj.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        ed = np.empty(len(me.edges) * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", ed)
        ed = ed.reshape(-1, 2)
        mean_edge = float(np.linalg.norm(co[ed[:, 0]] - co[ed[:, 1]], axis=1).mean())
        cs = mesh_obj.modifiers.new("MyrmexCorrectiveSmooth", "CORRECTIVE_SMOOTH")
        cs.smooth_type = "LENGTH_WEIGHTED"
        cs.factor = 0.5
        cs.iterations = int(min(120, max(5, (smooth_radius / max(mean_edge, 1e-6)) ** 2 * 0.5)))
        cs.rest_source = "ORCO"
        cs.vertex_group = vg.name
        cs.use_pin_boundary = True
