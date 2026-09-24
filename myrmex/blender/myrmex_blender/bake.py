"""Bake a :class:`myrmex.performance.performance.Performance` onto an armature.

For every bone the engine provides a rest-world delta ``D``; the posed
armature-space matrix is ``P = Mobj^-1 · D · Mobj · Rest`` and Blender's
``matrix_basis`` follows from the parent chain:

    basis = Rest^-1 · Rest_parent · P_parent^-1 · P      (child bones)
    basis = Rest^-1 · P                                   (root bones)

All frames are converted in one vectorised pass and written with
``keyframe_points.foreach_set`` – a 60 s, 30-bone performance bakes in about
a second.
"""
from __future__ import annotations

import bpy
import numpy as np

from myrmex.performance.performance import Performance
from myrmex.util.mathutil import quat_continuity, quats_from_matrices

from . import compat


def _batched_inv(M: np.ndarray) -> np.ndarray:
    return np.linalg.inv(M)


def compute_basis(arm_obj: bpy.types.Object, perf: Performance) -> tuple[list[str], np.ndarray]:
    bones = arm_obj.data.bones
    names = [n for n in perf.bone_names if n in bones]
    if not names:
        raise ValueError("no bone of the performance exists in the armature")
    pidx = {n: perf.bone_names.index(n) for n in names}
    Mobj = np.array(arm_obj.matrix_world)
    Mobj_inv = np.linalg.inv(Mobj)
    rest = {b.name: np.array(b.matrix_local) for b in bones}
    T = perf.frames
    posed: dict[str, np.ndarray] = {}
    for n in names:
        D = perf.deltas[:, pidx[n]].astype(float)
        posed[n] = np.einsum("ij,tjk,kl,lm->tim", Mobj_inv, D, Mobj, rest[n])
    basis = np.zeros((T, len(names), 4, 4))
    for j, n in enumerate(names):
        b = bones[n]
        R = rest[n]
        Rinv = np.linalg.inv(R)
        if b.parent is None:
            basis[:, j] = np.einsum("ij,tjk->tik", Rinv, posed[n])
            continue
        pn = b.parent.name
        Rp = rest[pn]
        if pn in posed:
            Pp = posed[pn]
        else:
            # Undriven parent: evaluate its own parents recursively at rest (pose = rest).
            Pp = np.broadcast_to(Rp, (T, 4, 4))
        Pp_inv = _batched_inv(Pp)
        basis[:, j] = np.einsum("ij,jk,tkl,tlm->tim", Rinv, Rp, Pp_inv, posed[n])
    return names, basis


def bake(arm_obj: bpy.types.Object, perf: Performance, frame_start: int = 1,
         action_name: str | None = None, set_scene_range: bool = True) -> bpy.types.Action:
    names, basis = compute_basis(arm_obj, perf)
    T = basis.shape[0]
    act = compat.new_action_for(arm_obj, action_name or f"myrmex_{arm_obj.name}")
    frames = np.arange(frame_start, frame_start + T, dtype=float)
    for j, n in enumerate(names):
        pb = arm_obj.pose.bones[n]
        pb.rotation_mode = "QUATERNION"
        loc = basis[:, j, :3, 3]
        q = quat_continuity(quats_from_matrices(basis[:, j, :3, :3]))
        channels = [(f'pose.bones["{n}"].location', i, loc[:, i]) for i in range(3)]
        channels += [(f'pose.bones["{n}"].rotation_quaternion', i, q[:, i]) for i in range(4)]
        for path, idx, vals in channels:
            fc = compat.ensure_fcurve(act, arm_obj, path, idx, n)
            fc.keyframe_points.clear()
            fc.keyframe_points.add(T)
            co = np.empty(2 * T)
            co[0::2] = frames
            co[1::2] = vals
            fc.keyframe_points.foreach_set("co", co)
            fc.keyframe_points.foreach_set("interpolation", [1] * T)   # LINEAR
            fc.update()
    if set_scene_range:
        sc = bpy.context.scene
        sc.frame_start = frame_start
        sc.frame_end = frame_start + T - 1
        sc.render.fps = int(round(perf.fps))
        sc.render.fps_base = sc.render.fps / perf.fps
    arm_obj["myrmex_performance_frames"] = T
    return act


def apply_frame(arm_obj: bpy.types.Object, perf: Performance, k: int) -> None:
    """Pose a single frame directly (live preview, no keyframes)."""
    names, basis = compute_basis_frame(arm_obj, perf, k)
    for j, n in enumerate(names):
        pb = arm_obj.pose.bones[n]
        pb.rotation_mode = "QUATERNION"
        from mathutils import Matrix
        pb.matrix_basis = Matrix(basis[j].tolist())


def compute_basis_frame(arm_obj, perf, k):
    sub = Performance(perf.fps, perf.bone_names, perf.deltas[k:k + 1])
    names, basis = compute_basis(arm_obj, sub)
    return names, basis[0]
