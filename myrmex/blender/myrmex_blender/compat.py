"""Blender version compatibility helpers (4.2 LTS … 5.2 LTS)."""
from __future__ import annotations

import bpy

VERSION = bpy.app.version


def eevee_engine_id() -> str:
    return "BLENDER_EEVEE" if VERSION >= (5, 0, 0) else "BLENDER_EEVEE_NEXT"


def new_action_for(obj: bpy.types.ID, name: str) -> bpy.types.Action:
    """Create an action, assign it to ``obj`` and (4.4+) give it a slot for that datablock."""
    ad = obj.animation_data or obj.animation_data_create()
    act = bpy.data.actions.new(name)
    ad.action = act
    if hasattr(act, "slots") and hasattr(ad, "action_slot"):
        try:
            if ad.action_slot is None:
                id_type = "OBJECT" if isinstance(obj, bpy.types.Object) else obj.id_type
                slot = act.slots.new(id_type=id_type, name=obj.name)
                ad.action_slot = slot
        except Exception:  # pragma: no cover - older API variants
            pass
    return act


def ensure_fcurve(act: bpy.types.Action, owner: bpy.types.ID, data_path: str, index: int, group: str):
    if hasattr(act, "fcurve_ensure_for_datablock"):
        return act.fcurve_ensure_for_datablock(owner, data_path, index=index, group_name=group)
    fc = act.fcurves.find(data_path, index=index)
    if fc is None:
        fc = act.fcurves.new(data_path, index=index, action_group=group)
    return fc


def action_fcurves(act: bpy.types.Action, owner: bpy.types.ID):
    """All F-curves of an action for the owner's slot (legacy API fallback)."""
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    try:
        from bpy_extras import anim_utils
        ad = owner.animation_data
        cb = anim_utils.action_get_channelbag_for_slot(act, ad.action_slot)
        return list(cb.fcurves) if cb else []
    except Exception:
        return []


def material_node_tree(mat: bpy.types.Material):
    if VERSION < (5, 0, 0) and not mat.use_nodes:
        mat.use_nodes = True
    return mat.node_tree


def world_node_tree(world: bpy.types.World):
    if VERSION < (5, 0, 0) and not world.use_nodes:
        world.use_nodes = True
    return world.node_tree


def set_view_transform(scene: bpy.types.Scene, view: str = "AgX", look: str | None = None) -> None:
    vs = scene.view_settings
    try:
        vs.view_transform = view
    except TypeError:
        pass
    if look:
        for cand in (look, f"{view} - {look}", look.replace("AgX - ", "")):
            try:
                vs.look = cand
                break
            except TypeError:
                continue


def sequence_strips(scene: bpy.types.Scene):
    se = scene.sequence_editor or scene.sequence_editor_create()
    return se.strips if hasattr(se, "strips") else se.sequences
