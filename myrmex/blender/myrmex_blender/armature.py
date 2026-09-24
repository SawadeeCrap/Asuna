"""Build a Blender armature from a :class:`myrmex.rig.rigdesc.RigDescription`."""
from __future__ import annotations

import json

import bpy
from mathutils import Vector

from myrmex.rig.rigdesc import RigDescription


def _set_mode(obj: bpy.types.Object, mode: str) -> None:
    vl = bpy.context.view_layer
    if not obj.users_collection:
        bpy.context.scene.collection.objects.link(obj)
    vl.update()
    for o in vl.objects:
        if o.select_get():
            o.select_set(False)
    vl.objects.active = obj
    obj.select_set(True)
    if obj.mode != mode:
        bpy.ops.object.mode_set(mode=mode)


def build_armature(rd: RigDescription, name: str = "MyrmexRig",
                   collection: bpy.types.Collection | None = None) -> bpy.types.Object:
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    (collection or bpy.context.scene.collection).objects.link(obj)
    arm.display_type = "OCTAHEDRAL"
    obj.show_in_front = True
    _set_mode(obj, "EDIT")
    ebs = arm.edit_bones
    for b in rd.bones:
        eb = ebs.new(b.name)
        eb.head = Vector(b.head)
        eb.tail = Vector(b.tail)
        eb.align_roll(Vector(b.roll_ref))
        eb.use_deform = bool(b.deform)
    for b in rd.bones:
        if b.parent:
            eb = ebs[b.name]
            eb.parent = ebs[b.parent]
            eb.use_connect = bool(b.connect) and (eb.head - eb.parent.tail).length < 1e-4
    _set_mode(obj, "OBJECT")
    for pb in obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    obj["myrmex_rig"] = True
    obj["myrmex_body_plan"] = rd.body_plan
    # The .blend carries its own rig description: the live engine can start from the armature alone.
    obj["myrmex_rig_desc"] = json.dumps(rd.to_dict())
    return obj
