"""Saved looks: your materials, lights, world, colour management and render settings, kept between sessions.

* Creatures (Black Nanomaterial / Mimetic Polyalloy / Polyalloy Colony): the scene is saved as
  ``~/Myrmex/looks/<variant>.blend``; the Myrmex app opens it for live work and for take renders
  instead of building the default studio.
* Humanoid: the look lives in the character's own .blend (saved in place, next to its rig.json).
"""
from __future__ import annotations

import os

import bpy

CREATURES = ("nanomaterial", "polyalloy", "colony")


def looks_dir() -> str:
    return os.path.expanduser(os.environ.get("MYRMEX_LOOKS", "~/Myrmex/looks"))


def look_kind(scene: bpy.types.Scene | None = None) -> str:
    scene = scene or bpy.context.scene
    v = scene.get("myrmex_variant")
    if v in CREATURES:
        return v
    if bpy.data.objects.get("CreatureBody") is not None:
        if bpy.data.objects.get("ColonyPlates") is not None:
            return "colony"
        return "polyalloy" if bpy.data.objects.get("PolyLattice") is not None else "nanomaterial"
    return "humanoid"


def look_path(kind: str) -> str:
    return os.path.join(looks_dir(), f"{kind}.blend")


def save_look(context) -> str:
    """Save the current scene as the look for its character type.  Returns the file written."""
    from . import ui
    kind = look_kind(context.scene)
    link = ui._LINK.get("link")
    hidden = []
    if link is not None:                    # live-only speed-ups must not end up in the saved file
        for ob_name, m_name in link._restore:
            ob = bpy.data.objects.get(ob_name)
            if ob is not None and m_name in ob.modifiers:
                ob.modifiers[m_name].show_viewport = True
                hidden.append(ob.modifiers[m_name])
    try:
        if kind == "humanoid":
            if not bpy.data.filepath:
                raise ValueError("open the character's .blend first (the humanoid look is saved in it)")
            bpy.ops.wm.save_mainfile()
            path = bpy.data.filepath
        else:
            path = look_path(kind)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    finally:
        for m in hidden:
            m.show_viewport = False
    return path


def forget_look(kind: str) -> bool:
    path = look_path(kind)
    if kind in CREATURES and os.path.exists(path):
        os.remove(path)
        return True
    return False


__all__ = ["save_look", "forget_look", "look_kind", "look_path", "looks_dir"]
