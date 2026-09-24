"""Open a character and go live at once (used by the Myrmex launcher).

    Blender character_live.blend --python blender/scripts/live_autostart.py

Registers the Myrmex add-on if needed, starts the live link (the engine runs in the
launcher's terminal), looks through the live camera in Rendered mode with light
EEVEE settings for a smooth viewport.
"""
import os
import sys

import bpy

HERE = os.path.dirname(os.path.realpath(__file__))
for p in (os.path.join(HERE, ".."), os.path.join(HERE, "..", "..", "src")):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def _register():
    if not hasattr(bpy.types.Scene, "myrmex_live"):
        import myrmex_blender
        myrmex_blender.register()


def _viewport_settings(scene):
    ee = scene.eevee
    for attr, val in (("use_raytracing", False), ("use_fast_gi", False), ("taa_samples", 8),
                      ("use_shadows", True), ("shadow_resolution_scale", 0.5)):
        if hasattr(ee, attr):
            try:
                setattr(ee, attr, val)
            except Exception:
                pass
    studio = bpy.data.collections.get("MyrmexStudio")
    if studio is not None:
        for ob in studio.objects:
            if ob.type == "LIGHT":
                ob.data.use_shadow = ob.name == "Key"


def _look_through_camera():
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "RENDERED"
                    space.region_3d.view_perspective = "CAMERA"
                    space.overlay.show_overlays = False
            area.tag_redraw()


def go_live():
    _register()
    scene = bpy.context.scene
    s = scene.myrmex_live
    if os.environ.get("MYRMEX_MODE") == "creature":
        from myrmex_blender import creature, live, ui
        creature.setup_creature_scene(scene)
        s.port = int(os.environ.get("MYRMEX_POSE_PORT", s.port))
        _viewport_settings(scene)
        link = live.LiveLink(None, port=s.port)
        link.start()
        ui._LINK["link"] = link
        if not bpy.app.background:
            bpy.app.timers.register(lambda: (_look_through_camera(), None)[1], first_interval=1.5)
        print("Myrmex: creature live on port", s.port)
        return None
    if s.armature is None:
        s.armature = next((o for o in bpy.data.objects if o.type == "ARMATURE" and o.get("myrmex_rig")), None) or \
            next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    s.engine_mode = os.environ.get("MYRMEX_ENGINE_MODE", "EXTERNAL")
    s.port = int(os.environ.get("MYRMEX_POSE_PORT", s.port))
    _viewport_settings(scene)
    from myrmex_blender import ui
    wm = bpy.context.window_manager
    win = wm.windows[0] if wm is not None and len(wm.windows) else None
    if win is not None:
        with bpy.context.temp_override(window=win):
            bpy.ops.myrmex.live_start()
    else:
        bpy.ops.myrmex.live_start()
    link = ui._LINK.get("link")
    # The live camera appears with the first pose packet; look through it a moment later.
    if win is not None:
        bpy.app.timers.register(lambda: (_look_through_camera(), None)[1], first_interval=1.5)
    print("Myrmex: live", "ON" if link is not None else "FAILED", "- waiting for poses on port", s.port)
    return None


if bpy.app.background:
    go_live()
else:
    bpy.app.timers.register(go_live, first_interval=0.5)
