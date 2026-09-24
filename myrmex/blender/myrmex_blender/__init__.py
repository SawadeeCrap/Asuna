"""Myrmex Blender integration (add-on package).

Submodules are imported lazily so headless scripts can use e.g.
``myrmex_blender.ingest`` without registering any UI.
"""
bl_info = {
    "name": "Myrmex – music-driven procedural creatures",
    "author": "Myrmex contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Myrmex",
    "description": "Auto-rig generated creatures and animate them procedurally from Ableton / VCV Rack",
    "category": "Animation",
}


def register():
    from . import ui
    ui.register()


def unregister():
    from . import ui
    ui.unregister()
