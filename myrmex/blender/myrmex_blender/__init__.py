"""Myrmex Blender integration (add-on package).

Install: symlink (or copy) this folder into Blender's add-ons directory, e.g. on macOS

    ln -s /path/to/myrmex/blender/myrmex_blender \
        "$HOME/Library/Application Support/Blender/5.2/scripts/addons/myrmex_blender"

and enable "Myrmex" in Preferences > Add-ons.  The ``myrmex`` engine package is
found next to the add-on (``../../src`` of the repository, symlinks resolved),
through the ``MYRMEX_SRC`` environment variable, or in Blender's own Python.

Submodules are imported lazily so headless scripts can use e.g.
``myrmex_blender.ingest`` without registering any UI.
"""
import os
import sys

bl_info = {
    "name": "Myrmex – music-driven procedural creatures",
    "author": "Myrmex contributors",
    "version": (0, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Myrmex",
    "description": "Auto-rig generated creatures and animate them live from Ableton / VCV Rack",
    "category": "Animation",
}


def _ensure_engine_path() -> None:
    try:
        import myrmex  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.realpath(__file__))
    for cand in (os.environ.get("MYRMEX_SRC"), os.path.join(here, "..", "..", "src"), os.path.join(here, "src")):
        if cand and os.path.isdir(os.path.join(cand, "myrmex")):
            sys.path.insert(0, os.path.abspath(cand))
            return


_ensure_engine_path()


def register():
    from . import take_player, ui
    ui.register()
    take_player.register()


def unregister():
    from . import take_player, ui
    take_player.unregister()
    ui.unregister()
