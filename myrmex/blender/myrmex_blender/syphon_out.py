"""Blender -> TouchDesigner (or Resolume, VDMX, MadMapper, OBS ...): the live camera's picture over Syphon.

The scene camera is rendered into an off-screen buffer at its own size (EEVEE, exactly as the viewport
shades it, colour managed; optionally with a transparent background), read back and published as a
Syphon server named "Myrmex" (TouchDesigner: Syphon Spout In TOP).  The library (syphon-python +
pyobjc) ships inside Myrmex.app (blender/vendor/cp311 for Blender 4.2-5.0, cp313 for 5.1+), so there
is nothing to install; a Blender with its own syphon-python (e.g. from the TextureSharing add-on)
works too.  Nothing here runs in background (render) mode.

    configure({"on": True, "name": "Myrmex", "width": 1280, "height": 720, "fps": 60, "alpha": False})
"""
from __future__ import annotations

import os
import sys
import time

import bpy

_S: dict = {"out": None, "error": ""}


def _vendor() -> None:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    for p in (os.path.join(base, "common"), os.path.join(base, tag)):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def available() -> tuple[bool, str]:
    if sys.platform != "darwin":
        return False, "Syphon is macOS only"
    if bpy.app.background:
        return False, "no picture to share in background mode"
    _vendor()
    try:
        import syphon  # noqa: F401
        return True, ""
    except Exception as e:                                   # a Python Myrmex.app does not carry
        return False, (f"Syphon library not available for Python {sys.version_info.major}."
                       f"{sys.version_info.minor} ({type(e).__name__}: {e})")


def _tag_redraw() -> None:
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class SyphonOut:
    def __init__(self, name: str = "Myrmex", width: int = 1280, height: int = 720, fps: float = 60.0,
                 alpha: bool = False, flip: bool = False):
        self.name, self.w, self.h = str(name or "Myrmex"), int(width), int(height)
        self.fps, self.alpha, self.flip = max(1.0, float(fps)), bool(alpha), bool(flip)
        self.srv = self.tex = self.off = self.handle = None
        self.metal = False
        self.last = 0.0
        self.frames = 0
        self.error = ""
        self.hidden: list[str] = []

    def start(self) -> None:
        ok, msg = available()
        if not ok:
            raise RuntimeError(msg)
        import gpu
        import syphon
        if gpu.platform.backend_type_get() == "METAL":
            from syphon.utils.raw import create_mtl_texture
            self.srv = syphon.SyphonMetalServer(self.name)
            self.tex = create_mtl_texture(self.srv.device, self.w, self.h)
            self.metal = True
        else:
            self.srv = syphon.SyphonOpenGLServer(self.name)
        self.off = gpu.types.GPUOffScreen(self.w, self.h)
        self.handle = bpy.types.SpaceView3D.draw_handler_add(self._draw, (), "WINDOW", "POST_PIXEL")
        if not bpy.app.timers.is_registered(self._pump):
            bpy.app.timers.register(self._pump, first_interval=0.05, persistent=True)
        if self.alpha:                                      # the organism alone: TD draws what is behind it
            for n in ("MyrmexFloor", "CrawlerTerrain"):
                ob = bpy.data.objects.get(n)
                if ob is not None and not ob.hide_viewport:
                    ob.hide_viewport = True
                    self.hidden.append(n)

    def _pump(self):
        if _S.get("out") is not self:
            return None
        _tag_redraw()                                       # frames keep coming when nothing else redraws
        return 1.0 / self.fps

    def _draw(self) -> None:
        now = time.perf_counter()
        if now - self.last < 0.9 / self.fps:               # one publish per frame (several 3D views)
            return
        ctx = bpy.context
        scene, space, region = ctx.scene, ctx.space_data, ctx.region
        cam = scene.camera if scene is not None else None
        if cam is None or space is None or space.type != "VIEW_3D" or region is None:
            return
        self.last = now
        try:
            view = cam.matrix_world.inverted()
            proj = cam.calc_matrix_camera(ctx.evaluated_depsgraph_get(), x=self.w, y=self.h)
            self.off.draw_view3d(scene, ctx.view_layer, space, region, view, proj, do_color_management=True,
                                 draw_background=not self.alpha)
            if self.metal:
                from syphon.utils.raw import copy_bytes_to_mtl_texture
                copy_bytes_to_mtl_texture(self.off.texture_color.read(), self.tex)
                self.srv.publish_frame_texture(self.tex, size=(self.w, self.h), is_flipped=self.flip)
            else:
                self.srv.publish_frame_texture(self.off.color_texture, size=(self.w, self.h),
                                               is_flipped=self.flip)
            self.frames += 1
            self.error = ""
        except Exception as e:                              # never break the viewport
            self.error = f"{type(e).__name__}: {e}"

    def stop(self) -> None:
        if self.handle is not None:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(self.handle, "WINDOW")
            except ValueError:
                pass
            self.handle = None
        if bpy.app.timers.is_registered(self._pump):
            bpy.app.timers.unregister(self._pump)
        for n in self.hidden:
            ob = bpy.data.objects.get(n)
            if ob is not None:
                ob.hide_viewport = False
        self.hidden = []
        try:
            if self.srv is not None:
                self.srv.stop()
        except Exception:
            pass
        if self.off is not None:
            try:
                self.off.free()
            except Exception:
                pass
        self.srv = self.tex = self.off = None


def status() -> dict:
    out = _S.get("out")
    if out is None:
        return {"on": False, "error": _S.get("error", "")}
    return {"on": True, "name": out.name, "size": [out.w, out.h], "fps": out.fps, "alpha": out.alpha,
            "frames": out.frames, "error": out.error or _S.get("error", "")}


def configure(d: dict) -> dict:
    """Start / change / stop the Syphon picture.  Returns the status."""
    old = _S.get("out")
    if old is not None:
        old.stop()
        _S["out"] = None
    _S["error"] = ""
    if not d.get("on", True):
        return status()
    size = d.get("size")
    w, h = (int(size[0]), int(size[1])) if size else (int(d.get("width", 1280)), int(d.get("height", 720)))
    out = SyphonOut(d.get("name", "Myrmex"), w, h, float(d.get("fps", 60)), bool(d.get("alpha", False)),
                    bool(d.get("flip", False)))
    try:
        out.start()
    except Exception as e:
        out.stop()
        _S["error"] = f"{type(e).__name__}: {e}" if not isinstance(e, RuntimeError) else str(e)
        return status()
    _S["out"] = out
    return status()


def from_env() -> dict | None:
    """MYRMEX_SYPHON='{"on": true, ...}' (set by the app when it opens Blender)."""
    raw = os.environ.get("MYRMEX_SYPHON")
    if not raw:
        return None
    import json
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    return configure(d) if isinstance(d, dict) and d.get("on") else None


__all__ = ["configure", "status", "available", "from_env", "SyphonOut"]
