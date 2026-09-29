"""Blender's live graphics: the presets and knobs (realtime/gfx.py), the body's detail following the shot, the app's
settings and FX page - and, when Blender's ``bpy`` module is importable, the add-on side: the fast position
writes, the metaball elements as arrays, the body's resolution, the afterimage limits, the command and the panel."""
import json
import math
import os

import numpy as np
import pytest

from myrmex.realtime import gfx as G


# ---------------------------------------------------------------------- the settings
def test_presets_and_normalize():
    d = G.normalize(None)
    assert d["preset"] == "balanced" and d == {**G.DEFAULTS}
    assert list(G.LEVELS) == ["quality", "balanced", "performance", "max_fps"]
    q = G.normalize({"preset": "quality"})
    assert q["body_auto"] is False and q["body_res"] == 0.03 and q["ghost_parts"] == "all" and q["picture"] == 1.0
    fast = G.normalize({"preset": "max_fps"})
    assert fast["body_res"] > q["body_res"] and fast["ghost_max"] < q["ghost_max"] and fast["ghost_parts"] == "body"
    for a, b in zip(G.LEVELS, G.LEVELS[1:]):                            # each step is lighter
        pa, pb = G.PRESETS[a], G.PRESETS[b]
        assert pb["body_res"] >= pa["body_res"] and pb["ghost_max"] <= pa["ghost_max"] and pb["picture"] <= pa["picture"]
    c = G.normalize({"ghost_max": 3}, q)                                # a knob moved: custom
    assert c["preset"] == "custom" and c["ghost_max"] == 3 and c["body_res"] == 0.03
    assert G.normalize({"preset": "performance"}, c)["preset"] == "performance"
    x = G.normalize({"body_res": 9, "ghost_max": -4, "picture": "x", "ghost_parts": "?", "target_fps": 500})
    assert x["body_res"] == 0.12 and x["ghost_max"] == 1 and x["picture"] == G.DEFAULTS["picture"]
    assert x["ghost_parts"] == "all" and x["target_fps"] == 120.0
    assert G.normalize({"auto": True})["preset"] == "balanced"          # auto / fps are not a look: no "custom"
    assert G.from_env() is None
    os.environ["MYRMEX_GFX"] = G.to_env({"preset": "performance", "auto": True})["MYRMEX_GFX"]
    try:
        e = G.from_env()
        assert e["preset"] == "performance" and e["auto"] is True
    finally:
        del os.environ["MYRMEX_GFX"]


def test_body_detail_follows_the_shot():
    cfg = G.normalize({"preset": "balanced"})
    fov = math.radians(39.6)
    near = G.body_resolution(cfg, 1.5, fov)
    far = G.body_resolution(cfg, 30.0, fov)
    assert near == pytest.approx(0.03)                                  # close-ups: as the look has it
    assert far == pytest.approx(0.03 * cfg["body_max"], rel=0.06)       # far away: at most body_max coarser
    mid = G.body_resolution(cfg, 12.0, fov)
    cell = mid / G.world_per_pixel(12.0, fov)
    assert 0.03 < mid < far and cell == pytest.approx(cfg["body_px"], rel=0.06)   # the cell ~body_px wide
    assert G.body_resolution(cfg, 12.5, fov, current=mid) == mid        # a small move: no change (dead band)
    assert G.body_resolution(cfg, None, fov) == 0.03 and G.body_resolution(cfg, float("nan"), fov) == 0.03
    q = G.normalize({"preset": "quality"})
    assert G.body_resolution(q, 30.0, fov) == 0.03                      # quality: fixed, as before
    steps = {round(G.body_resolution(cfg, d, fov) / 0.03, 6) for d in np.linspace(1, 40, 200)}
    assert all(abs(math.log(s) / math.log(1.05) - round(math.log(s) / math.log(1.05))) < 1e-4 for s in steps)
    assert G.step("balanced", True) == "performance" and G.step("max_fps", True) == "max_fps"
    assert G.step("balanced", False) == "quality" and G.step("quality", False) == "quality"


# ---------------------------------------------------------------------- the app
def test_app_gfx_settings():
    from myrmex.app import controllers as C
    from myrmex.app.settings import AppSettings
    s = AppSettings()
    d = C.gfx_settings(s)
    assert d["preset"] == "balanced" and d["eevee"] is False              # "keep my Blender settings" is on
    s.keep_blender_settings = False
    assert C.gfx_settings(s)["eevee"] is True
    s.gfx = {"preset": "performance", "auto": True}
    d = C.gfx_settings(s)
    assert d["preset"] == "performance" and d["ghost_parts"] == "body" and d["auto"] is True
    s.gfx = {"preset": "custom", "ghost_max": 3, "body_res": 0.05}
    d = C.gfx_settings(s)
    assert d["preset"] == "custom" and d["ghost_max"] == 3 and d["body_res"] == 0.05
    env = C.gfx_env(s)
    assert json.loads(env["MYRMEX_GFX"])["ghost_max"] == 3


def test_app_fx_page_graphics_group(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from myrmex.app import fx_tab
    from myrmex.app import window as W
    from myrmex.app.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    s = AppSettings()
    s.start_engine_on_launch = s.open_blender_on_start = False
    w = W.MainWindow(s)
    w.timer.stop()
    assert w.cmb_gfx.currentData() == "balanced"
    w.cmb_gfx.setCurrentIndex(w.cmb_gfx.findData("max_fps"))
    assert s.gfx["preset"] == "max_fps" and w.gfx_knobs["ghost_max"].value() == 4
    assert w.cmb_gfx_parts.currentData() == "body"
    w.gfx_knobs["ghost_max"].setValue(7)                                # fine-tuning: custom
    assert s.gfx["preset"] == "custom" and s.gfx["ghost_max"] == 7 and w.cmb_gfx.currentData() == "custom"
    w.chk_gfx_auto.setChecked(True)
    assert s.gfx["auto"] is True and s.gfx["preset"] == "custom"
    fx_tab.on_blender_gfx_stats(w, {"fps": 47.6, "poll_ms": 1.8, "eval_ms": 9.94, "res": 0.041, "level":
                                    "performance", "auto": True})
    txt = w.lbl_gfx.text()
    assert "48 fps" in txt and "9.9 ms" in txt and "0.041" in txt and "auto: performance" in txt
    app.processEvents()
    w.close()


# ---------------------------------------------------------------------- inside Blender (bpy module)
@pytest.fixture(scope="module")
def blender():
    bpy = pytest.importorskip("bpy")
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "blender"))
    import myrmex_blender
    if not hasattr(bpy.types.Scene, "myrmex_gfx"):
        myrmex_blender.register()
    yield bpy
    myrmex_blender.unregister()
    for lst in (bpy.app.handlers.load_post, bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post):
        lst.clear()


def test_blender_fast_positions(blender):
    bpy = blender
    from myrmex_blender import compat
    me = bpy.data.meshes.new("gfx_t")
    me.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [], [(0, 1, 2)])
    a = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 3]], np.float32)
    compat.set_positions(me, a)
    me.update()
    assert np.allclose(compat.get_positions(me), a.ravel())
    b = np.empty(9, np.float32)
    me.vertices.foreach_get("co", b)
    assert np.allclose(b, a.ravel()) and np.allclose(tuple(me.polygons[0].normal), (0, -1, 0), atol=1e-5)
    with pytest.raises((RuntimeError, ValueError, TypeError)):          # a wrong size is not silently accepted
        compat.set_positions(me, np.zeros(4, np.float32))
    bpy.data.meshes.remove(me)


def test_blender_metaball_elements_as_arrays(blender):
    bpy = blender
    from myrmex_blender.creature import set_elements
    mb = bpy.data.metaballs.new("gfx_mb")
    for _ in range(4):
        mb.elements.new().type = "ELLIPSOID"
    pos = np.arange(12, dtype=float).reshape(4, 3)
    rad = np.array([0.5, 0.005, 0.3, 0.2])
    st = np.array([[1, 1, 1], [2, 2, 2], [1, 2, 3], [0.5, 1, 1.5]], float)
    set_elements(mb, pos, rad, st, np.array([0, 1, 0, 1]))
    els = mb.elements
    assert [e.hide for e in els] == [False, True, False, False]
    assert np.allclose(els[2].co, pos[2]) and els[3].radius == pytest.approx(0.2)
    assert (els[2].size_x, els[2].size_y, els[2].size_z) == pytest.approx((0.55, 1.1, 1.65))
    assert [round(e.stiffness, 3) for e in els] == [2.0, 1.6, 2.0, 1.6]
    bpy.data.metaballs.remove(mb)


def _look_through(bpy, sc, distance):
    import mathutils
    cam = sc.camera
    if cam is None:
        cam = bpy.data.objects.new("GfxCam", bpy.data.cameras.new("GfxCam"))
        sc.collection.objects.link(cam)
        sc.camera = cam
    cam.matrix_world = mathutils.Matrix.Translation((0.0, -distance, 0.0)) @ \
        mathutils.Euler((math.pi / 2, 0.0, 0.0)).to_matrix().to_4x4()


def test_blender_body_resolution_by_hand_and_saved_looks(blender):
    bpy = blender
    from myrmex_blender import creature, gfx
    sc = bpy.context.scene
    creature.setup_creature_scene(sc, "spear")
    mb = bpy.data.metaballs[creature.META]
    gfx.configure({"preset": "balanced"})
    gfx.S["res"] = gfx.S["set"] = None
    _look_through(bpy, sc, 40.0)
    gfx.set_body(sc, mb, (0.0, 0.0, 0.0))
    look = float(mb[gfx.LOOK_RES])
    assert mb.resolution == pytest.approx(2.0 * look, rel=0.06)                      # a wide shot: coarser
    with gfx.saving():                                                              # a look saved now keeps its own
        assert mb.resolution == pytest.approx(look)
    assert mb.resolution == pytest.approx(2.0 * look, rel=0.06)
    mb.resolution = 0.025                                                           # by hand (Properties)
    _look_through(bpy, sc, 1.0)
    gfx.set_body(sc, mb, (0.0, 0.0, 0.0))
    assert float(mb[gfx.LOOK_RES]) == pytest.approx(0.025) and mb.resolution == pytest.approx(0.025)
    _look_through(bpy, sc, 40.0)
    gfx.set_body(sc, mb, (0.0, 0.0, 0.0))
    assert mb.resolution == pytest.approx(0.05, rel=0.06)                          # the new own value, scaled
    gfx.set_body(sc, mb, (0.0, 0.0, 0.0))
    assert float(mb[gfx.LOOK_RES]) == pytest.approx(0.025)                          # its own changes are not "by hand"
    mb[gfx.LOOK_RES] = look
    gfx.S["res"] = gfx.S["set"] = None


def test_blender_auto_quality(blender):
    from myrmex_blender import gfx
    gfx.S["handler"] = object()                                         # (as if a 3D view were counting frames)
    try:
        gfx.configure({"preset": "custom", "ghost_max": 3, "auto": True, "target_fps": 50})
        assert gfx.effective()["ghost_max"] == 3
        gfx._auto(0.0, 70.0)
        gfx._auto(1.0, 40.0)
        assert gfx.S["level"] is None and gfx.effective()["ghost_max"] == 3      # your knobs stay while it keeps up
        gfx._auto(4.1, 40.0)
        assert gfx.S["level"] == "performance" and gfx.effective()["ghost_max"] == 6
        gfx._auto(5.0, 40.0)
        gfx._auto(8.1, 40.0)
        assert gfx.S["level"] == "max_fps"
        gfx._auto(9.0, 40.0)
        gfx._auto(20.0, 40.0)
        assert gfx.S["level"] == "max_fps"                              # the bottom
        gfx._auto(21.0, 70.0)
        gfx._auto(36.1, 70.0)
        assert gfx.S["level"] == "performance"
        gfx._auto(37.0, 70.0)
        gfx._auto(52.1, 70.0)
        assert gfx.S["level"] is None and gfx.effective()["ghost_max"] == 3      # back to Custom, not above
        gfx._auto(53.0, 70.0)
        gfx._auto(70.0, 70.0)
        assert gfx.S["level"] is None
        gfx.configure({"preset": "balanced"})
        assert gfx.S["cfg"]["auto"] is True
        gfx._auto(100.0, 30.0)
        gfx._auto(103.1, 30.0)
        assert gfx.status()["level"] == "performance"
        gfx._auto(104.0, 58.0)                                          # within the band: stays
        gfx._auto(130.0, 58.0)
        assert gfx.S["level"] == "performance"
        gfx._auto(131.0, 90.0)
        gfx._auto(146.1, 90.0)
        assert gfx.S["level"] is None and gfx.status()["level"] == "balanced"
        gfx._auto(147.0, 90.0)
        gfx._auto(170.0, 90.0)
        assert gfx.S["level"] is None                                   # never above Balanced (the choice)
        gfx.configure({"auto": False})
        gfx._auto(171.0, 10.0)
        gfx._auto(180.0, 10.0)
        assert gfx.S["level"] is None
    finally:
        gfx.S["handler"] = None
        gfx.configure({"preset": "balanced", "auto": False})


def test_blender_body_detail_and_afterimage_limits(blender):
    bpy = blender
    from myrmex_blender import control, creature, fx, gfx, live
    from test_fx import _stream
    sc = bpy.context.scene
    creature.setup_creature_scene(sc, "spear")
    mb = bpy.data.metaballs[creature.META]
    cam = sc.camera or bpy.data.objects.new("GfxCam", bpy.data.cameras.new("GfxCam"))
    if sc.camera is None:
        sc.collection.objects.link(cam)
        sc.camera = cam
    try:
        out = control.handle({"cmd": "gfx", "preset": "performance"})
        assert out["ok"] and out["preset"] == "performance"
        assert sc.myrmex_gfx.preset == "performance" and sc.myrmex_gfx.ghost_parts == "body"   # the panel follows
        cam.matrix_world = __import__("mathutils").Matrix.Translation((0.0, -40.0, 0.0)) @ \
            __import__("mathutils").Euler((math.pi / 2, 0.0, 0.0)).to_matrix().to_4x4()
        gfx.S["res"] = None
        far = gfx.body_res(sc, mb, (0.0, 0.0, 0.0))
        base = float(mb[gfx.LOOK_RES]) * 0.04 / 0.03
        assert far == pytest.approx(base * 2.0, rel=0.06)                                # far: coarser
        cam.matrix_world = __import__("mathutils").Matrix.Translation((0.0, -1.0, 0.0)) @ \
            __import__("mathutils").Euler((math.pi / 2, 0.0, 0.0)).to_matrix().to_4x4()
        gfx.S["res"] = None
        assert gfx.body_res(sc, mb, (0.0, 0.0, 0.0)) == pytest.approx(base)               # close: the preset's base
        assert gfx.ghost_limits() == (16, False, 1.6)          # headless = a render: every copy, every part
        gfx.S["assume_view"] = True
        assert gfx.ghost_limits() == (6, True, 2.2)
        raw = _stream("spear", "Echo", seconds=5.0)
        link = live.LiveLink(None, port=0)
        for data in raw:
            link._apply_creature(data)
        ghosts = [o for o in bpy.data.objects if o.name.startswith("MyrmexGhost")]
        alive = [o for o in ghosts if (o.type == "META" and len(o.data.elements)) or
                 (o.type == "MESH" and len(o.data.vertices))]
        assert alive and all(o.type == "META" for o in alive)                             # the body only
        slots = {o.name[len("MyrmexGhost"):len("MyrmexGhost") + 2] for o in alive}
        assert len(slots) <= 6                                                            # at most 6 copies
        assert all(o.data.resolution >= base * 2.2 - 1e-6 for o in alive)                   # coarser copies
        st = gfx.status()
        assert st["preset"] == "performance" and "stats" in st
    finally:
        gfx.S["assume_view"] = False
        gfx.configure({"preset": "balanced"})
        fx.configure({"on": False})
        fx.S["live"] = False
