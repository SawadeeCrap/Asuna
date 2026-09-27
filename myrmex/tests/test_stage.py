"""The stage: the HDRI you like in Material Preview lights every organism and shows in its reflections, while
the camera sees black - the app's setting and page, and (when Blender's ``bpy`` module is importable) the world,
the lamps, looks, the control command and the view's light."""
import json
import math
import os

import pytest


def test_app_stage_settings_and_env():
    from myrmex.app import controllers as C
    from myrmex.app.settings import AppSettings
    s = AppSettings()
    assert C.stage_settings(s) == {"hdri": "", "strength": 1.0, "rotation": 0.0, "lights": True}
    s.stage = {"hdri": "forest.exr", "strength": 99, "rotation": -90, "lights": 0, "junk": 1}
    d = C.stage_settings(s)
    assert d == {"hdri": "forest.exr", "strength": 20.0, "rotation": 270.0, "lights": False}
    assert json.loads(C.stage_env(s)["MYRMEX_STAGE"]) == d
    s.stage = {"strength": "loud"}                                        # a broken file: the defaults
    assert C.stage_settings(s)["strength"] == 1.0
    assert C.hdri_label("forest.exr") == "Forest" and C.hdri_label("/x/my_studio_4k.hdr") == "My Studio 4K"


class FakeBlender:
    """A Blender the app opened (it listens to the app)."""

    def __init__(self):
        from PySide6.QtCore import QProcess
        self.running = QProcess.ProcessState.Running

    def state(self):
        return self.running

    def property(self, name):
        return name == "myrmex_control"


def test_fx_page_stage(tmp_path, monkeypatch):
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
    w._goto("FX")
    app.processEvents()
    sent = []
    w._blender_send = lambda p, msg: sent.append((p, msg))
    live, take = FakeBlender(), FakeBlender()
    w.blender_proc, w.take_procs = live, [take]
    assert w.cmb_stage.currentData() == "" and not w.stage_sliders["strength"][0].isEnabled()
    w.cmb_stage.setCurrentIndex(w.cmb_stage.findData("forest.exr"))
    assert w.s.stage["hdri"] == "forest.exr" and w.stage_sliders["strength"][0].isEnabled()
    assert [m["hdri"] for _, m in sent] == ["forest.exr", "forest.exr"] and all(m["cmd"] == "stage" for _, m in sent)
    w.stage_sliders["rotation"][0].setValue(90)
    w.stage_sliders["strength"][0].setValue(150)
    w.chk_stage_lamps.setChecked(False)
    assert w.s.stage == {"hdri": "forest.exr", "strength": 1.5, "rotation": 90.0, "lights": False}
    assert sent[-1][1] == {"cmd": "stage", **w.s.stage}
    # set in Blender (Use This View's Lighting): kept for every organism, the other Blender follows
    sent.clear()
    fx_tab.on_blender_stage(w, take, {"cmd": "stage", "ok": True, "from_blender": True, "hdri": "my_studio.hdr",
                                      "strength": 0.8, "rotation": 200.0, "lights": True})
    assert w.s.stage == {"hdri": "my_studio.hdr", "strength": 0.8, "rotation": 200.0, "lights": True}
    assert w.cmb_stage.currentData() == "my_studio.hdr" and w.cmb_stage.currentText() == "My Studio"
    assert w.stage_sliders["rotation"][0].value() == 200 and w.stage_sliders["strength"][1].text() == "0.80"
    assert w.chk_stage_lamps.isChecked()
    assert [p for p, _ in sent] == [live] and sent[0][1]["hdri"] == "my_studio.hdr"
    fx_tab.on_blender_stage(w, live, {"cmd": "stage", "ok": True, "hdri": "night.exr"})   # (only an answer)
    assert w.s.stage["hdri"] == "my_studio.hdr" and len(sent) == 1
    from myrmex.app.settings import settings_dir
    with open(os.path.join(settings_dir(), "settings.json")) as f:
        assert json.load(f)["stage"]["hdri"] == "my_studio.hdr"
    w.cmb_stage.setCurrentIndex(w.cmb_stage.findData(""))              # back to the studio panels
    assert w.s.stage["hdri"] == "" and not w.stage_sliders["rotation"][0].isEnabled()
    w.blender_proc, w.take_procs = None, []
    w.close()


# ---------------------------------------------------------------------- inside Blender (bpy module)
@pytest.fixture(scope="module")
def blender():
    bpy = pytest.importorskip("bpy")
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "blender"))
    import myrmex_blender
    if not hasattr(bpy.types.Scene, "myrmex_stage"):
        myrmex_blender.register()
    yield bpy
    from myrmex_blender import stage
    stage.configure(stage.DEFAULTS)
    myrmex_blender.unregister()                  # (the bpy module waits at exit for handlers left behind)
    for lst in (bpy.app.handlers.load_post, bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post):
        lst.clear()


def _lamps(sc):
    return {o.name: (o.hide_viewport, o.hide_render) for o in sc.objects if o.type == "LIGHT"}


def test_blender_stage_world_lamps_and_looks(blender, tmp_path, monkeypatch):
    """The HDRI lights the organism in a world of its own (camera: black); the organism's own world waits
    underneath and is what its look keeps; lamps off = the HDRI alone."""
    bpy = blender
    monkeypatch.setenv("MYRMEX_LOOKS", str(tmp_path / "looks"))
    from myrmex_blender import control, creature, looks, stage
    stage.configure(stage.DEFAULTS)
    assert "forest.exr" in stage.hdri_names() and os.path.isfile(stage.hdri_path("forest.exr"))
    assert stage.hdri_path("") == "" and stage.hdri_path("no_such.exr") == ""
    sc = bpy.context.scene
    creature.setup_creature_scene(sc, "spear")
    own = sc.world
    assert own.name != stage.STAGE_WORLD and "MyrmexCameraBlack" in own.node_tree.nodes
    mine = next(o for o in sc.objects if o.type == "LIGHT")               # a lamp you switched off yourself
    mine.hide_viewport = mine.hide_render = True
    st = stage.apply({"hdri": "forest.exr", "strength": 1.5, "rotation": 90, "lights": False})
    assert st == {"hdri": "forest.exr", "strength": 1.5, "rotation": 90.0, "lights": False}
    w = sc.world
    assert w.name == stage.STAGE_WORLD and sc[stage.OWN] == own.name and bpy.data.worlds[own.name].use_fake_user
    nd = w.node_tree.nodes
    rot, env, light = nd["MyrmexStageRotate"], nd["MyrmexStageHDRI"], nd["MyrmexStageLight"]
    assert rot.rotation_type == "Z_AXIS" and rot.inputs["Angle"].default_value == pytest.approx(math.pi / 2)
    assert rot.inputs["Vector"].links[0].from_socket.name == "Generated"     # as Material Preview builds it
    assert os.path.basename(env.image.filepath) == "forest.exr" and light.inputs["Strength"].default_value == 1.5
    mix = nd["MyrmexCameraBlackMix"]                                      # the camera still sees black
    assert mix.inputs[0].links[0].from_socket.name == "Is Camera Ray" and mix.inputs[1].links[0].from_node == light
    assert all(v == (True, True) for v in _lamps(sc).values())
    assert not mine.get(stage.OFF) and all(o.get(stage.OFF) for o in sc.objects if o.type == "LIGHT" and o != mine)
    p = sc.myrmex_stage                                                   # the Stage panel shows it
    assert p.hdri == "forest.exr" and p.strength == pytest.approx(1.5) and not p.lights
    assert math.degrees(p.rotation) == pytest.approx(90.0, abs=1e-3)
    ptr = env.as_pointer()
    stage.apply({"strength": 0.7, "rotation": 450})                       # turned: in place, no rebuild
    nd = sc.world.node_tree.nodes
    assert nd["MyrmexStageHDRI"].as_pointer() == ptr and nd["MyrmexStageLight"].inputs["Strength"].default_value \
        == pytest.approx(0.7) and nd["MyrmexStageRotate"].inputs["Angle"].default_value == pytest.approx(math.pi / 2)
    assert sum(1 for n in nd if n.name == "MyrmexCameraBlackMix") == 1
    # a look keeps its own world and lamps; Blender goes on showing the stage
    path = str(tmp_path / "look.blend")
    looks._write_look(path, sc)
    assert sc.world.name == stage.STAGE_WORLD and all(v == (True, True) for v in _lamps(sc).values())
    assert bpy.data.worlds[own.name].use_fake_user
    with bpy.data.libraries.load(path) as (src, dst):
        assert src.worlds == [own.name]
        dst.objects = [n for n in src.objects if n in _lamps(sc)]
    back = {o.name[:-4] if o.name[-4:-3] == "." else o.name: (o.hide_render, o.get(stage.OFF)) for o in dst.objects}
    assert back[mine.name] == (True, None) and all(v == (False, None) for k, v in back.items() if k != mine.name)
    for o in dst.objects:
        bpy.data.objects.remove(o)
    # the panel sets it too
    p.hdri = "city.exr"
    assert stage.STAGE["hdri"] == "city.exr" and os.path.basename(sc.world.node_tree.nodes["MyrmexStageHDRI"].image
                                                                   .filepath) == "city.exr"
    # back to the organism's own light: its world, its lamps (yours stays off)
    stage.apply({"hdri": "", "lights": True})
    assert sc.world == own and not own.use_fake_user and stage.KEPT not in own
    assert "MyrmexCameraBlack" in own.node_tree.nodes and p.hdri == "NONE" and p.lights
    lamps = _lamps(sc)
    assert lamps[mine.name] == (True, True) and all(v == (False, False) for k, v in lamps.items() if k != mine.name)
    assert not any(o.get(stage.OFF) for o in sc.objects)
    # a file of your own; a missing one falls back to the organism's own light
    stage.apply({"hdri": stage.hdri_path("night.exr")})
    assert sc.world.name == stage.STAGE_WORLD and p.hdri == "PATH"
    stage.apply({"hdri": "gone.exr"})
    assert sc.world == own
    # the app's command, and a new organism under the same light
    out = control.handle({"cmd": "stage", "hdri": "sunset.exr", "rotation": -30, "lights": False})
    assert out["ok"] and out["rotation"] == 330.0 and sc.world.name == stage.STAGE_WORLD
    creature.setup_creature_scene(sc, "hive")
    assert sc.world.name == stage.STAGE_WORLD and all(v == (True, True) for v in _lamps(sc).values())
    own2 = stage.own_world(sc)
    assert own2 is not None and own2.name != stage.STAGE_WORLD
    stage.apply(stage.DEFAULTS)
    assert sc.world == own2 and "MyrmexCameraBlack" in own2.node_tree.nodes


def test_blender_stage_from_env_and_view(blender, monkeypatch):
    from myrmex_blender import stage
    monkeypatch.setenv("MYRMEX_STAGE", json.dumps({"hdri": "studio.exr", "strength": 2, "rotation": 30}))
    assert stage.from_env() == {"hdri": "studio.exr", "strength": 2.0, "rotation": 30.0, "lights": True}
    assert json.loads(stage.env()["MYRMEX_STAGE"])["hdri"] == "studio.exr"
    monkeypatch.setenv("MYRMEX_STAGE", "not json")
    assert stage.from_env() is None

    class Shading:
        type = "MATERIAL"
        use_scene_world = False
        use_scene_lights = False
        use_scene_world_render = True
        use_scene_lights_render = True
        studio_light = "sunset.exr"
        studiolight_intensity = 1.2
        studiolight_rotate_z = -math.pi / 2
    sh = Shading()
    assert stage.view_light(sh) == {"hdri": "sunset.exr", "strength": 1.2, "rotation": 270.0, "lights": False}
    sh.use_scene_world = True                                             # the view shows the scene's world
    assert stage.view_light(sh) is None
    sh.type = "RENDERED"                                                  # Rendered with Scene World on: None
    assert stage.view_light(sh) is None
    sh.use_scene_world_render = False
    assert stage.view_light(sh)["lights"] is True
    sh.type = "SOLID"
    assert stage.view_light(sh) is None
    stage.configure(stage.DEFAULTS)
