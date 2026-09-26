"""Myrmex FX (effects drawn by Blender, no TouchDesigner): the rack and its presets / MIDI knobs, the block every
frame carries to Blender, takes, the app page - and, when Blender's ``bpy`` module is importable, the
afterimages, ribbons and GPU passes themselves."""
import os
import time

import numpy as np
import pytest

from myrmex.creature.protocol import decode_creature
from myrmex.realtime.fx import (CHANNELS, DEFAULT_RACK, DRIVES, PRESETS, RACK, FxRack, as_dict, decode_trailer,
                                rack_from_preset, size_of, trailer)
from myrmex.realtime.protocol import PoseFrame, decode_pose, encode_pose
from myrmex.realtime.session import LiveConfig, LiveSession
from myrmex.realtime.touch import CHANNELS as TD


class Sink:
    def __init__(self):
        self.raw = []
        self.sent = 0

    def send_raw(self, b):
        self.raw.append(b)

    def send(self, fr, now, extra=b""):
        self.raw.append(encode_pose(fr) + extra)

    def close(self):
        pass


def test_channels_rack_and_presets():
    assert len(set(CHANNELS)) == len(CHANNELS) and all(n in TD for n in DRIVES)
    for name, p in PRESETS.items():
        assert set(p) <= set(RACK), name
        r = rack_from_preset(name)
        assert set(r) == set(RACK) and all(0.0 <= x <= 1.0 for x in r.values())
    assert rack_from_preset("Afterimage")["ghosts"] > 0.5 and rack_from_preset("Clean")["ghosts"] == 0.0
    assert set(RACK) == {"ghosts", "ghost_life", "ghost_density", "ribbons", "trails", "bloom", "react", "exposure",
                         "contrast", "saturation"}                   # nothing that moves, tears or tints the frame
    assert rack_from_preset("Echo")["ghost_density"] > 0.9                    # dense copies: a smear
    assert size_of(False) == (1920, 1080) and size_of(True) == (1080, 1920)


def test_rack_frame_knobs_and_trailer():
    fx = FxRack({"preset": "Trace", "vertical": True, "preview": 0.75})
    drives = np.arange(len(TD), dtype=float)
    v = as_dict(fx.frame(drives, {"fx_ghosts": 0.12}))
    assert v["kick"] == TD.index("kick") and v["speed"] == TD.index("speed")      # the TD drives, by name
    assert v["r_ghosts"] == pytest.approx(0.12)                                   # a MIDI knob wins
    assert v["r_ribbons"] == pytest.approx(PRESETS["Trace"]["ribbons"]) and v["vertical"] == 1.0 and v["preview"] == 0.75
    fx.configure(rack={"trails": 3.0, "nonsense": 1.0})
    assert fx.cfg["rack"]["trails"] == 1.0 and "nonsense" not in fx.cfg["rack"]
    fx.configure(enabled=False)
    vals = fx.frame(drives)
    assert as_dict(vals)["on"] == 0.0
    blob = b"frame bytes" + trailer(vals)
    back = decode_trailer(blob)
    assert back is not None and np.allclose(back, vals.astype(np.float32))
    assert decode_trailer(b"no block here") is None and decode_trailer(b"") is None


def test_creature_frames_carry_the_block_and_takes_record_it(tmp_path):
    sink = Sink()
    ses = LiveSession(LiveConfig(backend="spear", clock="internal", out=[], record=str(tmp_path),
                                 fx={"preset": "Echo", "vertical": True}), start_inputs=False, sink=sink, now=0.0)
    now = 0.0
    for _ in range(240):
        now += 1 / 120
        ses.step(now)
    assert len(sink.raw) > 100
    data = sink.raw[-1]
    fr = decode_creature(data)                                   # the frame itself still decodes
    assert fr is not None and fr.links is not None
    v = as_dict(decode_trailer(data))
    assert v["r_ghost_density"] == pytest.approx(PRESETS["Echo"]["ghost_density"]) and v["vertical"] == 1.0
    assert v["on"] == 1.0 and 0.0 <= v["phase"] < 1.0
    ses.fx.configure(rack={"ghosts": 0.2})                       # the app's page, live
    for k in range(3):
        ses.step(now + (k + 1) / 120)
    assert as_dict(decode_trailer(sink.raw[-1]))["r_ghosts"] == pytest.approx(0.2)
    assert ses.status()["fx"]["vertical"] is True
    from myrmex.creature.take import CreatureTake
    take = CreatureTake(ses.save_take())
    assert take.resampled("fx", 30).shape[1] == len(CHANNELS)
    assert [str(n) for n in take.d["fx_names"]] == list(CHANNELS)


def test_pose_frames_keep_decoding_with_the_block():
    D = np.tile(np.eye(4), (3, 1, 1))
    fr = PoseFrame(7, 1.0, 2.0, 120.0, 5, D, 1, None, np.array([1.0, 2.0, 0.0]), 0.5, 1.2, 0.3)
    vals = FxRack().frame(np.zeros(len(TD)))
    back = decode_pose(encode_pose(fr) + trailer(vals))
    assert back is not None and back.seq == 7 and np.allclose(back.subject_pos, [1.0, 2.0, 0.0])
    assert decode_trailer(encode_pose(fr) + trailer(vals)) is not None


def test_app_settings_to_engine_and_blender():
    from myrmex.app import controllers as C
    from myrmex.app.settings import AppSettings
    s = AppSettings()
    d = C.fx_settings(s)
    assert d["enabled"] and d["preset"] == "Afterimage" and set(d["rack"]) == set(RACK)
    s.fx = {"preset": "Sandevistan", "rack": {"chroma": 0.9}}                 # an old preset: back to Afterimage
    assert C.fx_settings(s)["preset"] == "Afterimage" and "chroma" not in C.fx_settings(s)["rack"]
    s.fx = {"preset": "Phantom", "rack": {"trails": 0.3}, "vertical": True}
    e = C.fx_engine(s)
    assert e["rack"]["trails"] == 0.3 and e["rack"]["ghost_life"] == PRESETS["Phantom"]["ghost_life"] and e["vertical"]
    b = C.fx_blender(s)
    assert b["on"] and b["replay"] and b["monitor"] and b["rack"]["trails"] == 0.3
    import json
    assert json.loads(C.fx_env(s)["MYRMEX_FX"])["vertical"] is True
    pytest.importorskip("PySide6")
    from myrmex.app.tabs import TARGETS
    assert all(f"fx_{k}" in TARGETS for k in RACK)


def test_fx_page(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from myrmex.app import window as W
    from myrmex.app.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    s = AppSettings()
    s.start_engine_on_launch = s.open_blender_on_start = False
    w = W.MainWindow(s)
    assert "FX" in w.page_index
    w.cmb_fx_preset.setCurrentIndex(w.cmb_fx_preset.findData("Trace"))
    app.processEvents()
    assert w.s.fx["preset"] == "Trace" and w.s.fx["rack"]["ribbons"] == PRESETS["Trace"]["ribbons"]
    assert w.fx_sliders["ribbons"].value() == int(1000 * PRESETS["Trace"]["ribbons"])
    w.fx_sliders["ghosts"].setValue(100)                          # a moved slider: your own mix
    assert w.s.fx["rack"]["ghosts"] == pytest.approx(0.1) and w.s.fx["preset"] == ""
    from myrmex.app import fx_tab
    fx_tab._format(w, True)
    assert w.s.fx["vertical"] is True and w.s.render_size == "1080x1920"
    w.close()


# ---------------------------------------------------------------------- inside Blender (bpy module)
@pytest.fixture(scope="module")
def blender():
    bpy = pytest.importorskip("bpy")
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "blender"))
    import myrmex_blender
    if not hasattr(bpy.types.Scene, "myrmex_fx"):
        myrmex_blender.register()
    yield bpy
    myrmex_blender.unregister()                  # (the bpy module waits at exit for handlers left behind)
    for lst in (bpy.app.handlers.load_post, bpy.app.handlers.frame_change_pre, bpy.app.handlers.frame_change_post):
        lst.clear()


def _stream(variant, preset="Afterimage", seconds=4.0):
    sink = Sink()
    ses = LiveSession(LiveConfig(backend=variant, clock="internal", out=[], fx={"preset": preset}),
                      start_inputs=False, sink=sink, now=0.0)
    now = 0.0
    for _ in range(int(120 * seconds)):
        now += 1 / 120
        ses.step(now)
    return sink.raw


def test_blender_afterimages_and_ribbons_follow_the_stream(blender):
    bpy = blender
    from myrmex_blender import creature, fx, live
    raw = _stream("spear", "Trace")
    creature.setup_creature_scene(bpy.context.scene, "spear")
    link = live.LiveLink(None, port=0)
    for data in raw:
        link._apply_creature(data)
    st = fx.status()
    assert st["on"] and st["source"] == "block" and not st["error"]
    assert st["ghosts"]["spawned"] >= 10
    ghosts = [o for o in bpy.data.objects if o.name.startswith("MyrmexGhost")]
    alive = [o for o in ghosts if (o.type == "META" and len(o.data.elements)) or
             (o.type == "MESH" and len(o.data.vertices))]
    assert {o.name.split("_", 1)[1] for o in alive} >= {"CreatureBody", "MimeticTendons"}
    assert all(fx.S["t"] - o["myrmex_birth"] <= 2.0 for o in alive)            # only the living copies
    from myrmex_blender.fx_ghosts import sources
    own = {m.name for o in sources(bpy.context.scene)
           for m in (o.data.materials if o.type == "META" else [x.material for x in o.material_slots]) if m}
    worn = {m.name for o in alive for m in (o.data.materials if o.type == "META" else
                                            [x.material for x in o.material_slots]) if m}
    assert worn and all(n.startswith("MyrmexGhost·") for n in worn)            # the organism's own materials,
    assert {n[len("MyrmexGhost·"):] for n in worn} <= own                     # made ghostly - no colours of their own
    rb = bpy.data.objects.get("MyrmexRibbons")
    assert rb is not None and len(rb.data.vertices) > 0
    from myrmex_blender.fx_ghosts import accent
    col = np.empty(4 * len(rb.data.vertices), np.float32)
    rb.data.attributes["col"].data.foreach_get("color", col)
    assert np.allclose(col.reshape(-1, 4)[0, :3], accent(sources(bpy.context.scene)), atol=1e-4)   # its own colour
    assert bpy.context.scene.render.resolution_x == 1920                        # the stream said horizontal
    fx.configure({"on": False})
    fx.S["live"] = False


def test_blender_vertical_format_and_panel(blender):
    bpy = blender
    from myrmex_blender import fx
    sc = bpy.context.scene
    fx.apply_format(sc, True)
    assert (sc.render.resolution_x, sc.render.resolution_y) == (1080, 1920)
    sc.render.resolution_x, sc.render.resolution_y = 2160, 3840
    fx.apply_format(sc, True)                                   # a vertical size of your own stays
    assert sc.render.resolution_y == 3840
    fx.apply_format(sc, False)
    assert (sc.render.resolution_x, sc.render.resolution_y) == (1920, 1080)
    fx.configure({"preset": "Phantom", "on": False})
    assert sc.myrmex_fx.preset == "Phantom" and sc.myrmex_fx.ghost_life == pytest.approx(PRESETS["Phantom"]["ghost_life"])


def test_blender_gpu_passes(blender):
    bpy = blender
    import gpu
    if bpy.app.background:
        if not hasattr(gpu, "init"):
            pytest.skip("the gpu module needs Blender 5.0+ in background mode")
        gpu.init()
    from myrmex_blender import fx, fx_post
    W, H = 160, 90
    pipe = fx_post.Pipeline(W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 4), np.float32)
    img[..., 3] = 1.0
    img[(xx - 40) ** 2 + (yy - 45) ** 2 < 100, :3] = 1.0

    def tex(a):
        return gpu.types.GPUTexture((W, H), format="RGBA16F", data=gpu.types.Buffer("FLOAT", W * H * 4,
                                                                                     a.ravel().tolist()))
    sc = bpy.context.scene
    fx.S["cfg"]["rack"] = rack_from_preset("Clean")
    fx.S["vals"], fx.S["rack_src"] = {}, "cfg"
    pipe.run(fx.post_params(sc, W, H), src_tex=tex(img))
    out = pipe.read().astype(float) / 255
    assert out[45, 40, 0] > 0.9 and out[45, 53, 1] > 0.0                 # the disc and a soft glow round it
    assert out[5, 150, :3].max() == 0.0 and out[85, 5, :3].max() == 0.0   # black stays pure black
    fx.S["cfg"]["rack"] = {**rack_from_preset("Clean"), "exposure": 1.0, "contrast": 0.0, "saturation": 1.0}
    pipe.run(fx.post_params(sc, W, H), src_tex=tex(img))
    graded = pipe.read().astype(float) / 255
    assert graded[5, 150, :3].max() == 0.0                                 # colour never lifts the black
    fx.S["cfg"]["rack"] = rack_from_preset("Phantom")
    fx.S["vals"] = {"kick": 1.0, "impact": 1.0, "energy": 1.0}             # hits move nothing on the frame
    pipe.reset()
    for k in range(5):                                                   # the disc moves: its echo stays
        a = np.zeros_like(img)
        a[..., 3] = 1.0
        a[(xx - (40 + 20 * k)) ** 2 + (yy - 45) ** 2 < 100, :3] = 1.0
        fx.S["t"] = k / 30
        pipe.run(fx.post_params(sc, W, H), src_tex=tex(a))
    tr = pipe.read().astype(float) / 255
    assert tr[45, 40, 1] > 0.15 and tr[5, 5, :3].max() == 0.0 and tr[85, 150, :3].max() == 0.0
    pipe.free()


def test_blender_black_stage(blender):
    """Only the organism, on black: no floor, no terrain, no impact balls, no prey; the camera sees black."""
    bpy = blender
    from myrmex_blender import creature
    for variant in ("spear", "crawler", "hive", "ferro"):
        sc = bpy.context.scene
        creature.setup_creature_scene(sc, variant)
        floor = bpy.data.objects.get("MyrmexFloor")
        assert floor is None or floor.hide_render
        assert bpy.data.objects.get("CrawlerTerrain") is None
        for o in bpy.data.objects:
            if o.name.startswith("PolyObstacle") or o.name == "ColonyPrey":
                assert o.hide_render and o.hide_viewport, o.name
        nd = sc.world.node_tree.nodes
        assert "MyrmexCameraBlack" in nd and "MyrmexCameraBlackMix" in nd
        mix = nd["MyrmexCameraBlackMix"]
        assert mix.inputs[0].links[0].from_socket.name == "Is Camera Ray"
        assert mix.inputs[2].links[0].from_node.name == "MyrmexCameraBlack"
    creature.black_stage(sc)                                             # idempotent
    assert sum(1 for n in sc.world.node_tree.nodes if n.name.startswith("MyrmexCameraBlack")) == 2


def test_blender_take_playback_leaves_copies(blender, tmp_path):
    bpy = blender
    from myrmex_blender import creature_take, fx
    sink = Sink()
    ses = LiveSession(LiveConfig(backend="swarm", clock="internal", out=[], record=str(tmp_path),
                                 fx={"preset": "Afterimage"}), start_inputs=False, sink=sink, now=0.0)
    now = 0.0
    for _ in range(120 * 3):
        now += 1 / 120
        ses.step(now)
    path = ses.save_take()
    fx.configure({"on": True, "preset": "Echo", "replay": False})     # the app's page, not the recorded rack
    creature_take.import_take(path)
    sc = bpy.context.scene
    g0 = fx.S["ghosts"].spawned if fx.S.get("ghosts") is not None else 0
    for f in range(sc.frame_start, sc.frame_start + 60):
        sc.frame_set(f)
    assert fx.S["rack_src"] == "cfg" and fx.rack()["ghost_density"] == PRESETS["Echo"]["ghost_density"]
    assert fx.S["ghosts"].spawned - g0 >= 5 and not fx.S["error"]
    body = [o for o in bpy.data.objects if o.name.startswith("MyrmexGhost") and o.name.endswith("_CreatureBody")
            and o.type == "META" and len(o.data.elements)]
    assert body                                                      # the body, copied from the take's data
    sc.frame_set(sc.frame_start + 5)                                  # a jump back: the copies start over
    assert all(b < -1e8 or b <= fx.S["t"] for b in fx.S["ghosts"].birth)
    fx.configure({"on": False, "replay": True})


def test_blender_monitor_limits_samples_waits_and_paces(blender, tmp_path):
    """The live monitor must never choke Blender: EEVEE at 1 viewport sample while pictures are drawn every
    frame (draw_view3d renders all of them per call), the scene's own value back afterwards (and in saved
    looks), no picture before the view has compiled its shaders, and it slows down / pauses by itself."""
    bpy = blender
    import time as _t

    from myrmex_blender import fx_post, looks
    sc = bpy.context.scene
    sc.eevee.taa_samples = 16
    fx_post.limit_samples("monitor", True)
    fx_post.limit_samples("syphon", True)
    assert sc.eevee.taa_samples == 1 and sc[fx_post.TAA_KEY] == 16
    fx_post.limit_samples("monitor", False)
    assert sc.eevee.taa_samples == 1                                    # Syphon still draws
    path = str(tmp_path / "look.blend")
    with looks._without_take(sc):                                        # what a saved look keeps
        assert sc.eevee.taa_samples == 16 and fx_post.TAA_KEY not in sc
    assert sc.eevee.taa_samples == 1 and sc[fx_post.TAA_KEY] == 16
    fx_post.limit_samples("syphon", False)
    assert sc.eevee.taa_samples == 16 and fx_post.TAA_KEY not in sc
    # waiting for shaders, the right shading, EEVEE
    scr = bpy.data.screens.get("Layout")
    area = next((a for a in scr.areas if a.type == "VIEW_3D"), None) if scr else None
    if area is None:
        pytest.skip("no 3D view in this Blender's startup screen")
    space = area.spaces[0]
    sc.render.engine = "BLENDER_EEVEE" if bpy.app.version >= (5, 0, 0) else "BLENDER_EEVEE_NEXT"
    space.shading.type = "SOLID"
    assert "Rendered" in fx_post._why_not(sc, space)
    space.shading.type = "RENDERED"
    fx_post._M["warm_until"] = _t.perf_counter() + 10.0
    assert "preparing" in fx_post._why_not(sc, space)
    fx_post._M["warm_until"] = 0.0
    fx_post._M["paused"] = False
    assert fx_post._why_not(sc, space) == ""
    # pacing: a struggling Blender lowers the size, then pauses
    fx_post._M.update(scale=1.0, busy=0.0, period=0.0, slow=0, note="", paused=False)
    for _ in range(25):
        fx_post._adapt(0.01, 0.3)
    assert fx_post._M["scale"] < 1.0 and "lowered" in fx_post._M["note"]
    for _ in range(80):
        fx_post._adapt(0.01, 0.3)
    assert fx_post._M["paused"] and "paused" in fx_post._M["note"]
    assert "paused" in fx_post._why_not(sc, space)
    fx_post._M.update(scale=1.0, busy=0.0, period=0.0, slow=0, note="", paused=False)
    for _ in range(40):                                                   # a healthy pace stays as it is
        fx_post._adapt(0.004, 0.017)
    assert fx_post._M["scale"] == 1.0 and not fx_post._M["paused"] and fx_post._M["interval"] <= 1 / 50


def test_blender_monitor_picture_without_switching_the_view(blender):
    bpy = blender
    import gpu
    if bpy.app.background:
        if not hasattr(gpu, "init"):
            pytest.skip("the gpu module needs Blender 5.0+ in background mode")
        gpu.init()
    from myrmex_blender import fx, fx_post
    sc = bpy.context.scene
    scr = bpy.data.screens.get("Layout")
    area = next((a for a in scr.areas if a.type == "VIEW_3D"), None) if scr else None
    if area is None:
        pytest.skip("no 3D view in this Blender's startup screen")
    space = area.spaces[0]
    region = next(r for r in area.regions if r.type == "WINDOW")
    sc.render.engine = "BLENDER_EEVEE"
    space.shading.type = "RENDERED"
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 320, 180, 100
    fx.S["cfg"].update(on=True, preview=1.0)
    fx.S["cfg"]["rack"] = rack_from_preset("Afterimage")
    fx.S["vals"], fx.S["rack_src"] = {}, "cfg"
    sc.eevee.taa_samples = 16
    fx_post.limit_samples("monitor", True)
    try:
        fx_post._produce(bpy.context, sc, space, region)             # the first one compiles the shaders
        t0 = time.perf_counter()
        fx_post._produce(bpy.context, sc, space, region)
        took = time.perf_counter() - t0
    finally:
        fx_post.limit_samples("monitor", False)
    assert space.shading.type == "RENDERED"                          # nothing switched behind the view's back
    pipe = fx_post._M["pipe"]
    assert (pipe.w, pipe.h) == (320, 180) and pipe.read()[..., :3].max() > 0
    assert took < 5.0 and sc.eevee.taa_samples == 16
    pipe.free()
    fx_post._M["pipe"] = None
