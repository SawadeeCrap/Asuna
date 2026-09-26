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
    assert rack_from_preset("Sandevistan")["ghosts"] > 0.5 and rack_from_preset("Clean")["ghosts"] == 0.0
    assert rack_from_preset("Echo")["ghost_density"] > 0.9                    # dense copies: a smear
    assert size_of(False) == (1920, 1080) and size_of(True) == (1080, 1920)


def test_rack_frame_knobs_and_trailer():
    fx = FxRack({"preset": "Anime Impact", "vertical": True, "preview": 0.75})
    drives = np.arange(len(TD), dtype=float)
    v = as_dict(fx.frame(drives, {"fx_ghosts": 0.12}))
    assert v["kick"] == TD.index("kick") and v["speed"] == TD.index("speed")      # the TD drives, by name
    assert v["r_ghosts"] == pytest.approx(0.12)                                   # a MIDI knob wins
    assert v["r_impact_frames"] == pytest.approx(1.0) and v["vertical"] == 1.0 and v["preview"] == 0.75
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
    assert d["enabled"] and d["preset"] == "Sandevistan" and set(d["rack"]) == set(RACK)
    s.fx = {"preset": "Glitch", "rack": {"glitch": 0.3}, "vertical": True}
    e = C.fx_engine(s)
    assert e["rack"]["glitch"] == 0.3 and e["rack"]["chroma"] == PRESETS["Glitch"]["chroma"] and e["vertical"]
    b = C.fx_blender(s)
    assert b["on"] and b["replay"] and b["rack"]["glitch"] == 0.3
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
    w.cmb_fx_preset.setCurrentIndex(w.cmb_fx_preset.findData("Anime Impact"))
    app.processEvents()
    assert w.s.fx["preset"] == "Anime Impact" and w.s.fx["rack"]["impact_frames"] == 1.0
    assert w.fx_sliders["speed_lines"].value() == int(1000 * PRESETS["Anime Impact"]["speed_lines"])
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


def _stream(variant, preset="Sandevistan", seconds=4.0):
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
    raw = _stream("spear")
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
    assert len({tuple(round(c, 2) for c in o.color[:3]) for o in alive}) >= 3     # the colours cycle
    rb = bpy.data.objects.get("MyrmexRibbons")
    assert rb is not None and len(rb.data.vertices) > 0
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
    fx.configure({"preset": "Dream", "on": False})
    assert sc.myrmex_fx.preset == "Dream" and sc.myrmex_fx.ghost_life == pytest.approx(PRESETS["Dream"]["ghost_life"])


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
    assert out[45, 40, 0] > 0.9 and 0.02 < out[45, 55, 1] < 0.6 and out[5, 150, 1] < 0.02    # disc, halo, dark
    fx.S["cfg"]["rack"] = rack_from_preset("Anime Impact")
    fx.S["vals"] = {"impact": 1.0}
    pipe.run(fx.post_params(sc, W, H), src_tex=tex(img))
    inv = pipe.read().astype(float) / 255
    assert abs(inv[45, 40, 0] - inv[5, 150, 0]) > 0.5                     # a two-tone impact frame
    fx.S["cfg"]["rack"] = rack_from_preset("Dream")
    fx.S["vals"] = {}
    pipe.reset()
    for k in range(5):                                                   # the disc moves; its trail stays
        a = np.zeros_like(img)
        a[..., 3] = 1.0
        a[(xx - (40 + 20 * k)) ** 2 + (yy - 45) ** 2 < 100, :3] = 1.0
        fx.S["t"] = k / 30
        pipe.run(fx.post_params(sc, W, H), src_tex=tex(a))
    tr = pipe.read().astype(float) / 255
    assert tr[45, 40, 1] > 0.2 and tr[5, 5, 1] < 0.05
    pipe.free()


def test_blender_take_playback_leaves_copies(blender, tmp_path):
    bpy = blender
    from myrmex_blender import creature_take, fx
    sink = Sink()
    ses = LiveSession(LiveConfig(backend="swarm", clock="internal", out=[], record=str(tmp_path),
                                 fx={"preset": "Sandevistan"}), start_inputs=False, sink=sink, now=0.0)
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
