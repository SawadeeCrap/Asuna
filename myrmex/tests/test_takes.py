"""Takes: REC / STOP record exactly the stretch in between (not the whole engine session), a take keeps real time
(stalls are held, not lost), STOP saves in the background, the engine stop saves only a take still recording -
and the app's REC button, list of takes and their length."""
import os
import time

import numpy as np
import pytest

from myrmex.bus.transport import LiveEvent
from myrmex.creature.take import CreatureTake
from myrmex.realtime.inputs import InputConfig
from myrmex.realtime.session import LiveConfig, LiveSession

DT = 1.0 / 120.0


def _session(tmp_path, backend="spear", **kw):
    cfg = LiveConfig(backend=backend, clock="internal", out=[], record=str(tmp_path), rec_on_start=False,
                     inputs=InputConfig(osc_port=0), **kw)
    return LiveSession(cfg, start_inputs=False, now=0.0)


def _run(ses, seconds, t0):
    n = int(round(seconds / DT))
    for i in range(n):
        ses.step(t0 + (i + 1) * DT)
    return t0 + n * DT


def test_rec_stop_records_only_what_is_in_between(tmp_path):
    ses = _session(tmp_path)
    now = _run(ses, 5.0, 0.0)                              # the engine runs a while: nothing is recorded
    assert not ses.rec_status()["on"] and not os.listdir(tmp_path)
    assert ses.rec_start() and not ses.rec_start()         # (already recording)
    now = _run(ses, 2.0, now)
    st = ses.rec_status()
    assert st["on"] and st["seconds"] == pytest.approx(2.0, abs=0.05)
    path = ses.rec_stop()
    assert path and os.path.basename(path).startswith("spear_take_") and os.path.exists(path)
    take = CreatureTake(path)
    assert take.fps == 30.0 and take.duration == pytest.approx(2.0, abs=0.05)      # the render is 2 s long
    assert ses.rec_status()["last"] == path and ses.rec_status()["last_seconds"] == pytest.approx(2.0, abs=0.05)
    assert ses.rec_stop() is None                          # nothing more to save
    now = _run(ses, 3.0, now)                              # the engine goes on: still nothing recorded
    ses.rec_start()
    now = _run(ses, 1.0, now)
    second = ses.rec_stop()
    assert second != path and CreatureTake(second).duration == pytest.approx(1.0, abs=0.05)   # (same second: _2)
    assert ses.stop() is None                              # the engine stop saves nothing more
    assert sorted(os.listdir(tmp_path)) == sorted([os.path.basename(path), os.path.basename(second)])


def test_engine_stop_saves_the_take_still_recording(tmp_path):
    ses = _session(tmp_path, backend="ferro")
    now = _run(ses, 1.0, 0.0)
    ses.rec_start()
    _run(ses, 1.5, now)
    path = ses.stop()
    assert path and CreatureTake(path).duration == pytest.approx(1.5, abs=0.05)


def test_a_stall_is_held_so_the_take_keeps_real_time(tmp_path):
    ses = _session(tmp_path)
    ses.rec_start()
    now = _run(ses, 1.0, 0.0)
    ses.rec["pad"] += 0.5                                  # the run loop lost half a second (it resynced)
    _run(ses, 1.0, now + 0.5)
    take = CreatureTake(ses.rec_stop())
    assert take.duration == pytest.approx(2.5, abs=0.05)
    pos = take.d["pos"]
    held = [i for i in range(1, len(pos)) if np.array_equal(pos[i], pos[i - 1])]
    assert len(held) >= 14                                 # the frame before the stall, held


def test_background_save_and_pad_trigger(tmp_path):
    ses = _session(tmp_path)
    ses.inputs.push(LiveEvent("trigger", 0.0, {"name": "take"}))          # a MIDI pad: REC
    now = _run(ses, 1.0, 0.0)
    assert ses.rec_status()["on"]
    ses.inputs.push(LiveEvent("trigger", now, {"name": "take"}))          # again: STOP (written by a thread)
    now = _run(ses, 0.2, now)
    t_end = time.time() + 10
    while ses.rec_status()["saving"] and time.time() < t_end:
        time.sleep(0.02)
    st = ses.rec_status()
    assert not st["on"] and st["last"] and os.path.exists(st["last"]) and st["error"] is None
    assert CreatureTake(st["last"]).duration == pytest.approx(1.0, abs=0.05)
    ses.inputs.push(LiveEvent("trigger", now, {"name": "take:stop"}))    # (not recording: nothing)
    ses.inputs.push(LiveEvent("trigger", now, {"name": "take:start"}))
    _run(ses, 0.1, now)
    assert ses.rec_status()["on"]
    ses.rec_stop()
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".")]      # no half-written files left


def test_old_style_record_from_the_start(tmp_path):
    cfg = LiveConfig(backend="spear", clock="internal", out=[], record=str(tmp_path), inputs=InputConfig(osc_port=0))
    ses = LiveSession(cfg, start_inputs=False, now=0.0)    # (scripts, the CLI: --record records from the start)
    _run(ses, 1.0, 0.0)
    assert CreatureTake(ses.save_take()).duration == pytest.approx(1.0, abs=0.05)


def test_humanoid_rec_stop(tmp_path, biped_plan):
    from myrmex.performance.performance import Performance
    cfg = LiveConfig(clock="internal", link=False, out=[], latency=0.0, record=str(tmp_path), rec_on_start=False,
                     inputs=InputConfig(osc_port=0))
    ses = LiveSession(cfg, biped_plan, start_inputs=False, now=0.0)
    now = _run(ses, 1.0, 0.0)
    ses.rec_start()
    now = _run(ses, 1.0, now)
    ses.rec["pad"] += 0.5
    _run(ses, 0.5, now + 0.5)
    path = ses.rec_stop()
    perf = Performance.load(path)
    assert perf.frames / perf.fps == pytest.approx(2.0, abs=0.05)
    assert os.path.exists(path[:-4] + ".json") and not [f for f in os.listdir(tmp_path) if f.startswith(".")]


def test_app_take_helpers(tmp_path):
    from myrmex.app import controllers as C
    ses = _session(tmp_path)
    ses.rec_start()
    _run(ses, 1.2, 0.0)
    path = ses.rec_stop()
    info = C.take_info(path)
    assert info["frames"] == 36 and info["fps"] == 30.0 and info["seconds"] == pytest.approx(1.2)
    assert info["variant"] == "spear" and C.fmt_seconds(75.25) == "1:15.2" and C.fmt_seconds(3) == "0:03.0"
    assert C.list_takes(str(tmp_path)) == [path] and C.last_take(str(tmp_path)) == path
    assert C.take_videos(path) == []
    video = path[:-4] + "_1080x1920.mp4"
    open(video, "wb").close()
    assert C.take_videos(path) == [video] and C.list_takes(str(tmp_path)) == [path]


class FakeEngine:
    """The engine as the app sees it (no ports opened)."""

    def __init__(self, tmp_path):
        self.ses = _session(tmp_path)
        self.running = False
        self.now = 0.0

    def start(self):
        self.running = True
        return True

    def stop(self):
        self.running = False

    def rec_start(self):
        return self.ses.rec_start()

    def rec_stop(self):
        return self.ses.rec_stop()

    def rec_status(self):
        return self.ses.rec_status()

    def status(self):
        return {}

    def control(self, *a):
        pass

    def glove_snapshot(self):
        return None

    def step(self, seconds):
        self.now = _run(self.ses, seconds, self.now)


def test_app_rec_button_and_takes_list(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from myrmex.app import takes_tab as TK
    from myrmex.app import window as W
    from myrmex.app.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    s = AppSettings()
    s.start_engine_on_launch = s.open_blender_on_start = False
    s.record_dir = str(tmp_path / "takes")
    w = W.MainWindow(s)
    w.timer.stop()
    eng = FakeEngine(tmp_path / "takes")
    w.engine = eng
    w.start_engine = lambda: eng.start()
    assert w.btn_rec_rail.text() == "●" and w.list_takes.count() == 0
    TK.toggle_rec(w)                                       # REC with the engine stopped: it starts
    assert eng.running and eng.rec_status()["on"]
    eng.step(2.0)
    TK.refresh_rec(w)
    assert w.btn_rec_rail.property("recording") and w.btn_rec_rail.text() == "■"
    assert w.lbl_rec_rail.text() == "0:02" and w.btn_rec.text().endswith("STOP take")
    assert "Recording  0:02.0" in w.lbl_rec.text()
    TK.toggle_rec(w)                                       # STOP: saved, listed, selected
    TK.refresh_rec(w)
    assert not w.btn_rec_rail.property("recording") and w.list_takes.count() == 1
    path = eng.rec_status()["last"]
    assert TK.selected_take(w) == path and "0:02.0" in w.list_takes.item(0).text()
    assert "Saved:" in w.lbl_rec.text()
    w.close()


def test_app_render_refuses_while_recording_and_logs_the_length(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from myrmex.app import controllers as C
    from myrmex.app import takes_tab as TK
    from myrmex.app import window as W
    from myrmex.app.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    s = AppSettings()
    s.start_engine_on_launch = s.open_blender_on_start = False
    s.record_dir = str(tmp_path / "takes")
    w = W.MainWindow(s)
    w.timer.stop()
    eng = FakeEngine(tmp_path / "takes")
    w.engine = eng
    eng.running = True
    eng.rec_start()
    eng.step(1.0)
    first = eng.rec_stop()
    TK.fill_takes(w, select=first)
    monkeypatch.setattr(C, "find_blender", lambda hint="": "/bin/true")
    logs = []
    w.log = logs.append
    eng.rec_start()
    w.open_take(True)                                      # recording: no render
    assert any("STOP it first" in m for m in logs)
    eng.step(0.5)
    eng.rec_stop()
    logs.clear()
    w.open_take(True)                                      # the selected take, with its length
    assert any("rendering " + os.path.basename(first) in m and "0:01.0 = 30 frames at 30 fps" in m for m in logs)
    for p in getattr(w, "take_procs", []):
        p.waitForFinished(3000)
    w.close()
