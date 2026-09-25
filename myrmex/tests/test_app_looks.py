"""Looks from the app: Save current look -> Blender saves it, the Look list under Character type, loading a look.

A small stand-in for Blender speaks the same stdin / stdout protocol as blender/myrmex_blender/control.py.
"""
import json
import os
import sys
import time

import pytest

pytest.importorskip("PySide6")

FAKE_BLENDER = r'''
import json, os, sys
def reply(d): print("MYRMEX_REPLY " + json.dumps(d), flush=True)
print("Blender 5.2 (stand-in)", flush=True)
reply({"cmd": "hello", "ok": True})
for line in sys.stdin:
    d = json.loads(line)
    if d["cmd"] == "save_look":
        os.makedirs(os.path.dirname(d["path"]), exist_ok=True)
        open(d["path"], "wb").write(b"look")
        reply({"cmd": "save_look", "ok": True, "kind": d["kind"], "path": d["path"]})
    elif d["cmd"] == "load_look":
        reply({"cmd": "load_look", "ok": True, "kind": d["kind"], "path": d["path"], "autosaved": "x", "take": False})
'''


def _pump(app, until, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if until():
            return True
        time.sleep(0.02)
    return False


def test_looks_are_saved_listed_and_loaded_through_blender(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication, QMessageBox

    from myrmex.app import controllers as C
    from myrmex.app import window as W
    from myrmex.app.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    legacy = tmp_path / "Myrmex" / "looks" / "cyber_hive.blend"          # the older single look is listed too
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"old")
    s = AppSettings()
    s.backend, s.start_engine_on_launch, s.open_blender_on_start = "cyber_hive", False, False
    w = W.MainWindow(s)
    items = [w.cmb_look.itemText(i) for i in range(w.cmb_look.count())]
    assert items[0].startswith("Default studio") and "My look" in items and w.cmb_look.currentText() == "My look"

    class Ask:                                                            # the name dialog / confirmations
        @staticmethod
        def getText(*a, **k):
            return "Neon", True

    class Box:
        StandardButton = QMessageBox.StandardButton

        @staticmethod
        def question(*a, **k):
            return QMessageBox.StandardButton.Yes

        @staticmethod
        def information(*a, **k):
            Box.told = True
    monkeypatch.setattr(W, "QInputDialog", Ask)
    monkeypatch.setattr(W, "QMessageBox", Box)
    w._save_current_look()                                               # no Blender opened by Myrmex yet
    assert getattr(Box, "told", False)

    p = QProcess(w)                                                      # "Open in Blender"
    p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
    p.readyReadStandardOutput.connect(lambda: w._pipe(p, "blender"))
    p.setProperty("myrmex_variant", "cyber_hive")
    p.start(sys.executable, ["-u", "-c", FAKE_BLENDER])
    w.blender_proc = p
    try:
        assert _pump(app, lambda: bool(p.property("myrmex_control")))
        w._save_current_look()                                           # -> Blender saves "Neon"
        neon = C.look_path("cyber_hive", "Neon")
        assert _pump(app, lambda: w.s.looks.get("cyber_hive") == neon)
        assert os.path.isfile(neon) and w.cmb_look.currentText() == "Neon"
        assert [n for n, _ in C.list_looks("cyber_hive")] == ["My look", "Neon"]
        w.cmb_look.setCurrentIndex(w.cmb_look.findText("My look"))      # choosing a look loads it in Blender
        w._look_chosen()
        assert _pump(app, lambda: "Blender shows the look 'My look'" in w.logbox.toPlainText())
        assert w.s.looks["cyber_hive"] == str(legacy)
        assert C.look_file("cyber_hive", w.s.looks) == str(legacy)
        cmd, _ = C.blender_live_command("blender", "", 9101, "cyber_hive", True, w.s.looks)
        assert str(legacy) in cmd                                        # Open in Blender uses the chosen look
        w.cmb_look.setCurrentIndex(w.cmb_look.findText("Neon"))
        w._delete_look()
        assert not os.path.exists(neon) and w.s.looks["cyber_hive"] == str(legacy)   # the chosen one stays
        w.cmb_look.setCurrentIndex(w.cmb_look.findText("My look"))
        w._delete_look()
        assert not legacy.exists() and w.s.looks["cyber_hive"] == ""
        assert C.look_file("cyber_hive", w.s.looks) == "" and w.cmb_look.currentText().startswith("Default studio")
    finally:
        p.closeWriteChannel()
        p.waitForFinished(3000)
        w.close()
