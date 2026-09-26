"""The TouchDesigner page: the link, Blender's picture (Syphon), the effect rack, the output and recording."""
from __future__ import annotations

import os

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from ..realtime.touch import FX, FX_DEFAULTS, PRESET_NAMES, PRESETS
from . import controllers as C

FX_LABELS = {"bloom": "Bloom", "trails": "Trails (echoes)", "chroma": "Chromatic aberration", "glitch": "Glitch",
             "warp": "Liquid warp", "shock": "Shockwaves", "kaleido": "Kaleidoscope", "edges": "Neon edges",
             "grain": "Grain", "vignette": "Vignette", "hud": "HUD", "react": "Music moves the effects",
             "exposure": "Exposure", "contrast": "Contrast", "saturation": "Saturation", "hue": "Colour drift",
             "mix": "Trails mix"}
PRESET_NOTES = {"Clean": "a clean cinematic picture: soft glow, grain, vignette",
                "Neon Trails": "glowing echoes of every movement, colours drifting",
                "Glitch Storm": "the image tears on hits, heavy aberration, a scanner HUD",
                "Dream": "long soft trails, a liquid, dreamy glow",
                "Kaleido": "a kaleidoscope round the organism",
                "Scanner": "a neon outline and a targeting HUD, cold and technical",
                "Liquid": "the picture flows like liquid round the organism"}
SIZES = (("1280x720", "1280 × 720 (TouchDesigner Non-Commercial)"), ("960x540", "960 × 540 (lighter)"),
         ("720x1280", "720 × 1280 vertical"), ("1920x1080", "1920 × 1080 (TouchDesigner Commercial)"))


def td_tab(win) -> QWidget:
    d = C.td_settings(win.s)
    win.s.td = d
    w = QWidget()
    v = QVBoxLayout(w)
    intro = QLabel("Myrmex sends TouchDesigner everything that happens - the beat, the kick, the organism's form, "
                   "where it is on screen, its impacts, the camera cuts - and Blender sends its picture over Syphon. "
                   "The effects in TouchDesigner follow the music and the organism, and this page drives them.")
    intro.setWordWrap(True)
    intro.setProperty("muted", True)
    v.addWidget(intro)
    # --- the link
    box = QGroupBox("Link")
    form = QFormLayout(box)
    chk = QCheckBox("Sync with TouchDesigner")
    chk.setChecked(bool(d["enabled"]))
    chk.toggled.connect(lambda on: _set(win, "enabled", on))
    form.addRow(chk)
    win.lbl_td = QLabel("off")
    form.addRow("Status", win.lbl_td)
    row = QHBoxLayout()
    b = QPushButton("Set up TouchDesigner…")
    b.setToolTip("Copies the Myrmex FX builder to ~/Myrmex/touchdesigner and the line to paste into TD's Textport")
    b.clicked.connect(lambda: setup_td(win))
    row.addWidget(b)
    b = QPushButton("Open TouchDesigner")
    b.clicked.connect(lambda: open_td(win))
    row.addWidget(b)
    row.addStretch(1)
    form.addRow(row)
    host = QLineEdit(str(d["host"]))
    host.editingFinished.connect(lambda: _set(win, "host", host.text().strip() or "127.0.0.1"))
    ports = QHBoxLayout()
    for key, label in (("port", "data"), ("text_port", "text")):
        sp = QSpinBox()
        sp.setRange(1024, 65535)
        sp.setValue(int(d[key]))
        sp.valueChanged.connect(lambda val, k=key: _set(win, k, int(val)))
        ports.addWidget(QLabel(label))
        ports.addWidget(sp)
    ports.addStretch(1)
    form.addRow("TouchDesigner at", host)
    form.addRow("Ports", ports)
    v.addWidget(box)
    # --- Blender's picture
    box = QGroupBox("Picture from Blender (Syphon)")
    form = QFormLayout(box)
    chk = QCheckBox('Send Blender\'s camera to TouchDesigner (Syphon "Myrmex")')
    chk.setChecked(bool(d["syphon"]))
    chk.toggled.connect(lambda on: (_set(win, "syphon", on), _blender_syphon(win)))
    form.addRow(chk)
    cmb = QComboBox()
    for key, label in SIZES:
        cmb.addItem(label, key)
    cmb.setCurrentIndex(max(0, cmb.findData(d["syphon_size"])))
    cmb.currentIndexChanged.connect(lambda *_: (_set(win, "syphon_size", cmb.currentData()), _blender_syphon(win)))
    form.addRow("Size", cmb)
    fps = QComboBox()
    for f in (60, 30):
        fps.addItem(f"{f} fps", f)
    fps.setCurrentIndex(max(0, fps.findData(int(d["syphon_fps"]))))
    fps.currentIndexChanged.connect(lambda *_: (_set(win, "syphon_fps", fps.currentData()), _blender_syphon(win)))
    form.addRow("Rate", fps)
    chk = QCheckBox("Transparent background: only the organism (TouchDesigner draws what is behind it)")
    chk.setChecked(bool(d["alpha"]))
    chk.toggled.connect(lambda on: (_set(win, "alpha", on), _blender_syphon(win)))
    form.addRow(chk)
    win.lbl_td_syphon = QLabel("Blender sends it when it opens with the link on")
    win.lbl_td_syphon.setProperty("muted", True)
    form.addRow(win.lbl_td_syphon)
    v.addWidget(box)
    # --- effects
    box = QGroupBox("Effects in TouchDesigner")
    lay = QVBoxLayout(box)
    top = QHBoxLayout()
    win.cmb_td_preset = QComboBox()
    for name in PRESET_NAMES:
        win.cmb_td_preset.addItem(name, name)
    win.cmb_td_preset.setCurrentIndex(max(0, win.cmb_td_preset.findData(d.get("preset"))))
    note = QLabel(PRESET_NOTES.get(d.get("preset"), ""))
    note.setWordWrap(True)
    note.setProperty("muted", True)
    top.addWidget(QLabel("Preset"))
    top.addWidget(win.cmb_td_preset, 1)
    lay.addLayout(top)
    lay.addWidget(note)
    grid = QGridLayout()
    win.td_sliders = {}
    for i, k in enumerate(FX):
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, 1000)
        sl.setValue(int(1000 * float(d["fx"].get(k, FX_DEFAULTS[k]))))
        sl.valueChanged.connect(lambda val, key=k: _fx(win, key, val / 1000.0))
        sl.setToolTip(f"MIDI: map a knob to td_{k}")
        grid.addWidget(QLabel(FX_LABELS[k]), i // 2, (i % 2) * 2)
        grid.addWidget(sl, i // 2, (i % 2) * 2 + 1)
        win.td_sliders[k] = sl

    def preset(*_):
        name = win.cmb_td_preset.currentData()
        note.setText(PRESET_NOTES.get(name, ""))
        fx = {**win.s.td.get("fx", FX_DEFAULTS), **PRESETS[name]}
        win.s.td["preset"], win.s.td["fx"] = name, fx
        for key, sl in win.td_sliders.items():
            sl.blockSignals(True)
            sl.setValue(int(1000 * fx.get(key, FX_DEFAULTS[key])))
            sl.blockSignals(False)
        win.s.save()
        win.engine.td_config(preset=name, fx=fx)
    win.cmb_td_preset.currentIndexChanged.connect(preset)
    lay.addLayout(grid)
    v.addWidget(box)
    # --- output
    box = QGroupBox("Output")
    form = QFormLayout(box)
    row = QHBoxLayout()
    chk = QCheckBox("Fullscreen output window on display")
    chk.setChecked(bool(d["window"]))
    chk.toggled.connect(lambda on: _set(win, "window", on))
    mon = QSpinBox()
    mon.setRange(0, 4)
    mon.setValue(int(d["monitor"]))
    mon.valueChanged.connect(lambda val: _set(win, "monitor", int(val)))
    row.addWidget(chk)
    row.addWidget(mon)
    row.addStretch(1)
    form.addRow(row)
    chk = QCheckBox("Record what TouchDesigner shows (ProRes .mov in ~/Myrmex/td_recordings)")
    chk.setChecked(False)
    chk.toggled.connect(lambda on: _set(win, "rec", on, save=False))
    form.addRow(chk)
    lic = QLabel("TouchDesigner Non-Commercial: pictures up to 1280 × 1280; recordings are ProRes (H.264 needs a "
                 "Commercial licence). MIDI knobs can drive every slider (targets td_bloom, td_trails …).")
    lic.setWordWrap(True)
    lic.setProperty("muted", True)
    form.addRow(lic)
    v.addWidget(box)
    v.addStretch(1)
    return w


def _set(win, key: str, value, save: bool = True) -> None:
    win.s.td[key] = value
    if save:
        win.s.save()
    win.engine.td_config(**{key: value})


def _fx(win, key: str, value: float) -> None:
    win.s.td.setdefault("fx", dict(FX_DEFAULTS))[key] = value
    win.s.save()
    win.engine.td_config(fx={key: value})


def _blender_syphon(win) -> None:
    """Tell an open Blender to start / change / stop its Syphon picture."""
    p = win._blender_for()
    if p is not None:
        win._blender_send(p, {"cmd": "syphon", **C.syphon_config(win.s)})


def setup_td(win) -> None:
    try:
        path = C.install_td_files()
    except OSError as e:
        QMessageBox.warning(win, "Myrmex", f"Could not copy the TouchDesigner builder: {e}")
        return
    line = C.td_build_command(path)
    QGuiApplication.clipboard().setText(line)
    win.log(f"TouchDesigner builder ready: {path} (the line to paste is in the clipboard)")
    QMessageBox.information(
        win, "Set up TouchDesigner",
        "The line that builds the Myrmex FX network is in your clipboard.\n\n"
        "1. Open TouchDesigner (a new project is fine).\n"
        "2. Open the Textport: Dialogs → Textport and DATs (or Alt+T).\n"
        "3. Paste it (⌘V) and press Enter.\n\n"
        "TouchDesigner builds /project1/myrmex and saves it as\n"
        f"{C.TD_PROJECT}.\n"
        "From then on “Open TouchDesigner” opens that project directly.\n\n"
        f"{line}")


def open_td(win) -> None:
    project = C.TD_PROJECT if os.path.exists(C.TD_PROJECT) else None
    cmd = C.open_touchdesigner_command(project)
    if cmd is None:
        QMessageBox.warning(win, "Myrmex", "TouchDesigner was not found (macOS: /Applications/TouchDesigner.app).")
        return
    if not QProcess.startDetached(cmd[0], cmd[1:]):
        win.log("! could not open TouchDesigner")
        return
    if project is None:
        setup_td(win)
    else:
        win.log(f"opening {project} in TouchDesigner")
    if not win.s.td.get("enabled"):
        _set(win, "enabled", True)


def refresh_td(win) -> None:
    if not hasattr(win, "lbl_td"):
        return
    st = win.engine.status() or {}
    td = st.get("td") or {}
    if not win.s.td.get("enabled"):
        win.lbl_td.setText("off")
    elif not st:
        win.lbl_td.setText("start the engine")
    elif td.get("connected"):
        win.lbl_td.setText(f"TouchDesigner connected · {td.get('fps', 0):.0f} fps · {td.get('sent', 0)} packets sent")
    else:
        win.lbl_td.setText(f"sending ({td.get('sent', 0)} packets) · waiting for TouchDesigner with Myrmex FX"
                           + (f" · ! {td['error']}" if td.get("error") else ""))


def on_blender_syphon(win, d: dict) -> None:
    """A Blender reply about its Syphon picture."""
    if not hasattr(win, "lbl_td_syphon"):
        return
    if d.get("on"):
        size = d.get("size") or ["?", "?"]
        win.lbl_td_syphon.setText(f'Blender sends "{d.get("name")}" · {size[0]} × {size[1]} · {d.get("fps", 60):.0f} fps')
    else:
        win.lbl_td_syphon.setText("Blender: Syphon off" + (f" ({d['error']})" if d.get("error") else ""))


__all__ = ["td_tab", "refresh_td", "setup_td", "open_td", "on_blender_syphon"]
