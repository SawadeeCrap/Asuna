"""The FX page: Myrmex FX drawn by Blender itself - the organism's afterimages, traces, echo and glow, on black."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QRadioButton, QSlider, QVBoxLayout, QWidget)

from ..realtime.fx import DEFAULT_RACK, LABELS, PRESET_NAMES, RACK, rack_from_preset
from . import controllers as C

PRESET_NOTES = {
    "Afterimage": "the organism leaves copies of itself behind as it moves - made of what it is made of",
    "Echo": "dense copies melt into one continuous echo of the motion",
    "Phantom": "long copies dissolving slowly, a deep motion echo",
    "Trace": "thin traces of light from its extremities, in its own accent colour, a few copies",
    "Clean": "the organism alone, a soft glow of its highlights",
}
GROUPS = (("Afterimages and traces", ("ghosts", "ghost_life", "ghost_density", "ribbons")),
          ("Echo and glow", ("trails", "bloom", "react")),
          ("Colour (black stays black)", ("exposure", "contrast", "saturation")))


def fx_tab(win) -> QWidget:
    d = C.fx_settings(win.s)
    win.s.fx = d
    w = QWidget()
    v = QVBoxLayout(w)
    intro = QLabel("Blender draws the effects itself, only on the organism and in its own tones, always on black. "
                   "It leaves copies of itself as it moves - its own material, fading and dissolving - light traces "
                   "follow its extremities in its accent colour, its motion leaves an echo and its highlights glow. "
                   "You see it live in Blender's camera view, and every take renders with exactly the same effects.")
    intro.setWordWrap(True)
    intro.setProperty("muted", True)
    v.addWidget(intro)
    # --- the switch, the preset, the format
    box = QGroupBox("Effects")
    form = QFormLayout(box)
    chk = QCheckBox("Myrmex FX in Blender")
    chk.setChecked(bool(d["enabled"]))
    chk.toggled.connect(lambda on: _set(win, "enabled", on))
    form.addRow(chk)
    top = QHBoxLayout()
    win.cmb_fx_preset = QComboBox()
    for name in PRESET_NAMES:
        win.cmb_fx_preset.addItem(name, name)
    win.cmb_fx_preset.addItem("Custom", "")
    win.cmb_fx_preset.setCurrentIndex(max(0, win.cmb_fx_preset.findData(d.get("preset") or "")))
    top.addWidget(win.cmb_fx_preset, 1)
    form.addRow("Preset", top)
    note = QLabel(PRESET_NOTES.get(d.get("preset"), ""))
    note.setWordWrap(True)
    note.setProperty("muted", True)
    form.addRow(note)
    fmt = QHBoxLayout()
    grp = QButtonGroup(w)
    for vert, label in ((False, "Horizontal 1920 × 1080"), (True, "Vertical 1080 × 1920 (Reels, Shorts, TikTok)")):
        rb = QRadioButton(label)
        rb.setChecked(bool(d.get("vertical")) == vert)
        rb.toggled.connect(lambda on, x=vert: on and _format(win, x))
        grp.addButton(rb)
        fmt.addWidget(rb)
    fmt.addStretch(1)
    form.addRow("Format", fmt)
    size = QComboBox()
    for s, label in ((1.0, "100% (sharp)"), (0.75, "75%"), (0.5, "50% (lightest)")):
        size.addItem(label, s)
    size.setCurrentIndex(max(0, size.findData(float(d.get("preview", 1.0)))))
    size.currentIndexChanged.connect(lambda *_: _set(win, "preview", size.currentData()))
    form.addRow("Live picture", size)
    mon = QCheckBox("Show the picture effects live in Blender (off: afterimages and ribbons only - lightest; "
                    "renders always get everything)")
    mon.setChecked(bool(d.get("monitor", True)))
    mon.toggled.connect(lambda on: _set(win, "monitor", on))
    form.addRow(mon)
    rep = QCheckBox("Renders repeat the knob moves recorded in the take (off: the settings on this page)")
    rep.setChecked(bool(d.get("replay", True)))
    rep.toggled.connect(lambda on: _set(win, "replay", on))
    form.addRow(rep)
    v.addWidget(box)
    v.addWidget(_stage_box(win))
    # --- the sliders
    win.fx_sliders = {}
    for title, keys in GROUPS:
        box = QGroupBox(title)
        grid = QGridLayout(box)
        for i, k in enumerate(keys):
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(0, 1000)
            sl.setValue(int(1000 * float(d["rack"].get(k, DEFAULT_RACK[k]))))
            sl.valueChanged.connect(lambda val, key=k: _rack(win, key, val / 1000.0, custom=True))
            sl.setToolTip(f"MIDI: map a knob to fx_{k}")
            grid.addWidget(QLabel(LABELS[k]), i // 2, (i % 2) * 2)
            grid.addWidget(sl, i // 2, (i % 2) * 2 + 1)
            win.fx_sliders[k] = sl
        v.addWidget(box)

    def preset(*_):
        name = win.cmb_fx_preset.currentData()
        note.setText(PRESET_NOTES.get(name, "your own mix"))
        if not name:
            win.s.fx["preset"] = ""
            win.s.save()
            return
        rack = rack_from_preset(name)
        win.s.fx["preset"], win.s.fx["rack"] = name, rack
        _show_rack(win, rack)
        win.s.save()
        win.engine.fx_config(preset=name, rack=rack)
        _blender(win)
    win.cmb_fx_preset.currentIndexChanged.connect(preset)
    hint = QLabel("MIDI: map knobs to fx_ghosts, fx_trails, fx_bloom … on the MIDI page. Open in Blender "
                  "shows the camera view (in Rendered) as the finished picture; if Blender cannot keep up, "
                  "the picture gets smaller or pauses by itself - set Live picture lower for a smoother view. "
                  "Renders: Takes page - the size follows the format chosen here.")
    hint.setWordWrap(True)
    hint.setProperty("muted", True)
    v.addWidget(hint)
    win.lbl_fx = QLabel("")
    win.lbl_fx.setProperty("muted", True)
    v.addWidget(win.lbl_fx)
    v.addStretch(1)
    return w


FILE = "\x00file"                          # the "your own HDRI file…" item


def _stage_box(win) -> QGroupBox:
    """The light on the organism: the HDRI you like in Material Preview, on black - for every organism."""
    st = C.stage_settings(win.s)
    win.s.stage = st
    box = QGroupBox("Light on the organism - the background stays black")
    form = QFormLayout(box)
    note = QLabel("The HDRI Blender's Material Preview lights with - here it lights the organism and shows in "
                  "its reflections, while the camera sees pure black: live view, FX, Syphon and renders, every "
                  "organism. In Blender: Myrmex sidebar > Myrmex Stage > Use This View's Lighting takes the light "
                  "you set up in a view (its HDRI, strength, rotation) and brings it here.")
    note.setWordWrap(True)
    note.setProperty("muted", True)
    form.addRow(note)
    win.cmb_stage = QComboBox()
    _fill_hdris(win, st["hdri"])
    win.cmb_stage.currentIndexChanged.connect(lambda *_: _stage_hdri(win))
    form.addRow("HDRI", win.cmb_stage)
    win.stage_sliders = {}
    for key, label, top, scale, fmt in (("strength", "Strength", 400, 100.0, "{:.2f}"),
                                        ("rotation", "Rotation", 360, 1.0, "{:.0f}°")):
        row = QHBoxLayout()
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, top)
        sl.setValue(int(round(min(top / scale, float(st[key])) * scale)))
        val = QLabel(fmt.format(st[key]))
        val.setMinimumWidth(44)
        sl.valueChanged.connect(lambda x, k=key, f=fmt, sc=scale, lb=val: (lb.setText(f.format(x / sc)),
                                                                        _stage_set(win, **{k: x / sc})))
        row.addWidget(sl, 1)
        row.addWidget(val)
        form.addRow(label, row)
        win.stage_sliders[key] = (sl, val, scale, fmt)
    win.chk_stage_lamps = QCheckBox("Studio lamps too (off: the HDRI alone, like Material Preview)")
    win.chk_stage_lamps.setChecked(bool(st["lights"]))
    win.chk_stage_lamps.toggled.connect(lambda on: _stage_set(win, lights=bool(on)))
    form.addRow(win.chk_stage_lamps)
    _stage_enable(win)
    return box


def _fill_hdris(win, hdri: str) -> None:
    cmb = win.cmb_stage
    cmb.blockSignals(True)
    cmb.clear()
    for name, label in C.HDRIS:
        cmb.addItem(label, name)
    if hdri and cmb.findData(hdri) < 0:                  # one of yours (installed in Blender, or a file)
        cmb.addItem(C.hdri_label(hdri), hdri)
    cmb.addItem("Your HDRI file…", FILE)
    cmb.setCurrentIndex(max(0, cmb.findData(hdri)))
    cmb.blockSignals(False)


def _stage_hdri(win) -> None:
    hdri = win.cmb_stage.currentData()
    if hdri == FILE:
        path = QFileDialog.getOpenFileName(win, "HDRI", os.path.expanduser("~"), "HDRI (*.exr *.hdr)")[0]
        if not path:
            _fill_hdris(win, win.s.stage.get("hdri", ""))        # (cancelled: as it was)
            return
        _fill_hdris(win, path)
        hdri = path
    _stage_set(win, hdri=hdri or "")
    _stage_enable(win)


def _stage_enable(win) -> None:
    on = bool(win.s.stage.get("hdri"))
    for sl, val, *_ in win.stage_sliders.values():
        sl.setEnabled(on)
        val.setEnabled(on)


def _stage_set(win, **kw) -> None:
    win.s.stage = {**C.stage_settings(win.s), **kw}
    win.s.save()
    _stage_blender(win)


def _stage_blender(win, skip=None) -> None:
    """Every Blender the app opened (live or a take) gets the light at once."""
    from PySide6.QtCore import QProcess
    msg = {"cmd": "stage", **C.stage_settings(win.s)}
    for p in [getattr(win, "blender_proc", None)] + list(getattr(win, "take_procs", [])):
        if p is not None and p is not skip and p.state() != QProcess.ProcessState.NotRunning and \
                p.property("myrmex_control"):
            win._blender_send(p, msg)


def on_blender_stage(win, p, d: dict) -> None:
    """Blender's answer to the light: set there (Use This View's Lighting, its Stage panel) -> kept here for
    every organism, and the other Blenders follow."""
    if not d.get("ok"):
        win.log(f"! Blender: the light on the organism: {d.get('error', 'failed')}")
        return
    if not d.get("from_blender"):
        return
    win.s.stage = {**C.stage_settings(win.s), **{k: d[k] for k in C.STAGE if k in d}}
    st = win.s.stage = C.stage_settings(win.s)
    win.s.save()
    if getattr(win, "cmb_stage", None) is not None:
        _fill_hdris(win, st["hdri"])
        for key, (sl, val, scale, fmt) in win.stage_sliders.items():
            sl.blockSignals(True)
            sl.setValue(int(round(min(sl.maximum() / scale, float(st[key])) * scale)))
            sl.blockSignals(False)
            val.setText(fmt.format(st[key]))
        win.chk_stage_lamps.blockSignals(True)
        win.chk_stage_lamps.setChecked(bool(st["lights"]))
        win.chk_stage_lamps.blockSignals(False)
        _stage_enable(win)
    _stage_blender(win, skip=p)
    if st["hdri"]:
        win.log(f"light on the organism from Blender: {C.hdri_label(st['hdri'])}, strength {st['strength']:.2f}, "
                f"rotation {st['rotation']:.0f}°, lamps {'on' if st['lights'] else 'off'} - the background stays black")
    else:
        win.log("light on the organism from Blender: the studio panels - the background stays black")


def _show_rack(win, rack: dict) -> None:
    for key, sl in win.fx_sliders.items():
        sl.blockSignals(True)
        sl.setValue(int(1000 * rack.get(key, DEFAULT_RACK[key])))
        sl.blockSignals(False)


def _set(win, key: str, value) -> None:
    win.s.fx[key] = value
    win.s.save()
    win.engine.fx_config(**{key: value})
    _blender(win)


def _rack(win, key: str, value: float, custom: bool = False) -> None:
    win.s.fx.setdefault("rack", dict(DEFAULT_RACK))[key] = float(value)
    if custom and win.s.fx.get("preset"):                 # a moved slider makes it your own mix
        win.s.fx["preset"] = ""
        win.cmb_fx_preset.blockSignals(True)
        win.cmb_fx_preset.setCurrentIndex(win.cmb_fx_preset.findData(""))
        win.cmb_fx_preset.blockSignals(False)
    win.s.save()
    win.engine.fx_config(rack={key: float(value)}, preset=win.s.fx.get("preset", ""))
    _blender(win)


def _format(win, vertical: bool) -> None:
    """Horizontal / vertical: the live picture, Syphon and (as the default) the render size."""
    _set(win, "vertical", bool(vertical))
    size = "1080x1920" if vertical else "1920x1080"
    win.s.render_size = size
    cmb = getattr(win, "cmb_rsize", None)
    if cmb is not None:
        cmb.setCurrentText(size)
    win.s.save()


def _blender(win) -> None:
    """Every Blender the app opened (live or a take) follows the page at once."""
    from PySide6.QtCore import QProcess
    msg = {"cmd": "fx", **C.fx_blender(win.s)}
    for p in [getattr(win, "blender_proc", None)] + list(getattr(win, "take_procs", [])):
        if p is not None and p.state() != QProcess.ProcessState.NotRunning and p.property("myrmex_control"):
            win._blender_send(p, msg)


def refresh_fx(win) -> None:
    st = win.engine.status() or {}
    fx = st.get("fx")
    if fx is None:
        win.lbl_fx.setText("")
        return
    r = fx.get("rack", {})
    win.lbl_fx.setText(("on" if fx.get("enabled") else "off") + f" · {fx.get('preset') or 'custom'} · "
                       f"{'1080 × 1920' if fx.get('vertical') else '1920 × 1080'} · afterimages "
                       f"{r.get('ghosts', 0):.2f} · trails {r.get('trails', 0):.2f}")


__all__ = ["fx_tab", "refresh_fx", "on_blender_stage", "RACK"]
