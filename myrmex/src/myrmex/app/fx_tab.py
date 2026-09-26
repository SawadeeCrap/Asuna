"""The FX page: Myrmex FX drawn by Blender itself - the organism's afterimages, traces, echo and glow, on black."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QRadioButton, QSlider, QVBoxLayout, QWidget)

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


__all__ = ["fx_tab", "refresh_fx", "RACK"]
