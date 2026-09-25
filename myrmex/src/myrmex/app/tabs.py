"""Extra tabs of the Myrmex window: Creature, manual Camera, MIDI (monitor + fine mapping)."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSlider, QSpinBox,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..creature.config import PARAMS
from ..creature.engine import EVENTS
from ..creature.colony import ColonyEngine
from ..creature.hive import HiveEngine
from ..creature.polyalloy import PolyalloyEngine

POLY_EVENTS = tuple(e for e in PolyalloyEngine.EVENTS if e not in EVENTS)   # obstacle, impulse, pressure, turbulence
COLONY_EVENTS = tuple(e for e in ColonyEngine.EVENTS if e not in EVENTS + POLY_EVENTS)   # split, merge, wave, hunt, perch
HIVE_EVENTS = tuple(e for e in HiveEngine.EVENTS if e not in EVENTS + POLY_EVENTS + COLONY_EVENTS)  # build, recall
OSSEOUS_EVENTS = ("STRIKE", "OSSIFY", "QUILLS")
from ..realtime.midimap import CURVES, NOTE_MODES
from . import controllers as C

CAMERA_SHOTS = C.SHOTS[1:]
TARGETS = (list(PARAMS) + ["energy", "stride", "sway", "style", "hold", "cam_mode", "cam_distance", "cam_height",
                           "cam_orbit", "cam_lens", "cam_smooth", "camera", "pose", "flourish", "creature_debug"] +
           [f"camera:{k}" for k in CAMERA_SHOTS] + [f"creature:{e.lower()}" for e in EVENTS + POLY_EVENTS + COLONY_EVENTS + HIVE_EVENTS + OSSEOUS_EVENTS] +
           ["kick", "snare", "hats", "perc", "bass", "melody", "harmony", "fx"])


def creature_tab(win) -> QWidget:
    from .window import Knob
    w = QWidget()
    v = QVBoxLayout(w)
    v.addWidget(QLabel("Choose the organism on the Character page: Black Nanomaterial (v1), Mimetic Polyalloy (v2, "
                       "flying), Polyalloy Colony (v3, flock), Polyalloy Hive (v4) or their bony Osseous versions "
                       "(v5–v7: bone-link skeletons, scutes, claws, blades, strikes). Auto = the organism decides "
                       "from its behaviour and the music; any knob can also be a MIDI CC. kick mode · obstacle rate · "
                       "altitude: v2–v4; swarm · armor · mechanism · hunt: v3–v4; architecture · pattern · nanoswarm · "
                       "memory: v4."))
    box = QGroupBox("Parameters")
    grid = QGridLayout(box)
    win.cknobs = {}
    for i, p in enumerate(PARAMS):
        kb = Knob(p, win.s.creature_params.get(p), win._creature_knob)
        win.cknobs[p] = kb
        grid.addWidget(QLabel(p.replace("_", " ")), i // 2, (i % 2) * 2)
        grid.addWidget(kb, i // 2, (i % 2) * 2 + 1)
    v.addWidget(box)
    box = QGroupBox("Reconfiguration events")
    row = QHBoxLayout(box)
    for e in EVENTS:
        b = QPushButton(e.replace("_", " ").title())
        b.clicked.connect(lambda _=False, n=e: win.engine.trigger(f"creature:{n.lower()}"))
        row.addWidget(b)
    v.addWidget(box)
    for title, events in (("Physical events (v2, v3) — the kick does one of these by itself (kick mode)", POLY_EVENTS),
                          ("Colony (v3, v4) — flock, hardening wave, prey, landing", COLONY_EVENTS),
                          ("Hive (v4, v7) — living architecture: build a structure, call the material back", HIVE_EVENTS),
                          ("Osseous (v5–v7) — lunge and knock away · ossify in a wave · quill volley (v7)",
                           OSSEOUS_EVENTS)):
        box = QGroupBox(title)
        row = QHBoxLayout(box)
        for e in events:
            b = QPushButton(e.title())
            b.clicked.connect(lambda _=False, n=e: win.engine.trigger(f"creature:{n.lower()}"))
            row.addWidget(b)
        v.addWidget(box)
    dbg = QCheckBox("Debug: show the internal control network (nodes + links) in Blender")
    dbg.toggled.connect(lambda on: win.engine.control("creature_debug", 1.0 if on else 0.0))
    v.addWidget(dbg)
    v.addStretch(1)
    return w


def camera_group(win) -> QGroupBox:
    cc = win.s.camera_controls
    box = QGroupBox("Camera control (keys 1–9 = shots, 0 = automatic)")
    v = QVBoxLayout(box)
    win.cmb_cam_mode = QComboBox()
    win.cmb_cam_mode.addItems(["Automatic director", "Manual (hold the chosen shot)"])
    win.cmb_cam_mode.setCurrentIndex(1 if cc.get("cam_mode", 0.0) > 0.5 else 0)
    win.cmb_cam_mode.currentIndexChanged.connect(lambda i: win._cam_control("cam_mode", float(i)))
    v.addWidget(win.cmb_cam_mode)
    grid = QGridLayout()
    for i, k in enumerate(CAMERA_SHOTS):
        b = QPushButton(f"{i + 1}   {k.replace('_', ' ')}")
        b.setObjectName("shot")
        b.clicked.connect(lambda _=False, kind=k: win._pick_shot(kind))
        grid.addWidget(b, i // 3, i % 3)
        QShortcut(QKeySequence(str(i + 1)), win, activated=lambda kind=k: win._pick_shot(kind))
    QShortcut(QKeySequence("0"), win, activated=lambda: win.cmb_cam_mode.setCurrentIndex(0))
    v.addLayout(grid)
    form = QFormLayout()
    for key, label, default in (("cam_distance", "Distance", 0.2857), ("cam_height", "Height", 1 / 3),
                                ("cam_orbit", "Orbit", 0.5), ("cam_lens", "Lens (0 = shot default)", 0.0),
                                ("cam_smooth", "Smoothness", 0.2)):
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(0, 1000)
        sl.setValue(int(1000 * cc.get(key, default)))
        sl.valueChanged.connect(lambda val, k=key: win._cam_control(k, val / 1000.0))
        form.addRow(label, sl)
    v.addLayout(form)
    return box


def midi_tab(win) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    v.addWidget(QLabel("Incoming MIDI (and where it goes). Select a mapping row and press Learn, then move a knob / "
                       "hit a pad.\nCC: curve → invert → [min,max] → smoothing. Notes: trigger / gate / toggle / "
                       "velocity / group (music). Changes apply live; Save keeps them."))
    win.tbl_mon = QTableWidget(0, 6)
    win.tbl_mon.setHorizontalHeaderLabels(["time", "type", "ch", "#", "value", "→ target"])
    win.tbl_mon.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
    win.tbl_mon.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    win.tbl_mon.setMaximumHeight(200)
    v.addWidget(win.tbl_mon)
    win.tbl_map = QTableWidget(0, 10)
    win.tbl_map.setHorizontalHeaderLabels(["type", "ch (0=any)", "#", "target", "min", "max", "curve", "invert",
                                           "note mode", "smooth s"])
    win.tbl_map.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
    v.addWidget(win.tbl_map, 1)
    for b in win.s.midi_bindings:
        add_binding_row(win, b)
    row = QHBoxLayout()
    for label, fn in (("Add CC", lambda: add_binding_row(win, {"kind": "cc"})),
                      ("Add note", lambda: add_binding_row(win, {"kind": "note", "number": 60})),
                      ("Remove", lambda: win.tbl_map.removeRow(win.tbl_map.currentRow())),
                      ("Learn selected", lambda: learn(win)), ("Save + apply", lambda: save_bindings(win))):
        b = QPushButton(label)
        b.clicked.connect(fn)
        row.addWidget(b)
    v.addLayout(row)
    win._mon_seq = 0
    return w


def _combo(items, value):
    c = QComboBox()
    c.addItems(list(items))
    c.setEditable(False)
    if value in items:
        c.setCurrentText(value)
    return c


def add_binding_row(win, b: dict) -> None:
    t = win.tbl_map
    r = t.rowCount()
    t.insertRow(r)
    t.setCellWidget(r, 0, _combo(["cc", "note"], b.get("kind", "cc")))
    for col, key, rng, dv in ((1, "channel", (0, 16), 0), (2, "number", (0, 127), 1)):
        sp = QSpinBox()
        sp.setRange(*rng)
        sp.setValue(int(b.get(key, dv)))
        t.setCellWidget(r, col, sp)
    tg = QComboBox()
    tg.setEditable(True)
    tg.addItems(TARGETS)
    tg.setCurrentText(b.get("target", "energy"))
    t.setCellWidget(r, 3, tg)
    for col, key, dv in ((4, "lo", 0.0), (5, "hi", 1.0), (9, "smooth", 0.0)):
        sp = QDoubleSpinBox()
        sp.setRange(-10.0, 10.0)
        sp.setDecimals(3)
        sp.setSingleStep(0.05)
        sp.setValue(float(b.get(key, dv)))
        t.setCellWidget(r, col, sp)
    t.setCellWidget(r, 6, _combo(CURVES, b.get("curve", "linear")))
    inv = QCheckBox()
    inv.setChecked(bool(b.get("invert", False)))
    t.setCellWidget(r, 7, inv)
    t.setCellWidget(r, 8, _combo(NOTE_MODES, b.get("mode", "trigger")))


def read_bindings(win) -> list[dict]:
    t, out = win.tbl_map, []
    for r in range(t.rowCount()):
        g = t.cellWidget
        out.append({"kind": g(r, 0).currentText(), "channel": g(r, 1).value(), "number": g(r, 2).value(),
                    "target": g(r, 3).currentText().strip(), "lo": g(r, 4).value(), "hi": g(r, 5).value(),
                    "curve": g(r, 6).currentText(), "invert": g(r, 7).isChecked(), "mode": g(r, 8).currentText(),
                    "smooth": g(r, 9).value()})
    return out


def save_bindings(win) -> None:
    win.s.midi_bindings = read_bindings(win)
    win.s.save()
    sess = win.engine.session
    if sess is not None:
        sess.inputs.midimap.set_bindings(win.s.midi_bindings)
    win.log(f"MIDI mapping saved: {len(win.s.midi_bindings)} bindings (applied live)")


def learn(win) -> None:
    r = win.tbl_map.currentRow()
    sess = win.engine.session
    if r < 0 or sess is None:
        win.log("Learn: select a mapping row and start the engine first")
        return
    save_bindings(win)
    sess.inputs.midimap.learning = r
    win.log(f"Learn: move a knob or hit a pad for row {r + 1}…")


def refresh_midi(win) -> None:
    sess = win.engine.session
    if sess is None or not hasattr(win, "tbl_mon"):
        return
    mm = sess.inputs.midimap
    if getattr(win, "_learning_row", None) is not None and mm.learning is None:      # learned: pull it into the table
        b = mm.bindings[win._learning_row] if win._learning_row < len(mm.bindings) else None
        if b is not None:
            g = win.tbl_map.cellWidget
            g(win._learning_row, 0).setCurrentText(b.kind)
            g(win._learning_row, 1).setValue(b.channel)
            g(win._learning_row, 2).setValue(b.number)
            win.s.midi_bindings = read_bindings(win)
            win.log(f"learned: {b.kind} #{b.number} ch {b.channel or 'any'} → {b.target}")
    win._learning_row = mm.learning
    rows = [r for r in sess.inputs.monitor.rows if r[0] > win._mon_seq]
    if not rows:
        return
    win._mon_seq = rows[-1][0]
    t = win.tbl_mon
    for seq, ts, kind, ch, num, val, tg in rows[-40:]:
        t.insertRow(0)
        for col, x in enumerate((ts, kind, ch, num, val, tg)):
            t.setItem(0, col, QTableWidgetItem(str(x)))
    while t.rowCount() > 200:
        t.removeRow(t.rowCount() - 1)


# ============================================================================ Hand Glove
GLOVE_PRESETS = [("puppet", "Puppet", "The body turns with your hand, fingers are its limbs; raise / lower = height, "
                                       "left / right = steering, towards the screen = bigger and closer."),
                 ("sculpt", "Sculpt", "The hand turns the form in place; each finger blends in one of five forms."),
                 ("conductor", "Conductor", "Twist to spin it, tilt to melt or harden it, open fingers = energy; "
                                            "the hand steers and lifts."),
                 ("camera", "Camera", "The organism stays free; your hand orbits, raises and zooms the camera."),
                 ("off", "Off", "The glove's MIDI goes to the parameters (MIDI page) as before.")]


def glove_tab(win) -> QWidget:
    from PySide6.QtWidgets import QProgressBar
    from ..realtime.glove import PARAMS
    g = win.s.glove
    w = QWidget()
    v = QVBoxLayout(w)
    box = QGroupBox("Hand Glove")
    lay = QVBoxLayout(box)
    win.lbl_glove = QLabel("waiting for the glove …")
    lay.addWidget(win.lbl_glove)
    grid = QGridLayout()
    win.glove_bars = {}
    for i, p in enumerate(PARAMS):
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        bar.setFixedHeight(8)
        grid.addWidget(QLabel(p), i % 6, (i // 6) * 2)
        grid.addWidget(bar, i % 6, (i // 6) * 2 + 1)
        win.glove_bars[p] = bar
    lay.addLayout(grid)
    win.lbl_glove_ev = QLabel("")
    win.lbl_glove_ev.setProperty("muted", True)
    lay.addWidget(win.lbl_glove_ev)
    v.addWidget(box)

    box = QGroupBox("Synchronisation")
    form = QFormLayout(box)
    win.cmb_glove = QComboBox()
    for key, label, _ in GLOVE_PRESETS:
        win.cmb_glove.addItem(label, key)
    win.cmb_glove.setCurrentIndex(max(0, win.cmb_glove.findData(g.get("preset", "puppet"))))
    desc = QLabel()
    desc.setWordWrap(True)
    desc.setProperty("muted", True)

    def preset_changed(*_):
        key = win.cmb_glove.currentData()
        desc.setText(next(d for k, _, d in GLOVE_PRESETS if k == key))
        _glove_set(win, "preset", key)
    win.cmb_glove.currentIndexChanged.connect(preset_changed)
    desc.setText(next(d for k, _, d in GLOVE_PRESETS if k == win.cmb_glove.currentData()))
    form.addRow("Preset", win.cmb_glove)
    form.addRow("", desc)
    for key, label, default in (("intensity", "Intensity", 1.0), ("smoothing", "Smoothing (0 = raw, fastest)", 0.45),
                                ("sensitivity", "Gesture sensitivity", 0.5)):
        sl = QSlider(Qt.Orientation.Horizontal)
        top = 1500 if key == "intensity" else 1000
        sl.setRange(0, top)
        sl.setValue(int(1000 * g.get(key, default)))
        sl.valueChanged.connect(lambda val, k=key: _glove_set(win, k, val / 1000.0))
        form.addRow(label, sl)
    chk = QCheckBox("Gestures trigger events")
    chk.setToolTip("flick = strike / impulse · fist = harden · spread = burst · push = strike · pinch = split / merge")
    chk.setChecked(g.get("gestures", True))
    chk.toggled.connect(lambda on: _glove_set(win, "gestures", on))
    form.addRow(chk)
    chk = QCheckBox("Invert fingers (if an open hand makes the organism close)")
    chk.setChecked(g.get("invert_fingers", False))
    chk.toggled.connect(lambda on: _glove_set(win, "invert_fingers", on))
    form.addRow(chk)
    b = QPushButton("Calibrate neutral pose")
    b.setToolTip("Hold your hand relaxed, palm down, then click: this pose becomes 'straight'")
    b.clicked.connect(lambda: _glove_calibrate(win))
    form.addRow(b)
    v.addWidget(box)

    box = QGroupBox("Link (which MIDI controls are the glove)")
    lay = QVBoxLayout(box)
    lay.addWidget(QLabel("Myrmex reads the glove's MIDI port in parallel with your other programs. Auto-detect "
                         "takes the controls that stream, in the glove's order (thumb … z). If that is wrong: "
                         "click Learn on a row, then press MAP on the same row in Hand Glove within 1.5 s. "
                         "OSC works too: /glove/thumb … /glove/z (0..1 or 0..16383) to the OSC port."))
    row = QHBoxLayout()
    b = QPushButton("Auto-detect")
    b.clicked.connect(lambda: _glove_autodetect(win))
    row.addWidget(b)
    row.addStretch(1)
    lay.addLayout(row)
    win.tbl_glove = QTableWidget(len(PARAMS), 3)
    win.tbl_glove.setHorizontalHeaderLabels(["glove", "MIDI / OSC source", ""])
    win.tbl_glove.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    win.tbl_glove.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    win.tbl_glove.setMinimumHeight(380)
    win.tbl_glove.verticalHeader().setVisible(False)
    win.tbl_glove.setShowGrid(False)
    for i, p in enumerate(PARAMS):
        win.tbl_glove.setItem(i, 0, QTableWidgetItem(p))
        win.tbl_glove.setItem(i, 1, QTableWidgetItem(_source_label(g.get("profile", {}).get(p, ""))))
        b = QPushButton("Learn")
        b.clicked.connect(lambda _=False, name=p: _glove_learn(win, name))
        win.tbl_glove.setCellWidget(i, 2, b)
    lay.addWidget(win.tbl_glove)
    v.addWidget(box)
    v.addStretch(1)
    return w


def _source_label(key: str) -> str:
    if not key:
        return "—"
    p = key.split(":")
    if p[0] == "cc14":
        return f"CC {p[2]} + {int(p[2]) + 32} (14-bit) · ch {p[1]}"
    if p[0] == "cc":
        return f"CC {p[2]} · ch {p[1]}"
    if p[0] == "pb":
        return f"pitch bend · ch {p[1]}"
    return f"OSC /glove/{p[1]}"


def _glove_set(win, key: str, value) -> None:
    win.s.glove[key] = value
    win.engine.glove_config(**{key: value})


def _glove_calibrate(win) -> None:
    neutral = win.engine.glove_calibrate()
    if neutral:
        win.s.glove["neutral"] = neutral
        win.log("glove calibrated: this pose is neutral now")
    else:
        win.log("glove: nothing to calibrate (engine stopped or no glove data)")


def _glove_autodetect(win) -> None:
    prof = win.engine.glove_autodetect()
    if prof:
        win.s.glove["profile"] = prof
        win.log(f"glove: {len(prof)} controls linked (auto-detect)")
    else:
        win.log("glove: no streaming controls found - is the glove sending, and its MIDI port enabled on Inputs?")


def _glove_learn(win, param: str) -> None:
    win.engine.glove_learn(param)
    win.log(f"glove: press MAP on '{param}' in Hand Glove now …")


def refresh_glove(win) -> None:
    if not hasattr(win, "lbl_glove"):
        return
    snap = win.engine.glove_snapshot()
    if snap is None:
        win.lbl_glove.setText("start the engine to link the glove")
        return
    if snap.get("profile"):
        win.s.glove["profile"] = snap["profile"]
    state = "connected" if snap["present"] else "no glove data"
    if snap.get("learning"):
        state = f"listening for '{snap['learning']}' — press MAP in Hand Glove"
    win.lbl_glove.setText(f"{state} · {snap['rate']:.0f} msg/s · preset {snap['preset']} · "
                          f"turn {snap['angles'][0]:.0f}° / {snap['angles'][1]:.0f}° / {snap['angles'][2]:.0f}°")
    for p, bar in win.glove_bars.items():
        val = snap["values"].get(p)
        bar.setValue(int(1000 * val) if val is not None else 0)
    ev = snap.get("events") or [f"{g}" for g in snap.get("gestures", [])]
    win.lbl_glove_ev.setText(("gestures: " + " · ".join(ev)) if ev else "gestures: —")
    for i in range(win.tbl_glove.rowCount()):
        p = win.tbl_glove.item(i, 0).text()
        item = win.tbl_glove.item(i, 1)
        txt = _source_label(snap["profile"].get(p, ""))
        if item.text() != txt:
            item.setText(txt)
