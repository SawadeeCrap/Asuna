"""Myrmex desktop app (PySide6).

    python -m myrmex.app          (or double-click Myrmex.app / Myrmex.command on macOS)

One window for everything the live performance needs:

* Live      - engine start/stop, what the engine hears (clock, tempo, beat, play state,
              section), the character's knobs (style, energy, stride, sway, hold) and
              one-shot moves (poses, gestures, camera cuts);
* Character - humanoid or an organism, prepare a character from a GLB, your saved Blender look;
* Creature  - the organisms' knobs and events;   * Camera - director / manual shots;
* Inputs    - Ableton (Link, Remote Script), MIDI ports, audio input, clock source, latency;
* MIDI      - monitor + fine mapping;   * Takes - pose stream, recording, take -> video;   * Log.

A narrow rail of icons for navigation (``icons.py``), light / dark theme following macOS (``theme.py``).

Settings are saved on exit and restored on the next launch.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

from PySide6.QtCore import QProcess, QProcessEnvironment, QSize, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListView, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
                               QScrollArea,
                               QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from . import controllers as C
from . import icons
from . import tabs as T
from . import theme
from .settings import AppSettings, characters_dir

CLOCKS = [("auto", "Auto (best available)"), ("link", "Ableton Link"), ("osc", "Remote Script (Ableton)"),
          ("midi", "MIDI clock"), ("onsets", "Follow the kicks"), ("internal", "Internal tempo")]
MATERIALS = ["black_chrome", "keep", "chrome", "liquid_metal", "gunmetal", "ceramic", "clay", "iridescent"]
MOVES = [("Pose", "pose"), ("Look back", "pose:look_back"), ("Hand on hip", "flourish:hand_hip"),
         ("Hair touch", "flourish:hair_touch"), ("Shoulder roll", "flourish:shoulder_roll"),
         ("Side glance", "flourish:side_glance"), ("Chin up", "flourish:chin_up")]


class Knob(QWidget):
    """Slider 0..100 with an 'Auto' switch (auto = the music decides)."""

    def __init__(self, name: str, value: float | None, on_change):
        super().__init__()
        self.name = name
        self.on_change = on_change
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(round((0.5 if value is None else value) * 100)))
        self.auto = QPushButton("Auto")
        self.auto.setObjectName("pill")
        self.auto.setCheckable(True)
        self.auto.setToolTip("Auto: the organism / the music decides")
        self.auto.setChecked(value is None)
        self.slider.setEnabled(value is not None)
        self.val = QLabel()
        self.val.setFont(theme.mono_font(12))
        self.val.setFixedWidth(34)
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.val)
        lay.addWidget(self.auto)
        self.slider.valueChanged.connect(self._changed)
        self.auto.toggled.connect(self._changed)
        self._update_label()

    def value(self) -> float | None:
        return None if self.auto.isChecked() else self.slider.value() / 100.0

    def _update_label(self):
        self.val.setText("–" if self.auto.isChecked() else f"{self.slider.value()}")

    def _changed(self, *_):
        self.slider.setEnabled(not self.auto.isChecked())
        self._update_label()
        self.on_change(self.name, self.value())


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings | None = None):
        super().__init__()
        self.s = settings or AppSettings.load()
        self.setWindowTitle("Myrmex")
        self.resize(900, 740)
        self.setMinimumSize(720, 560)
        self.engine = C.EngineController(self.s, self.log)
        self.blender_proc: QProcess | None = None
        self.prepare_proc: QProcess | None = None
        self._last_notes = (0, time.time())
        self._notes_rate = 0.0
        self.logbox = QPlainTextEdit()
        self.logbox.setObjectName("log")
        self.logbox.setReadOnly(True)
        self.logbox.setMaximumBlockCount(4000)
        self.logbox.setFont(theme.mono_font(12))
        self._build_shell()
        act = QAction("Quit", self)
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)
        self.addAction(act)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(100)
        self._refresh_devices()
        if self.s.start_engine_on_launch:
            QTimer.singleShot(200, self.start_engine)
        if self.s.open_blender_on_start:
            QTimer.singleShot(800, self.open_blender)

    # ================================================================== shell
    PAGES = (("Live", "What the character hears and does, right now"),
             ("Character", "Who performs: a rigged character or one of the organisms"),
             ("Creature", "The organisms' parameters and events"),
             ("Camera", "Automatic director or your own shots"),
             ("Glove", "Hand Glove: the organism moves with your hand"),
             ("Inputs", "Ableton, VCV Rack, MIDI, audio and the clock"),
             ("MIDI", "Every incoming CC and note, and where it goes"),
             ("Takes", "Record performances and turn them into videos"),
             ("Log", "Everything the engine and Blender report"))

    def _build_shell(self) -> None:
        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        side = QWidget()                                   # a narrow rail of icons (names in the tooltips)
        side.setObjectName("sidebar")
        side.setFixedWidth(64)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 14, 0, 14)
        sv.setSpacing(2)
        brand = QLabel()
        brand.setObjectName("brand")
        brand.setPixmap(icons.logo(36))
        brand.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        brand.setToolTip("Myrmex — live character engine")
        sv.addWidget(brand)
        sv.addSpacing(12)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setViewMode(QListView.ViewMode.IconMode)
        self.nav.setFlow(QListView.Flow.TopToBottom)
        self.nav.setMovement(QListView.Movement.Static)
        self.nav.setWrapping(False)
        self.nav.setIconSize(QSize(22, 22))
        self.nav.setGridSize(QSize(64, 46))
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sv.addWidget(self.nav, 1)
        # Engine state, always visible: a dot (the words in its tooltip).
        self.dot = QLabel()
        self.dot.setFixedSize(10, 10)
        self.lbl_engine = QLabel("Engine stopped")           # (not shown: the dot's tooltip)
        self.lbl_engine.setObjectName("enginestate")
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.dot)
        row.addStretch(1)
        sv.addLayout(row)
        h.addWidget(side)
        self.stack = QStackedWidget()
        h.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self.btn_engine = QPushButton("Start engine")
        self.btn_engine.setObjectName("primary")
        self.btn_engine.clicked.connect(self.toggle_engine)
        self.btn_blender = QPushButton("Open in Blender")
        self.btn_blender.clicked.connect(self.open_blender)
        builders = {"Live": self._live_tab, "Character": self._character_tab, "Creature": lambda: T.creature_tab(self),
                    "Camera": self._camera_tab, "Glove": lambda: T.glove_tab(self), "Inputs": self._inputs_tab,
                    "MIDI": lambda: T.midi_tab(self),
                    "Takes": self._output_tab, "Log": lambda: self.logbox}
        self.page_index = {}
        for name, subtitle in self.PAGES:
            actions = [self.btn_blender, self.btn_engine] if name == "Live" else []
            self.page_index[name] = self.stack.addWidget(self._page(name, subtitle, builders[name](), actions))
            it = QListWidgetItem(self.nav)
            it.setToolTip(f"{name} — {subtitle}")
            it.setData(Qt.ItemDataRole.UserRole, name)
            it.setSizeHint(QSize(64, 46))
        self._nav_icons()
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self._polish()

    def _page(self, title: str, subtitle: str, body: QWidget, actions=()) -> QWidget:
        page = QWidget()
        page.setObjectName("page")
        v = QVBoxLayout(page)
        v.setContentsMargins(32, 26, 32, 20)
        v.setSpacing(14)
        head = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("pagetitle")
        st = QLabel(subtitle)
        st.setObjectName("pagesub")
        col.addWidget(t)
        col.addWidget(st)
        head.addLayout(col, 1)
        for a in actions:
            head.addWidget(a, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addLayout(head)
        if isinstance(body, QPlainTextEdit):
            v.addWidget(body, 1)
            return page
        if body.layout() is not None:
            body.layout().setContentsMargins(0, 0, 6, 0)
            body.layout().setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(body)
        v.addWidget(scroll, 1)
        return page

    def _polish(self) -> None:
        """Consistent forms, wrapped explanations, quiet tables."""
        for form in self.findChildren(QFormLayout):
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setHorizontalSpacing(16)
            form.setVerticalSpacing(10)
        for lab in self.findChildren(QLabel):
            if len(lab.text()) > 70 and lab.objectName() not in ("pagesub", "pagetitle"):
                lab.setWordWrap(True)
                lab.setProperty("muted", True)
                lab.style().unpolish(lab)
                lab.style().polish(lab)
        for name in ("tbl_mon", "tbl_map"):
            tbl = getattr(self, name, None)
            if tbl is not None:
                tbl.verticalHeader().setVisible(False)
                tbl.setShowGrid(False)
        self._engine_state(False)

    def _nav_icons(self) -> None:
        """(Re)paint the rail's icons in the current theme's colours."""
        for i in range(self.nav.count()):
            it = self.nav.item(i)
            it.setIcon(icons.nav_icon(it.data(Qt.ItemDataRole.UserRole), theme.T))

    def _goto(self, page: str) -> None:
        self.nav.setCurrentRow(self.page_index[page])

    def _engine_state(self, running: bool) -> None:
        self.btn_engine.setText("Stop engine" if running else "Start engine")
        self.btn_engine.setProperty("running", running)
        self.btn_engine.style().unpolish(self.btn_engine)
        self.btn_engine.style().polish(self.btn_engine)

    # ================================================================== pages
    def _live_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        # ---- status
        box = QGroupBox("What the character hears")
        g = QGridLayout(box)
        g.setHorizontalSpacing(18)
        g.setVerticalSpacing(12)
        self.st = {}
        items = [("clock", "Clock"), ("bpm", "Tempo"), ("beat", "Bar . beat"), ("transport", "Transport"),
                 ("state", "Character"), ("section", "Section"), ("behavior", "Doing"), ("camera", "Camera"),
                 ("notes", "Notes / s"), ("peers", "Link peers"), ("tick", "Engine tick"), ("sent", "Frames sent")]
        for i, (k, label) in enumerate(items):
            tile = QVBoxLayout()
            tile.setSpacing(1)
            lab = QLabel(label)
            lab.setObjectName("statlabel")
            val = QLabel("–")
            val.setObjectName("statvalue")
            val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            tile.addWidget(lab)
            tile.addWidget(val)
            g.addLayout(tile, i // 4, i % 4)
            self.st[k] = val
        beats = QHBoxLayout()
        beats.setSpacing(8)
        self.beat_leds = []
        for _ in range(4):
            led = QLabel()
            led.setFixedSize(46, 8)
            beats.addWidget(led)
            self.beat_leds.append(led)
        beats.addStretch(1)
        g.addLayout(beats, 3, 0, 1, 4)
        self._leds_off()
        v.addWidget(box)
        # ---- knobs
        box = QGroupBox("Character (Auto = the music decides)")
        form = QFormLayout(box)
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(C.STYLES)
        self.cmb_style.setCurrentText(self.s.style)
        self.cmb_style.currentTextChanged.connect(self._style_changed)
        form.addRow("Walk style", self.cmb_style)
        self.knobs = {}
        for k, label in (("energy", "Energy"), ("stride", "Stride"), ("sway", "Hip sway")):
            kb = Knob(k, getattr(self.s, k), self._knob_changed)
            self.knobs[k] = kb
            form.addRow(label, kb)
        self.chk_hold = QCheckBox("Stand and pose (hold)")
        self.chk_hold.setToolTip("Otherwise she walks while the music plays")
        self.chk_hold.toggled.connect(lambda on: self.engine.control("hold", 1.0 if on else 0.0))
        form.addRow("", self.chk_hold)
        v.addWidget(box)
        # ---- moves
        box = QGroupBox("Moves (one-shot)")
        grid = QGridLayout(box)
        for i, (label, name) in enumerate(MOVES):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, n=name: self.engine.trigger(n))
            grid.addWidget(b, i // 4, i % 4)
        cam = QPushButton("Camera cut ▸")
        cam.clicked.connect(self._camera_cut)
        self.cmb_shot = QComboBox()
        self.cmb_shot.addItems(C.SHOTS)
        grid.addWidget(cam, 2, 0)
        grid.addWidget(self.cmb_shot, 2, 1)
        v.addWidget(box)
        v.addStretch(1)
        return w

    def _leds_off(self) -> None:
        for led in self.beat_leds:
            led.setStyleSheet(f"background: {theme.T['border']}; border-radius: 4px")

    def _inputs_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        box = QGroupBox("Ableton Live")
        form = QFormLayout(box)
        self.chk_link = QCheckBox("Join Ableton Link (tempo + phase; enable LINK and Start Stop Sync in Live)")
        self.chk_link.setChecked(self.s.link)
        form.addRow(self.chk_link)
        rs = QHBoxLayout()
        self.lbl_rs = QLabel()
        b = QPushButton("Install / update Remote Script")
        b.clicked.connect(self._install_rs)
        rs.addWidget(self.lbl_rs, 1)
        rs.addWidget(b)
        form.addRow("Remote Script", rs)
        form.addRow(QLabel("After installing: Live > Settings > Link, Tempo & MIDI > Control Surface: Myrmex.\n"
                           "It sends play/stop, tempo and the notes of playing MIDI clips (ahead of time).\n"
                           "A track named “Myrmex” with a Rack: its macros steer the character "
                           "(Energy, Stride, Sway, Style, Hold, Camera, Pose, Flourish);\n"
                           "notes in its clips: C3 pose, D3 gesture, E3 camera cut, C4 hold."))
        v.addWidget(box)
        box = QGroupBox("Clock and timing")
        form = QFormLayout(box)
        self.cmb_clock = QComboBox()
        for key, label in CLOCKS:
            self.cmb_clock.addItem(label, key)
        self.cmb_clock.setCurrentIndex([k for k, _ in CLOCKS].index(self.s.clock) if self.s.clock in dict(CLOCKS) else 0)
        form.addRow("Beat source", self.cmb_clock)
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(40, 240)
        self.spin_bpm.setValue(self.s.bpm)
        form.addRow("Internal tempo (BPM)", self.spin_bpm)
        self.spin_lat = QSpinBox()
        self.spin_lat.setRange(0, 300)
        self.spin_lat.setSuffix(" ms")
        self.spin_lat.setValue(int(self.s.latency_ms))
        self.spin_lat.valueChanged.connect(lambda ms: (setattr(self.s, "latency_ms", ms), self.engine.set_latency(ms)))
        form.addRow("Latency compensation", self.spin_lat)
        v.addWidget(box)
        box = QGroupBox("MIDI / OSC / audio (VCV Rack, controllers, audio-only sets)")
        form = QFormLayout(box)
        self.spin_osc = QSpinBox()
        self.spin_osc.setRange(1024, 65535)
        self.spin_osc.setValue(int(self.s.osc_port))
        form.addRow("OSC in port", self.spin_osc)
        self.list_midi = QListWidget()
        self.list_midi.setMaximumHeight(90)
        form.addRow("MIDI inputs (IAC …)", self.list_midi)
        self.cmb_audio = QComboBox()
        form.addRow("Audio input (BlackHole …)", self.cmb_audio)
        mp = QHBoxLayout()
        self.ed_mapping = QLineEdit(self.s.mapping_file)
        self.ed_mapping.setPlaceholderText("optional mapping.json (MIDI channels, CC, OSC addresses)")
        b = QPushButton("…")
        b.clicked.connect(lambda: self._pick_file(self.ed_mapping, "Mapping JSON", "JSON (*.json)"))
        mp.addWidget(self.ed_mapping, 1)
        mp.addWidget(b)
        form.addRow("Mapping", mp)
        self.lbl_dev = QLabel()
        self.lbl_dev.setProperty("muted", True)
        form.addRow(self.lbl_dev)
        row = QHBoxLayout()
        b = QPushButton("Refresh devices")
        b.clicked.connect(self._refresh_devices)
        apply = QPushButton("Apply (restart engine)")
        apply.clicked.connect(self.apply_and_restart)
        row.addWidget(b)
        row.addStretch(1)
        row.addWidget(apply)
        v.addWidget(box)
        v.addLayout(row)
        v.addStretch(1)
        return w

    def _character_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        box = QGroupBox("Character")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self.cmb_char = QComboBox()
        self._fill_characters()
        b = QPushButton("Browse…")
        b.clicked.connect(self._browse_character)
        row.addWidget(self.cmb_char, 1)
        row.addWidget(b)
        form.addRow("Live character (.blend)", row)
        self.cmb_backend = QComboBox()
        self.cmb_backend.addItem("Humanoid (rigged character)", "humanoid")
        self.cmb_backend.addItem("Black Nanomaterial Creature (procedural organism)", "creature")
        self.cmb_backend.addItem("Mimetic Polyalloy (flying, self-reconfiguring material)", "polyalloy")
        self.cmb_backend.addItem("Polyalloy Colony (flock, armour, mechanisms, prey)", "colony")
        self.cmb_backend.addItem("Polyalloy Hive (nanomachine swarm, builds, patterns, remembers)", "hive")
        self.cmb_backend.addItem("Osseous Polyalloy (v5: bone-link skeleton, bony blades, strikes)", "osseous")
        self.cmb_backend.addItem("Osseous Colony (v6: bony flock, scutes, mandibles, strikes)", "osseous_colony")
        self.cmb_backend.addItem("Osseous Hive (v7: bony swarm, quill volleys, fanged gates)", "osseous_hive")
        self.cmb_backend.addItem("Cyber Hive (v8: white nanomaterial, light lines, machine forms)", "cyber_hive")
        self.cmb_backend.addItem("Mimetic Swarm (v9: distributed fluid flight, streaming filaments)", "swarm")
        self.cmb_backend.addItem("Mimetic Spear (v10: elongated, high-speed, dashes)", "spear")
        self.cmb_backend.addItem("Mimetic Cloud (v11: dispersion, camouflage, reassembly)", "cloud")
        self.cmb_backend.addItem("Mimetic Blade (v12: swept blades, high-velocity cutting)", "blade")
        self.cmb_backend.addItem("Mimetic Crawler (v13: many legs on rough terrain)", "crawler")
        self.cmb_backend.setCurrentIndex(max(0, self.cmb_backend.findData(self.s.backend)))
        form.addRow("Character type", self.cmb_backend)
        for cb in (self.cmb_char, self.cmb_backend):                # long names must not widen the page
            cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(18)
        row = QHBoxLayout()
        self.cmb_look = QComboBox()
        self.cmb_look.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_look.setMinimumContentsLength(10)
        self.cmb_look.setToolTip("Saved looks of this organism. Choosing one loads it into the Blender that Myrmex "
                                 "opened (what it showed is kept as 'Autosave'); it is also used next time and for "
                                 "take renders.")
        self.cmb_look.activated.connect(self._look_chosen)
        self.btn_save_look = QPushButton("Save look…")
        self.btn_save_look.setToolTip("Save what the Blender opened by Myrmex shows now (materials, lights, world, "
                                      "colour, render settings) as a look of this organism")
        self.btn_save_look.clicked.connect(self._save_current_look)
        self.btn_del_look = QPushButton("Delete")
        self.btn_del_look.setToolTip("Delete the selected look")
        self.btn_del_look.clicked.connect(self._delete_look)
        self.lbl_look_note = QLabel("in the character's .blend")
        self.lbl_look_note.setProperty("muted", True)
        row.addWidget(self.cmb_look, 1)
        row.addWidget(self.lbl_look_note, 1)
        row.addWidget(self.btn_save_look)
        row.addWidget(self.btn_del_look)
        form.addRow("Look", row)
        row = QHBoxLayout()
        self.ed_blender = QLineEdit(self.s.blender or (C.find_blender() or ""))
        b = QPushButton("…")
        b.clicked.connect(lambda: self._pick_file(self.ed_blender, "Blender executable", "All (*)"))
        row.addWidget(self.ed_blender, 1)
        row.addWidget(b)
        form.addRow("Blender", row)
        self.chk_open_blender = QCheckBox("Open Blender automatically when the app starts")
        self.chk_open_blender.setChecked(self.s.open_blender_on_start)
        form.addRow(self.chk_open_blender)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999)
        self.spin_seed.setValue(int(self.s.seed))
        form.addRow("Personality seed", self.spin_seed)
        v.addWidget(box)
        box = QGroupBox("Your look in Blender")
        form = QFormLayout(box)
        form.addRow(QLabel("Open in Blender, tune materials, lights, world and render settings there, then Save "
                           "look… (Character type → Look). Every organism keeps its own list of looks: choose "
                           "one to load it into Blender at once; live sessions and take renders open the chosen look. "
                           "Humanoid: the look is saved in the character's .blend."))
        self.chk_keep = QCheckBox("Keep my Blender settings")
        self.chk_keep.setToolTip("Myrmex doesn't change EEVEE, colour, shadows or samples")
        self.chk_keep.setChecked(self.s.keep_blender_settings)
        self.chk_keep.toggled.connect(lambda on: self.cmb_rquality.setEnabled(not on))
        form.addRow(self.chk_keep)
        self.cmb_backend.currentIndexChanged.connect(lambda *_: self._fill_looks())
        self._fill_looks()
        v.addWidget(box)
        box = QGroupBox("New character from a Hunyuan3D GLB")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self.ed_glb = QLineEdit()
        self.ed_glb.setPlaceholderText("model.glb")
        b = QPushButton("…")
        b.clicked.connect(lambda: self._pick_file(self.ed_glb, "Hunyuan3D model", "3D (*.glb *.gltf *.obj *.fbx)"))
        row.addWidget(self.ed_glb, 1)
        row.addWidget(b)
        form.addRow("GLB", row)
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.2, 20.0)
        self.spin_height.setSingleStep(0.05)
        self.spin_height.setValue(1.70)
        self.spin_height.setSuffix(" m")
        form.addRow("Height", self.spin_height)
        self.cmb_mat = QComboBox()
        self.cmb_mat.addItems(MATERIALS)
        form.addRow("Material", self.cmb_mat)
        self.spin_smooth = QSpinBox()
        self.spin_smooth.setRange(0, 30)
        self.spin_smooth.setValue(6)
        form.addRow("Surface smoothing", self.spin_smooth)
        self.btn_prepare = QPushButton("Prepare character (auto-rig, ~1 min)")
        self.btn_prepare.clicked.connect(self.prepare_character)
        form.addRow(self.btn_prepare)
        v.addWidget(box)
        v.addStretch(1)
        return w

    def _camera_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        box = QGroupBox("Camera")
        form = QFormLayout(box)
        self.chk_camera = QCheckBox("Cinematic live camera (cuts on downbeats)")
        self.chk_camera.setChecked(self.s.camera)
        form.addRow(self.chk_camera)
        v.addWidget(box)
        v.addWidget(T.camera_group(self))
        v.addStretch(1)
        return w

    def _output_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        box = QGroupBox("Pose stream (to Blender / other renderers)")
        form = QFormLayout(box)
        self.spin_pose = QSpinBox()
        self.spin_pose.setRange(1024, 65535)
        self.spin_pose.setValue(int(self.s.pose_port))
        form.addRow("Pose port", self.spin_pose)
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(15, 240)
        self.spin_fps.setValue(int(self.s.out_fps))
        form.addRow("Poses per second", self.spin_fps)
        self.ed_targets = QLineEdit(self.s.extra_targets)
        self.ed_targets.setPlaceholderText("extra targets, e.g. 192.168.1.20:9101")
        form.addRow("Also send to", self.ed_targets)
        v.addWidget(box)
        box = QGroupBox("Recording (for final renders)")
        form = QFormLayout(box)
        self.chk_record = QCheckBox("Record the performance (saved when the engine stops)")
        self.chk_record.setChecked(self.s.record)
        form.addRow(self.chk_record)
        row = QHBoxLayout()
        self.ed_rec = QLineEdit(self.s.record_dir)
        b = QPushButton("…")
        b.clicked.connect(lambda: self._pick_dir(self.ed_rec))
        row.addWidget(self.ed_rec, 1)
        row.addWidget(b)
        form.addRow("Takes folder", row)
        b = QPushButton("Save take now")
        b.clicked.connect(self._save_take)
        form.addRow(b)
        row = QHBoxLayout()
        self.ed_song = QLineEdit(self.s.take_audio)
        self.ed_song.setPlaceholderText("the track exported from Ableton (from bar 1) — lined up automatically")
        b = QPushButton("…")
        b.clicked.connect(lambda: self._pick_file(self.ed_song, "Song", "Audio (*.wav *.aif *.aiff *.mp3 *.flac)"))
        row.addWidget(self.ed_song, 1)
        row.addWidget(b)
        form.addRow("Song for renders", row)
        row = QHBoxLayout()
        self.cmb_rsize = QComboBox()
        self.cmb_rsize.addItems(["1920x1080", "1080x1920", "1080x1080", "3840x2160", "1280x720"])
        self.cmb_rsize.setCurrentText(self.s.render_size)
        self.cmb_rquality = QComboBox()
        for label, key in (("Draft (EEVEE fast)", "eevee_preview"), ("Final (EEVEE)", "eevee"),
                           ("Cinema (Cycles, slow)", "cycles")):
            self.cmb_rquality.addItem(label, key)
        self.cmb_rquality.setCurrentIndex(max(0, self.cmb_rquality.findData(self.s.render_quality)))
        self.cmb_rquality.setEnabled(not self.s.keep_blender_settings)
        self.cmb_rquality.setToolTip("Used when 'Keep my Blender settings' (Character page) is off")
        row.addWidget(self.cmb_rsize)
        row.addWidget(self.cmb_rquality, 1)
        form.addRow("Video", row)
        row = QHBoxLayout()
        for label, render, choose in (("Open last take in Blender", False, False), ("Render last take → .mp4", True, False),
                                      ("Choose take…", False, True)):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, r=render, c=choose: self.open_take(r, c))
            row.addWidget(b)
        form.addRow(row)
        self.chk_autostart = QCheckBox("Start the engine when the app starts")
        self.chk_autostart.setChecked(self.s.start_engine_on_launch)
        form.addRow(self.chk_autostart)
        v.addWidget(box)
        row = QHBoxLayout()
        row.addStretch(1)
        apply = QPushButton("Apply (restart engine)")
        apply.clicked.connect(self.apply_and_restart)
        row.addWidget(apply)
        v.addLayout(row)
        v.addStretch(1)
        return w

    # ================================================================== actions
    def log(self, msg: str) -> None:
        self.logbox.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    def _collect(self) -> None:
        s = self.s
        s.link = self.chk_link.isChecked()
        s.clock = self.cmb_clock.currentData()
        s.bpm = float(self.spin_bpm.value())
        s.latency_ms = float(self.spin_lat.value())
        s.osc_port = int(self.spin_osc.value())
        s.midi_ports = [self.list_midi.item(i).text() for i in range(self.list_midi.count())
                        if self.list_midi.item(i).checkState() == Qt.CheckState.Checked]
        a = self.cmb_audio.currentText()
        s.audio_device = "" if a.startswith("(") else a
        s.mapping_file = self.ed_mapping.text().strip()
        s.character = self.cmb_char.currentData() or s.character
        s.blender = self.ed_blender.text().strip()
        s.open_blender_on_start = self.chk_open_blender.isChecked()
        s.seed = int(self.spin_seed.value())
        s.camera = self.chk_camera.isChecked()
        s.pose_port = int(self.spin_pose.value())
        s.out_fps = float(self.spin_fps.value())
        s.extra_targets = self.ed_targets.text().strip()
        s.record = self.chk_record.isChecked()
        s.record_dir = self.ed_rec.text().strip() or s.record_dir
        s.start_engine_on_launch = self.chk_autostart.isChecked()
        s.style = self.cmb_style.currentText()
        s.backend = self.cmb_backend.currentData()
        s.take_audio = self.ed_song.text().strip()
        s.keep_blender_settings = self.chk_keep.isChecked()
        snap = self.engine.glove_snapshot() if self.engine.running else None
        if snap and snap.get("profile"):                     # keep the glove link for the next session
            s.glove["profile"] = snap["profile"]
        s.render_size = self.cmb_rsize.currentText()
        s.render_quality = self.cmb_rquality.currentData()
        s.midi_bindings = T.read_bindings(self)
        for k, kb in self.knobs.items():
            setattr(s, k, kb.value())

    def start_engine(self) -> None:
        self._collect()
        if self.engine.start():
            self._engine_state(True)
            if self.chk_hold.isChecked():
                self.engine.control("hold", 1.0)
        else:
            self._goto("Log")

    def stop_engine(self) -> None:
        self.engine.stop()
        self._engine_state(False)

    def toggle_engine(self) -> None:
        if self.engine.running:
            self.stop_engine()
        else:
            self.start_engine()

    def apply_and_restart(self) -> None:
        self._collect()
        self.s.save()
        if self.engine.running:
            self.stop_engine()
            self.start_engine()
        self.log("settings applied")

    def _knob_changed(self, name: str, value: float | None) -> None:
        setattr(self.s, name, value)
        self.engine.control(name, -1.0 if value is None else value)

    def _creature_knob(self, name: str, value: float | None) -> None:
        if value is None:
            self.s.creature_params.pop(name, None)
        else:
            self.s.creature_params[name] = value
        self.engine.control(name, -1.0 if value is None else value)

    def _cam_control(self, name: str, value: float) -> None:
        self.s.camera_controls[name] = value
        self.engine.control(name, value)

    def _pick_shot(self, kind: str) -> None:
        if self.cmb_cam_mode.currentIndex() != 1:
            self.cmb_cam_mode.setCurrentIndex(1)            # choosing a shot means: manual, hold it
        self.engine.trigger(f"camera:{kind}")

    def _style_changed(self, style: str) -> None:
        self.s.style = style
        self.engine.control("style", (C.STYLES.index(style) + 0.5) / len(C.STYLES))

    def _camera_cut(self) -> None:
        shot = self.cmb_shot.currentText()
        self.engine.trigger("camera" if shot == "auto" else f"camera:{shot}")

    def _save_take(self) -> None:
        path = self.engine.save_take()
        self.log(f"take saved: {path}" if path else "no take: enable recording and restart the engine")

    # ------------------------------------------------------------------ looks (saved in / loaded into Blender)
    def _fill_looks(self) -> None:
        b = self.cmb_backend.currentData()
        creature = b in C.CREATURE_BACKENDS
        self.cmb_look.setVisible(creature)
        self.btn_del_look.setVisible(creature)
        self.lbl_look_note.setVisible(not creature)
        self.cmb_look.blockSignals(True)
        self.cmb_look.clear()
        if creature:
            self.cmb_look.addItem("Default studio (built by Myrmex)", "")
            for name, path in C.list_looks(b):
                self.cmb_look.addItem("Autosave (before the last switch)" if name == C.LOOK_AUTOSAVE else name, path)
            self.cmb_look.setCurrentIndex(max(0, self.cmb_look.findData(C.look_file(b, self.s.looks))))
        self.cmb_look.blockSignals(False)
        self.btn_del_look.setEnabled(creature and bool(self.cmb_look.currentData()))

    def _blender_for(self, variant: str | None = None) -> QProcess | None:
        """The Blender that Myrmex opened and that listens to it (showing ``variant``, if given)."""
        procs = [self.blender_proc] + list(reversed(getattr(self, "take_procs", [])))
        for p in procs:
            if p is None or p.state() == QProcess.ProcessState.NotRunning or not p.property("myrmex_control"):
                continue
            if variant is None or p.property("myrmex_variant") == variant:
                return p
        return None

    def _blender_send(self, p: QProcess, cmd: dict) -> None:
        p.write((json.dumps(cmd) + "\n").encode("utf-8"))

    def _look_chosen(self, *_) -> None:
        b = self.cmb_backend.currentData()
        if b not in C.CREATURE_BACKENDS:
            return
        v, path, name = C.variant_of(b), self.cmb_look.currentData() or "", self.cmb_look.currentText()
        self.s.looks[v] = path
        self.s.save()
        self.btn_del_look.setEnabled(bool(path))
        p = self._blender_for(v)
        if p is not None:
            self._blender_send(p, {"cmd": "load_look", "path": path, "kind": v, "keep": self.chk_keep.isChecked()})
            self.log(f"loading the look '{name}' into Blender…")
        else:
            self.log(f"look '{name}': used when Blender opens this organism (Open in Blender, takes)")

    def _save_current_look(self) -> None:
        p = self._blender_for()
        if p is None:
            QMessageBox.information(self, "Myrmex", "Open Blender from Myrmex first (Open in Blender, or a take), "
                                                    "tune the look there, then save it here.")
            return
        v = p.property("myrmex_variant") or "humanoid"
        if v not in [C.variant_of(b) for b in C.CREATURE_BACKENDS]:
            self._blender_send(p, {"cmd": "save_look", "kind": "humanoid"})
            self.log("saving the look in the character's .blend…")
            return
        looks = C.list_looks(v)
        chosen = C.look_file(v, self.s.looks)
        default = next((n for n, path in looks if path == chosen and n != C.LOOK_AUTOSAVE), "")
        if not default:
            taken = {n.lower() for n, _ in looks}
            k = 1
            while f"look {k}" in taken:
                k += 1
            default = f"Look {k}"
        label = self.cmb_backend.itemText(self.cmb_backend.findData(
            next((b for b in C.CREATURE_BACKENDS if C.variant_of(b) == v), v))).split(" (")[0]
        name, ok = QInputDialog.getText(self, "Save current look", f"Name of the look ({label}):", text=default)
        name = name.strip()
        if not ok or not name:
            return
        path = C.look_path(v, name)
        if os.path.exists(path) and QMessageBox.question(self, "Myrmex", f"Replace the look '{name}'?") != \
                QMessageBox.StandardButton.Yes:
            return
        self._blender_send(p, {"cmd": "save_look", "path": path, "kind": v})
        self.log(f"saving the look '{name}'…")

    def _delete_look(self) -> None:
        path, name = self.cmb_look.currentData() or "", self.cmb_look.currentText()
        if not path:
            return
        if QMessageBox.question(self, "Myrmex", f"Delete the look '{name}'?\n{path}") != QMessageBox.StandardButton.Yes:
            return
        try:
            os.remove(path)
        except OSError as e:
            self.log(f"! could not delete the look: {e}")
            return
        v = C.variant_of(self.cmb_backend.currentData())
        if self.s.looks.get(v) == path:
            self.s.looks[v] = ""
            self.s.save()
        self.log(f"look deleted: {name}")
        self._fill_looks()

    def _on_blender_reply(self, p: QProcess, d: dict) -> None:
        cmd = d.get("cmd")
        if cmd == "hello":
            p.setProperty("myrmex_control", True)
            return
        if not d.get("ok"):
            self.log(f"! Blender: {d.get('error', 'failed')}")
            return
        name = os.path.splitext(os.path.basename(d.get("path") or ""))[0]
        kind = d.get("kind", "")
        if cmd == "save_look":
            if kind == "humanoid":
                self.log(f"look saved in the character: {d.get('path')}")
                return
            if name == kind:
                name = C.LOOK_DEFAULT_NAME
            self.s.looks[kind] = d.get("path", "")
            self.s.save()
            self.log(f"look saved: {name} ({d.get('path')})")
        elif cmd == "load_look":
            what = name if d.get("path") else "the default studio"
            if name == kind:
                what = C.LOOK_DEFAULT_NAME
            self.log(f"Blender shows the look '{what}' now" + (" (the take was reloaded in it)" if d.get("take") else "")
                     + (" · what was on screen is kept as 'Autosave'" if d.get("autosaved") else ""))
        if C.variant_of(self.cmb_backend.currentData() or "") == kind:
            self._fill_looks()

    def _install_rs(self) -> None:
        try:
            dst = C.install_remote_script()
            self.log(f"Remote Script installed: {dst} (restart Live, pick Control Surface: Myrmex)")
        except Exception as e:
            self.log(f"! Remote Script install failed: {e}")
        self._refresh_devices()

    def _refresh_devices(self) -> None:
        self.lbl_rs.setText("installed ✓" if C.remote_script_installed() else "not installed")
        names, err1 = C.midi_inputs()
        self.list_midi.clear()
        for n in names:
            it = QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = n in self.s.midi_ports or ("auto" in self.s.midi_ports and "IAC" in n)
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.list_midi.addItem(it)
        audio, err2 = C.audio_inputs()
        self.cmb_audio.clear()
        self.cmb_audio.addItem("(off)")
        self.cmb_audio.addItems(audio)
        if self.s.audio_device in audio:
            self.cmb_audio.setCurrentText(self.s.audio_device)
        notes = [e for e in (err1, err2) if e]
        if not C.link_available():
            notes.append("Ableton Link unavailable (pip install aalink)")
        self.lbl_dev.setText("\n".join(notes) if notes else "devices ok")

    def _fill_characters(self) -> None:
        self.cmb_char.clear()
        for p in C.known_characters(self.s.character):
            self.cmb_char.addItem(os.path.basename(p), p)
        i = self.cmb_char.findData(self.s.character)
        if i >= 0:
            self.cmb_char.setCurrentIndex(i)

    def _browse_character(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Live character", characters_dir(), "Blender (*.blend)")
        if path:
            if not os.path.exists(os.path.splitext(path)[0] + ".rig.json"):
                QMessageBox.warning(self, "Myrmex", "This .blend has no .rig.json next to it.\n"
                                                    "Prepare the character from its GLB first.")
                return
            self.s.character = path
            self._fill_characters()

    def _pick_file(self, edit: QLineEdit, title: str, flt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, os.path.expanduser("~"), flt)
        if path:
            edit.setText(path)

    def _pick_dir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Folder", edit.text() or os.path.expanduser("~"))
        if path:
            edit.setText(path)

    # ------------------------------------------------------------------ blender
    def open_blender(self) -> None:
        self._collect()
        blender = C.find_blender(self.s.blender)
        if not blender:
            QMessageBox.warning(self, "Myrmex", "Blender not found. Install Blender 5.2 or set its path "
                                                "on the Character tab.")
            return
        if not self.engine.running:
            self.start_engine()
        cmd, env = C.blender_live_command(blender, self.s.character, self.s.pose_port, self.s.backend,
                                          self.s.keep_blender_settings, self.s.looks)
        look = C.look_file(self.s.backend, self.s.looks) if self.s.backend in C.CREATURE_BACKENDS else ""
        if look:
            self.log(f"using your saved look: {look}")
        p = QProcess(self)
        qenv = QProcessEnvironment.systemEnvironment()
        for k, val in env.items():
            qenv.insert(k, val)
        p.setProcessEnvironment(qenv)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self._pipe(p, "blender"))
        p.finished.connect(lambda *_: self.log("Blender closed"))
        p.setProperty("myrmex_variant", C.variant_of(self.s.backend))
        p.start(cmd[0], cmd[1:])
        self.blender_proc = p
        self.log(f"opening {os.path.basename(self.s.character)} in Blender (live link on port {self.s.pose_port})")

    def open_take(self, render: bool = False, choose: bool = False) -> None:
        """A recorded take -> Blender: opened ready to render, or rendered to .mp4 in the background."""
        self._collect()
        blender = C.find_blender(self.s.blender)
        if not blender:
            QMessageBox.warning(self, "Myrmex", "Blender not found. Install Blender 5.2 or set its path on the Character tab.")
            return
        take = C.last_take(self.s.record_dir)
        if choose or not take:
            take = QFileDialog.getOpenFileName(self, "Take", os.path.expanduser(self.s.record_dir or "~"),
                                               "Takes (*.npz)")[0]
        if not take:
            return
        cmd = C.take_command(blender, take, self.s.character, self.s.take_audio, render, self.s.render_size,
                             self.s.render_quality, self.s.keep_blender_settings, self.s.looks)
        p = QProcess(self)
        if not render:                                     # the app can save / load looks in this Blender
            qenv = QProcessEnvironment.systemEnvironment()
            qenv.insert("MYRMEX_CONTROL", "stdin")
            qenv.insert("MYRMEX_LOOKS", C.looks_dir())
            p.setProcessEnvironment(qenv)
            p.setProperty("myrmex_variant", C.take_variant(take) or "humanoid")
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self._pipe(p, "render", ("Myrmex", "rror")) if render
                                          else self._pipe(p, "blender"))
        p.finished.connect(lambda *_: self.log("render finished" if render else "Blender (take) closed"))
        p.start(cmd[0], cmd[1:])
        self.take_procs = [q for q in getattr(self, "take_procs", []) if q.state() != QProcess.ProcessState.NotRunning]
        self.take_procs.append(p)
        self.log(("rendering " if render else "opening ") + os.path.basename(take) +
                 (" (video next to the take)" if render else " in Blender"))

    def prepare_character(self) -> None:
        self._collect()
        glb = self.ed_glb.text().strip()
        blender = C.find_blender(self.s.blender)
        if not glb or not os.path.exists(glb):
            QMessageBox.warning(self, "Myrmex", "Choose a GLB file first.")
            return
        if not blender:
            QMessageBox.warning(self, "Myrmex", "Blender not found (Character tab).")
            return
        out = os.path.join(characters_dir(), os.path.splitext(os.path.basename(glb))[0] + "_live.blend")
        cmd = C.prepare_command(blender, glb, out, self.spin_height.value(), self.cmb_mat.currentText(),
                                self.spin_smooth.value())
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self._pipe(p, "prepare", only=("[", "!!", "Error", "rror")))
        p.finished.connect(lambda code, *_: self._prepared(out, code))
        self.btn_prepare.setEnabled(False)
        self.btn_prepare.setText("Preparing… (see Log)")
        p.start(cmd[0], cmd[1:])
        self.prepare_proc = p
        self.log(f"preparing {os.path.basename(glb)} -> {out}")
        self._goto("Log")

    def _prepared(self, out: str, code: int) -> None:
        self.btn_prepare.setEnabled(True)
        self.btn_prepare.setText("Prepare character (auto-rig, ~1 min)")
        if code == 0 and os.path.exists(out):
            self.s.character = out
            self._fill_characters()
            self.log(f"character ready: {out} — selected; press Apply (restart engine) or Open in Blender")
        else:
            self.log(f"! prepare failed (exit {code})")

    def _pipe(self, p: QProcess, tag: str, only=None) -> None:
        data = (p.property("myrmex_buf") or "") + bytes(p.readAllStandardOutput()).decode("utf-8", "replace")
        lines = data.split("\n")
        p.setProperty("myrmex_buf", lines.pop())             # a line still arriving
        for line in lines:
            line = line.strip()
            if line.startswith("MYRMEX_REPLY "):                # an answer to Save / load look
                try:
                    self._on_blender_reply(p, json.loads(line[len("MYRMEX_REPLY "):]))
                except ValueError:
                    pass
                continue
            if line and (only is None or any(o in line for o in only)):
                self.log(f"[{tag}] {line}")

    # ------------------------------------------------------------------ status
    def _refresh(self) -> None:
        T.refresh_midi(self)
        if self.stack.currentIndex() == self.page_index.get("Glove"):
            T.refresh_glove(self)
        st = self.engine.status()
        tok = theme.T
        if not st:
            for k in self.st:
                self.st[k].setText("–")
            self._leds_off()
            self.dot.setStyleSheet(f"background: {tok['text3']}; border-radius: 5px")
            self.lbl_engine.setText("Engine stopped")
            self.dot.setToolTip("Engine stopped")
            return
        self.dot.setStyleSheet(f"background: {tok['ok'] if not st['errors'] else tok['warn']}; border-radius: 5px")
        self.lbl_engine.setText(f"Running · {st['bpm']:.1f} BPM" if st["bpm"] else "Running")
        self.dot.setToolTip("Engine " + self.lbl_engine.text().lower())
        now = time.time()
        n0, t0 = self._last_notes
        if now - t0 >= 1.0:
            self._notes_rate = (st["notes"] - n0) / (now - t0)
            self._last_notes = (st["notes"], now)
        beat = st.get("beat_raw", 0.0) or 0.0
        bpb = max(1, int(round(st.get("bpb", 4.0))))
        bar = int(beat // bpb) + 1
        inbar = int(beat % bpb)
        names = {"auto": "auto", "link": "Ableton Link", "osc": "Remote Script", "midi": "MIDI clock",
                 "onsets": "following kicks", "internal": "internal"}
        self.st["clock"].setText(names.get(st["clock"], str(st["clock"])))
        self.st["bpm"].setText(f"{st['bpm']:.1f} BPM" if st["bpm"] else "–")
        self.st["beat"].setText(f"{bar}.{inbar + 1}")
        self.st["transport"].setText("▶ playing" if st["playing"] else "■ stopped")
        self.st["state"].setText("holding a pose" if st["hold"] else "walking")
        self.st["section"].setText(str(st["section"]))
        self.st["behavior"].setText(str(st["behavior"]))
        self.st["camera"].setText(str(st["camera"]))
        self.st["notes"].setText(f"{self._notes_rate:.1f}")
        self.st["peers"].setText(str(st["peers"]))
        self.st["tick"].setText(f"{st['tick_ms']} ms")
        self.st["sent"].setText(str(st["sent"]))
        frac = beat - math.floor(beat)
        for i, led in enumerate(self.beat_leds):
            on = i == inbar % 4 and st["playing"]
            col = (tok["accent"] if i == 0 else tok["text"]) if on and frac < 0.35 else \
                (tok["text3"] if on else tok["border"])
            led.setStyleSheet(f"background: {col}; border-radius: 4px")

    def closeEvent(self, ev) -> None:
        self._collect()
        try:
            self.s.save()
        except OSError:
            pass
        self.engine.stop()
        super().closeEvent(ev)


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Myrmex")
    app.setApplicationDisplayName("Myrmex")
    app.setWindowIcon(icons.app_icon())
    holder: dict = {}
    theme.follow_system(app, on_change=lambda: holder["w"]._nav_icons() if "w" in holder else None)
    w = MainWindow()
    holder["w"] = w
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
