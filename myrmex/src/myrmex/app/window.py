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

Sidebar navigation, light / dark theme following macOS (``theme.py``).

Settings are saved on exit and restored on the next launch.
"""
from __future__ import annotations

import math
import os
import sys
import time

from PySide6.QtCore import QProcess, QProcessEnvironment, QSize, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
                               QSlider, QSpinBox, QStackedWidget, QVBoxLayout, QWidget)

from . import controllers as C
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
        self.resize(1060, 740)
        self.setMinimumSize(880, 600)
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
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(212)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 20, 0, 16)
        sv.setSpacing(2)
        brand = QLabel("Myrmex")
        brand.setObjectName("brand")
        sub = QLabel("live character engine")
        sub.setObjectName("brandsub")
        for lab in (brand, sub):
            lab.setContentsMargins(22, 0, 16, 0)
            sv.addWidget(lab)
        sv.addSpacing(16)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.nav.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sv.addWidget(self.nav, 1)
        # Engine state, always visible.
        box = QWidget()
        bl = QHBoxLayout(box)
        bl.setContentsMargins(22, 0, 16, 0)
        bl.setSpacing(8)
        self.dot = QLabel()
        self.dot.setFixedSize(8, 8)
        self.lbl_engine = QLabel("Engine stopped")
        self.lbl_engine.setObjectName("enginestate")
        bl.addWidget(self.dot)
        bl.addWidget(self.lbl_engine, 1)
        sv.addWidget(box)
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
            QListWidgetItem(name, self.nav).setSizeHint(QSize(180, 34))
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
        self.chk_hold = QCheckBox("Stand and pose (hold) — otherwise she walks while the music plays")
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
        self.cmb_backend.setCurrentIndex(max(0, self.cmb_backend.findData(self.s.backend)))
        form.addRow("Character type", self.cmb_backend)
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
        form.addRow(QLabel("Tune materials, lights, world and render settings in Blender, then press Save Look in "
                           "the Myrmex panel (N sidebar). Humanoid: the look is saved in the character's .blend. "
                           "Organisms: each type keeps its own look; live sessions and take renders open it."))
        self.chk_keep = QCheckBox("Keep my Blender settings (Myrmex doesn't change EEVEE, colour, shadows, samples)")
        self.chk_keep.setChecked(self.s.keep_blender_settings)
        self.chk_keep.toggled.connect(lambda on: self.cmb_rquality.setEnabled(not on))
        form.addRow(self.chk_keep)
        row = QHBoxLayout()
        self.lbl_looks = QLabel()
        b = QPushButton("Forget saved look")
        b.setToolTip("Delete the saved look of the selected organism: next time the default studio is built")
        b.clicked.connect(self._forget_look)
        row.addWidget(self.lbl_looks, 1)
        row.addWidget(b)
        form.addRow("Saved looks", row)
        self.cmb_backend.currentIndexChanged.connect(lambda *_: self._show_looks())
        self._show_looks()
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

    def _show_looks(self) -> None:
        names = {"creature": "Nanomaterial", "polyalloy": "Polyalloy", "colony": "Colony", "hive": "Hive",
                 "osseous": "Osseous", "osseous_colony": "Osseous Colony", "osseous_hive": "Osseous Hive"}
        parts = [f"{label} {'✓' if C.look_file(b) else '–'}" for b, label in names.items()]
        self.lbl_looks.setText("   ".join(parts))

    def _forget_look(self) -> None:
        b = self.cmb_backend.currentData()
        path = C.look_file(b)
        if not path:
            self.log("no saved look for this character type" if b in C.CREATURE_BACKENDS else
                     "the humanoid look lives in the character's .blend")
            return
        if QMessageBox.question(self, "Myrmex", f"Delete the saved look?\n{path}") == QMessageBox.StandardButton.Yes:
            os.remove(path)
            self.log(f"saved look removed: {path}")
            self._show_looks()

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
                                          self.s.keep_blender_settings)
        if C.look_file(self.s.backend):
            self.log(f"using your saved look: {C.look_file(self.s.backend)}")
        p = QProcess(self)
        qenv = QProcessEnvironment.systemEnvironment()
        for k, val in env.items():
            qenv.insert(k, val)
        p.setProcessEnvironment(qenv)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(lambda: self._pipe(p, "blender"))
        p.finished.connect(lambda *_: self.log("Blender closed"))
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
                             self.s.render_quality, self.s.keep_blender_settings)
        p = QProcess(self)
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
        data = bytes(p.readAllStandardOutput()).decode("utf-8", "replace")
        for line in data.splitlines():
            if line.strip() and (only is None or any(o in line for o in only)):
                self.log(f"[{tag}] {line.strip()}")

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
            self.dot.setStyleSheet(f"background: {tok['text3']}; border-radius: 4px")
            self.lbl_engine.setText("Engine stopped")
            return
        self.dot.setStyleSheet(f"background: {tok['ok'] if not st['errors'] else tok['warn']}; border-radius: 4px")
        self.lbl_engine.setText(f"Running · {st['bpm']:.1f} BPM" if st["bpm"] else "Running")
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
    theme.follow_system(app)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
