"""The Takes page and the rail's REC: record exactly what you want, see how long every take is, render it.

* REC starts a take, STOP ends it and saves it - the take holds only what happened in between.  From the red
  button on the left rail (every page), the page's big button, ⌘R, or a MIDI pad mapped to ``take``.
* Stopping the engine while recording saves the take too.  Nothing records by itself any more.
* The list shows every take with its length; the one selected (the newest by default) is what Open / Render use.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from . import controllers as C

IDLE_NOTE = "Press REC when you start playing and STOP when you are done: the take is exactly that."


# ---------------------------------------------------------------------------- the rail's REC (every page)
def rec_rail(win) -> QWidget:
    box = QWidget()
    v = QVBoxLayout(box)
    v.setContentsMargins(0, 0, 0, 6)
    v.setSpacing(3)
    win.btn_rec_rail = QPushButton("●")
    win.btn_rec_rail.setObjectName("rec")
    win.btn_rec_rail.setFixedSize(38, 38)
    win.btn_rec_rail.setToolTip("REC a take now (⌘R)")
    win.btn_rec_rail.clicked.connect(lambda: toggle_rec(win))
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(win.btn_rec_rail)
    row.addStretch(1)
    v.addLayout(row)
    win.lbl_rec_rail = QLabel("")
    win.lbl_rec_rail.setObjectName("rectime")
    win.lbl_rec_rail.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    v.addWidget(win.lbl_rec_rail)
    sc = QShortcut(QKeySequence("Ctrl+R"), win)
    sc.setContext(Qt.ShortcutContext.ApplicationShortcut)
    sc.activated.connect(lambda: toggle_rec(win))
    win._rec_seen = {"last": None, "error": None, "on": None}
    return box


def toggle_rec(win) -> None:
    """REC <-> STOP.  REC with the engine stopped starts it first."""
    st = win.engine.rec_status()
    if st.get("on"):
        path = win.engine.rec_stop()
        if path:
            win.log(f"■ STOP: {os.path.basename(path)} · {C.fmt_seconds(st.get('seconds', 0.0))} - saving…")
    else:
        if not win.engine.running:
            win.start_engine()
            if not win.engine.running:
                return
        if win.engine.rec_start():
            win.log(f"● REC: recording a take (into {C.takes_dir(win.s)})")
        else:
            win.log("! could not start recording (is the takes folder set on the Takes page?)")
    refresh_rec(win)


def refresh_rec(win) -> None:
    """(Every tick.)  The buttons, the clock, and the take just saved."""
    st = win.engine.rec_status()
    on = bool(st.get("on"))
    seen = win._rec_seen
    if seen["on"] != on:
        seen["on"] = on
        for b in (win.btn_rec_rail, getattr(win, "btn_rec", None)):
            if b is None:
                continue
            b.setProperty("recording", on)
            b.style().unpolish(b)
            b.style().polish(b)
        win.btn_rec_rail.setText("■" if on else "●")
        win.btn_rec_rail.setToolTip("STOP the take and save it (⌘R)" if on else "REC a take now (⌘R)")
        if getattr(win, "btn_rec", None) is not None:
            win.btn_rec.setText("■   STOP take" if on else "●   REC take")
    win.lbl_rec_rail.setText(C.fmt_seconds(st.get("seconds", 0.0))[:-2] if on else "")
    lbl = getattr(win, "lbl_rec", None)
    if lbl is not None:
        if on:
            lbl.setText(f"Recording  {C.fmt_seconds(st.get('seconds', 0.0))}  ·  {st.get('frames', 0)} frames")
        elif st.get("saving"):
            lbl.setText("Saving the take…")
        elif st.get("last"):
            lbl.setText(f"Saved: {os.path.basename(st['last'])}  ·  {C.fmt_seconds(st.get('last_seconds', 0.0))}")
        else:
            lbl.setText(IDLE_NOTE)
    last = st.get("last")
    if last and last != seen["last"] and not st.get("saving"):
        seen["last"] = last
        win.log(f"take saved: {os.path.basename(last)} · {C.fmt_seconds(st.get('last_seconds', 0.0))}")
        fill_takes(win, select=last)
    err = st.get("error")
    if err and err != seen["error"]:
        seen["error"] = err
        win.log(f"! the take could not be saved: {err}")


# ---------------------------------------------------------------------------- the page
def takes_tab(win) -> QWidget:
    w = QWidget()
    v = QVBoxLayout(w)
    # --- REC / STOP
    box = QGroupBox("Record a take")
    lay = QVBoxLayout(box)
    row = QHBoxLayout()
    win.btn_rec = QPushButton("●   REC take")
    win.btn_rec.setObjectName("recbig")
    win.btn_rec.setMinimumHeight(40)
    win.btn_rec.setMinimumWidth(150)
    win.btn_rec.clicked.connect(lambda: toggle_rec(win))
    row.addWidget(win.btn_rec)
    win.lbl_rec = QLabel(IDLE_NOTE)
    win.lbl_rec.setWordWrap(True)
    row.addWidget(win.lbl_rec, 1)
    lay.addLayout(row)
    hint = QLabel("REC and STOP any time while the engine runs: the red button on the left (on every page), ⌘R, "
                  "or a MIDI pad mapped to 'take' on the MIDI page. REC with the engine stopped starts it. "
                  "Stopping the engine while recording saves the take too.")
    hint.setWordWrap(True)
    hint.setProperty("muted", True)
    lay.addWidget(hint)
    form = QFormLayout()
    row = QHBoxLayout()
    win.ed_rec = QLineEdit(win.s.record_dir)
    b = QPushButton("…")
    b.clicked.connect(lambda: win._pick_dir(win.ed_rec))
    win.ed_rec.editingFinished.connect(lambda: _folder_changed(win))
    row.addWidget(win.ed_rec, 1)
    row.addWidget(b)
    form.addRow("Takes folder", row)
    lay.addLayout(form)
    v.addWidget(box)
    # --- the takes
    box = QGroupBox("Takes")
    lay = QVBoxLayout(box)
    win.list_takes = QListWidget()
    win.list_takes.setObjectName("takes")
    win.list_takes.setMinimumHeight(150)
    win.list_takes.itemDoubleClicked.connect(lambda *_: win.open_take(False))
    lay.addWidget(win.list_takes)
    row = QHBoxLayout()
    for label, fn in (("Open in Blender", lambda: win.open_take(False)),
                      ("Render → .mp4", lambda: win.open_take(True)),
                      ("Other take…", lambda: win.open_take(False, True))):
        b = QPushButton(label)
        if label.startswith("Render"):
            b.setObjectName("primary")
        b.clicked.connect(fn)
        row.addWidget(b)
    if sys.platform == "darwin":
        b = QPushButton("Show in Finder")
        b.clicked.connect(lambda: _reveal(win))
        row.addWidget(b)
    row.addStretch(1)
    lay.addLayout(row)
    form = QFormLayout()
    row = QHBoxLayout()
    win.ed_song = QLineEdit(win.s.take_audio)
    win.ed_song.setPlaceholderText("the track exported from Ableton (from bar 1) — lined up automatically")
    b = QPushButton("…")
    b.clicked.connect(lambda: win._pick_file(win.ed_song, "Song", "Audio (*.wav *.aif *.aiff *.mp3 *.flac)"))
    row.addWidget(win.ed_song, 1)
    row.addWidget(b)
    form.addRow("Song for renders", row)
    row = QHBoxLayout()
    win.cmb_rsize = QComboBox()
    win.cmb_rsize.addItems(["1920x1080", "1080x1920", "1080x1080", "3840x2160", "1280x720"])
    win.cmb_rsize.setCurrentText(win.s.render_size)
    win.cmb_rquality = QComboBox()
    for label, key in (("Draft (EEVEE fast)", "eevee_preview"), ("Final (EEVEE)", "eevee"),
                       ("Cinema (Cycles, slow)", "cycles")):
        win.cmb_rquality.addItem(label, key)
    win.cmb_rquality.setCurrentIndex(max(0, win.cmb_rquality.findData(win.s.render_quality)))
    win.cmb_rquality.setEnabled(not win.s.keep_blender_settings)
    win.cmb_rquality.setToolTip("Used when 'Keep my Blender settings' (Character page) is off")
    row.addWidget(win.cmb_rsize)
    row.addWidget(win.cmb_rquality, 1)
    form.addRow("Video", row)
    lay.addLayout(form)
    note = QLabel("A render is exactly as long as the take: its frames at the take's 30 fps, the song lined up "
                  "under it. The video is written next to the take.")
    note.setWordWrap(True)
    note.setProperty("muted", True)
    lay.addWidget(note)
    v.addWidget(box)
    # --- where the poses go
    box = QGroupBox("Pose stream (to Blender)")
    form = QFormLayout(box)
    win.spin_pose = QSpinBox()
    win.spin_pose.setRange(1024, 65535)
    win.spin_pose.setValue(int(win.s.pose_port))
    form.addRow("Pose port", win.spin_pose)
    win.spin_fps = QSpinBox()
    win.spin_fps.setRange(15, 240)
    win.spin_fps.setValue(int(win.s.out_fps))
    form.addRow("Poses per second", win.spin_fps)
    win.ed_targets = QLineEdit(win.s.extra_targets)
    win.ed_targets.setPlaceholderText("extra targets, e.g. 192.168.1.20:9101")
    form.addRow("Also send to", win.ed_targets)
    win.chk_autostart = QCheckBox("Start the engine when the app starts")
    win.chk_autostart.setChecked(win.s.start_engine_on_launch)
    form.addRow(win.chk_autostart)
    v.addWidget(box)
    row = QHBoxLayout()
    row.addStretch(1)
    apply = QPushButton("Apply (restart engine)")
    apply.clicked.connect(win.apply_and_restart)
    row.addWidget(apply)
    v.addLayout(row)
    v.addStretch(1)
    fill_takes(win)
    return w


def _folder_changed(win) -> None:
    folder = win.ed_rec.text().strip()
    if folder and folder != win.s.record_dir:
        win.s.record_dir = folder
        win.s.save()
        if win.engine.running and win.engine.session is not None and not win.engine.rec_status().get("on"):
            win.engine.session.cfg.record = C.takes_dir(win.s)       # the next take goes there
        fill_takes(win)


def take_label(path: str) -> str:
    info = C.take_info(path)
    who = (info.get("variant") or "take").replace("_", " ").title()
    when = time.strftime("%d %b  %H:%M:%S", time.localtime(os.path.getmtime(path))) if os.path.exists(path) else ""
    length = C.fmt_seconds(info["seconds"]) if info else "?"
    video = " · video ✓" if C.take_videos(path) else ""
    return f"{who}   ·   {length}   ·   {when}{video}"


def fill_takes(win, select: str | None = None) -> None:
    lst = getattr(win, "list_takes", None)
    if lst is None:
        return
    cur = select or selected_take(win)
    lst.clear()
    for path in C.list_takes(C.takes_dir(win.s))[:200]:
        it = QListWidgetItem(take_label(path))
        it.setData(Qt.ItemDataRole.UserRole, path)
        it.setToolTip(path)
        lst.addItem(it)
        if cur and os.path.abspath(path) == os.path.abspath(cur):
            lst.setCurrentItem(it)
    if lst.currentItem() is None and lst.count():
        lst.setCurrentRow(0)


def selected_take(win) -> str:
    lst = getattr(win, "list_takes", None)
    it = lst.currentItem() if lst is not None else None
    return it.data(Qt.ItemDataRole.UserRole) if it is not None else ""


def _reveal(win) -> None:
    path = selected_take(win) or C.takes_dir(win.s)
    videos = C.take_videos(path) if path.endswith(".npz") else []
    subprocess.Popen(["open", "-R", videos[0] if videos else path])


__all__ = ["rec_rail", "toggle_rec", "refresh_rec", "takes_tab", "fill_takes", "selected_take", "take_label"]
