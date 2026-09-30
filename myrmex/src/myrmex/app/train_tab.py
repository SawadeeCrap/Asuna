"""The Train page: teach the brain your taste - one change of the organism's body at a time, Good or Bad.

brain/trainer.py does the work (LiveSession.set_training); this page starts and stops it, shows the change on
the body and what was learned, and sends your verdicts: the buttons, the keys G / B / S (or → ← ↓) and the MIDI
pads ``train:good`` / ``train:bad`` / ``train:skip`` (MIDI page).
"""
from __future__ import annotations

import os
import shlex

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from . import controllers as C
from . import responsive as R

WORDS = {"geo:size": "big", "geo:elong": "elongated", "geo:flat": "flat", "geo:asym": "asymmetric",
         "geo:clump": "clumped", "geo:reach": "reaching out", "geo:lumpy": "lumpy", "geo:twist": "twisted",
         "dfm:stretch": "stretched", "dfm:width": "wide", "dfm:height": "tall", "dfm:twist": "twisted",
         "dfm:bend": "bent", "dfm:ripple": "rippled", "dfm:size": "bigger", "change": "big changes",
         "mix": "blends of two forms"}
MATERIAL_WORDS = {"FLUID": "fluid", "ELASTIC": "elastic", "COHESIVE": "cohesive", "STRUCTURED": "structured",
                  "HIGH_STIFFNESS": "hard", "DISPERSED": "dispersed"}
WHO = {"kev": "Kev picked it", "taste": "your taste picked it", "explore": "exploring (new ground)"}


def _word(key: str) -> str:
    if key in WORDS:
        return WORDS[key]
    if key.startswith("mat:"):
        return MATERIAL_WORDS.get(key[4:], key[4:].lower())
    return key.split(":", 1)[-1]


def train_tab(win) -> QWidget:
    d = C.train_settings(win.s)
    w = QWidget()
    v = QVBoxLayout(w)
    intro = QLabel("The organism alone - no music, no hand, no knobs, no effects - and one change of its body at a "
                   "time. It stays until you judge it: Good or Bad brings the next one at once. Watch it in Blender "
                   "(the camera circles it). Your verdicts teach the brain your taste for live play and are kept as "
                   "Kev's fine-tune data.")
    intro.setWordWrap(True)
    intro.setProperty("muted", True)
    v.addWidget(intro)

    row = QHBoxLayout()
    win.btn_train = QPushButton("Start training")
    win.btn_train.setObjectName("primary")
    win.btn_train.clicked.connect(lambda: _toggle(win))
    row.addWidget(win.btn_train)
    win.lbl_train_state = QLabel("")
    win.lbl_train_state.setProperty("muted", True)
    win.lbl_train_state.setWordWrap(True)
    row.addWidget(win.lbl_train_state, 1)
    v.addLayout(row)

    now = QGroupBox("On the body now")
    nv = QVBoxLayout(now)
    win.lbl_train_now = QLabel("–")
    win.lbl_train_now.setWordWrap(True)
    win.lbl_train_now.setStyleSheet("font-size: 17px; font-weight: 600;")
    win.lbl_train_now.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    nv.addWidget(win.lbl_train_now)
    win.lbl_train_who = QLabel("")
    win.lbl_train_who.setProperty("muted", True)
    win.lbl_train_who.setWordWrap(True)
    nv.addWidget(win.lbl_train_who)
    win.btn_train_bad = QPushButton("Bad   B / ←")
    win.btn_train_skip = QPushButton("Skip   S / ↓")
    win.btn_train_good = QPushButton("Good   G / →")
    win.btn_train_good.setObjectName("primary")
    for b, verdict in ((win.btn_train_bad, False), (win.btn_train_skip, None), (win.btn_train_good, True)):
        b.setMinimumHeight(40)
        b.setEnabled(False)
        b.clicked.connect(lambda _=False, vv=verdict: _rate(win, vv))
    nv.addWidget(R.grid_box([win.btn_train_bad, win.btn_train_skip, win.btn_train_good], min_cell=100, spacing=8,
                            max_cols=3))
    win.lbl_train_count = QLabel("")
    win.lbl_train_count.setProperty("muted", True)
    win.lbl_train_count.setWordWrap(True)
    nv.addWidget(win.lbl_train_count)
    v.addWidget(now)
    for keys, verdict in ((("G", "Right"), True), (("B", "Left"), False), (("S", "Down"), None)):
        for k in keys:
            sc = QShortcut(QKeySequence(k), w)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda vv=verdict: _rate(win, vv))

    opts = QGroupBox("What it shows")
    form = QFormLayout(opts)
    win.cmb_train_range = QComboBox()
    win.cmb_train_range.addItem("every form it knows", "all")
    win.cmb_train_range.addItem("its own forms", "own")
    win.cmb_train_range.setCurrentIndex(max(0, win.cmb_train_range.findData(d["range"])))
    win.cmb_train_range.currentIndexChanged.connect(lambda *_: _set(win, range=win.cmb_train_range.currentData()))
    form.addRow("Forms", win.cmb_train_range)
    win.sl_train_spread = QSlider(Qt.Orientation.Horizontal)
    win.sl_train_spread.setRange(0, 100)
    win.sl_train_spread.setValue(int(round(100 * d["spread"])))
    win.sl_train_spread.setToolTip("How far a change goes beyond the form itself: stretched, squashed, twisted, bent, "
                                   "rippled, scaled (0: the forms as they are)")
    win.sl_train_spread.valueChanged.connect(lambda val: _set(win, spread=val / 100.0))
    form.addRow("How far it changes", win.sl_train_spread)
    win.chk_train_kev = QCheckBox("Kev picks what to show (the local Kev server - Creature page)")
    win.chk_train_kev.setChecked(d["kev"])
    win.chk_train_kev.toggled.connect(lambda on: _set(win, kev=bool(on)))
    form.addRow(win.chk_train_kev)
    win.chk_train_fx = QCheckBox("Effects off while training")
    win.chk_train_fx.setChecked(d["fx_off"])
    win.chk_train_fx.toggled.connect(lambda on: _set(win, fx_off=bool(on)))
    form.addRow(win.chk_train_fx)
    v.addWidget(opts)

    learned = QGroupBox("What it has learned")
    lv = QVBoxLayout(learned)
    win.lbl_train_taste = QLabel("Nothing yet - judge a few changes.")
    win.lbl_train_taste.setWordWrap(True)
    lv.addWidget(win.lbl_train_taste)
    b = C.brain_settings(win.s)
    win.chk_taste_live = QCheckBox("Use my taste in live play (the morphology brain, Creature page)")
    win.chk_taste_live.setChecked(bool(b.get("taste", True)))
    win.chk_taste_live.toggled.connect(lambda on: _brain(win, taste=bool(on)))
    lv.addWidget(win.chk_taste_live)
    win.chk_kev_liked = QCheckBox("Kev asks what I would like (once Kev is fine-tuned on my verdicts)")
    win.chk_kev_liked.setChecked(b.get("kev_ask") == "liked")
    win.chk_kev_liked.toggled.connect(lambda on: _brain(win, kev_ask="liked" if on else "choice"))
    lv.addWidget(win.chk_kev_liked)
    v.addWidget(learned)

    kev = QGroupBox("Kev's fine-tune data")
    kv = QVBoxLayout(kev)
    win.lbl_train_files = QLabel("")
    win.lbl_train_files.setWordWrap(True)
    win.lbl_train_files.setProperty("muted", True)
    win.lbl_train_files.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    kv.addWidget(win.lbl_train_files)
    win.txt_train_cmd = QLineEdit("")
    win.txt_train_cmd.setReadOnly(True)
    win.txt_train_cmd.setToolTip("In the kev folder, once there are a few hundred verdicts (docs/MORPHOLOGY_BRAIN.md)")
    kv.addWidget(win.txt_train_cmd)
    reveal = QPushButton("Show in Finder")
    reveal.clicked.connect(lambda: _reveal(win))
    kv.addWidget(reveal, 0, Qt.AlignmentFlag.AlignLeft)
    v.addWidget(kev)
    v.addStretch(1)
    _files(win, None)
    return w


# ---------------------------------------------------------------------------- actions
def _status(win) -> dict:
    return (win.engine.status() or {}).get("training") or {"on": False} if win.engine.running else {"on": False}


def _toggle(win) -> None:
    if not win.engine.running:
        win.log("! Train: start the engine first (Live page) - and open Blender to watch")
        refresh_train(win)
        return
    st = win.engine.training(not _status(win).get("on"))
    if st.get("error"):
        win.log(f"! Train: {st['error']}")
    elif st.get("on"):
        win.log(f"Train: {st.get('organism', '')} - {st.get('forms', 0)} forms"
                + (", stretched / twisted / bent" if st.get("deform") else ""))
    refresh_train(win)


def _rate(win, verdict) -> None:
    if not win.engine.running or not _status(win).get("on"):
        return
    win.engine.train_rate(verdict)
    refresh_train(win)


def _set(win, **kw) -> None:
    win.s.train = {**(win.s.train or {}), **kw}
    win.s.save()
    if win.engine.running and _status(win).get("on"):
        win.engine.training(True)                         # (settings on the fly; Kev on / off starts afresh)
    refresh_train(win)


def _brain(win, **kw) -> None:
    win.s.brain = {**(win.s.brain or {}), **kw}
    win.s.save()
    if win.engine.running:
        win.engine.brain_config(dict(kw))


def _reveal(win) -> None:
    import subprocess
    import sys
    folder = C.train_settings(win.s)["log_dir"]
    os.makedirs(folder, exist_ok=True)
    if sys.platform == "darwin":
        subprocess.Popen(["open", folder])
    else:
        win.log(f"Train data: {folder}")


def _files(win, st: dict | None) -> None:
    d = C.train_settings(win.s)
    org = (st or {}).get("organism") or win.s.backend
    kev = os.path.join(d["log_dir"], f"kev-{org}.jsonl")
    n = ((st or {}).get("kev") or {}).get("records")
    if n is None:
        try:
            with open(kev, encoding="utf-8") as f:
                n = sum(1 for line in f if line.strip())
        except OSError:
            n = 0
    win.lbl_train_files.setText(f"{n} verdict{'s' if n != 1 else ''} for {org} in {kev} - the question Kev is asked, "
                                "labelled with your verdict. Fine-tune Kev on them (kev folder):")
    win.txt_train_cmd.setText(f"uv run python -m kev.train --data {shlex.quote(kev)} --init_from jaredpalmer/kev-0.8b "
                              f"--out runs/myrmex-{org}")


# ---------------------------------------------------------------------------- the page, as it is now
def focus(win) -> None:
    """The page was opened: its keys (G / B / S, arrows) work at once."""
    if hasattr(win, "btn_train"):
        (win.btn_train_good if win.btn_train_good.isEnabled() else win.btn_train).setFocus()
    refresh_train(win)


def refresh_train(win) -> None:
    if not hasattr(win, "btn_train"):
        return
    st = _status(win)
    on = bool(st.get("on"))
    win.btn_train.setText("Stop training" if on else "Start training")
    win.btn_train.setProperty("running", on)
    win.btn_train.style().unpolish(win.btn_train)
    win.btn_train.style().polish(win.btn_train)
    cur = st.get("current") if on else None
    ready = on and cur is not None and not st.get("busy")
    for b in (win.btn_train_bad, win.btn_train_skip, win.btn_train_good):
        b.setEnabled(ready)
    if not win.engine.running:
        win.lbl_train_state.setText("Start the engine (Live page) and open Blender to watch.")
    elif not on:
        win.lbl_train_state.setText(f"Organism: {win.s.backend} (Character page).")
    else:
        extra = "" if st.get("deform") else " - this organism has no stretch / twist / bend: forms and materials only"
        win.lbl_train_state.setText(f"Training {st.get('organism', '')}: {st.get('forms', 0)} forms{extra}.")
    if not on:
        win.lbl_train_now.setText("–")
        win.lbl_train_who.setText("")
    elif cur is None:
        win.lbl_train_now.setText("preparing the next change …")
        win.lbl_train_who.setText("")
    else:
        win.lbl_train_now.setText(f"{cur['step']}.  {cur['label']}")
        who = [WHO.get(cur.get("source", ""), cur.get("source", ""))]
        if cur.get("p_kev") is not None:
            who.append(f"Kev thinks you'd like it: {cur['p_kev']:.0%}")
        if cur.get("p_taste") is not None:
            who.append(f"your taste model: {cur['p_taste']:.0%}")
        win.lbl_train_who.setText(" · ".join(who))
    if on:
        c = st.get("counts", {})
        tst, kv = st.get("taste", {}), st.get("kev", {})
        parts = [f"{c.get('good', 0)} good, {c.get('bad', 0)} bad, {c.get('skip', 0)} skipped this time"]
        if tst.get("accuracy") is not None:
            parts.append(f"the taste model guessed {tst['accuracy']:.0%} of your last verdicts")
        if kv.get("accuracy") is not None:
            parts.append(f"Kev {kv['accuracy']:.0%}")
        win.lbl_train_count.setText(" · ".join(parts))
        if st.get("error"):
            win.lbl_train_count.setText(win.lbl_train_count.text() + f"  ! {st['error']}")
        n = tst.get("n", 0)
        if n:
            likes = ", ".join(_word(k) for k, _ in tst.get("likes", [])) or "-"
            dislikes = ", ".join(_word(k) for k, _ in tst.get("dislikes", [])) or "-"
            win.lbl_train_taste.setText(f"From {n} verdicts: you like {likes}; you don't like {dislikes}.")
        _files(win, st)
    else:
        win.lbl_train_count.setText("")


__all__ = ["train_tab", "refresh_train", "focus"]
