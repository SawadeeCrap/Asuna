"""Training the brain's taste (brain/trainer.py, the Train page): the engines' deformations (creature/colony.py
deform), wide proposals, verdicts -> Kev fine-tune records + the taste model, Kev picking through yes / no
questions, the live session's training mode, the live brain's reshape candidates and taste, the app page."""
import json
import math
import time

import numpy as np
import pytest

from test_brain import MockKev, mock_kev  # noqa: F401  (the mock Kev server: canned answers, no model)

from myrmex.brain.fingerprint import GEO, Fingerprint, embed, geometry
from myrmex.creature.colony import DEFORM, DEFORM_RANGE, ColonyConfig, ColonyEngine, deform, deform_vector, shape3

PH0 = {"t": 0.0, "beat": 0.0, "spin": 0.0, "flap": 0.0, "pulse": 0.0, "gait": 0.0, "mech": 0.5, "snap": 0.3}


# ---------------------------------------------------------------------------- the engines can deform
def test_deform_is_the_form_itself_at_zero_and_changes_it_otherwise():
    U = np.random.default_rng(3).random((128, 3))
    P = shape3("SPINDLE", U, 1.6, 1.1, PH0, -3.0)
    P -= P.mean(0)
    assert deform(P, np.zeros(len(DEFORM))) is P                          # nothing asked: nothing done
    g0 = geometry(P)
    long = geometry(deform(P, deform_vector({"stretch": 0.9, "width": -0.5, "height": -0.5})))
    assert long[GEO.index("elong")] > g0[GEO.index("elong")] + 0.8        # longer and thinner
    flat = geometry(deform(P, deform_vector({"width": 0.7, "height": -0.7})))
    assert flat[GEO.index("flat")] > g0[GEO.index("flat")] + 0.5
    tw = geometry(deform(P, deform_vector({"twist": 1.5, "width": 0.6, "height": -0.6})))
    assert tw[GEO.index("twist")] > 0.5                                   # a twisted ribbon reads as twisted
    big = geometry(deform(P, deform_vector({"size": 0.35})))
    assert big[0] == pytest.approx(g0[0] + 0.35, abs=1e-6)                # log size
    v = deform_vector({"stretch": 9.0, "ripple": -1.0, "nope": 3.0, "twist": float("nan")})
    assert v[DEFORM.index("stretch")] == DEFORM_RANGE["stretch"][1] and v[DEFORM.index("ripple")] == 0.0
    assert not np.isnan(v).any()


def _hold(eng, form: str, d: dict, secs: float = 3.0) -> np.ndarray:
    b = eng.bodies[0]
    for _ in range(int(secs * 120)):
        b.intent, b.intent_t, b.dwell = "HOVER", 0.0, 1e9
        b.goal_shape(form, 2.6)
        b.dfm_goal = deform_vector(d)
        eng.update(1 / 120)
    return geometry(eng.x[eng.own == 0])


def test_the_real_body_takes_the_deformation():
    from myrmex.creature.mimetic import SpearConfig, SpearEngine
    eng = SpearEngine(SpearConfig(seed=0))
    eng.morph_tau, eng.plastic_scale = 0.5, 0.35
    ball = _hold(eng, "CORE", {})
    rod = _hold(eng, "CORE", {"stretch": 0.9})
    assert np.isfinite(eng.x).all()
    assert ball[GEO.index("elong")] < 0.2 < 0.6 < rod[GEO.index("elong")]  # the same form, stretched: a rod
    small = _hold(eng, "CORE", {"size": -0.35})
    assert math.exp(small[0]) < 0.85 * math.exp(ball[0])
    b = eng.bodies[0]
    b.dfm_goal = deform_vector({"twist": 1.0})
    eng._set_intent(b, "CRUISE")                                          # its own choice: the form as it is
    assert not b.dfm_goal.any()


def test_default_engines_are_unchanged():
    """No deformation asked: the colony family moves exactly as before (same seed, same nodes)."""
    a, b = ColonyEngine(ColonyConfig(seed=5)), ColonyEngine(ColonyConfig(seed=5))
    for _ in range(240):
        a.update(1 / 120)
        b.update(1 / 120)
    assert np.allclose(a.x, b.x) and not a.bodies[0].dfm.any() and a.morph_tau is None


# ---------------------------------------------------------------------------- the trainer
def _vocab(organism="spear"):
    from myrmex.brain.vocab import vocabulary
    from myrmex.creature.mimetic import VARIANTS
    return vocabulary(VARIANTS[organism][0], organism)


def _cur(vocab):
    from myrmex.creature.colony import shape3 as s3
    w = vocab.weights({vocab.signature(): 1.0})
    X = s3(vocab.signature(), np.random.default_rng(1).random((128, 3)), 1.6, 1.1, PH0, -3.0)
    return Fingerprint(w, geometry(X), np.array([1.0, 0.35, 0.6, 0.8, 1.9, 0.0]), 1)


def test_proposals_are_clear_changes_and_wide(tmp_path):
    from myrmex.brain.fingerprint import Scale
    from myrmex.brain.trainer import Trainer
    voc = _vocab()
    tr = Trainer(voc, str(tmp_path), "all", 0.9, seed=1)
    cur = _cur(voc)
    scale = Scale.from_geometry(voc.geometry)
    shown, forms, deformed = [], set(), 0
    for _ in range(16):
        p = tr.propose(cur)
        assert p.fp is not None and p.label and len(p.options) == 6 and p.source in ("taste", "explore")
        shown.append(p)
        forms.update(p.blend)
        deformed += bool(p.deform)
        tr.rate(None)
    d = [float(np.linalg.norm(a.emb - b.emb)) for a, b in zip(shown, shown[1:])]
    assert min(d) > 0.3 and np.median(d) > 0.6                            # every step a clear change
    assert len(forms) >= 8 and deformed >= 12                             # many forms, mostly deformed
    own = Trainer(voc, "", "own", 0.0, seed=2)
    assert set(own.forms()) == set(f for f in voc.free if f in voc.forms) and len(own.forms()) < len(tr.forms())
    p = own.propose(cur)
    assert not p.deform and set(p.blend) <= set(own.forms())              # spread 0: the forms as they are
    assert embed(p.fp, scale).shape == shown[0].emb.shape


def test_verdicts_write_kev_records_and_teach_the_taste(tmp_path):
    from myrmex.brain.taste import Taste
    from myrmex.brain.trainer import Trainer
    voc = _vocab()
    tr = Trainer(voc, str(tmp_path), "all", 0.9, seed=4)
    cur = _cur(voc)
    for _ in range(60):                                                   # this performer likes long bodies
        p = tr.propose(cur)
        tr.rate(p.fp.geo[GEO.index("elong")] > 0.9)
    assert tr.rate(True) is False                                         # nothing shown: nothing rated
    st = tr.status()
    assert st["counts"]["good"] + st["counts"]["bad"] == 60 and st["taste"]["n"] == 60
    assert st["taste"]["accuracy"] is not None and st["taste"]["accuracy"] >= 0.65
    assert tr.taste.w.get("geo:elong", 0.0) > 0.3                         # it learned what was liked
    assert any(k == "geo:elong" for k, _ in st["taste"]["likes"])
    with open(st["files"]["kev"], encoding="utf-8") as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) == 60 == st["kev"]["records"]
    for r in recs:                                                        # kev.data.load_records' own checks
        assert "state" in r and isinstance(r["questions"], dict) and set(r) == {"state", "questions"}
        q = r["questions"]["liked"]
        assert q["type"] == "noul" and isinstance(q["label"], bool) and "Would the performer like" in q["instructions"]
    with open(st["files"]["log"], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert rows[0]["kind"] == "verdict" and "features" in rows[0] and "options" in rows[0]
    again = Taste.load(st["files"]["taste"])                              # kept: a new session starts from it
    assert again.n == 60 and again.w.get("geo:elong", 0) == pytest.approx(tr.taste.w["geo:elong"], rel=1e-6)
    tr2 = Trainer(voc, str(tmp_path), "all", 0.9, seed=5)
    assert tr2.taste.n == 60 and tr2.kev_records == 60


def test_kev_picks_with_yes_no_questions_and_falls_back(tmp_path, mock_kev):  # noqa: F811
    from myrmex.brain.kev_client import KevClient
    from myrmex.brain.trainer import Trainer
    srv, url = mock_kev
    voc = _vocab()
    tr = Trainer(voc, str(tmp_path), "all", 0.9, kev=KevClient(url, timeout=2.0), seed=6)
    cur = _cur(voc)
    kev_picks = 0
    for _ in range(12):
        p = tr.propose(cur)
        q = srv.requests[-1]["questions"]
        assert len(q) == 6 and all(v["type"] == "noul" for v in q.values())
        if p.source == "kev":
            kev_picks += 1
            assert p.p_kev == pytest.approx(max(0.85 if "stretched" in o else 0.2 for o in p.options))
        tr.rate(True)
    assert kev_picks >= 7 and tr.status()["kev"]["on"]
    srv.shutdown()                                                        # Kev gone: the taste model picks
    tr.kev.timeout = 0.3
    p = tr.propose(cur)
    assert p.source in ("taste", "explore") and tr.status()["error"].startswith("Kev")


# ---------------------------------------------------------------------------- the live session's training mode
def _session(organism="spear"):
    from myrmex.realtime.inputs import InputConfig
    from myrmex.realtime.session import LiveConfig, LiveSession
    return LiveSession(LiveConfig(backend=organism, clock="internal", out=[], inputs=InputConfig(osc_port=0)),
                       start_inputs=False, now=0.0)


def _run(ses, secs):
    ses._test_now = getattr(ses, "_test_now", 0.0)
    for _ in range(int(secs * 120)):
        ses._test_now += 1 / 120
        ses.step(ses._test_now)


def _wait(ses):
    t0 = time.time()
    while ses.training is not None and ses.training.busy and time.time() - t0 < 10:
        time.sleep(0.01)


def test_session_training_mode(tmp_path):
    ses = _session()
    _run(ses, 1.0)
    eng = ses.creature.engine
    eng.set_parameter("mutation", 0.9)                                   # a knob the app had turned
    st = ses.set_training({"on": True, "spread": 0.9, "log_dir": str(tmp_path)})
    assert st["on"] and st["organism"] == "spear" and st["deform"] and st["forms"] >= 15
    assert ses.fx.cfg["enabled"] is False and ses.camera.mode == "manual" and eng.morph_tau == 0.5
    assert "mutation" not in eng.params.manual                           # knobs back to the organism's own
    _wait(ses)
    _run(ses, 1.0)
    first = ses.status()["training"]["current"]
    assert first["step"] == 1 and first["label"]
    lead = eng.bodies[0]
    assert lead.dwell > 1e8                                               # held until the verdict
    ses.inputs.controls["expansion"] = 1.0                               # the knobs stay out
    _run(ses, 2.0)
    assert "expansion" not in eng.params.manual
    assert ses.status()["training"]["current"]["step"] == 1               # nothing moves on by itself
    ses.train_rate(True)
    _wait(ses)
    _run(ses, 0.5)
    st = ses.status()["training"]
    assert st["current"]["step"] == 2 and st["counts"]["good"] == 1
    ses.inputs.triggers.append(("train:bad", 0.0))                        # a MIDI pad
    _run(ses, 0.1)
    _wait(ses)
    assert ses.status()["training"]["counts"]["bad"] == 1
    ses.inputs.triggers.append(("brain:good", 0.0))                       # (the brain's pad counts here too)
    _run(ses, 0.1)
    _wait(ses)
    assert ses.status()["training"]["counts"]["good"] == 2
    ses.set_training({"on": False})
    assert ses.training is None and ses.fx.cfg["enabled"] is True and eng.morph_tau is None
    assert not lead.dfm_goal.any() and lead.dwell < 10                    # its own state machine goes on
    assert ses.status()["training"] == {"on": False}
    _run(ses, 1.0)
    assert np.isfinite(eng.x).all()


def test_training_needs_an_organism():
    from myrmex.brain.trainer import TrainingRun
    ses = _session("polyalloy")
    st = ses.set_training({"on": True})
    assert st["on"] and st["deform"] is False                             # forms and materials only
    _wait(ses)
    ses.set_training({"on": False})
    assert TrainingRun.MORPH_TAU > 0


# ---------------------------------------------------------------------------- the live brain
def test_live_brain_reshapes_and_uses_the_taste(tmp_path):
    from myrmex.brain.core import BrainConfig, BrainCore, Snapshot
    from myrmex.brain.taste import Taste, taste_path
    voc = _vocab()
    cur = _cur(voc)
    cfg = BrainConfig.from_dict({"enabled": True, "log_dir": str(tmp_path), "controls": {"autonomy": 1.0}})
    assert cfg.reshape and cfg.taste and cfg.kev_ask == "choice"
    core = BrainCore(voc, cfg)
    assert core.taste is None                                             # no verdicts yet
    snap = Snapshot(t=100.0, fp=cur, blend={voc.signature(): 1.0}, free=True, due=True)
    core.rng = np.random.default_rng(0)
    d = None
    for k in range(40):
        snap.t = 100.0 + 20.0 * k
        d = core.step(snap) or d
    ops = {c.op for c in core._cands}
    assert "RESHAPE" in ops and any(c.deform for c in core._cands if c.op == "RESHAPE")
    t = Taste(taste_path(str(tmp_path), voc.organism))                   # a performer who likes stretched forms
    for i in range(30):
        t.rows.append(({"dfm:stretch": 1.0 if i % 2 else 0.0}, 1.0 if i % 2 else 0.0))
    t.fit()
    t.save()
    core2 = BrainCore(voc, cfg)
    assert core2.taste is not None and core2.taste.w["dfm:stretch"] > 1.0
    from myrmex.brain.candidates import Candidate, evaluate
    from myrmex.brain.core import BrainControls
    plain = Candidate("SHIFT", {"CORE": 1.0}, label="become CORE")
    long = Candidate("RESHAPE", {"CORE": 1.0}, deform={"stretch": 0.9}, label="reshape CORE")
    ctl = BrainControls(autonomy=1.0)
    evaluate([plain, long], voc, core2.scale, cur, core2.memory, 200.0, 0.0, snap.user, ctl)
    U2 = core2._utilities([plain, long], ctl, snap)
    core3 = BrainCore(voc, BrainConfig.from_dict({**cfg.to_dict(), "taste": False}))
    U3 = core3._utilities([plain, long], ctl, snap)
    assert (U2[1] - U2[0]) > (U3[1] - U3[0]) + 0.3                        # the taste moved the choice
    off = BrainConfig.from_dict({"reshape": False})
    assert BrainConfig.from_dict(off.to_dict()).reshape is False


def test_live_brain_asks_kev_what_you_would_like(mock_kev):  # noqa: F811
    from myrmex.brain.core import BrainConfig, BrainCore, Snapshot
    srv, url = mock_kev
    voc = _vocab()
    cur = _cur(voc)
    core = BrainCore(voc, BrainConfig.from_dict({"enabled": True, "mode": "F", "kev_url": url, "kev_timeout": 2.0,
                                                 "kev_ask": "liked", "controls": {"autonomy": 1.0}}))
    snap = Snapshot(t=100.0, fp=cur, blend={voc.signature(): 1.0}, free=True, due=True)
    asked = 0
    for k in range(30):
        snap.t = 100.0 + 20.0 * k
        core.step(snap)
        if srv.requests and all(q["type"] == "noul" for q in srv.requests[-1]["questions"].values()):
            asked += 1
    assert asked >= 3 and core.stats["kev_calls"] >= 3 and core.stats["kev_fail"] == 0


# ---------------------------------------------------------------------------- the app
def test_app_train_page(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from myrmex.app import controllers as C
    from myrmex.app import train_tab as TR
    from myrmex.app import window as W
    from myrmex.app.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    s = AppSettings()
    s.start_engine_on_launch = s.open_blender_on_start = False
    w = W.MainWindow(s)
    w.timer.stop()
    assert "Train" in w.page_index
    assert not w.btn_train_good.isEnabled() and w.btn_train.text() == "Start training"
    w.sl_train_spread.setValue(40)
    w.cmb_train_range.setCurrentIndex(w.cmb_train_range.findData("own"))
    d = C.train_settings(s)
    assert d["spread"] == pytest.approx(0.4) and d["range"] == "own" and d["fx_off"] is True
    w.chk_taste_live.setChecked(False)
    assert s.brain["taste"] is False
    w.chk_kev_liked.setChecked(True)
    assert s.brain["kev_ask"] == "liked"
    TR._toggle(w)                                                          # engine stopped: it says so
    assert "Start the engine" in w.lbl_train_state.text()
    assert "kev.train --data" in w.txt_train_cmd.text() and "kev-" in w.txt_train_cmd.text()

    class FakeEngine:                                                     # (a running engine's answers)
        running = True
        rated = []

        def status(self):
            return {"training": {"on": True, "busy": False, "organism": "spear", "forms": 21, "deform": True,
                                 "current": {"step": 3, "label": "LANCE, stretched x2.1, hard", "source": "taste",
                                             "p_kev": None, "p_taste": 0.72},
                                 "counts": {"good": 2, "bad": 1, "skip": 0},
                                 "taste": {"n": 3, "accuracy": None, "likes": [("dfm:stretch", 0.4)],
                                           "dislikes": [("dfm:ripple", -0.3)]},
                                 "kev": {"on": False, "accuracy": None, "records": 3}}}

        def train_rate(self, v):
            self.rated.append(v)
            return {}

    real = w.engine
    w.engine = FakeEngine()
    try:
        TR.refresh_train(w)
        assert w.btn_train_good.isEnabled() and w.btn_train.text() == "Stop training"
        assert "LANCE, stretched x2.1, hard" in w.lbl_train_now.text() and "72%" in w.lbl_train_who.text()
        assert "stretched" in w.lbl_train_taste.text() and "rippled" in w.lbl_train_taste.text()
        w.btn_train_good.click()
        w.btn_train_bad.click()
        w.btn_train_skip.click()
        assert w.engine.rated == [True, False, None]
    finally:
        w.engine = real
    app.processEvents()
    w.close()


def test_the_live_brain_picks_up_new_verdicts(tmp_path):
    """Training while the brain runs: when training stops, the live brain reads the verdicts given since."""
    from myrmex.brain.core import BrainConfig
    from myrmex.brain.taste import Taste, taste_path
    from myrmex.brain.worker import BrainLink
    ses = _session()
    cfg = BrainConfig.from_dict({"enabled": True, "log_dir": str(tmp_path)})
    ses.cfg.brain = cfg.to_dict()
    link = ses.brain = BrainLink(cfg, ses.creature.engine, "spear", process=False)   # (in-process: to look inside)
    assert link.core is not None and link.core.taste is None
    ses.set_training({"on": True, "log_dir": str(tmp_path)})
    _wait(ses)
    t = Taste(taste_path(str(tmp_path), "spear"))
    for i in range(8):
        t.rows.append(({"dfm:stretch": float(i % 2)}, float(i % 2)))
    t.fit()
    t.save()
    ses.set_training({"on": False})
    assert link.core.taste is not None and link.core.taste.n >= 8
    ses.set_brain({"taste": False})
    assert link.core.taste is None                                        # "Use my taste in live play" off
    ses.stop()
