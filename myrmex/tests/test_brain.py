"""The morphology brain: its vocabulary comes from the engines, its memory / novelty / candidates behave, the
sandbox measures the same way for every mode, the adapter only uses what an engine already has, and nothing
the brain does - a crash, a hang, a slow or broken Kev - can stall the organism.

The Kev tests talk to ``MockKev``: a MOCK of the documented System One API (canned, deterministic answers;
not Kev, no model).  It checks the client and the brain's confidence / fallback paths only - how good Kev's
decisions are is measured with the real server (``morphology_brain_test.py --kev-bench / --benchmark --kev``)."""
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from myrmex.brain.adapter import inverse_blend, make_adapter, softmax3, user_cue
from myrmex.brain.candidates import DIRECTIONS, RADICAL, UserCue, direction, evaluate, generate
from myrmex.brain.core import BrainConfig, BrainControls, BrainCore, Decision, with_env
from myrmex.brain.fingerprint import Fingerprint, Scale, blend_cloud, embed, geometry
from myrmex.brain.kev_client import KevClient, KevError, describe, is_local
from myrmex.brain.memory import MorphMemory
from myrmex.brain.metrics import diagnose
from myrmex.brain.novelty import novelty
from myrmex.brain.sandbox import AbstractColony, organism_class, run
from myrmex.brain.vocab import MATERIALS, MORPH_EVENTS, SITUATIONAL, vocabulary
from myrmex.brain.worker import BrainLink
from myrmex.creature.control import CreatureControlInput

DT = 1.0 / 120.0


def _spear(seed=0):
    from myrmex.creature.mimetic import SpearConfig, SpearEngine
    eng = SpearEngine(SpearConfig(seed=seed))
    eng.set_input(CreatureControlInput(energy=0.5, playing=True, amplitude=0.5))
    return eng


# ---------------------------------------------------------------------------- vocabulary
def test_vocabulary_is_read_from_the_engines():
    from myrmex.creature.mimetic import SpearEngine
    v = vocabulary(SpearEngine, "spear")
    assert v.family == "colony" and v.forms == tuple(SpearEngine.SHAPE_SET)
    assert v.signature() == "LANCE" and v.identity.max() == 1.0
    assert set(v.free) <= set(v.forms) and not set(v.free) & set(SITUATIONAL)
    assert set(v.events) <= set(MORPH_EVENTS) and v.materials == MATERIALS
    assert v.geometry.shape == (len(v.forms), 128, 3)
    for org in ("colony", "hive", "swarm", "blade"):
        assert vocabulary(organism_class(org), org).free                        # every organism has room to move
    from myrmex.creature.bionic.ferro import FerroEngine
    from myrmex.creature.engine import CreatureEngine
    from myrmex.creature.polyalloy import PolyalloyEngine
    assert vocabulary(FerroEngine, "ferro").family == "bionic"
    assert vocabulary(PolyalloyEngine, "polyalloy").family == "polyalloy"
    assert vocabulary(CreatureEngine, "creature").family == "creature"


# ---------------------------------------------------------------------------- fingerprints
def test_geometry_describes_the_shape_not_its_pose():
    rng = np.random.default_rng(0)
    ball = rng.standard_normal((200, 3))
    rod = ball * np.array([0.2, 0.2, 3.0])
    u, z = rng.uniform(-1, 1, 200), np.linspace(-3, 3, 200)
    ribbon = np.stack([u * np.cos(z * np.pi / 6), u * np.sin(z * np.pi / 6), z], 1)     # half a turn along it
    flat = np.stack([u, 0.03 * rng.standard_normal(200), z], 1)
    th = np.linspace(0, 2 * np.pi, 200)
    coil = np.stack([np.cos(th), np.sin(th), th], 1) + 0.05 * rng.standard_normal((200, 3))
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    g = geometry(rod)
    assert np.allclose(geometry(rod @ q.T + 5.0), g, atol=1e-6)               # moved and turned: the same shape
    assert g[1] > geometry(ball)[1] + 1.0                                      # elongation
    assert geometry(ribbon)[7] == pytest.approx(1.0, abs=0.25)                 # twist: half-turns round the axis
    assert geometry(coil)[7] > 0.6
    assert max(geometry(ball)[7], geometry(rod)[7], geometry(flat)[7]) < 0.2


def test_embedding_distance_orders_changes():
    v = vocabulary(organism_class("spear"), "spear")
    sc = Scale.from_geometry(v.geometry)

    def fp(blend):
        w = v.weights(blend)
        return Fingerprint(w, geometry(blend_cloud(w, v.geometry)), np.zeros(6), 1)
    a, b = fp({"LANCE": 1.0}), fp({"SPINDLE": 1.0})
    mid = fp({"LANCE": 0.6, "SPINDLE": 0.4})
    d = lambda x, y: float(np.linalg.norm(embed(x, sc) - embed(y, sc)))    # noqa: E731
    assert d(a, a) == 0.0
    assert d(a, mid) < d(a, b) and d(mid, b) < d(a, b)
    assert d(a, b) > RADICAL > d(a, mid)                                        # a whole new form vs a hybrid


# ---------------------------------------------------------------------------- memory and novelty
def _emb(i, n=4):
    e = np.zeros(n)
    e[i] = 1.0
    return e


def _fp():
    return Fingerprint(np.ones(3) / 3, np.zeros(8))


def test_memory_prototypes_residence_repetition_oscillation():
    m = MorphMemory(radius=0.3)
    for t in range(0, 20):
        m.observe(float(t), _fp(), _emb(0))
    assert len(m.protos) == 1 and m.residence(19.0) == pytest.approx(19.0)
    for t in range(20, 30):
        m.observe(float(t), _fp(), _emb(1))
    a, b = m.nearest(_emb(0))[0], m.nearest(_emb(1))[0]
    assert a != b and len(m.protos) == 2
    assert m.oscillating(a, 30.0)                                               # going back now: A -> B -> A
    m.observe(30.0, _fp(), _emb(0))
    assert m.protos[a].visits == 2 and m.repetition(a, 30.0) > m.repetition(b, 30.0)
    assert m.protos[a].dwell == pytest.approx(20.0) and m.protos[b].dwell == pytest.approx(10.0)


def test_returns_ripen_with_time_and_tire_with_repeats():
    m = MorphMemory()
    for t in range(0, 30):
        m.observe(float(t), _fp(), _emb(0))
    for t in range(30, 40):
        m.observe(float(t), _fp(), _emb(1))
    a = m.nearest(_emb(0))[0]
    soon = dict((p.id, pull) for p, pull in m.return_scores(40.0, 0.0))[a]
    later = dict((p.id, pull) for p, pull in m.return_scores(200.0, 0.0))[a]
    assert later > soon and later > 0.5                                       # long gone, long held: it may return
    m.mark(-1.0, a)
    assert a not in {p.id for p, _ in m.return_scores(200.0, 0.0)}             # "bad": not brought back


def test_memory_capacity_keeps_the_current_form():
    m = MorphMemory(radius=0.1, capacity=8)
    for t in range(40):
        m.observe(float(t), _fp(), np.array([t * 1.0, 0.0]))
    assert len(m.protos) == 8 and m.current in m.protos and m.nearest(np.array([39.0, 0.0]))[0] == m.current


def test_novelty_is_distance_to_memory_fading_with_time():
    m = MorphMemory()
    for t in range(10):
        m.observe(float(t), _fp(), _emb(0))
    now, old = novelty(_emb(0), m, 10.0), novelty(_emb(0), m, 1000.0)
    assert now < 0.05 and old > 0.4                                             # an old form is new again
    assert novelty(_emb(2), m, 10.0) > 1.0                                      # never seen: new


# ---------------------------------------------------------------------------- candidates
def _core_setup(org="spear", seed=0):
    col = AbstractColony.of(org, seed)
    ad = make_adapter(col, org)
    return col, ad, BrainCore(ad.vocab, BrainConfig(enabled=True, seed=seed))


def test_candidates_are_only_what_the_organism_can_do():
    col, ad, core = _core_setup()
    for _ in range(40):
        col.step(0.05, 0.5, 0.0, 0.2, True, UserCue())
    snap = ad.snapshot(2.0, None, UserCue(), {})
    ctl = BrainControls(autonomy=0.9, mutation=0.8)
    cands = generate(ad.vocab, snap.blend, snap.fp, core.memory, 2.0, 0.5, ctl, core.rng)
    evaluate(cands, ad.vocab, core.scale, snap.fp, core.memory, 2.0, 0.5, UserCue(), ctl)
    ops = {c.op for c in cands}
    assert {"HOLD", "SHIFT", "HYBRID", "INTENSIFY", "DISSOLVE", "EVENT"} <= ops
    words = {w for pair in DIRECTIONS.values() for w in pair}
    for c in cands:
        assert set(c.blend) <= set(ad.vocab.forms)
        assert c.event is None or c.event in ad.vocab.events
        assert c.material is None or c.material in MATERIALS
        assert {"novelty", "continuity", "identity", "user", "repetition", "oscillation", "pull", "radical"} <= set(c.feats)
        if c.direction:
            assert c.op in ("SHIFT", "HYBRID", "RETURN", "MUTATE")
            assert set(c.direction.split(" + ")) <= words
    assert {f for c in cands if c.op == "SHIFT" for f in c.blend} <= set(ad.vocab.free)


def test_direction_names_the_strongest_shape_move():
    d = np.zeros(8)
    d[7], d[0] = 2.0, -1.0
    assert direction(d) == "twist + contract"
    assert direction(np.full(8, 0.1)) == ""


def test_the_hand_biases_by_the_existing_gestures():
    assert UserCue(True, 1.0).bias()[0] > 0 > UserCue(True, 0.0).bias()[0]    # open hand: bigger; fist: smaller
    assert UserCue(True, 0.5, gesture="SPREAD").bias()[0] > 0
    assert UserCue(True, 0.5, spin=1.0).bias()[7] > 1.0                        # turning: twist
    assert UserCue(True, 0.5, gesture="PINCH").bias()[6] > 0                   # pinch: local (lumpy)
    assert not UserCue().bias().any()                                          # no hand: no bias


# ---------------------------------------------------------------------------- the core
def _snap(ad, t, due=True, user=None, controls=None):
    s = ad.snapshot(t, None, user or UserCue(), controls or {})
    s.due = due
    return s


def test_autonomy_zero_and_mode_current_never_change_anything():
    for mode, aut in (("deterministic", 0.0), ("current", 1.0)):
        col, ad, core = _core_setup()
        core.cfg.mode, core.cfg.controls.autonomy = mode, aut
        assert all(core.step(_snap(ad, float(t))) is None for t in range(1, 60))


def test_the_hand_sculpting_always_wins():
    col, ad, core = _core_setup()
    core.cfg.controls.autonomy = 1.0
    assert all(core.step(_snap(ad, float(t), user=UserCue(True, 0.5, sculpt=True))) is None for t in range(1, 30))
    assert core.stats["yields"] == 29


def test_full_autonomy_takes_the_engines_decision_points():
    col, ad, core = _core_setup()
    core.cfg.controls.autonomy = 1.0
    ds = [core.step(_snap(ad, float(t) * 3.0)) for t in range(1, 30)]
    made = [d for d in ds if d is not None]
    assert len(made) >= 10
    for d in made:
        assert isinstance(d, Decision) and 1.8 <= d.strength <= 3.8 and 3.0 <= d.hold <= 20.0
    assert core.log and {"current", "candidates", "chosen", "source"} <= set(core.log[-1])


def test_midi_ccs_override_the_controls():
    ctl = BrainControls().merged({"brain_autonomy": 0.9, "brain_novelty": 2.0, "brain_rate": 1.0, "brain_memory": -1})
    assert ctl.autonomy == 0.9 and ctl.novelty == 1.0 and ctl.rate_hz == pytest.approx(2.0) and ctl.memory == 0.6


def test_environment_flags():
    d = with_env({"controls": {"novelty": 0.3}}, {"MYRMEX_BRAIN_ENABLED": "1", "MYRMEX_BRAIN_MODE": "F",
                                                   "MYRMEX_BRAIN_HZ": "2", "MYRMEX_BRAIN_AUTONOMY": "0.8",
                                                   "MYRMEX_BRAIN_MEMORY_SIZE": "64", "MYRMEX_BRAIN_NOVELTY": "x"})
    cfg = BrainConfig.from_dict(d)
    assert cfg.enabled and cfg.mode == "kev_candidates" and cfg.memory_size == 64
    assert cfg.controls.rate_hz == 2.0 and cfg.controls.autonomy == 0.8 and cfg.controls.novelty == 0.3
    assert not BrainConfig.from_dict(with_env({"enabled": True}, {"MYRMEX_BRAIN_ENABLED": "0"})).enabled


# ---------------------------------------------------------------------------- the sandbox and its metrics
def test_sandbox_runs_every_mode_and_measures_them_alike():
    res = {m: run("spear", m, minutes=2.0, seed=3) for m in ("current", "random", "novelty", "deterministic")}
    keys = {"unique_forms", "top_share", "repetition_rate", "oscillation_rate", "return_per_10min", "novelty_mean",
            "novelty_max", "transition_diversity", "residence_s", "jitter", "radical_share", "brain_share"}
    for m, r in res.items():
        assert keys <= set(r["metrics"]) and set(diagnose(r["metrics"])) == {"problems", "qualities"}
    assert res["current"]["stats"]["commits"] == 0 and res["current"]["metrics"]["brain_share"] == 0.0
    assert res["deterministic"]["stats"]["commits"] > 0 and res["deterministic"]["metrics"]["brain_share"] > 0


def test_a_prefix_of_a_long_run_is_the_short_run():
    long = run("colony", "deterministic", minutes=2.0, seed=5, checkpoints=[1.0])
    short = run("colony", "deterministic", minutes=1.0, seed=5)
    assert long["by_minutes"][1.0] == short["metrics"]


def test_the_benchmark_cli(tmp_path, monkeypatch, capsys):
    from myrmex.brain import bench
    monkeypatch.setattr(bench, "OUT", str(tmp_path))
    assert bench.main(["--benchmark", "--durations", "0.5", "--seeds", "1", "--organisms", "spear",
                       "--jobs", "1"]) == 0
    out = capsys.readouterr().out
    assert "A current" in out and "G deterministic" in out and "unique forms" in out
    assert any(f.endswith(".json") for f in os.listdir(tmp_path))
    log = tmp_path / "demo.jsonl"
    assert bench.main(["--minutes", "0.5", "--log", str(log), "--autonomy", "1", "--no-compare", "--quiet"]) == 0
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines[0]["kind"] == "run" and lines[-1]["kind"] == "result"
    assert any(x["kind"] == "decision" and x["candidates"] for x in lines)
    with pytest.raises(SystemExit):                                            # Kev modes need a server
        bench.main(["--mode", "F"])


# ---------------------------------------------------------------------------- the adapter on a real engine
def test_adapter_drives_a_real_engine_through_what_it_has():
    eng = _spear()
    ad = make_adapter(eng, "spear")
    assert type(ad).__name__ == "ColonyAdapter"
    for _ in range(60):
        eng.update(DT)
    snap = ad.snapshot(0.5, None, UserCue(), {})
    assert snap.fp.w.sum() == pytest.approx(1.0) and np.isfinite(snap.fp.geo).all() and snap.fp.bodies >= 1
    blend = {"SPINDLE": 0.6, "THORN": 0.4}
    ad.apply(Decision(0.5, "HYBRID", blend, "STRUCTURED", None, 2.6, 7.0), 0.5)
    b = eng.bodies[0]
    w = softmax3(b.z_goal)
    assert w[ad.vocab.index("SPINDLE")] == pytest.approx(0.6, abs=0.02)
    assert w[ad.vocab.index("THORN")] == pytest.approx(0.4, abs=0.02)
    assert b.intent_t == 0.0 and b.dwell == 7.0
    for _ in range(240):                                                       # 2 s: it flows there by itself
        eng.update(DT)
    got = softmax3(b.z)
    assert got[ad.vocab.index("SPINDLE")] + got[ad.vocab.index("THORN")] > 0.6


def test_maintain_waits_for_the_engines_own_reaction():
    eng = _spear()
    ad = make_adapter(eng, "spear")
    b = eng.bodies[0]
    lance, thorn = ad.vocab.index("LANCE"), ad.vocab.index("THORN")
    ad.apply(Decision(0.0, "SHIFT", {"THORN": 1.0}, None, None, 2.6, 30.0), 0.0)
    b.goal_shape("LANCE", 3.6)                                                 # a dash on the kick: a flash ...
    ad.maintain(0.1)
    b.intent, b.intent_t = "CRUISE", 0.0
    ad.maintain(1.0)
    assert softmax3(b.z_goal)[lance] > 0.9                                     # ... seen ...
    ad.maintain(1.4)
    assert softmax3(b.z_goal)[thorn] > 0.9 and ad.restored == 1                # ... and the direction is back
    b.goal_shape("SPINE", 2.5)                                                 # a choice of the engine's own
    ad.maintain(2.0)
    ad.maintain(4.0)
    assert softmax3(b.z_goal)[ad.vocab.index("SPINE")] > 0.9                   # it holds a while ...
    ad.maintain(6.1)
    assert softmax3(b.z_goal)[thorn] > 0.9 and ad.restored == 2                # ... then the direction again
    b.intent = "EVADE"
    b.goal_shape("CORE", 2.5)
    ad.maintain(6.2)
    ad.maintain(12.0)
    assert softmax3(b.z_goal)[ad.vocab.index("CORE")] > 0.9                    # busy (evading): left alone


def test_inverse_blend_round_trip():
    w = np.array([0.5, 0.3, 0.2, 0.0])
    got = softmax3(inverse_blend(w, 2.6))
    assert np.allclose(got[:3], w[:3], atol=0.02) and got[3] < 0.02


def test_user_cue_reads_the_session_glove_state():
    class G:
        present, ext, omega, pos_v, gestures = True, np.full(5, 0.9), np.array([0.0, 7.0, 0.0]), np.zeros(3), [(9.5, "CLENCH")]
    u = user_cue(G(), None, 10.0)
    assert u.present and u.open == pytest.approx(0.9) and u.spin == 1.0 and u.gesture == "CLENCH" and not u.sculpt
    assert not user_cue(None, None, 0.0).present


# ---------------------------------------------------------------------------- the worker: never in the way
def _drive(link, eng, seconds, t0=0.0, until=None):
    """Step the engine and the link in real time; -> (sim time, max link tick ms)."""
    t, worst = t0, 0.0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        eng.update(DT)
        t += DT
        a = time.perf_counter()
        link.tick(time.perf_counter(), t)
        worst = max(worst, (time.perf_counter() - a) * 1000.0)
        if until is not None and until(link):
            break
        time.sleep(0.001)
    return t, worst


def test_worker_process_answers_and_survives_a_crash():
    eng = _spear()
    cfg = BrainConfig(enabled=True, controls=BrainControls(autonomy=1.0, rate_hz=4.0))
    link = BrainLink(cfg, eng, "spear")
    try:
        t, _ = _drive(link, eng, 20.0, until=lambda l: l.stats["answered"] >= 3)
        assert link.ready and link.stats["answered"] >= 3 and link.proc.poll() is None
        pid = link.proc.pid
        os.kill(pid, signal.SIGKILL)                                           # the worker dies
        t, worst = _drive(link, eng, 1.0, t)
        assert link.error.startswith("the brain worker stopped") and link.proc is None and not link.gave_up
        assert worst < 50.0                                                    # (the engine did not wait)
        t, _ = _drive(link, eng, 20.0, t, until=lambda l: l.ready and l.stats["restarts"] >= 1)
        assert link.stats["restarts"] == 1 and link.ready and link.proc.pid != pid
    finally:
        link.close()
    assert link.proc is None and not link.ready


def test_a_hung_worker_times_out_and_the_tick_never_blocks():
    eng = _spear()
    cfg = BrainConfig(enabled=True, controls=BrainControls(autonomy=1.0, rate_hz=10.0))
    link = BrainLink(cfg, eng, "spear", timeout=0.3)
    try:
        t, _ = _drive(link, eng, 20.0, until=lambda l: l.ready)
        os.kill(link.proc.pid, signal.SIGSTOP)                                 # hung (overloaded, a slow Kev ...)
        t, worst = _drive(link, eng, 1.5, t)
        assert link.stats["timeouts"] >= 1 and worst < 20.0
        os.kill(link.proc.pid, signal.SIGCONT)
        answered = link.stats["answered"]
        t, _ = _drive(link, eng, 10.0, t, until=lambda l: l.stats["answered"] > answered)
        assert link.stats["answered"] > answered                               # and it carries on
    finally:
        if link.proc is not None:
            os.kill(link.proc.pid, signal.SIGCONT)
        link.close()


def test_repeated_failures_give_up_and_leave_the_engine_alone():
    eng = _spear()
    link = BrainLink(BrainConfig(enabled=True), eng, "spear", process=False)
    for k in range(5):
        link._fail(float(k), "boom")
    assert link.gave_up and link.tick(10.0, 10.0) is None
    z = eng.bodies[0].z_goal.copy()
    link.tick(11.0, 11.0)
    assert np.array_equal(eng.bodies[0].z_goal, z)
    link.configure(BrainConfig(enabled=True))                                  # switching it on again: a fresh try
    assert not link.gave_up


def test_inline_link_applies_decisions():
    eng = _spear()
    link = BrainLink(BrainConfig(enabled=True, controls=BrainControls(autonomy=1.0, rate_hz=2.0)), eng, "spear",
                     process=False)
    t, applied = 0.0, 0
    for i in range(120 * 40):
        eng.update(DT)
        t += DT
        if link.tick(t, t) is not None:
            applied += 1
    assert applied >= 2 and link.status()["applied"] == applied


# ---------------------------------------------------------------------------- Kev: the client and the policy
class MockKev(BaseHTTPRequestHandler):
    """MOCK of the documented Kev System One API - canned answers, no model (see the module docstring)."""

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):                               # (the shape a real kev.serve answers, MLX on a Mac)
        self._send(200, {"models": [{"name": "kev-latest", "description": "mock", "run": "jaredpalmer/kev-0.8b",
                                     "base": "Qwen/Qwen3.5-0.8B-Base", "device": "mps", "backend": "mlx",
                                     "dtype": "bfloat16"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        srv = self.server
        srv.requests.append(body)
        if srv.behaviour == "slow":
            time.sleep(1.0)
        answers = {}
        for qid, q in body["questions"].items():
            assert q["type"] in ("choice", "score", "noul") and isinstance(q.get("instructions"), str)
            if q["type"] == "choice":
                keys = list(q["criteria"])
                assert isinstance(q["criteria"], dict) and 1 <= len(keys) <= 255
                k = len(keys)
                if srv.behaviour == "lowconf":
                    p = np.full(k, 1.0 / k)
                else:
                    top = 0.9 if srv.behaviour != "medium" else 1.0 / k + 0.35 * (1 - 1.0 / k)
                    p = np.full(k, (1 - top) / max(1, k - 1))
                    p[min(1, k - 1)] = top
                probs = {key: float(x) for key, x in zip(keys, p)}
                answers[qid] = {"type": "choice", "choice": keys[int(np.argmax(p))], "probabilities": probs,
                                "confidence": float((p.max() - 1 / k) / (1 - 1 / k)) if k > 1 else 1.0}
            elif q["type"] == "score":
                assert isinstance(q["criteria"], list) and len(q["criteria"]) >= 2
                n = len(q["criteria"])
                p = np.zeros(n)
                p[1] = 1.0
                answers[qid] = {"type": "score", "score": 1.0, "legend": q["criteria"],
                                "probabilities": {str(i): float(x) for i, x in enumerate(p)}, "confidence": 1.0}
        if srv.behaviour == "malformed":
            answers.pop(next(iter(answers)))
        elif srv.behaviour == "wrongtype":
            answers[next(iter(answers))]["type"] = "noul"
        elif srv.behaviour == "badprobs":
            a = answers[next(iter(answers))]
            a["probabilities"] = {k: v * 0.5 for k, v in a["probabilities"].items()}
        self._send(200, {"model": body.get("model"), "answers": answers, "usage": {"input_tokens": 0}, "latency_ms": 1.0})


@pytest.fixture
def mock_kev():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), MockKev)
    srv.behaviour, srv.requests = "ok", []
    srv.handle_error = lambda request, address: None               # (a client that timed out has hung up)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _q():
    return {"next": {"type": "choice", "instructions": "which?", "criteria": {"A": "one", "B": "two", "C": None}},
            "level": {"type": "score", "instructions": "how much?", "criteria": ["low", "mid", "high"]}}


def test_kev_client_validates_answers(mock_kev):
    srv, url = mock_kev
    cli = KevClient(url, timeout=0.5)
    info = cli.models()
    assert info["models"][0]["name"] == "kev-latest"
    assert describe(info) == "kev-latest (jaredpalmer/kev-0.8b, mlx, bfloat16)"
    assert describe({"object": "list", "data": [{"id": "kev-latest"}]}) == "kev-latest"
    assert describe({}) == describe({"models": []}) == describe(None) == "a model"
    ans = cli.ask({"organism": "spear"}, _q())
    assert ans["next"]["choice"] == "B" and ans["level"]["score"] == 1.0
    assert srv.requests[-1]["model"] == "kev-latest" and srv.requests[-1]["state"] == {"organism": "spear"}
    for bad in ("malformed", "wrongtype", "badprobs"):
        srv.behaviour = bad
        with pytest.raises(KevError):
            cli.ask({}, _q())
    srv.behaviour = "slow"
    t0 = time.perf_counter()
    with pytest.raises(KevError):
        cli.ask({}, _q())
    assert time.perf_counter() - t0 < 0.9                                      # the timeout, not the server's pace
    with pytest.raises(KevError):
        KevClient("http://127.0.0.1:9", timeout=0.3).ask({}, _q())             # nothing there


def test_kev_is_local_only():
    for ok in ("http://127.0.0.1:8009", "http://localhost:8009", "http://192.168.1.20:8009", "http://studio.local:8009",
               "http://[::1]:8009"):
        assert is_local(ok)
    for bad in ("https://api.typesafe.ai", "http://8.8.8.8:8009", "https://example.com"):
        assert not is_local(bad)
        with pytest.raises(KevError):
            KevClient(bad)
    core = BrainCore(vocabulary(organism_class("spear"), "spear"),
                     BrainConfig(enabled=True, mode="kev_candidates", kev_url="https://api.typesafe.ai"))
    assert core.kev is None                                                    # -> the deterministic arbiter


def _kev_run(url, behaviour, srv, n=40):
    srv.behaviour = behaviour
    col, ad, _ = _core_setup(seed=1)
    core = BrainCore(ad.vocab, BrainConfig(enabled=True, mode="kev_candidates", kev_url=url, kev_timeout=0.5, seed=1,
                                           controls=BrainControls(autonomy=1.0)))
    out = []
    for i in range(1, n):
        d = core.step(_snap(ad, float(i) * 3.0))
        if d is not None:
            out.append(d)
    return core, out


def test_kev_confidence_policy(mock_kev):
    srv, url = mock_kev
    core, ds = _kev_run(url, "ok", srv)
    assert ds and all(d.source == "kev" and d.confidence > 0.8 and d.probs for d in ds)
    assert all(e["candidates"] for e in core.log if e["source"] == "kev")
    req = srv.requests[-1]
    assert set(req["questions"]) == {"next", "intensity"} and "memory" in req["state"]
    core, ds = _kev_run(url, "medium", srv)
    assert ds and {d.source for d in ds} == {"kev+det"}                        # blended with the arbiter
    core, ds = _kev_run(url, "lowconf", srv)
    assert core.stats["kev_lowconf"] > 0 and {d.source for d in ds} <= {"det-lowconf"}


def test_kev_down_or_broken_falls_back(mock_kev):
    srv, url = mock_kev
    for behaviour in ("malformed", "slow"):
        core, ds = _kev_run(url, behaviour, srv, n=12)
        assert core.stats["kev_fail"] > 0 and ds and {d.source for d in ds} == {"fallback"}
    col, ad, _ = _core_setup()
    core = BrainCore(ad.vocab, BrainConfig(enabled=True, mode="kev_candidates", kev_url="http://127.0.0.1:9",
                                           kev_timeout=0.2, controls=BrainControls(autonomy=1.0)))
    ds = [core.step(_snap(ad, float(i) * 3.0)) for i in range(1, 12)]
    assert any(d is not None and d.source == "fallback" for d in ds)


def test_kev_direct_modes_ask_for_operations(mock_kev):
    srv, url = mock_kev
    col, ad, _ = _core_setup()
    for mode, memory in (("kev_direct", False), ("kev_memory", True)):
        core = BrainCore(ad.vocab, BrainConfig(enabled=True, mode=mode, kev_url=url, controls=BrainControls(autonomy=1.0)))
        srv.behaviour = "ok"
        d = next(d for d in (core.step(_snap(ad, float(i) * 3.0)) for i in range(1, 20)) if d is not None)
        q = srv.requests[-1]["questions"]["next"]["criteria"]
        assert d.source == "kev" and "stay as it is" in q and ("memory" in srv.requests[-1]["state"]) == memory


# ---------------------------------------------------------------------------- the live session
def test_live_session_brain_switch_and_status(tmp_path):
    from myrmex.realtime.inputs import InputConfig
    from myrmex.realtime.session import LiveConfig, LiveSession
    cfg = LiveConfig(backend="spear", clock="internal", out=[], inputs=InputConfig(osc_port=0),
                     brain={"enabled": True, "controls": {"autonomy": 1.0, "rate_hz": 4.0}})
    ses = LiveSession(cfg, start_inputs=False, now=time.perf_counter())
    try:
        assert ses.brain is not None and ses.status()["brain"]["enabled"]
        end = time.perf_counter() + 20.0
        while time.perf_counter() < end and not ses.brain.stats["answered"]:
            ses.step(time.perf_counter())
        assert ses.status()["brain"]["running"] and ses.brain.stats["answered"] > 0
        ses.set_brain({"controls": {"novelty": 0.9}})                          # a slider: the others stay
        assert ses.brain.cfg.controls.autonomy == 1.0 and ses.brain.cfg.controls.novelty == 0.9
        ses.inputs.triggers.append(("brain", 0.0))                             # the pad: off
        ses.step(time.perf_counter())
        assert not ses.status()["brain"]["enabled"] and ses.brain.proc is None
        ses.inputs.triggers.append(("brain", 0.0))                             # ... and on again
        ses.step(time.perf_counter())
        assert ses.status()["brain"]["enabled"] and ses.brain.proc is not None
        ses.inputs.triggers.append(("brain:good", 0.0))
        ses.step(time.perf_counter())
    finally:
        ses.stop()
    assert ses.brain.proc is None


def test_brain_is_off_by_default_and_env_can_switch_it_on(monkeypatch):
    from myrmex.realtime.inputs import InputConfig
    from myrmex.realtime.session import LiveConfig, LiveSession
    ses = LiveSession(LiveConfig(backend="spear", clock="internal", out=[], inputs=InputConfig(osc_port=0)),
                      start_inputs=False, now=0.0)
    assert ses.brain is None and ses.status()["brain"] == {"enabled": False}
    ses.stop()
    monkeypatch.setenv("MYRMEX_BRAIN_ENABLED", "1")
    monkeypatch.setenv("MYRMEX_BRAIN_AUTONOMY", "0.7")
    ses = LiveSession(LiveConfig(backend="spear", clock="internal", out=[], inputs=InputConfig(osc_port=0)),
                      start_inputs=False, now=0.0)
    try:
        assert ses.brain is not None and ses.brain.cfg.controls.autonomy == 0.7
    finally:
        ses.stop()


# ---------------------------------------------------------------------------- the app
def test_app_creature_page_brain_group(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from myrmex.app import controllers as C
    from myrmex.app import tabs as T
    from myrmex.app import window as W
    from myrmex.app.settings import AppSettings
    app = QApplication.instance() or QApplication([])
    s = AppSettings()
    s.start_engine_on_launch = s.open_blender_on_start = False
    assert C.brain_settings(s)["enabled"] is False                              # off by default
    w = W.MainWindow(s)
    w.timer.stop()
    w.stack.setCurrentIndex(w.page_index["Creature"])
    app.processEvents()
    assert not w.chk_brain.isChecked() and w.lbl_brain.text() == "off"
    w.chk_brain.setChecked(True)
    w.brain_sliders["autonomy"].setValue(80)
    w.cmb_brain.setCurrentIndex(w.cmb_brain.findData("kev_candidates"))
    assert s.brain["enabled"] and s.brain["controls"]["autonomy"] == 0.8 and s.brain["mode"] == "kev_candidates"
    assert s.brain["kev_url"] == "http://127.0.0.1:8009"
    cfg = C.brain_settings(s)
    assert cfg["enabled"] and cfg["controls"]["autonomy"] == 0.8 and cfg["controls"]["novelty"] == 0.6
    T.refresh_brain(w)
    assert w.lbl_brain.text() == "starts with the engine"
    w.close()


@pytest.mark.parametrize("organism,adapter", [("colony", "ColonyAdapter"), ("polyalloy", "PolyalloyAdapter"),
                                              ("ferro", "BionicAdapter"), ("creature", "CreatureAdapter")])
def test_every_family_runs_the_brain_live(organism, adapter):
    from myrmex.realtime.inputs import InputConfig
    from myrmex.realtime.session import LiveConfig, LiveSession
    ses = LiveSession(LiveConfig(backend=organism, clock="internal", out=[], inputs=InputConfig(osc_port=0),
                                 brain={"enabled": True, "controls": {"autonomy": 1.0, "rate_hz": 4.0}}),
                      start_inputs=False, now=time.perf_counter())
    try:
        assert type(ses.brain.adapter).__name__ == adapter
        end = time.perf_counter() + 30.0
        while time.perf_counter() < end and not ses.brain.stats["applied"]:
            ses.step(time.perf_counter())
        st = ses.brain.status()
        assert st["running"] and st["applied"] >= 1 and not st["error"] and st["last"]["op"]
    finally:
        ses.stop()


def test_a_bug_in_the_brain_never_reaches_the_engine_thread():
    eng = _spear()
    link = BrainLink(BrainConfig(enabled=True, controls=BrainControls(rate_hz=10.0)), eng, "spear", process=False)

    def boom(*a, **k):
        raise RuntimeError("a bug")
    link.adapter.maintain = boom
    for i in range(1, 8):
        assert link.tick(float(i), float(i)) is None                          # no exception comes out
    assert link.gave_up and "a bug" in link.error


def test_the_worker_keeps_a_decision_log(tmp_path):
    eng = _spear()
    cfg = BrainConfig(enabled=True, controls=BrainControls(autonomy=1.0, rate_hz=10.0), log_dir=str(tmp_path))
    link = BrainLink(cfg, eng, "spear")
    try:
        _drive(link, eng, 30.0, until=lambda l: l.stats["applied"] >= 2)
        link.mark(True)
        _drive(link, eng, 1.0)
    finally:
        link.close()
    files = list(tmp_path.glob("spear-*.jsonl"))
    assert len(files) == 1
    recs = [json.loads(x) for x in files[0].read_text().splitlines()]
    kinds = {r["kind"] for r in recs}
    assert kinds == {"decision", "mark"}
    d = next(r for r in recs if r["kind"] == "decision" and r["op"] != "HOLD")
    assert d["organism"] == "spear" and d["candidates"] and d["chosen"] and "memory" in d
    assert next(r for r in recs if r["kind"] == "mark")["good"] is True
