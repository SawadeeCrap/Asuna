"""Black Nanomaterial Creature: control input, dynamics, material conservation, morphology, behaviour,
determinism, protocol, live-session backend, manual camera and MIDI mapping."""
import math

import numpy as np
import pytest

from myrmex.bus.transport import LiveEvent
from myrmex.creature import CreatureConfig, CreatureControlInput, CreatureEngine
from myrmex.creature.control import ParameterSet
from myrmex.creature.nodes import NodeSystem
from myrmex.creature.protocol import decode_creature, encode_creature
from myrmex.realtime.inputs import InputConfig
from myrmex.realtime.midimap import MidiBinding, MidiMapper
from myrmex.realtime.session import LiveConfig, LiveSession


def music(t, on=True):
    beat = t * 2.0
    kick = 1.0 if on and (beat % 1.0) < 0.04 else 0.0
    return CreatureControlInput(bass=0.8 * on, high=0.5 * on, energy=0.85 * on, transient=kick, spectral_flux=0.4 * on,
                                amplitude=0.7 * on, tempo=120, beat=beat, beat_phase=beat % 1, playing=on)


def run(e, seconds, on=True, dt=1 / 60):
    out = []
    for i in range(int(seconds / dt)):
        e.set_input(music(e.t, on))
        out.append(e.update(dt))
    return out


def test_input_is_normalised_and_extremes_are_safe():
    x = CreatureControlInput(bass=5.0, high=-3, energy=float("nan"), transient=float("inf"), tempo=0, beat=float("nan"))
    s = x.sanitized()
    assert s.bass == 1.0 and s.high == 0.0 and s.energy == 0.0 and s.transient == 0.0 and s.tempo == 120.0
    e = CreatureEngine(CreatureConfig(seed=1))
    e.set_input(CreatureControlInput(bass=1e9, transient=1e9, energy=-1e9, spectral_flux=1e9))
    for _ in range(300):
        st = e.update(1 / 60)
    assert np.isfinite(st.pos).all() and np.isfinite(st.radius).all()
    assert e.update(float("nan")).t == st.t                # a bad dt is ignored, not propagated


def test_parameters_manual_override_and_back_to_auto():
    p = ParameterSet()
    auto = p["aggression"]
    assert p.set("aggression", 0.9) and p["aggression"] == 0.9
    p.set("aggression", -1)
    assert p["aggression"] == auto
    assert not p.set("not_a_param", 0.5)


def test_same_seed_same_performance_other_seed_differs():
    a, b, c = (CreatureEngine(CreatureConfig(seed=s)) for s in (4, 4, 5))
    for e in (a, b, c):
        run(e, 6.0)
    assert np.array_equal(a.nodes.pos, b.nodes.pos)
    assert not np.allclose(a.nodes.pos, c.nodes.pos)


def test_spring_node_has_inertia_overshoot_and_settles():
    n = NodeSystem(1)
    n.freq[:], n.zeta[:], n.radius[:] = 1.5, 0.35, 0.1
    n.pos[:] = n.target[:] = [0, 0, 1.0]
    n.target[0, 0] = 1.0                                  # sudden input change
    xs = []
    for _ in range(600):
        n.step(1 / 120, 0.0, 50.0, ground=False)
        xs.append(n.pos[0, 0])
    xs = np.array(xs)
    assert np.max(np.abs(np.diff(xs))) < 0.1               # no teleport (a jump would be 1.0)
    assert xs.max() > 1.02                                 # overshoot (underdamped)
    assert abs(xs[-1] - 1.0) < 0.01                        # settles


def test_material_is_conserved_and_appendages_take_it_from_the_core():
    e = CreatureEngine(CreatureConfig(seed=2))
    run(e, 2.0, on=False)
    core0 = e.vol["core"]
    e.trigger_event("APPENDAGE_BURST", "WHIP")
    e.trigger_event("APPENDAGE_BURST", "TENDRIL")
    states = run(e, 3.0, on=False)
    tot = [s.volumes["total"] for s in states]
    assert max(abs(x - 1.0) for x in tot) < 1e-9
    assert states[-1].volumes["appendages"] > 0.005 and states[-1].volumes["core"] < core0
    assert len(e.apps.active) >= 1


def test_morphology_transitions_are_continuous():
    e = CreatureEngine(CreatureConfig(seed=3))
    e.morph.set_target("AERODYNAMIC", 0.0, 2.0)
    vs = [e.morph.update(t, 0.0, 0.0).copy() for t in np.arange(0, 2.5, 1 / 60)]
    steps = np.abs(np.diff(np.array(vs), axis=0)).max()
    assert steps < 0.05 and abs(vs[-1][1] - 2.4) < 1e-6    # smooth, arrives at the elongated target


def test_behaviour_changes_state_with_music_and_is_not_frozen_in_silence():
    e = CreatureEngine(CreatureConfig(seed=6))
    quiet = run(e, 10.0, on=False)
    loud = run(e, 40.0, on=True)
    assert len({s.behavior for s in loud}) >= 3
    assert loud[-1].arousal > quiet[-1].arousal
    # idle is alive: the surface nodes keep moving even in silence
    d = np.linalg.norm(quiet[-1].pos[e.i_secondary] - quiet[-30].pos[e.i_secondary], axis=1)
    assert d.max() > 1e-3


def test_protocol_roundtrip():
    e = CreatureEngine(CreatureConfig(seed=1))
    s = run(e, 1.0)[-1]
    fr = decode_creature(encode_creature(s, 7, 12.5, 124.0, 16))
    assert fr.seq == 7 and fr.behavior == s.behavior and fr.flags & 16
    assert np.allclose(fr.pos, s.pos, atol=1e-5) and np.allclose(fr.radius, s.radius, atol=1e-6)
    assert decode_creature(b"junk") is None


def test_live_session_creature_backend_runs_without_blender_or_ableton(tmp_path):
    cfg = LiveConfig(backend="creature", clock="internal", link=False, out=["127.0.0.1:9"], latency=0.0,
                     record=str(tmp_path), inputs=InputConfig(osc_port=0))
    s = LiveSession(cfg, start_inputs=False, now=0.0)       # nobody listens on port 9: must not fail
    s.inputs.push(LiveEvent("control", 0.0, {"name": "aggression", "value": 0.95}))
    for i in range(1, 600):
        if i % 60 == 0:
            s.inputs.push(LiveEvent("note", 0.0, {"channel": 1, "pitch": 36.0, "velocity": 1.0}))
        s.step(i / 120)
    assert s.creature.engine.params["aggression"] == pytest.approx(0.95)
    assert s.sink.sent > 50 and s.status()["backend"] == "creature"
    assert s.save_take() is not None


def test_manual_camera_holds_the_chosen_shot():
    cfg = LiveConfig(backend="creature", clock="internal", bpm=120, link=False, out=[], latency=0.0,
                     inputs=InputConfig(osc_port=0))
    s = LiveSession(cfg, start_inputs=False, now=0.0)
    s.inputs.push(LiveEvent("control", 0.0, {"name": "cam_mode", "value": 1.0}))
    s.inputs.push(LiveEvent("trigger", 0.0, {"name": "camera:feet_close"}))
    kinds = set()
    for i in range(1, 120 * 20):                          # 20 s = 10 bars at 120 BPM
        s.step(i / 120)
        if i > 60 and s.camera.kind:
            kinds.add(s.camera.kind)
    assert kinds == {"feet_close"}


def test_midi_mapping_curves_ranges_modes_and_learn():
    m = MidiMapper([MidiBinding("cc", 74, 0, "aggression", 0.2, 0.8, "exp", False),
                    MidiBinding("note", 36, 10, "creature:collapse", mode="trigger"),
                    MidiBinding("note", 38, 0, "hold", mode="gate")])
    (a, tgt, v), = m.cc(3, 74, 0.5)
    assert tgt == "aggression" and v == pytest.approx(0.2 + 0.6 * 0.25)
    assert m.note(1, 36, 1.0, True) == []                  # wrong channel
    assert m.note(10, 36, 1.0, True) == [("trigger", "creature:collapse", 1.0)]
    assert m.note(2, 38, 0.7, False) == [("control", "hold", 0.0)]
    m.learning = 0
    assert m.learn("cc", 5, 21) and m.bindings[0].number == 21 and m.bindings[0].channel == 5


def test_stress_long_performance_stays_stable():
    e = CreatureEngine(CreatureConfig(seed=9))
    for k in range(6):
        run(e, 10.0, on=k % 2 == 0)
        e.trigger_event(["COLLAPSE", "RECONSTRUCTION", "APPENDAGE_BURST", "MASS_REBALANCE", "MORPHOLOGY_SHIFT",
                         "COLLAPSE"][k])
    st = e.state()
    assert np.isfinite(st.pos).all() and abs(st.volumes["total"] - 1.0) < 1e-9
    assert np.abs(st.com[:2]).max() < 3 * e.cfg.stage_radius


# ---------------------------------------------------------------------------- Mimetic Polyalloy (v2) + takes
def _poly_run(seed=1, seconds=12.0, dt=1 / 60):
    from myrmex.creature.polyalloy import PolyalloyConfig, PolyalloyEngine
    e = PolyalloyEngine(PolyalloyConfig(seed=seed))
    out = []
    for i in range(int(seconds / dt)):
        beat = i * dt * 2
        kick = 1.0 if (beat % 1.0) < dt * 2.5 else 0.0
        e.set_input(CreatureControlInput(bass=0.6, high=0.3, energy=0.8, transient=kick, spectral_flux=0.3,
                                         amplitude=0.7, tempo=120, beat=beat, playing=True))
        out.append(e.update(dt))
    return e, out


def test_polyalloy_flies_conserves_material_and_answers_kicks():
    e, states = _poly_run()
    s = states[-1]
    assert np.isfinite(s.pos).all() and abs(s.volumes["total"] - 1.0) < 1e-9
    alts = np.array([st.com[2] for st in states[120:]])
    assert alts.min() > 0.8                                   # airborne
    names = {n for _, n, _ in e.events}
    assert names & {"IMPULSE", "OBSTACLE", "PRESSURE", "TURBULENCE"}
    assert len({st.material for st in states}) >= 3           # the material changes state


def test_polyalloy_is_deterministic():
    _, a = _poly_run(seed=5, seconds=3)
    _, b = _poly_run(seed=5, seconds=3)
    assert np.allclose(a[-1].pos, b[-1].pos)


def test_polyalloy_protocol_roundtrip():
    from myrmex.creature.protocol import decode_creature, encode_creature
    from myrmex.realtime.protocol import CameraState
    _, states = _poly_run(seconds=2)
    s = states[-1]
    cam = CameraState(np.array([1.0, 2, 3]), np.zeros(3), 35.0, 4.0, 2.8, 7, "orbit")
    fr = decode_creature(encode_creature(s, 3, 1.5, 124.0, 1, cam))
    assert fr.material == s.material and fr.camera.kind == "orbit"
    assert fr.links.shape == s.links.shape and np.array_equal(fr.links[:, :2], s.links[:, :2])
    assert fr.obstacles.shape == s.obstacles.shape and fr.style == 0
    assert np.allclose(fr.pos, s.pos, atol=1e-5)


def test_aerial_camera_modes_and_app_buttons():
    from myrmex.camera.aerial import MODES, AerialCinematographer
    c = AerialCinematographer(1.6, 0)
    c.mode = "manual"
    c.request_cut("front_low")                                  # app button 2 -> follow
    st = c.update(0.0, 1 / 60, np.array([0.0, 0, 3]), 0.0, 0.0, 4.0)
    assert st.kind == MODES[1] and np.isfinite(st.position).all()


def test_creature_take_roundtrip(tmp_path):
    from myrmex.creature.take import CreatureTake
    from myrmex.realtime.session import LiveConfig, LiveSession

    class Sink:
        sent = 0

        def send_raw(self, b):
            pass

        def close(self):
            pass
    cfg = LiveConfig(backend="polyalloy", clock="internal", bpm=120, record=str(tmp_path), out=[])
    s = LiveSession(cfg, start_inputs=False, sink=Sink(), now=0.0)
    now = 0.0
    for _ in range(int(3 * cfg.rate)):
        now += 1.0 / cfg.rate
        s.step(now)
    take = CreatureTake(s.save_take())
    assert take.variant == "polyalloy" and take.n > 80 and take.nodes == 96
    tr = take.camera_track(24)
    assert tr is not None and len(tr.positions) == round(take.duration * 24) and np.isfinite(tr.positions).all()
    # song position: 8 beats in at 120 BPM when the take started -> the audio starts 4 s before frame 0
    take.d["song_beat"] = 8.0 + take.d["t"] * 2.0
    take.d["playing"] = np.ones(take.n)
    assert abs(take.audio_offset() - 4.0) < 0.1


# ---------------------------------------------------------------------------- Polyalloy Colony (v3)
def _colony_run(seed=3, seconds=40.0, dt=1 / 60, events=()):
    from myrmex.creature.colony import ColonyConfig, ColonyEngine
    e = ColonyEngine(ColonyConfig(seed=seed))
    e.set_parameter("swarm", 0.8)
    e.set_parameter("hunt", 0.9)
    out = []
    for i in range(int(seconds / dt)):
        t = i * dt
        beat = t * 2
        energy = 0.9 if t < 20 else 0.15                      # a drop, then a breakdown
        kick = 1.0 if (beat % 1.0) < dt * 2.5 and energy > 0.5 else 0.0
        for at, name in events:
            if i == int(at / dt):
                e.trigger_event(name)
        e.set_input(CreatureControlInput(bass=0.6, high=0.3, energy=energy, transient=kick, spectral_flux=0.3,
                                         amplitude=0.7, tempo=120, beat=beat, playing=True))
        out.append(e.update(dt))
    return e, out


def test_colony_splits_merges_hardens_and_stays_finite():
    e, states = _colony_run()
    assert all(np.isfinite(s.pos).all() for s in states[::30])
    assert abs(states[-1].volumes["total"] - 1.0) < 1e-9          # finite material
    bodies = [s.bodies for s in states]
    assert max(bodies) >= 2                                        # the drop splits it into a flock
    assert bodies[-1] == 1                                         # the breakdown brings it back together
    assert max(float(s.plate.max()) for s in states) > 0.3         # armour rises (hardening waves / hits)
    assert "WAVE" in {n for _, n, _ in e.events} or max(float(s.plate.max()) for s in states) > 0.5
    alts = np.array([s.com[2] for s in states[120:]])
    assert 0.5 < alts.min() and alts.max() < 12.0


def test_colony_hunts_and_perches():
    e, states = _colony_run(seed=2, seconds=40, events=((1.0, "HUNT"), (28.0, "PERCH")))
    names = [s.behavior for s in states]
    assert "HUNT" in names and "PERCH" in names
    assert any(s.lure[3] > 0 for s in states)


def test_colony_protocol_roundtrip():
    from myrmex.creature.protocol import decode_creature, encode_creature
    _, states = _colony_run(seconds=1.5)
    s = states[-1]
    fr = decode_creature(encode_creature(s, 1, 0.0, 120.0, 1))
    assert fr.bodies == s.bodies and np.allclose(fr.plate, s.plate, atol=1e-6)
    assert np.allclose(fr.nrm, s.nrm, atol=1e-5) and np.array_equal(fr.owner, s.owner) and fr.lure.shape == (4,)
    assert fr.links is not None and fr.behavior == s.behavior


def test_take_command_uses_saved_look(tmp_path, monkeypatch):
    from myrmex.app import controllers as C
    monkeypatch.setattr(C, "looks_dir", lambda: str(tmp_path))
    take = "/x/colony_take_1.npz"
    assert "--keep-settings" in C.take_command("B", take) and str(tmp_path) not in " ".join(C.take_command("B", take))
    (tmp_path / "colony.blend").write_bytes(b"")
    cmd = C.take_command("B", take, render=True)
    assert cmd[:3] == ["B", "-b", str(tmp_path / "colony.blend")]
    live, env = C.blender_live_command("B", "/c.blend", 9101, "colony", keep_settings=False)
    assert live[1] == str(tmp_path / "colony.blend") and env["MYRMEX_KEEP_SETTINGS"] == "0"


# ---------------------------------------------------------------------------- big frames (macOS UDP limit)
def test_frames_survive_macos_datagram_limit():
    import socket

    from myrmex.creature.protocol import decode_creature
    from myrmex.realtime.protocol import Reassembler
    from myrmex.realtime.session import LiveConfig, LiveSession, PoseSink

    class MacSock:                                   # net.inet.udp.maxdgram = 9216 on macOS
        def __init__(self, real):
            self.real = real

        def sendto(self, data, addr):
            if len(data) > 9216:
                raise OSError(40, "Message too long")
            return self.real.sendto(data, addr)

        def close(self):
            self.real.close()
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    for backend in ("creature", "polyalloy", "colony", "hive"):
        sink = PoseSink([f"127.0.0.1:{rx.getsockname()[1]}"], [])
        sink.sock = MacSock(sink.sock)
        s = LiveSession(LiveConfig(backend=backend, clock="internal", out=[]), start_inputs=False, sink=sink, now=0.0)
        now = 0.0
        for _ in range(4):
            now += 1 / 60
            s.step(now)
        ra, fr = Reassembler(), None
        while fr is None:
            got = ra.feed(rx.recvfrom(65536)[0])
            fr = decode_creature(got) if got is not None else None
        assert sink.errors == 0 and len(fr.pos) >= 61, backend
        sink.close()
    rx.close()


# ---------------------------------------------------------------------------- Polyalloy Hive (v4)
def _hive_run(seconds=30.0, events=(), energy_fn=None, seed=4, dt=1 / 60, params=None):
    from myrmex.creature.hive import HiveConfig, HiveEngine
    e = HiveEngine(HiveConfig(seed=seed))
    for k, val in (params or {}).items():
        e.set_parameter(k, val)
    out = []
    for i in range(int(seconds / dt)):
        t = i * dt
        beat = t * 2
        energy = energy_fn(t) if energy_fn else 0.5
        kick = 1.0 if (beat % 1.0) < dt * 2.5 and energy > 0.4 else 0.0
        for at, name in events:
            if i == int(at / dt):
                e.trigger_event(name)
        e.set_input(CreatureControlInput(bass=0.5, high=0.3, energy=energy, transient=kick, spectral_flux=0.3,
                                         amplitude=0.7, tempo=120, beat=beat, playing=True))
        out.append(e.update(dt))
    return e, out


def test_hive_builds_and_recalls_structures():
    e, states = _hive_run(24, events=((2.0, "BUILD"), (16.0, "RECALL")), params={"architecture": 0.0})
    assert max(s.structures for s in states) == 1 and states[-1].structures == 0
    assert abs(states[-1].volumes["total"] - 1.0) < 1e-9 and np.isfinite(states[-1].particles).all()
    assert "PATROL" in {s.behavior for s in states}


def test_hive_swarm_breathes_with_the_music():
    e, _ = _hive_run(20, energy_fn=lambda t: 0.15)
    calm = float((~e.p_bound).mean())
    e2, _ = _hive_run(20, energy_fn=lambda t: 0.95)
    assert calm < 0.4 and float((~e2.p_bound).mean()) > calm


def test_hive_remembers_repeated_phrases():
    e, _ = _hive_run(72, energy_fn=lambda t: 0.2 if (t % 24) < 12 else 0.7)
    names = [n for _, n, _ in e.events]
    assert len(e.memory) >= 2 and "REMEMBER" in names


def test_hive_protocol_and_take(tmp_path):
    from myrmex.creature.protocol import decode_creature, encode_creature
    from myrmex.creature.take import CreatureTake
    from myrmex.realtime.session import LiveConfig, LiveSession
    _, states = _hive_run(1.0)
    s = states[-1]
    fr = decode_creature(encode_creature(s, 1, 0.0, 120.0, 1))
    assert np.abs(fr.particles - s.particles).max() < 1e-3 and fr.rd.shape == (len(s.pos),)

    class Sink:
        sent = 0

        def send_raw(self, b):
            pass

        def close(self):
            pass
    ses = LiveSession(LiveConfig(backend="hive", clock="internal", record=str(tmp_path), out=[]), start_inputs=False,
                      sink=Sink(), now=0.0)
    now = 0.0
    for _ in range(240):
        now += 1 / 120
        ses.step(now)
    take = CreatureTake(ses.save_take())
    p = take.particles(30)
    assert take.variant == "hive" and p.shape[1] == 1536 and np.isfinite(p).all()


# ---------------------------------------------------------------------------- Osseous line (v5-v7)
def _run(engine, seconds, dt=1 / 60, events=(), energy=0.8):
    out = []
    for i in range(int(seconds / dt)):
        t = i * dt
        beat = t * 2
        kick = 1.0 if (beat % 1.0) < dt * 2.5 else 0.0
        for at, name in events:
            if i == int(at / dt):
                engine.trigger_event(name)
        engine.set_input(CreatureControlInput(bass=0.6, high=0.4, energy=energy, transient=kick, spectral_flux=0.3,
                                              amplitude=0.7, tempo=120, beat=beat, playing=True))
        out.append(engine.update(dt))
    return out


def test_skeleton_is_a_tree_without_long_bones():
    from myrmex.creature.skeleton import mst_edges
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 3))
    own = np.repeat([0, 1], 30)
    edges = mst_edges(x, own, max_len=10.0)
    assert len(edges) == 58 and all(own[a] == own[b] for a, b in edges)
    assert all(np.linalg.norm(x[a] - x[b]) <= 0.8 for a, b in mst_edges(x, own, max_len=0.8))


def test_osseous_polyalloy_grows_bones_and_strikes():
    from myrmex.creature.osseous import OsseousPolyalloyEngine
    e = OsseousPolyalloyEngine()
    e.set_parameter("aggression", 0.9)
    states = _run(e, 12, events=((6.0, "STRIKE"),))
    s = states[-1]
    assert s.style == 1 and len(s.links) == 128 and np.isfinite(s.pos).all()
    assert max(int((st.links[:, 2] > 0.3).sum()) for st in states) > 20       # bone links
    assert "STRIKE" in {n for _, n, _ in e.events} or any(st.behavior == "STRIKE" for st in states)


def test_classic_organisms_stay_classic():
    from myrmex.creature.colony import ColonyEngine
    from myrmex.creature.polyalloy import CLASSIC, PolyalloyEngine
    e = PolyalloyEngine()
    assert not e.trigger_event("STRIKE") and e.trigger_event("IMPULSE")
    states = _run(e, 6)
    assert states[-1].style == 0 and len(states[-1].links) == 480
    assert all(st.morphology in CLASSIC for st in states)
    c = ColonyEngine()
    assert not c.trigger_event("OSSIFY")
    assert _run(c, 2)[-1].style == 0


def test_osseous_colony_and_hive():
    from myrmex.creature.osseous import OsseousColonyEngine, OsseousHiveEngine
    from myrmex.creature.protocol import decode_creature, encode_creature
    c = OsseousColonyEngine()
    c.set_parameter("aggression", 0.8)
    states = _run(c, 10, events=((2.0, "OSSIFY"),))
    s = states[-1]
    assert s.style == 1 and len(s.links) == 192
    assert max(float(st.plate.max()) for st in states) > 0.3                  # scutes
    fr = decode_creature(encode_creature(s, 1, 0.0, 120.0, 1))
    assert fr.style == 1 and np.array_equal(fr.links[:, :2], s.links[:, :2])
    h = OsseousHiveEngine()
    h.set_parameter("aggression", 0.8)
    states = _run(h, 6, events=((2.0, "QUILLS"),))
    assert "QUILLS" in {n for _, n, _ in h.events} and np.isfinite(states[-1].particles).all()


# ---------------------------------------------------------------------------- Cyber Hive (v8)
def test_cyber_hive_machine_forms_light_scan_and_glitch():
    from myrmex.creature.colony import CYBER_SHAPES
    from myrmex.creature.cyber import CyberHiveEngine
    e = CyberHiveEngine()
    assert e.trigger_event("SCAN") and e.trigger_event("GLITCH") and e.trigger_event("STRIKE")
    states = _run(e, 14, events=((4.0, "SCAN"), (7.0, "GLITCH"), (9.0, "OSSIFY")))
    s = states[-1]
    assert s.style == 2 and len(s.links) == 192 and s.light.shape == (len(s.pos),) and np.isfinite(s.pos).all()
    assert {"SCAN", "GLITCH"} <= {n for _, n, _ in e.events}
    scans = [st for st in states if math.isfinite(st.scan)]
    assert scans and max(float(np.ptp(st.light)) for st in scans) > 0.4           # a bright band, not uniform
    from myrmex.creature.puppet import GloveControl
    from myrmex.realtime.glove import SCULPT_SHAPES
    f = CyberHiveEngine()                                   # the machine forms exist: a sculpting hand holds each
    shapes = SCULPT_SHAPES["cyber_hive"]
    for name in CYBER_SHAPES:
        ext = np.array([1.0 if s_ == name else 0.0 for s_ in shapes])
        f.set_glove(GloveControl(active=True, fingers=ext, finger_mode="morph", grip=1.0), shapes)
        assert _run(f, 2.0)[-1].morphology == name
    # the classic and bony organisms never take the machine forms
    from myrmex.creature.osseous import OsseousHiveEngine
    assert all(st.morphology not in CYBER_SHAPES for st in _run(OsseousHiveEngine(), 4))


def test_cyber_hive_protocol_session_and_take(tmp_path):
    from myrmex.creature.cyber import CyberHiveEngine
    from myrmex.creature.take import CreatureTake
    e = CyberHiveEngine()
    e.trigger_event("SCAN")
    s = _run(e, 0.3)[-1]
    fr = decode_creature(encode_creature(s, 1, 0.0, 120.0, 1))
    assert fr.style == 2 and np.abs(fr.light - s.light).max() < 0.01 and fr.particles is not None
    assert (math.isnan(fr.scan) and math.isnan(s.scan)) or abs(fr.scan - s.scan) < 1e-4

    class Sink:
        def send_raw(self, b):
            pass

        def close(self):
            pass
    ses = LiveSession(LiveConfig(backend="cyber_hive", clock="internal", record=str(tmp_path), out=[]),
                      start_inputs=False, sink=Sink(), now=0.0)
    now = 0.0
    for _ in range(240):
        now += 1 / 120
        ses.step(now)
    assert ses.creature.trigger("GLITCH")
    take = CreatureTake(ses.save_take())
    assert take.variant == "cyber_hive" and take.d["light"].shape[1] == 128 and "scan" in take.d
    from myrmex.app import controllers as C
    assert C.take_variant(take.path) == "cyber_hive" and "cyber_hive" in C.CREATURE_BACKENDS
