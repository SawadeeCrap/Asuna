"""Realtime path: protocol, inputs, clocks, the Ableton Remote Script helpers and a live session."""
import importlib.util
import math
import os

import numpy as np
import pytest

from myrmex.bus.osc import decode
from myrmex.bus.transport import LiveEvent, OscInput
from myrmex.realtime.clock import ClockHub, OnsetClock
from myrmex.realtime.inputs import InputConfig, InputHub, ScoreFollower
from myrmex.realtime.protocol import (CameraState, PoseFrame, decode_names, decode_pose, encode_names,
                                      encode_pose, rig_id)
from myrmex.realtime.session import LiveConfig, LiveSession

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _remote_script():
    path = os.path.join(ROOT, "ableton", "remote_script", "Myrmex", "surface.py")
    spec = importlib.util.spec_from_file_location("myrmex_surface", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------- protocol
def test_pose_packet_roundtrip():
    names = ["pelvis", "spine_01", "head"]
    D = np.tile(np.eye(4), (3, 1, 1))
    D[1, :3, 3] = [0.1, -2.0, 0.3]
    c = math.cos(0.3)
    D[2, :3, :3] = [[c, -math.sin(0.3), 0], [math.sin(0.3), c, 0], [0, 0, 1]]
    cam = CameraState(np.array([1.0, 2.0, 1.5]), np.array([0.0, 0.0, 1.0]), 85.0, 3.2, 2.8, 7, "hips_close")
    fr = PoseFrame(42, 12.5, 33.25, 124.0, rig_id(names), D, 1, cam, np.array([0.0, -5.0, 0.9]), -1.57, 0.9, 0.7)
    data = encode_pose(fr)
    assert len(data) < 1400
    back = decode_pose(data)
    assert back.seq == 42 and back.playing and back.rig == rig_id(names)
    assert np.allclose(back.deltas, D, atol=1e-6)
    assert back.camera.kind == "hips_close" and back.camera.shot_id == 7
    assert np.allclose(back.camera.position, cam.position) and abs(back.camera.lens - 85.0) < 1e-5
    assert np.allclose(back.subject_pos, [0.0, -5.0, 0.9], atol=1e-6)
    rid, nm = decode_names(encode_names(names))
    assert nm == names and rid == rig_id(names)
    assert decode_pose(b"garbage") is None


# ---------------------------------------------------------------------------- Remote Script
def test_remote_script_osc_is_decodable_and_scores_map_to_song_beats():
    rs = _remote_script()
    import struct
    blob = struct.pack(">8f", 16.0, 0.25, 36.0, 0.9, 17.0, 0.25, 38.0, 0.7)
    msgs = decode(rs._osc("/myrmex/score/notes", ["t0", blob, 16.0, 24.0]))
    ev = OscInput._to_event(msgs[0], 0.0)
    assert ev.kind == "score_notes" and ev.data["window"] == (16.0, 24.0)
    assert ev.data["notes"].shape == (2, 4) and ev.data["notes"][1, 2] == 38.0
    tr = OscInput._to_event(decode(rs._osc("/myrmex/transport", [12.5, 124.0, 1, 4, 4]))[0], 0.0)
    assert tr.kind == "transport" and tr.data["playing"] and tr.data["bpm"] == pytest.approx(124.0)

    class Clip:                                   # a 1-bar looping session clip
        looping, loop_start, loop_end = True, 0.0, 4.0
    notes = [(0.0, 0.25, 36, 100), (1.0, 0.25, 38, 90), (2.0, 0.25, 36, 100), (3.0, 0.25, 38, 90)]
    up = rs.upcoming(notes, Clip(), clip_pos=3.5, song_now=100.0, window=8.0)
    beats = sorted(b for b, *_ in up)
    assert beats[0] == pytest.approx(100.5)      # next kick: half a beat after "now"
    assert len(beats) == 8 and beats[-1] == pytest.approx(107.5)


def test_score_follower_windows_and_gm_mapping():
    sf = ScoreFollower()
    sf.set_track("drums", "Drums", "gm")
    sf.set_notes("drums", np.array([[4.0, 0.2, 36, 0.9], [5.0, 0.2, 38, 0.8], [6.0, 0.2, 42, 0.5]]), (4.0, 12.0))
    # A new window replaces its range (the clip changed): the old 6.0 hat is gone.
    sf.set_notes("drums", np.array([[6.5, 0.2, 36, 0.9]]), (6.0, 14.0))
    assert sf.due(3.9, 120.0, 0.0) == []                  # first call primes
    got = sf.due(5.5, 120.0, 1.0) + sf.due(7.0, 120.0, 1.1)
    assert [(n.beat, n.group) for n in got] == [(4.0, "kick"), (5.0, "snare"), (6.5, "kick")]
    assert sf.due(3.0, 120.0, 2.0) == []                  # relocation backwards: nothing replayed


# ---------------------------------------------------------------------------- inputs
def test_input_hub_midi_cc_and_generic_osc_mapping():
    hub = InputHub(InputConfig(osc_port=0), start=False)
    hub.push(LiveEvent("note", 0.0, {"channel": 1, "pitch": 36.0, "velocity": 0.9}))
    hub.push(LiveEvent("note", 0.0, {"channel": 2, "pitch": 40.0, "velocity": 0.6}))
    hub.push(LiveEvent("cc", 0.0, {"control": 1, "channel": 1, "value": 0.8}))
    hub.push(LiveEvent("osc", 0.0, {"address": "/ch/2", "value": 1.0}))
    hub.push(LiveEvent("osc", 0.0, {"address": "/ch/2", "value": 1.0}))     # gate held: no retrigger
    hub.push(LiveEvent("cc", 0.0, {"control": 6, "channel": 1, "value": 1.0}))
    notes = hub.poll(0.0, 0.0)
    assert [n.group for n in notes] == ["kick", "bass", "snare"]
    assert notes[1].duration > 1.0                       # open until note-off
    hub.push(LiveEvent("note_off", 0.0, {"channel": 2, "pitch": 40.0}))
    hub.poll(0.0, 0.5)
    assert notes[1].duration == pytest.approx(0.5)
    assert hub.controls["energy"] == pytest.approx(0.8)
    assert [n for n, _ in hub.take_triggers()] == ["camera"]


# ---------------------------------------------------------------------------- clocks
def test_onset_clock_locks_to_kicks():
    oc = OnsetClock(bpm=100.0)
    period = 60.0 / 128.0
    rng = np.random.default_rng(0)
    t = 10.0
    for i in range(48):
        oc.hit(t + rng.normal(0.0, 0.004), 1.0)
        if i % 4 == 2:
            oc.hit(t + period / 2, 0.6)                 # off-beat snare-ish hits
        t += period
    st = oc.state(t)
    assert abs(st.bpm - 128.0) < 1.0
    phase = st.beat - math.floor(st.beat)
    assert min(phase, 1.0 - phase) < 0.08               # a grid line is "now"


def test_clock_hub_relocation_keeps_beat_continuous_and_phase_of_source():
    hub = ClockHub("auto", 120.0, link=False, now=0.0)
    hub.feed(LiveEvent("transport", 0.0, {"beat": 32.0, "bpm": 120.0, "playing": True}))
    b0 = hub.state(0.5).beat
    # Song restarts at beat 0.25 (relocation): our beat must not jump back to zero.
    hub.feed(LiveEvent("transport", 0.6, {"beat": 0.25, "bpm": 120.0, "playing": True}))
    st = hub.state(0.6)
    assert st.source == "osc"
    assert st.beat > b0 - 2.01
    assert (st.beat - 0.25) % 4.0 == pytest.approx(0.0, abs=1e-6)       # bar phase of the song survives
    assert st.song_beat == pytest.approx(0.25)


# ---------------------------------------------------------------------------- live session
def test_live_session_walks_on_the_beat_holds_in_silence_and_resumes(biped_plan):
    cfg = LiveConfig(clock="onsets", bpm=124.0, link=False, out=[], latency=0.0, inputs=InputConfig(osc_port=0))
    T0 = 500.0
    s = LiveSession(cfg, biped_plan, start_inputs=False, now=T0)
    spb = 60.0 / 124.0
    dt = 1.0 / 120.0
    nxt = T0 + 0.3
    kicks = []
    speeds = {}
    for i in range(int(26.0 / dt)):
        now = T0 + (i + 1) * dt
        rel = now - T0
        while nxt <= now:
            if rel < 12.0 or rel > 18.0:
                s.inputs.push(LiveEvent("note", nxt, {"channel": 1, "pitch": 36.0, "velocity": 0.9}))
                kicks.append(nxt - T0)
            nxt += spb
        s.step(now)
        for mark in (10.0, 16.5, 25.0):
            if abs(rel - mark) < dt / 2:
                speeds[mark] = (s.motor.speed_ref, s.hold)
    assert speeds[10.0][0] > 0.5 and not speeds[10.0][1]           # walking
    assert speeds[16.5][1] and speeds[16.5][0] < 0.1                # music stopped: holding a pose
    assert speeds[25.0][0] > 0.5 and not speeds[25.0][1]           # walked off again
    td = [t for (t, k, d) in s.motor.events if k == "touchdown" and (4.0 < t < 12.0)]
    k0 = kicks[0]
    err = [abs(((t - k0) / (spb / 2)) - round((t - k0) / (spb / 2))) * spb / 2 for t in td]
    assert len(td) >= 8 and float(np.median(err)) < 0.04
    fr = s.last_frame
    assert fr is not None and fr.camera is not None and np.isfinite(fr.deltas).all()


def test_audio_onsets_from_a_rendered_song(tmp_path):
    import wave

    from myrmex.music import synth, synthetic
    from myrmex.realtime.audio_in import BandOnsetDetector
    tl = synthetic.test_a_four_on_floor(bars=6, bpm=124.0)
    p = synth.render_audio(tl, str(tmp_path / "a.wav"))
    with wave.open(p) as w:
        sr, ch, raw = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32).reshape(-1, ch).mean(axis=1) / 32768.0
    det = BandOnsetDetector(sr=sr)
    hits = []
    for i in range(0, len(y), 256):
        hits += det.process(y[i:i + 256])
    kicks = np.array(sorted(n.time for n in tl.notes if tl.group_of(n) == "kick"))
    got = np.array([t for t, g, v in hits if g == "kick"])
    recall = np.mean([np.min(np.abs(got - k)) < 0.04 for k in kicks])
    lat = np.median([got[np.argmin(np.abs(got - k))] - k for k in kicks])
    assert recall > 0.8 and 0.0 <= lat < 0.02


def test_recorded_take_keeps_live_camera_and_audio_alignment(biped_plan, tmp_path):
    from myrmex.performance.performance import Performance
    from myrmex.performance.takes import recorded_camera_track, take_audio_offset
    cfg = LiveConfig(clock="auto", link=False, out=[], latency=0.0, record=str(tmp_path),
                     inputs=InputConfig(osc_port=0))
    T0 = 100.0
    s = LiveSession(cfg, biped_plan, start_inputs=False, now=T0)
    song0 = T0 + 1.0                                     # Ableton starts playing 1 s after we record
    dt = 1.0 / 120.0
    next_tr, next_kick = song0, song0
    for i in range(int(8.0 / dt)):
        now = T0 + (i + 1) * dt
        while now >= next_tr:
            s.inputs.push(LiveEvent("transport", next_tr, {"beat": (next_tr - song0) * 2.0, "bpm": 120.0,
                                                           "playing": True}))
            next_tr += 0.05
        while now >= next_kick:
            s.inputs.push(LiveEvent("note", next_kick, {"channel": 1, "pitch": 36.0, "velocity": 0.9}))
            next_kick += 0.5
        s.step(now)
    path = s.save_take()
    perf = Performance.load(path)
    track = recorded_camera_track(perf)
    assert track is not None and track.positions.shape == (perf.frames, 3)
    assert np.isfinite(track.positions).all() and len(track.shots) >= 1
    assert sum(sh.end - sh.start for sh in track.shots) == perf.frames
    off = take_audio_offset(perf)
    assert off == pytest.approx(-1.0, abs=0.06)


def test_myrmex_track_knobs_and_choreography_notes(biped_plan):
    rs = _remote_script()
    assert rs.macro_role("Energy", 5) == "energy" and rs.macro_role("Macro 2", 1) == "stride"
    assert rs.macro_role("Камера", 0) == "camera" and rs.macro_role("Filter", 0) is None

    class P:
        min, max, value = 0.0, 127.0, 0.0
    assert rs.macro_value("energy", P()) == -1.0          # knob at zero = automatic
    cfg = LiveConfig(clock="internal", link=False, out=[], latency=0.0, inputs=InputConfig(osc_port=0))
    s = LiveSession(cfg, biped_plan, start_inputs=False, now=0.0)
    s.inputs.push(LiveEvent("control", 0.0, {"name": "energy", "value": 0.9}))
    s.step(1 / 120)
    assert s.engine.runway.live["energy"] == pytest.approx(0.9)
    s.inputs.push(LiveEvent("control", 0.0, {"name": "energy", "value": -1.0}))
    s.step(2 / 120)
    assert "energy" not in s.engine.runway.live
    # Choreography on MIDI channel 16 (the "control" channel): E3 = camera cut, C4 held = hold.
    s.inputs.push(LiveEvent("note", 0.0, {"channel": 16, "pitch": 64.0, "velocity": 1.0}))
    s.inputs.push(LiveEvent("note", 0.0, {"channel": 16, "pitch": 72.0, "velocity": 1.0}))
    s.step(3 / 120)
    assert s.hold and s.camera.pending_cut
    s.inputs.push(LiveEvent("note_off", 0.0, {"channel": 16, "pitch": 72.0}))
    for k in range(4, 12):
        s.step(k / 120)
    assert not s._hold_notes
