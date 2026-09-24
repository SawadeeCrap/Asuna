"""Ableton .als reader tests.

A small Live-12-style set is generated here (same element paths as real sets
from Live 12.3), so no third-party project files are needed.  If the
``MYRMEX_DAWTOOL_ALS`` environment variable points to dawtool's ``tests/als``
folder, tempo-automation timings are additionally checked against dawtool's
Live-accurate reference values.
"""
import gzip
import os
import xml.etree.ElementTree as ET

import pytest

from myrmex.adapters import ableton_als as A


def _sub(parent, tag, value=None, **attrs):
    e = ET.SubElement(parent, tag, **{k: str(v) for k, v in attrs.items()})
    if value is not None:
        e.set("Value", str(value))
    return e


def make_set(path, tempo_points=((-63072000, 120.0),)):
    root = ET.Element("Ableton", MajorVersion="5", MinorVersion="12.0_12300", Creator="Ableton Live 12.3.6")
    ls = _sub(root, "LiveSet")
    tracks = _sub(ls, "Tracks")
    # ---- drum track with a looped 1-bar clip placed for 4 bars
    mt = _sub(tracks, "MidiTrack", Id="10")
    name = _sub(mt, "Name")
    _sub(name, "EffectiveName", "Beat")
    dc = _sub(mt, "DeviceChain")
    devs = _sub(_sub(dc, "DeviceChain"), "Devices")
    dg = _sub(devs, "DrumGroupDevice")
    branches = _sub(dg, "Branches")
    for note, pad in ((36, "Kick 909"), (38, "Snare Tight"), (42, "Closed Hat")):
        br = _sub(branches, "DrumBranch")
        bn = _sub(br, "Name")
        _sub(bn, "EffectiveName", pad)
        _sub(_sub(br, "BranchInfo"), "ReceivingNote", 128 - note)
    ms = _sub(dc, "MainSequencer")
    ev = _sub(_sub(_sub(ms, "ClipTimeable"), "ArrangerAutomation"), "Events")
    clip = _sub(ev, "MidiClip", Id="0", Time="0")
    _sub(clip, "CurrentStart", 0)
    _sub(clip, "CurrentEnd", 16)
    loop = _sub(clip, "Loop")
    _sub(loop, "LoopStart", 0)
    _sub(loop, "LoopEnd", 4)
    _sub(loop, "StartRelative", 0)
    _sub(loop, "LoopOn", "true")
    _sub(loop, "OutMarker", 4)
    _sub(clip, "Name", "groove")
    _sub(clip, "Disabled", "false")
    kts = _sub(_sub(clip, "Notes"), "KeyTracks")
    for key, times in ((36, (0, 1, 2, 3)), (38, (1, 3)), (42, (0.5, 1.5, 2.5, 3.5))):
        kt = _sub(kts, "KeyTrack", Id=str(key))
        notes = _sub(kt, "Notes")
        for t in times:
            _sub(notes, "MidiNoteEvent", Time=t, Duration=0.25, Velocity=100, OffVelocity=64, IsEnabled="true")
        _sub(kt, "MidiKey", key)
    # disabled note must be ignored
    _sub(notes, "MidiNoteEvent", Time=3.75, Duration=0.25, Velocity=100, IsEnabled="false")
    # ---- bass track, un-looped clip starting at bar 2
    bt = _sub(tracks, "MidiTrack", Id="11")
    _sub(_sub(bt, "Name"), "EffectiveName", "Sub Bass")
    bdc = _sub(bt, "DeviceChain")
    _sub(_sub(bdc, "DeviceChain"), "Devices")
    bev = _sub(_sub(_sub(_sub(bdc, "MainSequencer"), "ClipTimeable"), "ArrangerAutomation"), "Events")
    bclip = _sub(bev, "MidiClip", Id="1", Time="4")
    _sub(bclip, "CurrentStart", 4)
    _sub(bclip, "CurrentEnd", 12)
    bl = _sub(bclip, "Loop")
    _sub(bl, "LoopStart", 0)
    _sub(bl, "LoopEnd", 8)
    _sub(bl, "StartRelative", 0)
    _sub(bl, "LoopOn", "false")
    _sub(bl, "OutMarker", 8)
    _sub(bclip, "Disabled", "false")
    bkt = _sub(_sub(_sub(bclip, "Notes"), "KeyTracks"), "KeyTrack", Id="0")
    bn_ = _sub(bkt, "Notes")
    for t in (0, 2, 4, 6):
        _sub(bn_, "MidiNoteEvent", Time=t, Duration=1.5, Velocity=90)
    _sub(bkt, "MidiKey", 33)
    # ---- main track: tempo + automation
    main = _sub(ls, "MainTrack")
    mixer = _sub(_sub(main, "DeviceChain"), "Mixer")
    tempo = _sub(mixer, "Tempo")
    _sub(tempo, "Manual", tempo_points[0][1])
    _sub(tempo, "AutomationTarget", Id="8")
    _sub(_sub(mixer, "TimeSignature"), "Manual", 201)
    envs = _sub(_sub(main, "AutomationEnvelopes"), "Envelopes")
    env = _sub(envs, "AutomationEnvelope", Id="0")
    _sub(_sub(env, "EnvelopeTarget"), "PointeeId", 8)
    evs = _sub(_sub(_sub(env, "Automation"), "Events"), "Events") if False else _sub(_sub(env, "Automation"), "Events")
    for t, v in tempo_points:
        _sub(evs, "FloatEvent", Time=t, Value=v)
    locs = _sub(_sub(ls, "Locators"), "Locators")
    for t, n in ((0, "Intro"), (8, "Drop")):
        loc = _sub(locs, "Locator", Id=str(t))
        _sub(loc, "Time", t)
        _sub(loc, "Name", n)
    with gzip.open(path, "wb") as fh:
        fh.write(ET.tostring(root))
    return path


def test_als_parse_generated_set(tmp_path):
    p = make_set(str(tmp_path / "set.als"))
    rep = A.inspect_als(p)
    assert rep.tempo == 120.0 and rep.time_signature == (4, 4)
    assert len(rep.tracks) == 2 and rep.tracks[0]["drum_pads"][36] == "Kick 909"
    tl = A.load_als(p, analyze_audio=False)
    kicks = [n for n in tl.notes if tl.group_of(n) == "kick"]
    snares = [n for n in tl.notes if tl.group_of(n) == "snare"]
    hats = [n for n in tl.notes if tl.group_of(n) == "hats"]
    assert len(kicks) == 16 and len(snares) == 8 and len(hats) == 16      # 4 loop passes
    assert abs(kicks[5].time - 2.5) < 1e-9                                 # beat 5 at 120 bpm
    bass = [n for n in tl.notes if n.track == "11"]
    assert len(bass) == 4 and abs(bass[0].time - 2.0) < 1e-9               # clip placed at beat 4
    assert tl.tracks["11"].group == "bass"
    assert [m.name for m in tl.markers] == ["Intro", "Drop"]
    assert abs(tl.markers[1].time - 4.0) < 1e-9
    assert abs(tl.duration - 8.0) < 1e-9


def test_als_tempo_ramp_quantised_like_live(tmp_path):
    # 60 bpm until beat 4, ramp to 120 at beat 8 (Live applies it per 1/16 note).
    p = make_set(str(tmp_path / "ramp.als"), ((-63072000, 60.0), (4, 60.0), (8, 120.0)))
    tl = A.load_als(p, analyze_audio=False)
    # Expected: 4 s for the first 4 beats + sum over 16 sixteenths of the stepped tempo.
    steps = [60.0 + 60.0 * k / 16 for k in range(16)]
    expected = 4.0 + sum(0.25 * 60.0 / b for b in steps)
    assert abs(tl.tempo.seconds(8.0) - expected) < 1e-9


DAWTOOL = os.environ.get("MYRMEX_DAWTOOL_ALS")


@pytest.mark.skipif(not DAWTOOL, reason="set MYRMEX_DAWTOOL_ALS to dawtool/tests/als")
def test_against_dawtool_references():
    cases = {
        "automation-intense-unaligned.als": [0.0, 0.7452763515350631, 2.335821771598745, 4.173841066076857,
                                             5.706896019728953, 6.905534691333889, 7.692021891450212,
                                             8.974199935476374, 10.705147339679828],
        "live8/live8-patch-markers-auto.als": [81.09683000726724, 125.34683000726724],
    }
    for f, exp in cases.items():
        tl = A.load_als(os.path.join(DAWTOOL, f), analyze_audio=False)
        got = sorted(m.time for m in tl.markers)
        assert len(got) == len(exp)
        assert max(abs(a - b) for a, b in zip(got, exp)) < 1e-6
