import numpy as np

from myrmex.music import features, phrases, synthetic
from myrmex.music.predictor import GroovePredictor
from myrmex.music.timeline import MusicTimeline, TempoMap


def test_tempo_map_roundtrip_with_changes():
    tm = TempoMap([(0.0, 120.0), (16.0, 90.0), (32.0, 140.0)], [(0.0, 4, 4), (16.0, 3, 4)])
    for b in [0.0, 3.5, 16.0, 20.25, 40.0]:
        assert abs(tm.beats(tm.seconds(b)) - b) < 1e-9
    assert abs(tm.seconds(16.0) - 8.0) < 1e-9
    assert abs(tm.seconds(17.0) - (8.0 + 60.0 / 90.0)) < 1e-9
    bar, pos = tm.bar_position(19.0)          # 4 bars of 4/4, then 3/4
    assert bar == 5 and abs(pos - 0.0) < 1e-9


def test_timeline_serialisation_roundtrip(tmp_path):
    tl = synthetic.test_a_four_on_floor(bars=2)
    p = tmp_path / "tl.json"
    tl.save(str(p))
    tl2 = MusicTimeline.load(str(p))
    assert len(tl2.notes) == len(tl.notes)
    assert abs(tl2.duration - tl.duration) < 1e-9
    assert tl2.tracks["kick"].group == "kick"


def test_features_react_to_silence_and_builds():
    F = features.analyze_features(synthetic.test_d_sudden_silence(), 30.0)
    t = F["time"]
    in_gap = (t > 18.0) & (t < 22.0)
    before = (t > 8.0) & (t < 14.0)
    assert F["silence"][in_gap].mean() > 0.8
    assert F["energy"][before].mean() > 3 * F["energy"][in_gap].mean()
    E = features.analyze_features(synthetic.test_e_energy_transition(), 30.0)
    te = E["time"]
    assert E["trend"][(te > 12) & (te < 22)].mean() > 0.1


def test_structure_labels_demo():
    tl = synthetic.demo_arrangement()
    F = features.analyze_features(tl, 30.0)
    st = phrases.analyze_structure(tl, F, 30.0)
    labels = [s.label for s in st.sections]
    assert labels[0] == "intro"
    assert "build" in labels and "drop" in labels and "break" in labels
    starts = [s.start_bar for s in st.sections]
    assert 8 in starts and 16 in starts


def test_predictor_learns_and_detects_surprise_and_omission():
    p = GroovePredictor(groups=("kick",))
    surprises = []
    for bar in range(6):
        for beat in range(4):
            p.advance(bar, beat / 4.0)
            s, _ = p.observe("kick", bar, beat / 4.0, 0.9)
            surprises.append(s)
    assert max(surprises[-8:]) < 0.15                   # learned four-on-the-floor
    p.advance(6, 0.0)
    p.observe("kick", 6, 0.0, 0.9)
    s_off, _ = p.observe("kick", 6, 0.37, 1.0)         # off-grid unexpected hit
    assert s_off > 0.6
    for beat in (1, 2, 3):                              # the groove continues in bar 6
        p.advance(6, beat / 4.0)
        p.observe("kick", 6, beat / 4.0, 0.9)
    # Omission: bar 7 without the kick on beat 2.
    om = []
    for step in range(16):
        ph = step / 16.0
        om += p.advance(7, ph)
        if step in (0, 8, 12):
            p.observe("kick", 7, ph, 0.9)
    assert any(g == "kick" for g, _ in om)
    ant = p.anticipation(0.23, 2.0)                     # just before beat 2 (0.25)
    assert ant.get("kick", 0.0) > 0.3


def test_entry_event_after_silence():
    p = GroovePredictor(groups=("hats",))
    for bar in range(4):
        p.advance(bar, 0.0)
    _, entry = p.observe("hats", 4, 0.0, 0.5)
    assert entry
