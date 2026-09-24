"""Runway strut: forward travel, beat-locked footfalls, and the cinematographer's shot plan."""
import math

import numpy as np
import pytest

from myrmex.camera import cinematographer as cine
from myrmex.music import synthetic
from myrmex.performance.generator import GenerateOptions, generate_biped


@pytest.fixture(scope="module")
def strut(biped_plan):
    tl = synthetic.test_a_four_on_floor(bars=8, bpm=120.0)
    perf = generate_biped(tl, biped_plan, GenerateOptions(seed=5, fps=30, tail=0.0))
    return tl, perf


def test_runway_walks_forward_without_turning(strut, biped_plan):
    _, perf = strut
    i = perf.bone_names.index("pelvis")
    P = perf.deltas[:, i, :3, 3].astype(float)
    h0 = float(perf.meta["heading0"])
    fwd = np.array([math.cos(h0), math.sin(h0)])
    along = (P[:, :2] - P[0, :2]) @ fwd
    dur = perf.frames / perf.fps
    # Keeps going: at least ~0.5 m/s on average, and never walks back.
    assert along[-1] > 0.5 * dur
    step = int(perf.fps)
    assert (np.diff(along[::step]) > -0.05).all()
    # Stays on the runway: lateral drift stays small compared to the distance travelled.
    lat = (P[:, :2] - P[0, :2]) @ np.array([-fwd[1], fwd[0]])
    assert np.abs(lat).max() < 0.15 * along[-1]


def test_footfalls_lock_to_the_beat_grid(strut):
    tl, perf = strut
    td = [e for e in perf.events if e["type"] == "touchdown" and e["time"] > 2.0]
    assert len(td) > 10
    # Error to the nearest half-beat (footfalls on quarters or eighths).
    err = []
    for e in td:
        b = tl.tempo.beats(e["time"])
        err.append(abs(b * 2 - round(b * 2)) / 2 * 60.0 / tl.tempo.bpm_at(e["time"]))
    assert float(np.median(err)) < 0.04
    # Feet alternate.
    sides = [e["foot"] for e in td]
    alternations = sum(1 for a, b in zip(sides, sides[1:]) if a != b)
    assert alternations >= 0.8 * (len(sides) - 1)


def _straight_walk(seconds=40.0, fps=30.0, speed=0.9, heading=-math.pi / 2):
    t = np.arange(int(seconds * fps)) / fps
    d = np.array([math.cos(heading), math.sin(heading), 0.0])
    path = t[:, None] * speed * d[None] + np.array([0.0, 0.0, 0.95])
    sway = 0.04 * np.sin(2 * math.pi * t)[:, None] * np.array([d[1], -d[0], 0.0])[None]
    path = path + sway
    head = path + np.array([0.0, 0.0, 0.62])
    feet = path * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, 0.1])
    return path, head, feet


def test_shot_plan_cuts_on_downbeats_and_varies():
    fps, bpm = 30.0, 120.0
    beats = np.arange(0, 100) * 60.0 / bpm
    sections = [{"label": "intro", "start": 0.0, "energy": 0.3}, {"label": "build", "start": 8.0, "energy": 0.6},
                {"label": "drop", "start": 16.0, "energy": 1.0}, {"label": "break", "start": 32.0, "energy": 0.3}]
    path, head, feet = _straight_walk()
    track = cine.compose(path, head, feet, fps, sections, beats, -math.pi / 2, 1.7, seed=3)
    shots = track.shots
    assert len(shots) >= 5
    assert shots[0].kind in cine.ESTABLISHING
    bars = beats[::4]
    starts = {s["start"] for s in sections}
    for a, b in zip(shots, shots[1:]):
        assert a.kind != b.kind
        assert a.end == b.start
        tc = b.start / fps
        on_bar = np.min(np.abs(bars - tc)) < 1.5 / fps
        on_section = min(abs(s - tc) for s in starts) < 1.5 / fps
        assert on_bar or on_section
    assert np.isfinite(track.positions).all() and np.isfinite(track.targets).all()
    # The camera never enters the subject and always looks towards it.
    dist = np.linalg.norm(track.positions[:, :2] - path[:, :2], axis=1)
    assert dist.min() > 0.8
    to_subject = path - track.positions
    view = track.targets - track.positions
    cosang = (to_subject * view).sum(1) / (np.linalg.norm(to_subject, axis=1) * np.linalg.norm(view, axis=1))
    assert cosang.min() > 0.7


def test_shot_plan_is_deterministic():
    fps = 30.0
    beats = np.arange(0, 80) * 0.5
    sections = [{"label": "groove", "start": 0.0, "energy": 0.5}]
    path, head, feet = _straight_walk(30.0)
    a = cine.compose(path, head, feet, fps, sections, beats, -math.pi / 2, seed=9)
    b = cine.compose(path, head, feet, fps, sections, beats, -math.pi / 2, seed=9)
    c = cine.compose(path, head, feet, fps, sections, beats, -math.pi / 2, seed=10)
    assert [s.kind for s in a.shots] == [s.kind for s in b.shots]
    assert np.array_equal(a.positions, b.positions)
    assert [s.kind for s in a.shots] != [s.kind for s in c.shots] or not np.array_equal(a.positions, c.positions)
