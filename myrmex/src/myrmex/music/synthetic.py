"""Synthetic musical test material (tests A–F and a full demo arrangement).

Each generator returns a :class:`MusicTimeline` built exactly like real
adapters build theirs, so the whole pipeline can be exercised (and judged by
ear, via :func:`myrmex.music.synth.render_audio`) without Ableton or VCV.
"""
from __future__ import annotations

from .timeline import MusicTimeline, NoteEvent, TempoMap, Track
from ..util.rng import Rng

KICK, SNARE, CLAP, HAT_C, HAT_O, PERC, BASS, PAD, LEAD, FX = (
    "kick", "snare", "clap", "hat_c", "hat_o", "perc", "bass", "pad", "lead", "fx")

_GROUP = {KICK: "kick", SNARE: "snare", CLAP: "snare", HAT_C: "hats", HAT_O: "hats", PERC: "perc",
          BASS: "bass", PAD: "harmony", LEAD: "melody", FX: "fx"}
_PITCH = {KICK: 36, SNARE: 38, CLAP: 39, HAT_C: 42, HAT_O: 46, PERC: 64}


class _Builder:
    def __init__(self, bpm: float, name: str, seed: int = 1):
        self.tl = MusicTimeline(TempoMap([(0.0, bpm)]), source=f"synthetic:{name}")
        self.bpm = bpm
        self.rng = Rng(seed)
        for tid, g in _GROUP.items():
            self.tl.tracks[tid] = Track(tid, tid.upper(), "midi", g, 1.0)

    def sec(self, beat: float) -> float:
        return self.tl.tempo.seconds(beat)

    def hit(self, track: str, beat: float, vel: float, dur_beats: float = 0.1, pitch: float | None = None,
            humanize: float = 0.0, sharp: float = 0.8) -> None:
        jitter = self.rng.normal(0.0, humanize) if humanize else 0.0
        b = max(0.0, beat + jitter)
        p = pitch if pitch is not None else _PITCH.get(track, 60)
        self.tl.notes.append(NoteEvent(self.sec(b), dur_beats * 60.0 / self.bpm, float(p),
                                       max(0.02, min(1.0, vel)), track, beat=b, sharpness=sharp))

    def four_floor(self, bar: int, vel: float = 0.95, hats: bool = True, clap: bool = True,
                   hat_density: int = 2) -> None:
        b0 = bar * 4.0
        for k in range(4):
            self.hit(KICK, b0 + k, vel * (1.0 if k % 2 == 0 else 0.93))
        if clap:
            self.hit(CLAP, b0 + 1, vel * 0.8, humanize=0.004)
            self.hit(CLAP, b0 + 3, vel * 0.82, humanize=0.004)
        if hats:
            step = 1.0 / hat_density
            k = 0
            while k * step < 4.0:
                pos = b0 + k * step
                off = (k * step) % 1.0 > 0.0
                self.hit(HAT_O if off and hat_density == 2 else HAT_C, pos,
                         (0.55 if off else 0.35) * vel + self.rng.uniform(-0.05, 0.05), humanize=0.006, sharp=1.0)
                k += 1

    def bassline(self, bar: int, root: int, vel: float = 0.8, pattern=(0.5, 1.5, 2.5, 3.5)) -> None:
        for p in pattern:
            self.hit(BASS, bar * 4.0 + p, vel, dur_beats=0.4, pitch=root, sharp=0.5)

    def pad(self, bar: int, chord: tuple[int, ...], vel: float = 0.5, bars: int = 1) -> None:
        for n in chord:
            self.hit(PAD, bar * 4.0, vel, dur_beats=4.0 * bars - 0.1, pitch=n, sharp=0.1)

    def done(self, bars: int) -> MusicTimeline:
        self.tl.duration = self.sec(bars * 4.0)
        return self.tl.finalize()


CHORDS = [(57, 60, 64), (53, 57, 60), (55, 59, 62), (52, 55, 59)]
ROOTS = [45, 41, 43, 40]


def test_a_four_on_floor(bars: int = 16, bpm: float = 124.0) -> MusicTimeline:
    """TEST A – steady four-on-the-floor."""
    b = _Builder(bpm, "A_four_on_floor")
    for bar in range(bars):
        b.four_floor(bar)
        b.bassline(bar, ROOTS[(bar // 2) % 4])
        if bar % 2 == 0:
            b.pad(bar, CHORDS[(bar // 2) % 4], 0.35, bars=2)
    return b.done(bars)


def test_b_broken(bars: int = 16, bpm: float = 136.0) -> MusicTimeline:
    """TEST B – dense broken percussion (breakbeat-like, ghost notes, 16th hats)."""
    b = _Builder(bpm, "B_broken", seed=2)
    kick_pat = [0.0, 0.75, 2.5, 2.75]
    snare_pat = [1.0, 3.0]
    ghosts = [1.75, 2.25, 3.5, 3.75]
    for bar in range(bars):
        b0 = bar * 4.0
        kp = kick_pat if bar % 4 != 3 else [0.0, 0.5, 1.75, 2.5, 3.25]
        for p in kp:
            b.hit(KICK, b0 + p, 0.9, humanize=0.01)
        for p in snare_pat:
            b.hit(SNARE, b0 + p, 0.95, humanize=0.008)
        for p in ghosts:
            if b.rng.chance(0.6):
                b.hit(SNARE, b0 + p, 0.25 + b.rng.uniform(0, 0.15), humanize=0.012)
        for k in range(16):
            if b.rng.chance(0.85):
                b.hit(HAT_C, b0 + k * 0.25, 0.25 + 0.35 * b.rng.random() + (0.2 if k % 4 == 2 else 0.0),
                      humanize=0.008, sharp=1.0)
        for k in range(8):
            if b.rng.chance(0.35):
                b.hit(PERC, b0 + k * 0.5 + 0.25, 0.4 + 0.3 * b.rng.random(), pitch=62 + b.rng.randint(0, 7))
        b.bassline(bar, ROOTS[(bar // 2) % 4], 0.75, pattern=(0.0, 0.75, 2.5, 3.0))
    return b.done(bars)


def test_c_ambient(bars: int = 12, bpm: float = 72.0) -> MusicTimeline:
    """TEST C – slow ambient: long pads, sparse melody, no drums."""
    b = _Builder(bpm, "C_ambient", seed=3)
    for bar in range(0, bars, 2):
        b.pad(bar, tuple(n - 12 for n in CHORDS[(bar // 2) % 4]) + (CHORDS[(bar // 2) % 4][0] + 12,), 0.4, bars=2)
    melody = [69, 72, 76, 74, 72, 67, 69, 64]
    for i, bar in enumerate(range(1, bars, 1)):
        if b.rng.chance(0.7):
            b.hit(LEAD, bar * 4.0 + b.rng.choice([0.0, 1.5, 2.0, 3.0]), 0.35 + 0.2 * b.rng.random(),
                  dur_beats=2.5, pitch=melody[i % len(melody)], sharp=0.2)
    return b.done(bars)


def test_d_sudden_silence(bpm: float = 124.0) -> MusicTimeline:
    """TEST D – groove, sudden silence (4 bars), groove returns."""
    b = _Builder(bpm, "D_sudden_silence", seed=4)
    for bar in list(range(0, 8)) + list(range(12, 18)):
        b.four_floor(bar, hat_density=4 if bar >= 4 else 2)
        b.bassline(bar, ROOTS[(bar // 2) % 4])
    return b.done(18)


def test_e_energy_transition(bpm: float = 128.0) -> MusicTimeline:
    """TEST E – sparse intro, 8-bar build (snare roll, rising sweep), drop, breakdown."""
    b = _Builder(bpm, "E_energy_transition", seed=5)
    for bar in range(0, 4):                       # intro: kick + pad
        b.hit(KICK, bar * 4.0, 0.7)
        b.hit(KICK, bar * 4.0 + 2.0, 0.6)
        b.pad(bar, CHORDS[bar % 4], 0.3)
    for i, bar in enumerate(range(4, 12)):        # build
        b.four_floor(bar, vel=0.6 + 0.04 * i, clap=i >= 2, hat_density=2 if i < 4 else 4)
        roll = 1 if i < 2 else 2 if i < 4 else 4 if i < 6 else 8
        for k in range(4 * roll):
            b.hit(SNARE, bar * 4.0 + k / roll, 0.3 + 0.08 * i, humanize=0.003)
        b.hit(FX, bar * 4.0, 0.3 + 0.07 * i, dur_beats=4.0, pitch=60 + 3 * i, sharp=0.2)
    b.hit(FX, 48.0, 1.0, dur_beats=2.0, pitch=84, sharp=1.0)
    for bar in range(12, 20):                     # drop
        b.four_floor(bar, vel=1.0, hat_density=4)
        b.bassline(bar, ROOTS[(bar // 2) % 4], 0.95, pattern=(0.0, 0.5, 1.5, 2.5, 3.0, 3.5))
        if bar % 2 == 0:
            b.pad(bar, CHORDS[(bar // 2) % 4], 0.55, bars=2)
    for bar in range(20, 24):                     # breakdown
        b.pad(bar, CHORDS[bar % 4], 0.45)
        b.hit(LEAD, bar * 4.0 + 1.0, 0.5, dur_beats=2.0, pitch=72 + (bar % 3) * 2, sharp=0.3)
    return b.done(24)


def test_f_repetition_with_surprise(bars: int = 20, bpm: float = 120.0, surprise_bar: int = 12) -> MusicTimeline:
    """TEST F – one bar repeated exactly, with a single unexpected off-grid event."""
    b = _Builder(bpm, "F_repetition_surprise", seed=6)
    for bar in range(bars):
        b.four_floor(bar, clap=True, hat_density=2)
        b.bassline(bar, 45)
    b.hit(FX, surprise_bar * 4.0 + 2.37, 1.0, dur_beats=0.5, pitch=90, sharp=1.0)
    b.hit(PERC, surprise_bar * 4.0 + 2.37, 1.0, pitch=70, sharp=1.0)
    return b.done(bars)


def demo_arrangement(bpm: float = 122.0, seed: int = 11) -> MusicTimeline:
    """~64 s arrangement: intro, build, peak, break, return, outro."""
    b = _Builder(bpm, "demo", seed=seed)
    for bar in range(0, 4):
        b.pad(bar, CHORDS[bar % 4], 0.35)
        if bar >= 2:
            b.hit(KICK, bar * 4.0, 0.55)
    for i, bar in enumerate(range(4, 8)):
        b.four_floor(bar, vel=0.7 + 0.05 * i, clap=i >= 2, hat_density=2)
        b.bassline(bar, ROOTS[(bar // 2) % 4], 0.6 + 0.05 * i)
        if i == 3:
            for k in range(16):
                b.hit(SNARE, bar * 4.0 + k * 0.25, 0.3 + 0.04 * k)
    for bar in range(8, 16):
        b.four_floor(bar, vel=1.0, hat_density=4)
        b.bassline(bar, ROOTS[(bar // 2) % 4], 0.9, pattern=(0.0, 0.5, 1.5, 2.5, 3.0, 3.5))
        if bar % 2 == 0:
            b.pad(bar, CHORDS[(bar // 2) % 4], 0.5, bars=2)
        if bar in (11, 15):
            b.hit(PERC, bar * 4.0 + 3.5, 0.9, pitch=70)
    for bar in range(16, 22):
        b.pad(bar, CHORDS[bar % 4], 0.4)
        if bar % 2 == 1:
            b.hit(LEAD, bar * 4.0 + 1.5, 0.45, dur_beats=2.0, pitch=74 + (bar % 3) * 3, sharp=0.3)
    for bar in range(22, 30):
        b.four_floor(bar, vel=0.95, hat_density=4 if bar >= 24 else 2)
        b.bassline(bar, ROOTS[(bar // 2) % 4], 0.85)
    for bar in range(30, 32):
        b.pad(bar, CHORDS[bar % 4], 0.3)
        b.hit(KICK, bar * 4.0, 0.5)
    return b.done(32)


TESTS = {
    "A": test_a_four_on_floor,
    "B": test_b_broken,
    "C": test_c_ambient,
    "D": test_d_sudden_silence,
    "E": test_e_energy_transition,
    "F": test_f_repetition_with_surprise,
    "demo": demo_arrangement,
}
