"""Procedural cinematography for a travelling subject (offline, knows the future).

Shots are defined in the subject's own frame (forward / left / up of its
smoothed path), so a camera can dolly backwards in front of a walking model,
track beside her, sit low for a hero angle, or go tight on the heels while
they hit the floor on the beat.  Cuts land on downbeats at phrase / section
boundaries; shot choice follows the music's intensity, avoids repeats and
keeps screen direction (180-degree rule) for side shots.

Because the whole performance is known, the subject path is smoothed with a
zero-phase filter: the camera anticipates instead of lagging, and never
shakes with the footsteps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..util.noise import Noise1D
from ..util.rng import Rng, stable_hash64

UP = np.array([0.0, 0.0, 1.0])


@dataclass
class ShotType:
    name: str
    fwd: float           # camera offset along subject forward (m, for a 1.7 m subject)
    left: float          # along subject left (sign flipped by the shot side)
    up: float
    target_up: float     # look-at height above the path
    lens: float          # mm
    fstop: float
    drift: float = 0.12  # orbit drift (rad)
    push: float = 0.12   # dolly-in over the shot (fraction of distance)
    target: str = "path"  # path | head | feet
    energy: tuple[float, float] = (0.0, 1.0)   # preferred intensity range


SHOTS = {
    "front_dolly": ShotType("front_dolly", 4.8, 0.25, 1.0, 0.88, 50.0, 4.0, 0.1, 0.08, energy=(0.2, 1.0)),
    "front_low": ShotType("front_low", 3.0, 0.45, 0.35, 0.95, 32.0, 4.0, 0.12, 0.08, energy=(0.5, 1.0)),
    "side_track": ShotType("side_track", 0.35, 4.2, 1.0, 0.9, 50.0, 4.0, 0.08, 0.05, energy=(0.0, 1.0)),
    "three_quarter": ShotType("three_quarter", 3.2, 3.0, 1.2, 0.92, 50.0, 4.0, 0.15, 0.1, energy=(0.0, 1.0)),
    "rear_follow": ShotType("rear_follow", -3.2, 0.35, 1.15, 0.9, 50.0, 4.0, 0.1, 0.06, energy=(0.3, 1.0)),
    "feet_close": ShotType("feet_close", 1.6, 0.9, 0.3, 0.06, 85.0, 2.8, 0.1, 0.05, target="feet", energy=(0.4, 1.0)),
    "hips_close": ShotType("hips_close", 1.2, 1.7, 0.9, 0.95, 85.0, 2.8, 0.08, 0.05, energy=(0.5, 1.0)),
    "face_close": ShotType("face_close", 1.9, 0.35, 1.55, 1.52, 100.0, 2.8, 0.06, 0.05, target="head",
                           energy=(0.0, 0.7)),
    "wide_orbit": ShotType("wide_orbit", 4.6, 3.0, 1.6, 0.9, 50.0, 5.6, 0.6, 0.0, energy=(0.0, 0.6)),
}

# Relative preference per section.  Walking *towards* the lens is the money shot of a strut, so
# frontal shots dominate; close-ups punctuate energetic parts, wide shots breathe in quiet ones.
SECTION_SHOTS = {
    "intro": {"front_dolly": 2.0, "wide_orbit": 1.2, "side_track": 1.2, "face_close": 1.0, "three_quarter": 1.0},
    "build": {"front_dolly": 2.4, "feet_close": 1.3, "side_track": 1.0, "three_quarter": 1.0, "face_close": 1.0},
    "drop": {"front_low": 2.4, "front_dolly": 2.2, "hips_close": 1.3, "feet_close": 1.3, "rear_follow": 1.2,
             "three_quarter": 0.8},
    "peak": {"front_low": 2.0, "front_dolly": 2.0, "hips_close": 1.2, "feet_close": 1.2, "rear_follow": 1.2,
             "side_track": 0.8},
    "groove": {"front_dolly": 2.0, "side_track": 1.3, "three_quarter": 1.2, "rear_follow": 1.0, "feet_close": 0.8},
    "return": {"front_dolly": 2.0, "front_low": 1.6, "three_quarter": 1.0, "hips_close": 1.0},
    "break": {"front_dolly": 1.5, "wide_orbit": 1.3, "face_close": 1.5, "side_track": 1.0, "three_quarter": 1.0},
    "outro": {"front_dolly": 1.8, "wide_orbit": 1.5, "face_close": 1.0},
    "silence": {"face_close": 2.0, "front_dolly": 1.2, "wide_orbit": 1.0},
}


# The first shot tells the viewer where we are and who walks.
ESTABLISHING = {"front_dolly": 2.0, "wide_orbit": 1.0, "three_quarter": 0.8}


@dataclass
class Shot:
    start: int
    end: int
    kind: str
    side: float
    seed: int


@dataclass
class CameraTrack:
    fps: float
    positions: np.ndarray      # (T, 3)
    targets: np.ndarray        # (T, 3)
    lens: np.ndarray           # (T,)
    focus: np.ndarray          # (T,)
    fstop: np.ndarray          # (T,)
    shots: list[Shot] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"fps": self.fps, "shots": [s.__dict__ for s in self.shots]}


def _gauss_smooth(P: np.ndarray, sigma_frames: float) -> np.ndarray:
    if sigma_frames <= 0.5:
        return P.copy()
    r = int(3 * sigma_frames)
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma_frames) ** 2)
    k /= k.sum()
    pad = np.pad(P, ((r, r), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(P.shape[1])], axis=1)


def subject_frames(path: np.ndarray, fps: float, fallback_heading: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed path and unit forward vectors (T, 3) from the subject's motion."""
    S = _gauss_smooth(path, 0.45 * fps)
    V = np.gradient(S, axis=0) * fps
    V[:, 2] = 0.0
    spd = np.linalg.norm(V, axis=1)
    fwd = np.zeros_like(V)
    fb = np.array([math.cos(fallback_heading), math.sin(fallback_heading), 0.0])
    last = fb
    for i in range(len(V)):
        if spd[i] > 0.12:
            last = V[i] / spd[i]
        fwd[i] = last
    # Backfill the start (before the first confident direction).
    first = next((i for i in range(len(V)) if spd[i] > 0.12), None)
    if first:
        fwd[:first] = fwd[first]
    fwd = _gauss_smooth(fwd, 0.8 * fps)
    fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
    return S, fwd


def plan_shots(n_frames: int, fps: float, sections: list[dict], beat_times: np.ndarray, seed: int = 0,
               min_len: float = 2.2, max_len: float = 7.5) -> list[Shot]:
    rng = Rng(stable_hash64("camera", seed))
    # Candidate cut times: section starts + downbeats every 2/4 bars.
    bars = beat_times[::4] if len(beat_times) else np.arange(0, n_frames / fps, 2.0)
    starts = sorted({float(s["start"]) for s in sections})
    cuts = [0.0]
    t = 0.0
    total = n_frames / fps
    while t < total:
        sec = next((s for s in reversed(sections) if s["start"] <= t), sections[0] if sections else {"label": "groove"})
        energetic = sec.get("label") in ("drop", "peak", "return")
        target_len = rng.uniform(min_len, max_len * (0.65 if energetic else 1.0))
        nxt = t + target_len
        # Prefer a section boundary inside the window, else the nearest downbeat.
        boundary = next((s for s in starts if t + min_len * 0.7 < s <= nxt + 1.0), None)
        if boundary is not None:
            nxt = boundary
        elif len(bars):
            nxt = float(bars[np.argmin(np.abs(bars - nxt))])
            if nxt <= t + min_len * 0.7:
                nxt = t + target_len
        if nxt >= total - 1.0:
            break
        cuts.append(nxt)
        t = nxt
    shots: list[Shot] = []
    prev_kind, side = None, rng.choice([-1.0, 1.0])
    recent: list[str] = []
    for i, c in enumerate(cuts):
        c1 = cuts[i + 1] if i + 1 < len(cuts) else total
        sec = next((s for s in reversed(sections) if s["start"] <= c + 0.05), sections[0] if sections else None)
        label = sec["label"] if sec else "groove"
        energy = float(sec.get("energy", 0.5)) if sec else 0.5
        table = SECTION_SHOTS.get(label, SECTION_SHOTS["groove"])
        if i == 0:
            table = ESTABLISHING
        names, w = [], []
        for k, base in table.items():
            st = SHOTS[k]
            fit = 1.0 if st.energy[0] <= energy <= st.energy[1] else 0.4
            pen = 0.0 if k == prev_kind else (0.55 if k in recent[-3:] else 1.0)
            # Alternate close / wide.
            if prev_kind and ("close" in prev_kind) and ("close" in k):
                pen *= 0.3
            names.append(k)
            w.append(base * fit * pen)
        kind = names[rng.weighted_index(w)]
        if kind in ("front_dolly", "rear_follow") and rng.chance(0.3):
            side = -side          # crossing the line is allowed through front/rear shots
        shots.append(Shot(int(round(c * fps)), int(round(c1 * fps)), kind, side, rng.randint(0, 10 ** 6)))
        recent.append(kind)
        prev_kind = kind
    return shots


def compose(path: np.ndarray, head: np.ndarray, feet: np.ndarray, fps: float, sections: list[dict],
            beat_times: np.ndarray, heading0: float, height: float = 1.7, seed: int = 0,
            fixed_shot: str | None = None) -> CameraTrack:
    T = len(path)
    S, F = subject_frames(path, fps, heading0)
    Hs = _gauss_smooth(head, 0.25 * fps)
    Fs = _gauss_smooth(feet, 0.15 * fps)
    L = np.cross(UP[None, :], F)
    sc = height / 1.7
    shots = plan_shots(T, fps, sections, beat_times, seed) if not fixed_shot else \
        [Shot(0, T, fixed_shot, 1.0, seed)]
    pos = np.zeros((T, 3))
    tgt = np.zeros((T, 3))
    lens = np.zeros(T)
    fstop = np.zeros(T)
    fpt = np.zeros((T, 3))
    for sh in shots:
        st = SHOTS[sh.kind]
        n1 = Noise1D(sh.seed, 0.07, 2)
        n2 = Noise1D(sh.seed + 1, 0.05, 2)
        a, b = max(0, sh.start), min(T, sh.end)
        if b <= a:
            continue
        # Anchor the shot's frame at its first frame for side / wide shots: the camera then
        # travels with the subject but does not swing with every path wiggle.
        for k in range(a, b):
            u = (k - a) / max(b - a - 1, 1)
            tt = k / fps
            ang = st.drift * n1.sample(tt)
            f, l = F[k], L[k]
            fr = math.cos(ang) * f + math.sin(ang) * l
            lr = math.cos(ang) * l - math.sin(ang) * f
            push = 1.0 - st.push * (u * u * (3 - 2 * u))
            off = (fr * st.fwd + lr * st.left * sh.side) * push * sc + UP * (st.up * sc + 0.06 * n2.sample(tt))
            if sh.kind == "wide_orbit":
                oa = 0.12 * tt + sh.seed % 7
                r = math.hypot(st.fwd, st.left) * push
                off = (math.cos(oa) * f + math.sin(oa) * l) * r * sc + UP * st.up * sc
            pos[k] = S[k] + off
            if st.target == "head":
                tgt[k] = Hs[k]
                fpt[k] = Hs[k]
            elif st.target == "feet":
                tgt[k] = Fs[k] * np.array([1.0, 1.0, 0.0]) + UP * (st.target_up + 0.08) * sc
                fpt[k] = Fs[k]
            else:
                tgt[k] = S[k] * np.array([1.0, 1.0, 0.0]) + UP * st.target_up * sc
                fpt[k] = 0.6 * Hs[k] + 0.4 * tgt[k]
            lens[k] = st.lens
            fstop[k] = st.fstop
    focus = np.linalg.norm(pos - fpt, axis=1)
    return CameraTrack(fps, pos, tgt, lens, focus, fstop, shots)
