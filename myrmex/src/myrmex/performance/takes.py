"""Recorded live takes: the live camera and the audio alignment, for rendering what was performed."""
from __future__ import annotations

import numpy as np

from ..camera.cinematographer import CameraTrack, Shot
from .performance import Performance


def recorded_camera_track(perf: Performance) -> CameraTrack | None:
    """The live camera exactly as it was during a take (``None`` if the take has no camera)."""
    ch = perf.channels
    if "cam_px" not in ch:
        return None
    P = np.stack([ch["cam_px"], ch["cam_py"], ch["cam_pz"]], axis=1).astype(float)
    T = np.stack([ch["cam_tx"], ch["cam_ty"], ch["cam_tz"]], axis=1).astype(float)
    ids = np.asarray(ch["cam_shot"]).astype(int)
    kinds = perf.labels.get("camera") or ["free"] * len(ids)
    shots = []
    start = 0
    for k in range(1, len(ids) + 1):
        if k == len(ids) or ids[k] != ids[start]:
            shots.append(Shot(start, k, kinds[start] or "free", 1.0, int(ids[start])))
            start = k
    return CameraTrack(perf.fps, P, T, np.asarray(ch["cam_lens"], float), np.asarray(ch["cam_focus"], float),
                       np.asarray(ch["cam_fstop"], float), shots)


def take_audio_offset(perf: Performance) -> float | None:
    """Audio time (s) at frame 0 of a take, from the first recorded song position (constant tempo).

    Negative when the song started after the recording did.
    """
    ch = perf.channels
    if "song_beat" not in ch or "playing" not in ch:
        return None
    return audio_offset(ch["song_beat"], ch["bpm"], ch["playing"], perf.fps)


def audio_offset(song_beat, bpm, playing, fps: float) -> float | None:
    sb, bpm, playing = (np.asarray(a, float) for a in (song_beat, bpm, playing))
    ok = np.nonzero(np.isfinite(sb) & (playing > 0.5))[0]
    if not len(ok):
        return None
    i = int(ok[0])
    return float(sb[i] * 60.0 / bpm[i] - i / fps)


__all__ = ["recorded_camera_track", "take_audio_offset", "audio_offset"]
