"""Creature takes (``*_take_*.npz`` written by the app): load, resample, camera, audio alignment.

Pure numpy - used by the Blender importer and by tests.
"""
from __future__ import annotations

import numpy as np

from ..camera.cinematographer import CameraTrack, Shot
from ..performance.takes import audio_offset

CAM = ("cam_px", "cam_py", "cam_pz", "cam_tx", "cam_ty", "cam_tz", "cam_lens", "cam_focus", "cam_fstop", "cam_shot")


def is_creature_take(path: str) -> bool:
    with np.load(path, allow_pickle=False) as z:
        return "pos" in z.files and "radius" in z.files


class CreatureTake:
    def __init__(self, path: str):
        self.path = path
        with np.load(path, allow_pickle=False) as z:
            self.d = {k: z[k] for k in z.files}
        d = self.d
        self.variant = str(d["variant"]) if "variant" in d else "nanomaterial"
        self.fps = float(d["fps"]) if "fps" in d else 30.0
        self.n = len(d["t"])
        self.nodes = d["pos"].shape[1]
        # Frames recorded before the camera existed: hold its first state.
        if "cam_px" in d:
            ok = np.isfinite(d["cam_px"])
            if ok.any() and not ok.all():
                first = int(np.argmax(ok))
                for k in CAM:
                    d[k] = d[k].copy()
                    d[k][:first] = d[k][first]
                if "cam_kind" in d:
                    d["cam_kind"] = d["cam_kind"].copy()
                    d["cam_kind"][:first] = d["cam_kind"][first]

    @property
    def duration(self) -> float:
        return self.n / self.fps

    def sample_index(self, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Frame-rate conversion: for each output frame, the two source frames and the blend weight."""
        m = max(1, int(round(self.duration * fps)))
        x = np.arange(m) * self.fps / fps
        i0 = np.clip(np.floor(x).astype(int), 0, self.n - 1)
        i1 = np.clip(i0 + 1, 0, self.n - 1)
        return i0, i1, (x - np.floor(x))

    def resampled(self, key: str, fps: float) -> np.ndarray:
        a = self.d[key]
        i0, i1, w = self.sample_index(fps)
        if a.dtype.kind in "US" or key in ("links", "anchor", "kind", "fragments"):
            return a[np.where(w < 0.5, i0, i1)]
        a = a.astype(float)
        w = w.reshape((-1,) + (1,) * (a.ndim - 1))
        return a[i0] * (1 - w) + a[i1] * w

    def camera_track(self, fps: float | None = None) -> CameraTrack | None:
        d = self.d
        if "cam_px" not in d or not np.isfinite(d["cam_px"]).any():
            return None
        fps = fps or self.fps
        R = {k: self.resampled(k, fps) for k in CAM}
        i0, i1, w = self.sample_index(fps)
        ids = np.nan_to_num(d["cam_shot"])[i0].astype(int)
        kinds = d["cam_kind"][i0] if "cam_kind" in d else np.array(["free"] * len(ids))
        shots, start = [], 0
        for k in range(1, len(ids) + 1):
            if k == len(ids) or ids[k] != ids[start]:
                shots.append(Shot(start, k, str(kinds[start]) or "free", 1.0, int(ids[start])))
                start = k
        P = np.stack([R["cam_px"], R["cam_py"], R["cam_pz"]], 1)
        T = np.stack([R["cam_tx"], R["cam_ty"], R["cam_tz"]], 1)
        return CameraTrack(fps, P, T, R["cam_lens"], R["cam_focus"], R["cam_fstop"], shots)

    def particles(self, fps: float) -> np.ndarray | None:
        """Hive nanomachines (frames, P, 3) in metres."""
        if "particles" not in self.d:
            return None
        return self.resampled("particles", fps) / 1000.0 + self.resampled("com", fps)[:, None, :]

    def audio_offset(self) -> float | None:
        """Song time (s) at the first frame of the take (negative: the song started later)."""
        d = self.d
        if "song_beat" not in d:
            return None
        return audio_offset(d["song_beat"], d["bpm"], d["playing"], self.fps)


__all__ = ["CreatureTake", "is_creature_take"]
