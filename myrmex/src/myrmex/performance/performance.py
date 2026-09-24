"""A baked performance: per-frame rigid deltas for every bone + debug channels.

The interchange format between the engine and any DCC is deliberately
simple: for frame ``k`` and bone ``b`` the world pose is
``deltas[k, b] @ rest_matrix[b]`` (rest-world delta, see
:mod:`myrmex.motion.skeleton`).  Stored as ``.npz`` (arrays) + ``.json``
(metadata, events, labels) so it can be inspected and re-baked without
re-simulating.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Performance:
    fps: float
    bone_names: list[str]
    deltas: np.ndarray                                   # (T, B, 4, 4) float32
    channels: dict[str, np.ndarray] = field(default_factory=dict)   # (T,) floats
    labels: dict[str, list[str]] = field(default_factory=dict)      # (T,) strings (behaviour, gait ...)
    events: list[dict] = field(default_factory=list)                 # {"time", "type", ...}
    meta: dict = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return int(self.deltas.shape[0])

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    def save(self, path: str) -> None:
        base, _ = os.path.splitext(path)
        arrays = {"deltas": self.deltas.astype(np.float32)}
        for k, v in self.channels.items():
            arrays[f"ch__{k}"] = np.asarray(v, dtype=np.float32)
        np.savez_compressed(base + ".npz", **arrays)
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump({"format": "myrmex.performance/1", "fps": self.fps, "bone_names": self.bone_names,
                       "labels": self.labels, "events": self.events, "meta": self.meta}, fh, indent=1,
                      default=_json_default)

    @classmethod
    def load(cls, path: str) -> "Performance":
        base, _ = os.path.splitext(path)
        data = np.load(base + ".npz")
        with open(base + ".json", encoding="utf-8") as fh:
            meta = json.load(fh)
        channels = {k[4:]: data[k] for k in data.files if k.startswith("ch__")}
        return cls(float(meta["fps"]), list(meta["bone_names"]), data["deltas"], channels,
                   meta.get("labels", {}), meta.get("events", []), meta.get("meta", {}))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


class Recorder:
    """Samples a pose stream at a fixed frame rate while the simulation runs faster."""

    def __init__(self, bone_names: list[str], fps: float):
        self.bone_names = list(bone_names)
        self.fps = float(fps)
        self._deltas: list[np.ndarray] = []
        self._channels: dict[str, list[float]] = {}
        self._labels: dict[str, list[str]] = {}
        self.events: list[dict] = []

    def add(self, deltas: np.ndarray, channels: dict[str, float] | None = None,
            labels: dict[str, str] | None = None) -> None:
        self._deltas.append(np.array(deltas, dtype=np.float32, copy=True))
        n = len(self._deltas)
        for k, v in (channels or {}).items():
            lst = self._channels.setdefault(k, [])
            lst.extend([lst[-1] if lst else 0.0] * (n - 1 - len(lst)))
            lst.append(float(v))
        for k, v in (labels or {}).items():
            lst = self._labels.setdefault(k, [])
            lst.extend([lst[-1] if lst else ""] * (n - 1 - len(lst)))
            lst.append(str(v))

    def event(self, time: float, kind: str, **data) -> None:
        self.events.append({"time": float(time), "type": kind, **data})

    def build(self, meta: dict | None = None) -> Performance:
        T = len(self._deltas)
        ch = {}
        for k, lst in self._channels.items():
            lst = lst + [lst[-1] if lst else 0.0] * (T - len(lst))
            ch[k] = np.asarray(lst, dtype=np.float32)
        lb = {k: v + [v[-1] if v else ""] * (T - len(v)) for k, v in self._labels.items()}
        return Performance(self.fps, self.bone_names, np.stack(self._deltas) if T else
                           np.zeros((0, len(self.bone_names), 4, 4), np.float32), ch, lb, self.events, meta or {})
