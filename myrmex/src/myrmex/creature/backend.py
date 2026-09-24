"""CreatureBackend: plugs the creature engine into the live session (the character backend switch).

The session keeps its input layer (InputHub, ClockHub); this backend turns the generic
music features into CreatureControlInput, applies parameters / events and records takes.
"""
from __future__ import annotations

import os
import time

import numpy as np

from ..music.features import FeatureExtractor
from ..music.timeline import MusicTimeline
from .config import PARAMS, CreatureConfig
from .control import CreatureControlInput
from .engine import EVENTS, CreatureEngine

# Notes on the control track / channel: choreography events for the creature.
CONTROL_NOTES = {60: "MORPHOLOGY_SHIFT", 62: "APPENDAGE_BURST", 65: "COLLAPSE", 67: "RECONSTRUCTION",
                 69: "MASS_REBALANCE"}


class CreatureBackend:
    height = 1.4

    def __init__(self, cfg: CreatureConfig | None = None, record: bool = False):
        self.engine = CreatureEngine(cfg)
        self.fx = FeatureExtractor(MusicTimeline(source="live"))
        self.debug = False
        self.frames: list | None = [] if record else None
        self.state = self.engine.state()

    def controls(self, c: dict) -> None:
        for p in PARAMS:
            self.engine.set_parameter(p, c.get(p, c.get("creature_" + p)))
        self.debug = c.get("creature_debug", 0.0) > 0.5

    def trigger(self, name: str) -> bool:
        ev = name.split(":", 1)[1] if ":" in name else name
        return self.engine.trigger_event(ev.upper()) if ev.upper() in EVENTS else False

    def control_note(self, pitch: float) -> None:
        ev = CONTROL_NOTES.get(int(round(pitch)))
        if ev:
            self.engine.trigger_event(ev)

    def tick(self, t: float, dt: float, notes, st):
        fr = self.fx.update(t, notes, beat=st.beat, tempo=st.bpm, beats_per_bar=st.beats_per_bar)
        self.engine.set_input(CreatureControlInput.from_frame(fr, notes, st.playing))
        self.state = self.engine.update(dt)
        return self.state

    def record(self, fps_due: bool) -> None:
        if self.frames is not None and fps_due:
            s = self.state
            self.frames.append((s.t, s.pos.astype(np.float32), s.radius.astype(np.float32),
                                s.stretch.astype(np.float32), s.surface, s.glow))

    def save_take(self, folder: str) -> str | None:
        if not self.frames:
            return None
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("creature_take_%Y%m%d_%H%M%S.npz"))
        np.savez_compressed(path, t=np.array([f[0] for f in self.frames]), pos=np.stack([f[1] for f in self.frames]),
                            radius=np.stack([f[2] for f in self.frames]), stretch=np.stack([f[3] for f in self.frames]),
                            surface=np.array([f[4] for f in self.frames]), glow=np.array([f[5] for f in self.frames]),
                            kind=self.state.kind, anchor=self.state.anchor, seed=self.engine.cfg.seed)
        return path
