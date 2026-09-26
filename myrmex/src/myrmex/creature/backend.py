"""CreatureBackend: plugs a creature engine into the live session (the character backend switch).

Two organisms share it: ``nanomaterial`` (v1, ground-bound Black Nanomaterial Creature) and
``polyalloy`` (v2, airborne Mimetic Polyalloy).  The session keeps its input layer (InputHub,
ClockHub); this backend turns the generic music features into CreatureControlInput, applies
parameters / events and records takes (everything needed to render them later in Blender:
the body, the live camera, the song position for the audio).
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
from .colony import ColonyConfig, ColonyEngine
from .hive import HiveConfig, HiveEngine
from .bionic import VARIANTS as BIONIC
from .cyber import VARIANTS as CYBER
from .mimetic import VARIANTS as MIMETIC
from .osseous import VARIANTS as OSSEOUS
from .polyalloy import PolyalloyConfig, PolyalloyEngine

# Notes on the control track / channel: choreography events for the creature.
CONTROL_NOTES = {60: "MORPHOLOGY_SHIFT", 62: "APPENDAGE_BURST", 65: "COLLAPSE", 67: "RECONSTRUCTION",
                 69: "MASS_REBALANCE",
                 # Mimetic Polyalloy: physical events (kick = impulse / obstacle / pressure / turbulence)
                 71: "OBSTACLE", 72: "IMPULSE", 74: "PRESSURE", 76: "TURBULENCE",
                 # Polyalloy Colony: flock, hardening wave, prey, landing
                 77: "SPLIT", 79: "MERGE", 81: "WAVE", 83: "HUNT", 84: "PERCH",
                 # Polyalloy Hive: living architecture
                 86: "BUILD", 88: "RECALL",
                 # Osseous line (v5-v7): lunge, ossify, quill volley
                 89: "STRIKE", 91: "OSSIFY", 93: "QUILLS",
                 # Cyber Hive (v8): light scan, digital glitch
                 95: "SCAN", 96: "GLITCH",
                 # Mimetic line (v9-v13): surge, dash, scatter / gather, slash, reconfigure, pounce
                 98: "SURGE", 100: "DASH", 101: "SCATTER", 103: "GATHER", 105: "SLASH", 107: "RECONFIGURE",
                 108: "POUNCE",
                 # Bionic line (v14-v18): tensegrity, fold, vessels, ferrofluid, bone truss
                 # (77 SPLIT and 98 SURGE also throw off droplets / surge the magnet on Ferro)
                 110: "LASH", 112: "COIL", 113: "UNFURL", 115: "CLAP", 117: "FURL", 118: "BLOOM", 119: "SPROUT",
                 120: "SHED", 121: "PULSE", 122: "CALM", 123: "BLOW", 124: "ANNEAL", 125: "OVERLOAD"}


CAM_KEYS = ("cam_px", "cam_py", "cam_pz", "cam_tx", "cam_ty", "cam_tz", "cam_lens", "cam_focus", "cam_fstop", "cam_shot")


class CreatureBackend:
    height = 1.4

    def __init__(self, cfg: CreatureConfig | PolyalloyConfig | ColonyConfig | HiveConfig | None = None, record: bool = False,
                 variant: str = "nanomaterial"):
        self.variant = variant
        kind = {"polyalloy": PolyalloyEngine, "colony": ColonyEngine, "hive": HiveEngine,
                **{k: v[0] for k, v in OSSEOUS.items()}, **{k: v[0] for k, v in CYBER.items()},
                **{k: v[0] for k, v in MIMETIC.items()}, **{k: v[0] for k, v in BIONIC.items()}}.get(variant,
                                                                                               CreatureEngine)
        self.engine = kind(cfg)
        self.events = getattr(kind, "EVENTS", EVENTS)
        self.fx = FeatureExtractor(MusicTimeline(source="live"))
        self.debug = False
        self.frames: list | None = [] if record else None
        self.state = self.engine.state()
        self._ev_seen = -1.0
        self.fresh: list = []                      # events of the last tick
        self.ev_log: list = []

    def controls(self, c: dict) -> None:
        for p in PARAMS:
            self.engine.set_parameter(p, c.get(p, c.get("creature_" + p)))
        self.debug = c.get("creature_debug", 0.0) > 0.5

    def trigger(self, name: str) -> bool:
        ev = name.split(":", 1)[1] if ":" in name else name
        return self.engine.trigger_event(ev.upper()) if ev.upper() in self.events else False

    def control_note(self, pitch: float) -> None:
        ev = CONTROL_NOTES.get(int(round(pitch)))
        if ev and ev in self.events:
            self.engine.trigger_event(ev)

    def tick(self, t: float, dt: float, notes, st):
        fr = self.fx.update(t, notes, beat=st.beat, tempo=st.bpm, beats_per_bar=st.beats_per_bar)
        self.engine.set_input(CreatureControlInput.from_frame(fr, notes, st.playing))
        self.state = self.engine.update(dt)
        self.fresh = self.new_events()
        if self.frames is not None:
            self.ev_log.extend(self.fresh)
        return self.state

    def new_events(self) -> list:
        """Engine events since the last call (the aerial camera reacts to impacts)."""
        out = [e for e in self.engine.events if e[0] > self._ev_seen]
        if out:
            self._ev_seen = out[-1][0]
        return out

    def record(self, fps_due: bool, st=None, cam=None, td=None) -> None:
        """One take frame: body, material, music position (audio alignment) and the live camera."""
        if self.frames is None or not fps_due:
            return
        s = self.state
        f = {"t": s.t, "pos": s.pos.astype(np.float32), "radius": s.radius.astype(np.float32),
             "stretch": s.stretch.astype(np.float16), "surface": s.surface, "glow": s.glow, "arousal": s.arousal,
             "com": np.asarray(s.com, np.float32), "heading": s.heading, "behavior": s.behavior,
             "morphology": s.morphology}
        if st is not None:
            f.update(beat=st.beat, bpm=st.bpm, playing=float(st.playing),
                     song_beat=st.song_beat if st.song_beat is not None else np.nan)
        if cam is not None:
            for i, a in enumerate("xyz"):
                f["cam_p" + a], f["cam_t" + a] = float(cam.position[i]), float(cam.target[i])
            f.update(cam_lens=cam.lens, cam_focus=cam.focus, cam_fstop=cam.fstop, cam_shot=float(cam.shot_id),
                     cam_kind=cam.kind)
        if getattr(s, "links", None) is not None:
            f.update(dispersion=s.dispersion.astype(np.float16), links=s.links.astype(np.float16),
                     obstacles=s.obstacles.astype(np.float32), material=s.material, fragments=s.fragments)
        if getattr(s, "plate", None) is not None:
            f.update(plate=s.plate.astype(np.float16), nrm=s.nrm.astype(np.float16), owner=s.owner.astype(np.uint8),
                     lure=s.lure.astype(np.float32), bodies=s.bodies)
        if getattr(s, "particles", None) is not None:       # millimetres from the centre of mass
            f.update(particles=np.clip(np.round((s.particles - s.com) * 1000.0), -32767, 32767).astype(np.int16),
                     rd=np.clip(s.rd * 255.0, 0, 255).astype(np.uint8), structures=s.structures,
                     memories=s.memories)
        if getattr(s, "light", None) is not None:          # Cyber Hive: light lines per node, the scan front
            f.update(light=np.clip(s.light * 255.0, 0, 255).astype(np.uint8), scan=float(s.scan))
        if td is not None:                                 # what TouchDesigner saw (realtime/touch.py CHANNELS)
            f["td"] = np.asarray(td, np.float32)
        if getattr(s, "members", None) is not None:        # Bionic line: the structure + the creature's floats
            f.update(members=np.asarray(s.members, np.float32).astype(np.float16),
                     extra=np.asarray(s.extra, np.float32), obstacles=np.asarray(s.obstacles, np.float32),
                     bkind=float(s.bkind))
        self.frames.append(f)

    def save_take(self, folder: str, fps: float = 30.0) -> str | None:
        if not self.frames:
            return None
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime(f"{self.variant}_take_%Y%m%d_%H%M%S.npz"))
        fr = self.frames
        keys = list(dict.fromkeys(k for f in fr for k in f))
        data = {}
        for k in keys:
            v0 = next(f[k] for f in fr if k in f)          # e.g. the camera starts a few frames late
            if isinstance(v0, str):
                data[k] = np.array([f.get(k, "") for f in fr])
            elif isinstance(v0, np.ndarray):
                data[k] = np.stack([f[k] if k in f else np.zeros_like(v0) for f in fr])
            else:
                data[k] = np.array([f.get(k, np.nan) for f in fr], float)
        ev = self.ev_log
        if "td" in data:
            from ..realtime.touch import CHANNELS
            data["td_names"] = np.array(CHANNELS)
        np.savez_compressed(path, **data, kind=self.state.kind, anchor=self.state.anchor, seed=self.engine.cfg.seed,
                            variant=self.variant, fps=float(fps), format="myrmex-creature-take-2",
                            ev_t=np.array([e[0] for e in ev], float), ev_name=np.array([str(e[1]) for e in ev]))
        return path
