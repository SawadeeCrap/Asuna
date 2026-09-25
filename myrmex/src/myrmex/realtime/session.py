"""The live performance loop: Ableton / VCV in, a moving character out - in real time.

    inputs (OSC 9100 / MIDI / Link / audio) --> InputHub + ClockHub
        --> PerformanceCore.tick()   (the *same* step the offline generator uses)
        --> LiveCinematographer
        --> PoseSink (UDP 9101: Blender live link, any renderer)  [+ recorded take]

The loop runs in its own thread at ``rate`` Hz.  Footfalls are scheduled on the
live beat grid, and the grid is evaluated ``latency`` seconds ahead, so a heel
strike *appears on screen* on the beat despite render / display latency.
When the transport stops (or the music does), the character does not freeze:
it holds a pose and stays alive (breathing, weight shifts, glances) until the
music comes back, then walks off on the next beat.
"""
from __future__ import annotations

import math
import os
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..behavior.engine import EngineConfig
from ..camera.live import LiveCinematographer
from ..motion.bodyplan import BipedPlan
from ..music.timeline import MusicTimeline
from ..performance.core import PerformanceCore
from ..performance.performance import Recorder
from ..rig.rigdesc import RigDescription
from .clock import ClockHub, ClockState
from .inputs import InputConfig, InputHub
from .protocol import (FLAG_HOLD, FLAG_PLAYING, FLAG_RECORDING, PoseFrame, encode_names, encode_pose, fragment,
                       rig_id)


@dataclass
class LiveConfig:
    rig: str | None = None                  # rig description JSON (written by the Blender auto-rig step)
    seed: int = 0
    rate: float = 120.0                     # simulation rate (Hz)
    out: list[str] = field(default_factory=lambda: ["127.0.0.1:9101"])   # pose stream targets
    out_rate: float = 60.0                  # pose packets per second
    clock: str = "auto"                     # auto | osc | link | midi | onsets | internal
    bpm: float = 120.0                      # internal clock tempo
    link: bool = True                       # join Ableton Link sessions
    latency: float = 0.045                  # visual latency compensation (s)
    style: str = "catwalk"
    engine: dict = field(default_factory=dict)
    camera: bool = True
    auto_hold: bool = True                  # stand and pose when the music stops
    record: str | None = None               # directory for recorded takes
    backend: str = "humanoid"               # humanoid | creature (Black Nanomaterial) | polyalloy (Mimetic Polyalloy)
    creature: dict = field(default_factory=dict)
    glove: dict = field(default_factory=dict)       # Hand Glove link (preset, intensity, profile, ...)
    record_fps: float = 30.0
    inputs: InputConfig = field(default_factory=InputConfig)


class PoseSink:
    def __init__(self, targets: list[str], names: list[str]):
        self.addrs = []
        for tgt in targets:
            host, _, port = tgt.rpartition(":")
            self.addrs.append((host or "127.0.0.1", int(port)))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.names_packet = encode_names(names)
        self.last_names = -1e9
        self.sent = 0
        self.msg_id = 0
        self.last_error = ""
        self.errors = 0

    def send(self, fr: PoseFrame, now: float) -> None:
        data = encode_pose(fr)
        if now - self.last_names > 1.0:
            for a in self.addrs:
                self._tx(self.names_packet, a)
            self.last_names = now
        for a in self.addrs:
            self._tx(data, a)
        self.sent += 1

    def send_raw(self, data: bytes) -> None:
        self.msg_id += 1
        parts = fragment(data, self.msg_id)
        for a in self.addrs:
            for part in parts:
                self._tx(part, a)
        self.sent += 1

    def _tx(self, data: bytes, addr) -> None:
        try:
            self.sock.sendto(data, addr)
        except ConnectionRefusedError:
            pass                          # receiver not up yet - keep going
        except OSError as e:              # anything else (e.g. a datagram too large) is reported
            self.errors += 1
            self.last_error = f"{type(e).__name__}: {e}"

    def close(self) -> None:
        self.sock.close()


CREATURES = ("creature", "polyalloy", "colony", "hive", "osseous", "osseous_colony", "osseous_hive", "cyber_hive")
FLYING = CREATURES[1:]


class LiveSession:
    def __init__(self, cfg: LiveConfig, plan: BipedPlan | None = None, *, start_inputs: bool = True,
                 sink: PoseSink | None = None, now: float | None = None):
        self.cfg = cfg
        self.creature = None
        if cfg.backend in CREATURES:
            from ..creature.backend import CreatureBackend
            from ..creature.config import CreatureConfig
            if cfg.backend == "polyalloy":
                from ..creature.polyalloy import PolyalloyConfig
                self.creature = CreatureBackend(PolyalloyConfig(seed=cfg.seed), record=bool(cfg.record),
                                                variant="polyalloy")
            elif cfg.backend == "colony":
                from ..creature.colony import ColonyConfig
                self.creature = CreatureBackend(ColonyConfig(seed=cfg.seed), record=bool(cfg.record), variant="colony")
            elif cfg.backend == "hive":
                from ..creature.hive import HiveConfig
                self.creature = CreatureBackend(HiveConfig(seed=cfg.seed), record=bool(cfg.record), variant="hive")
            elif cfg.backend.startswith("osseous"):             # the bony line (v5-v7)
                from ..creature.osseous import VARIANTS
                self.creature = CreatureBackend(VARIANTS[cfg.backend][1](seed=cfg.seed), record=bool(cfg.record),
                                                variant=cfg.backend)
            elif cfg.backend == "cyber_hive":                    # v8: white nanomaterial, light lines
                from ..creature.cyber import VARIANTS as CYBER
                self.creature = CreatureBackend(CYBER[cfg.backend][1](seed=cfg.seed), record=bool(cfg.record),
                                                variant=cfg.backend)
            else:
                extra = {k: v for k, v in cfg.creature.items() if k in ("variation", "stage_radius")}
                self.creature = CreatureBackend(CreatureConfig(seed=cfg.seed, **extra), record=bool(cfg.record))
            self.plan, self.names, self.rig_id = None, [], 0
            self.core = self.motor = self.engine = None
            height = CreatureBackend.height
        else:
            if plan is None:
                if not cfg.rig:
                    raise ValueError("LiveConfig.rig (rig description JSON) is required")
                plan = BipedPlan.from_rig(RigDescription.load(cfg.rig))
            self.plan = plan
            self.names = list(plan.sk.names)
            self.rig_id = rig_id(self.names)
            ecfg = dict(cfg.engine)
            ecfg.setdefault("mode", "runway")
            ecfg.setdefault("style", cfg.style)
            tl = MusicTimeline(source="live")
            self.core = PerformanceCore(plan, tl, cfg.seed, EngineConfig.from_dict(ecfg))
            self.motor, self.engine = self.core.motor, self.core.engine
            height = plan.height
        now = time.perf_counter() if now is None else now
        self.t0 = now
        self.clock = ClockHub(cfg.clock, cfg.bpm, link=cfg.link, now=now)
        self.inputs = InputHub(cfg.inputs, self.clock, start=start_inputs)
        from .glove import GloveLink
        self.glove = GloveLink(cfg.glove)
        if cfg.glove.get("profile"):
            self.inputs.glove.profile = dict(cfg.glove["profile"])
        if cfg.camera and cfg.backend in FLYING:
            from ..camera.aerial import AerialCinematographer
            self.camera = AerialCinematographer(self.creature.engine.cfg.size, cfg.seed)
        else:
            self.camera = LiveCinematographer(height, cfg.seed) if cfg.camera else None
        self.sink = sink if sink is not None else (PoseSink(cfg.out, self.names) if cfg.out else None)
        self.dt = 1.0 / cfg.rate
        self.t = 0.0
        self.seq = 0
        self.next_send = 0.0
        self.recorder = Recorder(self.names, cfg.record_fps) if cfg.record and self.creature is None else None
        self.next_rec = 0.0
        self.style_idx = None
        self._hold_notes: list = []
        self.last_state: ClockState | None = None
        self.last_frame: PoseFrame | None = None
        self.hold = True
        if self.creature is None:
            self._pelvis = self.names.index(plan.pelvis) if plan.pelvis in self.names else 0
            self._head = self.names.index("head") if "head" in self.names else self._pelvis
            self._feet = [self.names.index(n) for n in self.names if n.startswith("foot_")]
            sk = plan.sk
            self._rest_heads = np.asarray(sk.heads, float)
            self._head_pt = self._rest_heads[self._head] + 0.45 * (np.asarray(sk.tails[self._head]) - self._rest_heads[self._head])
        else:
            self.hold = False
        self.stats = {"ticks": 0, "overruns": 0, "max_tick_ms": 0.0, "mean_tick_ms": 0.0}
        self._thread: threading.Thread | None = None
        self._running = False
        self.lock = threading.Lock()

    # ------------------------------------------------------------------ one step
    def step(self, now: float) -> PoseFrame | None:
        cfg = self.cfg
        dt = self.dt
        self.t += dt
        t = self.t
        notes = self.inputs.poll(now, t)                   # also feeds transport / clock events
        # Evaluate the grid where the frame will be *seen*: footfalls land on screen on the beat.
        st = self.clock.state(now + cfg.latency)
        notes.extend(self.inputs.score_due(st, t))
        if any(n.group == "control" for n in notes):
            self._control_notes([n for n in notes if n.group == "control"], t)
            notes = [n for n in notes if n.group != "control"]
        self.last_state = st
        self._apply_controls(now, t)
        if self.creature is not None:
            return self._step_creature(now, t, dt, notes, st)
        lvl = self.inputs.controls.get("audio_level")
        res = self.core.tick(t, dt, notes, beat=st.beat, tempo=st.bpm, beats_per_bar=st.beats_per_bar,
                             curves={"audio": lvl} if lvl is not None else None)
        fr = None
        if t + 1e-9 >= self.next_send:
            self.next_send += 1.0 / cfg.out_rate
            if self.next_send < t:
                self.next_send = t + 1.0 / cfg.out_rate
            fr = self._frame(t, dt * max(1, round(cfg.rate / cfg.out_rate)), st, res)
            if self.sink is not None:
                self.sink.send(fr, now)
            self.last_frame = fr
        if self.recorder is not None and t + 1e-9 >= self.next_rec:
            self.next_rec += 1.0 / cfg.record_fps
            ch = {"beat": st.beat, "bpm": st.bpm, "hold": float(self.hold), "playing": float(st.playing),
                  "song_beat": st.song_beat if st.song_beat is not None else float("nan"),
                  "wall": now - self.t0}
            cam = self.camera.state if self.camera is not None else None
            if cam is not None:
                for i, a in enumerate("xyz"):
                    ch["cam_p" + a] = float(cam.position[i])
                    ch["cam_t" + a] = float(cam.target[i])
                ch.update(cam_lens=cam.lens, cam_focus=cam.focus, cam_fstop=cam.fstop, cam_shot=float(cam.shot_id))
            self.recorder.add(res.pose.delta, ch, {"behavior": self.engine.behavior_name,
                                                   "camera": cam.kind if cam is not None else ""})
        return fr

    # Notes on the character's own track ("Myrmex" in Ableton, or a MIDI channel mapped to "control").
    CONTROL_NOTES = {60: "pose", 62: "flourish", 64: "camera", 65: "pose:look_back", 67: "flourish:hair_touch",
                     69: "flourish:hand_hip", 71: "flourish:shoulder_roll"}
    HOLD_NOTE = 72
    AERIAL_SUGGEST = {"SPLIT": "retreat", "ENVELOP": "approach", "PERCH": "track", "MERGE": "observe",
                      "BUILD": "orbit", "RECALL": "retreat"}

    def _control_notes(self, notes, t: float) -> None:
        if self.creature is not None:
            for n in notes:
                if int(round(n.pitch)) == 64:
                    self.inputs.triggers.append(("camera", t))
                else:
                    self.creature.control_note(n.pitch)
            return
        for n in notes:
            p = int(round(n.pitch))
            if p == self.HOLD_NOTE:
                self._hold_notes.append(n)
            elif p in self.CONTROL_NOTES:
                self.inputs.triggers.append((self.CONTROL_NOTES[p], t))

    def _camera_controls(self, c: dict) -> None:
        cam = self.camera
        if cam is None:
            return
        cam.mode = "manual" if c.get("cam_mode", 0.0) > 0.5 else "auto"
        lens = c.get("cam_lens", 0.0)
        cam.manual.update(distance=0.4 + 2.1 * c.get("cam_distance", 0.2857), height=-1.0 + 3.0 * c.get("cam_height", 1 / 3),
                          orbit=(c.get("cam_orbit", 0.5) - 0.5) * 2.0 * math.pi, lens=18.0 + 117.0 * lens if lens > 0.01 else 0.0,
                          smooth=0.05 + 1.5 * c.get("cam_smooth", 0.2))

    def _step_creature(self, now: float, t: float, dt: float, notes, st) -> object:
        from ..creature.protocol import FLAG_DEBUG, FLAG_PLAYING, encode_creature
        cfg = self.cfg
        # The hand first: the organism is moved by it in this very tick.
        view = None
        cs = self.camera.state if self.camera is not None else None
        if cs is not None:
            dv = np.asarray(cs.target, float) - np.asarray(cs.position, float)
            view = math.atan2(dv[1], dv[0]) if abs(dv[0]) + abs(dv[1]) > 1e-6 else None
        ctrl, gestures, cam_mod = self.glove.tick(self.inputs.glove, now, dt, self.creature.variant,
                                                  clock=(st.beat, st.bpm, st.playing), view_yaw=view)
        self.inputs.glove_owns = ctrl.active or self.glove.cfg["preset"] == "camera"
        self.creature.engine.set_glove(ctrl, self.glove.sculpt_shapes(self.creature.variant))
        for g, names in gestures:
            if names == ("camera",):
                if self.camera is not None:
                    self.camera.request_cut(None)
                continue
            for name in names:
                if self.creature.trigger(name):
                    self.glove.last_events.append((t, g, name))
                    self.glove.last_events = self.glove.last_events[-8:]
                    break
        if self.camera is not None:
            self.camera.extra = cam_mod
        s = self.creature.tick(t, dt, notes, st)
        aerial = self.creature.variant in FLYING
        if aerial and self.camera is not None:
            for _, name, _a in self.creature.fresh:
                if name in ("IMPULSE", "PRESSURE", "TURBULENCE"):
                    self.camera.impact(0.5, t)
                elif name in ("RESPONSE", "COLLAPSE", "HIT", "QUILLS"):
                    self.camera.impact(1.0, t, reframe=True)
                elif name in self.AERIAL_SUGGEST:
                    self.camera.suggest(self.AERIAL_SUGGEST[name], t)
        due = t + 1e-9 >= self.next_rec
        if due:
            self.next_rec += 1.0 / cfg.record_fps
        self.creature.record(due, st, self.camera.state if self.camera is not None else None)
        if t + 1e-9 < self.next_send:
            return None
        self.next_send = max(self.next_send + 1.0 / cfg.out_rate, t)
        cam = None
        if self.camera is not None and aerial:
            extent = float(np.sqrt(((s.pos - s.com) ** 2).sum(1).mean())) / max(self.creature.engine.cfg.size * 0.45, 1e-3)
            cam = self.camera.update(t, 1.0 / cfg.out_rate, s.com, s.heading, st.beat, st.beats_per_bar,
                                     s.behavior, extent)
        elif self.camera is not None:
            top = s.com + np.array([0.0, 0.0, 0.5])
            cam = self.camera.update(t, 1.0 / cfg.out_rate, s.com, top, s.com * np.array([1.0, 1.0, 0.0]), s.heading,
                                     st.beat, st.beats_per_bar, "groove", s.arousal, False)
        flags = (FLAG_PLAYING if st.playing else 0) | (FLAG_DEBUG if self.creature.debug else 0)
        self.seq += 1
        if self.sink is not None:
            self.sink.send_raw(encode_creature(s, self.seq, st.beat, st.bpm, flags, cam))
        self.last_frame = s
        return s

    def _apply_controls(self, now: float, t: float) -> None:
        c = self.inputs.controls
        self._camera_controls(c)
        if self.creature is not None:
            self.creature.controls(c)
            for name, _ in self.inputs.take_triggers():
                if name.startswith("camera") and self.camera is not None:
                    self.camera.request_cut(name.split(":", 1)[1] if ":" in name else None)
                else:
                    self.creature.trigger(name)
            return
        rw = self.engine.runway
        self._hold_notes = [n for n in self._hold_notes if n.time + n.duration > t]
        if rw is not None:
            for k in ("energy", "stride", "sway"):
                if k in c:
                    rw.live[k] = c[k]
                else:
                    rw.live.pop(k, None)
            recent = now - self.inputs.stats["last_note"] < 2.5
            st = self.last_state
            link = self.clock.sources.get("link")
            authoritative = st is not None and (st.source in ("osc", "midi") or
                                                (st.source == "link" and getattr(link, "_seen_playing", False)))
            # A real transport decides (Stop means stop, at once); without one, the music does.
            walking = (st.playing if authoritative else recent) if st is not None else recent
            manual = c.get("hold")
            if (manual is not None and manual > 0.5) or self._hold_notes:
                self.hold = True
            elif self.cfg.auto_hold:
                self.hold = not walking
            else:
                self.hold = False
            rw.live["hold"] = self.hold
        if "style" in c:
            styles = self.inputs.cfg.mapping.get("styles", ["catwalk"])
            idx = min(len(styles) - 1, int(c["style"] * len(styles)))
            if idx != self.style_idx:
                self.style_idx = idx
                self.motor.set_style(styles[idx])
        for name, _ in self.inputs.take_triggers():
            if name == "camera" and self.camera is not None:
                self.camera.request_cut()
            elif name.startswith("camera:") and self.camera is not None:
                self.camera.request_cut(name.split(":", 1)[1])
            elif rw is not None and (name == "pose" or name.startswith("pose:")):
                rw.live["pose"] = name.split(":", 1)[1] if ":" in name else "hip_out"
            elif rw is not None and (name == "flourish" or name.startswith("flourish:")):
                rw.live["flourish"] = name.split(":", 1)[1] if ":" in name else "hand_hip"

    def _frame(self, t: float, dt: float, st: ClockState, res) -> PoseFrame:
        D = res.pose.delta
        pel = D[self._pelvis, :3, :3] @ self._rest_heads[self._pelvis] + D[self._pelvis, :3, 3]
        head = D[self._head, :3, :3] @ self._head_pt + D[self._head, :3, 3]
        if self._feet:
            feet = np.mean([D[i, :3, :3] @ self._rest_heads[i] + D[i, :3, 3] for i in self._feet], axis=0)
        else:
            feet = pel * np.array([1.0, 1.0, 0.0])
        cam = None
        if self.camera is not None:
            phrase = any(e.type in ("phrase", "drop", "break") for e in res.events)
            heading = self.engine.runway.heading if self.engine.runway is not None else self.motor.psi
            cam = self.camera.update(t, dt, pel, head, feet, heading, st.beat, st.beats_per_bar,
                                     self.engine.section, self.engine.drives.arousal, phrase)
        flags = (FLAG_PLAYING if st.playing else 0) | (FLAG_HOLD if self.hold else 0) | \
                (FLAG_RECORDING if self.recorder is not None else 0)
        self.seq += 1
        return PoseFrame(self.seq, t, st.beat, st.bpm, self.rig_id, D.copy(), flags, cam, pel,
                         float(self.motor.psi), float(self.motor.speed_ref), float(self.engine.drives.arousal))

    # ------------------------------------------------------------------ thread
    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="myrmex-live", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import gc
        # Long-lived objects (rig, springs, histories) out of the collector's way; young
        # generations only: avoids 10-20 ms full collections in the middle of a tick.
        gc.collect()
        gc.freeze()
        gc.set_threshold(50000, 50, 1000)
        dt = self.dt
        nxt = time.perf_counter()
        n, acc = 0, 0.0
        while self._running:
            now = time.perf_counter()
            if now < nxt:
                time.sleep(min(nxt - now, 0.002))
                continue
            t0 = time.perf_counter()
            with self.lock:
                self.step(now)
            el = (time.perf_counter() - t0) * 1000.0
            n += 1
            acc += el
            self.stats["ticks"] = n
            self.stats["mean_tick_ms"] = acc / n
            self.stats["max_tick_ms"] = max(self.stats["max_tick_ms"], el)
            nxt += dt
            if now - nxt > 0.25:              # fell far behind (debugger, sleep): resync, don't spiral
                nxt = now
                self.stats["overruns"] += 1

    def stop(self) -> str | None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.inputs.close()
        self.clock.close()
        if self.sink is not None:
            self.sink.close()
        return self.save_take()

    def save_take(self) -> str | None:
        if self.creature is not None:
            return self.creature.save_take(self.cfg.record, self.cfg.record_fps) if self.cfg.record else None
        if self.recorder is None or not self.cfg.record or not self.recorder._deltas:
            return None
        os.makedirs(self.cfg.record, exist_ok=True)
        path = os.path.join(self.cfg.record, time.strftime("take_%Y%m%d_%H%M%S"))
        perf = self.recorder.build({"generator": "myrmex.live", "seed": self.cfg.seed, "height": self.plan.height,
                                    "heading0": float(self.motor.psi0), "body_plan": self.plan.rig.body_plan,
                                    "behavior_log": self.engine.log, "sections": []})
        perf.save(path + ".npz")
        return path + ".npz"

    def status(self) -> dict:
        st = self.last_state
        return {
            "t": round(self.t, 2), "clock": st.source if st else None, "bpm": round(st.bpm, 2) if st else None,
            "beat": round(st.beat, 2) if st else None, "playing": st.playing if st else None,
            "peers": st.peers if st else 0, "hold": self.hold,
            "section": self.engine.section if self.engine else self.creature.state.morphology,
            "behavior": self.engine.behavior_name if self.engine else self.creature.state.behavior,
            "backend": self.cfg.backend, "notes": self.inputs.stats["notes"],
            "glove": {"present": self.glove.state.present, "rate": round(self.inputs.glove.rate, 1),
                      "preset": self.glove.cfg["preset"]},
            "osc_packets": self.inputs.stats["osc_packets"], "sent": self.sink.sent if self.sink else 0,
            "camera": self.camera.kind if self.camera else None, "tick_ms": round(self.stats["mean_tick_ms"], 2),
            "max_tick_ms": round(self.stats["max_tick_ms"], 2), "overruns": self.stats["overruns"],
            "errors": self.inputs.stats["errors"] + ([self.clock.link_error] if self.clock.link_error else []) +
                      ([f"pose stream: {self.sink.last_error}"] if getattr(self.sink, "last_error", "") else []),
        }


__all__ = ["LiveConfig", "LiveSession", "PoseSink"]
