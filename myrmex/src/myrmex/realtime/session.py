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
from .protocol import (FLAG_HOLD, FLAG_PLAYING, FLAG_RECORDING, PoseFrame, encode_names, encode_pose,
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

    def send(self, fr: PoseFrame, now: float) -> None:
        data = encode_pose(fr)
        if now - self.last_names > 1.0:
            for a in self.addrs:
                self._tx(self.names_packet, a)
            self.last_names = now
        for a in self.addrs:
            self._tx(data, a)
        self.sent += 1

    def _tx(self, data: bytes, addr) -> None:
        try:
            self.sock.sendto(data, addr)
        except OSError:
            pass                          # receiver not up yet - keep going

    def close(self) -> None:
        self.sock.close()


class LiveSession:
    def __init__(self, cfg: LiveConfig, plan: BipedPlan | None = None, *, start_inputs: bool = True,
                 sink: PoseSink | None = None, now: float | None = None):
        self.cfg = cfg
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
        now = time.perf_counter() if now is None else now
        self.t0 = now
        self.clock = ClockHub(cfg.clock, cfg.bpm, link=cfg.link, now=now)
        self.inputs = InputHub(cfg.inputs, self.clock, start=start_inputs)
        self.camera = LiveCinematographer(plan.height, cfg.seed) if cfg.camera else None
        self.sink = sink if sink is not None else (PoseSink(cfg.out, self.names) if cfg.out else None)
        self.dt = 1.0 / cfg.rate
        self.t = 0.0
        self.seq = 0
        self.next_send = 0.0
        self.recorder = Recorder(self.names, cfg.record_fps) if cfg.record else None
        self.next_rec = 0.0
        self.style_idx = None
        self.last_state: ClockState | None = None
        self.last_frame: PoseFrame | None = None
        self.hold = True
        self._pelvis = self.names.index(plan.pelvis) if plan.pelvis in self.names else 0
        self._head = self.names.index("head") if "head" in self.names else self._pelvis
        self._feet = [self.names.index(n) for n in self.names if n.startswith("foot_")]
        sk = plan.sk
        self._rest_heads = np.asarray(sk.heads, float)
        self._head_pt = self._rest_heads[self._head] + 0.45 * (np.asarray(sk.tails[self._head]) - self._rest_heads[self._head])
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
        self.last_state = st
        self._apply_controls(now, t)
        res = self.core.tick(t, dt, notes, beat=st.beat, tempo=st.bpm, beats_per_bar=st.beats_per_bar)
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
            self.recorder.add(res.pose.delta, {"beat": st.beat, "bpm": st.bpm, "hold": float(self.hold)},
                              {"behavior": self.engine.behavior_name})
        return fr

    def _apply_controls(self, now: float, t: float) -> None:
        c = self.inputs.controls
        rw = self.engine.runway
        if rw is not None:
            for k in ("energy", "stride", "sway"):
                if k in c:
                    rw.live[k] = c[k]
            recent = now - self.inputs.stats["last_note"] < 2.5
            st = self.last_state
            link = self.clock.sources.get("link")
            authoritative = st is not None and (st.source in ("osc", "midi") or
                                                (st.source == "link" and getattr(link, "_seen_playing", False)))
            walking = recent or (st is not None and st.playing and authoritative)
            manual = c.get("hold")
            if manual is not None and manual > 0.5:
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
            "peers": st.peers if st else 0, "hold": self.hold, "section": self.engine.section,
            "behavior": self.engine.behavior_name, "notes": self.inputs.stats["notes"],
            "osc_packets": self.inputs.stats["osc_packets"], "sent": self.sink.sent if self.sink else 0,
            "camera": self.camera.kind if self.camera else None, "tick_ms": round(self.stats["mean_tick_ms"], 2),
            "max_tick_ms": round(self.stats["max_tick_ms"], 2), "overruns": self.stats["overruns"],
            "errors": self.inputs.stats["errors"] + ([self.clock.link_error] if self.clock.link_error else []),
        }


__all__ = ["LiveConfig", "LiveSession", "PoseSink"]
