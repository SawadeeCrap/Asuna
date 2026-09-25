"""Live inputs: OSC over UDP and MIDI (IAC) with clock-based beat tracking.

All inputs push :class:`LiveEvent` objects into a thread-safe queue; the
realtime session drains it on the animation thread.  Nothing here blocks the
caller (sockets are non-blocking, MIDI is read in its own thread).

Myrmex OSC namespace (sent by the Ableton Remote Script, the Max for Live
follower and the VCV Rack bridge)::

    /myrmex/transport  f song_beats  f bpm  i playing  i sig_num  i sig_den
    /myrmex/note       s track  s group  f pitch  f velocity  f duration  [f beat]
    /myrmex/hit        s group  f intensity  f sharpness  [f pitch]
    /myrmex/cv         s name  f value(0..1)
    /myrmex/clock      i tick  i ppqn
    /myrmex/reset
    /myrmex/score/track  s id  s name  s group_hint
    /myrmex/score/notes  s id  b float32[beat, dur, pitch, vel]*  [f from  f to]
    /myrmex/score/clear
    /myrmex/control    s name  f value          (macro knobs: energy, stride, sway, style ...)
    /myrmex/trigger    s name  [f value]        (camera, pose, flourish ...)
    <any address> f value                       (generic: mapped by address in the input config)
"""
from __future__ import annotations

import queue
import socket
import time
from dataclasses import dataclass, field

import numpy as np

from .osc import OscMessage, decode

DEFAULT_PORT = 9100


@dataclass
class LiveEvent:
    kind: str                  # note | hit | cv | transport | clock | score_track | score_notes | score_clear | reset | start | stop
    t: float                   # arrival time (time.perf_counter)
    data: dict = field(default_factory=dict)


class OscInput:
    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1", out: queue.Queue | None = None):
        self.q: queue.Queue = out or queue.Queue()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.port = port
        self.packets = 0

    def poll(self, max_packets: int = 512) -> int:
        n = 0
        while n < max_packets:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            now = time.perf_counter()
            try:
                for m in decode(data):
                    ev = self._to_event(m, now)
                    if ev is not None:
                        self.q.put(ev)
            except Exception:
                continue
            n += 1
        self.packets += n
        return n

    @staticmethod
    def _to_event(m: OscMessage, now: float) -> LiveEvent | None:
        a = m.args
        addr = m.address
        if addr == "/myrmex/note" and len(a) >= 5:
            d = {"track": str(a[0]), "group": str(a[1]), "pitch": float(a[2]), "velocity": float(a[3]),
                 "duration": float(a[4])}
            if len(a) > 5:
                d["beat"] = float(a[5])
            return LiveEvent("note", now, d)
        if addr == "/myrmex/hit" and len(a) >= 2:
            return LiveEvent("hit", now, {"group": str(a[0]), "velocity": float(a[1]),
                                          "sharpness": float(a[2]) if len(a) > 2 else 0.8,
                                          "pitch": float(a[3]) if len(a) > 3 else 60.0})
        if addr == "/myrmex/cv" and len(a) >= 2:
            return LiveEvent("cv", now, {"name": str(a[0]), "value": float(a[1])})
        if addr == "/myrmex/transport" and len(a) >= 3:
            return LiveEvent("transport", now, {"beat": float(a[0]), "bpm": float(a[1]), "playing": bool(a[2]),
                                                "num": int(a[3]) if len(a) > 3 else 4,
                                                "den": int(a[4]) if len(a) > 4 else 4})
        if addr == "/myrmex/clock" and len(a) >= 1:
            return LiveEvent("clock", now, {"tick": int(a[0]), "ppqn": int(a[1]) if len(a) > 1 else 24})
        if addr == "/myrmex/score/track" and len(a) >= 2:
            return LiveEvent("score_track", now, {"id": str(a[0]), "name": str(a[1]),
                                                  "group": str(a[2]) if len(a) > 2 else ""})
        if addr == "/myrmex/score/notes" and len(a) >= 2 and isinstance(a[1], (bytes, bytearray)):
            arr = np.frombuffer(bytes(a[1]), dtype=">f4").astype(float).reshape(-1, 4)
            d = {"id": str(a[0]), "notes": arr}
            if len(a) >= 4:
                d["window"] = (float(a[2]), float(a[3]))
            return LiveEvent("score_notes", now, d)
        if addr == "/myrmex/score/clear":
            return LiveEvent("score_clear", now, {})
        if addr == "/myrmex/reset":
            return LiveEvent("reset", now, {})
        if addr == "/myrmex/control" and len(a) >= 2:
            return LiveEvent("control", now, {"name": str(a[0]), "value": float(a[1])})
        if addr == "/myrmex/trigger" and len(a) >= 1:
            return LiveEvent("trigger", now, {"name": str(a[0]), "value": float(a[1]) if len(a) > 1 else 1.0})
        # Anything else with a number (VCV cvOSCcv, TouchOSC, Max, Pd ...): mapped by address.
        nums = [float(x) for x in a if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if nums:
            return LiveEvent("osc", now, {"address": addr, "value": nums[0], "args": nums})
        return None

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class MidiInput:
    """Reads a MIDI port (e.g. macOS IAC Driver) in a background thread via mido.

    Note-on -> note events (channel 10 = drums, GM groups); MIDI clock / start /
    stop / song position -> clock events for beat tracking.
    """

    def __init__(self, port_name: str | None = None, out: queue.Queue | None = None):
        import mido  # optional dependency
        self.q: queue.Queue = out or queue.Queue()
        names = mido.get_input_names()
        if not names:
            raise RuntimeError("no MIDI input ports (enable the IAC Driver in Audio MIDI Setup)")
        if port_name and port_name not in names:
            match = [n for n in names if port_name.lower() in n.lower()]
            if not match:
                raise RuntimeError(f"MIDI port {port_name!r} not found; available: {names}")
            port_name = match[0]
        name = port_name or next((n for n in names if "IAC" in n), names[0])
        self.port = mido.open_input(name, callback=self._callback)
        self.name = name
        self.ticks = 0

    def _callback(self, msg) -> None:
        now = time.perf_counter()
        t = msg.type
        if t == "note_on" and msg.velocity > 0:
            self.q.put(LiveEvent("note", now, {"track": f"midi{msg.channel + 1}", "group": "",
                                               "channel": msg.channel + 1, "pitch": float(msg.note),
                                               "velocity": msg.velocity / 127.0, "duration": -1.0}))
        elif t == "note_off" or (t == "note_on" and msg.velocity == 0):
            self.q.put(LiveEvent("note_off", now, {"channel": msg.channel + 1, "pitch": float(msg.note)}))
        elif t == "clock":
            self.ticks += 1
            self.q.put(LiveEvent("clock", now, {"tick": self.ticks, "ppqn": 24}))
        elif t == "start":
            self.ticks = 0
            self.q.put(LiveEvent("start", now, {}))
        elif t == "continue":
            self.q.put(LiveEvent("start", now, {"continue": True}))
        elif t == "stop":
            self.q.put(LiveEvent("stop", now, {}))
        elif t == "songpos":
            self.ticks = msg.pos * 6          # SPP counts 16ths; 6 clocks per 16th
            self.q.put(LiveEvent("clock", now, {"tick": self.ticks, "ppqn": 24, "songpos": True}))
        elif t == "control_change":
            self.q.put(LiveEvent("cc", now, {"control": msg.control, "channel": msg.channel + 1,
                                             "value": msg.value / 127.0, "raw": msg.value, "port": self.name}))
        elif t == "pitchwheel":                        # 14-bit controllers (e.g. the Hand Glove)
            self.q.put(LiveEvent("pb", now, {"channel": msg.channel + 1, "value": (msg.pitch + 8192) / 16383.0,
                                             "port": self.name}))

    def close(self) -> None:
        try:
            self.port.close()
        except Exception:
            pass


class BeatClock:
    """Beat position and tempo from transport messages or MIDI clock (with smoothing)."""

    def __init__(self, bpm: float = 120.0):
        self.bpm = bpm
        self.beat = 0.0
        self.playing = True
        self._last_update = time.perf_counter()
        self._clock_times: list[float] = []
        self.num, self.den = 4, 4
        self.source = "free"

    def on_transport(self, ev: LiveEvent) -> None:
        """Transport reports arrive at display rate with ~10-30 ms jitter: follow them with a
        phase-locked filter (small errors are blended, relocations snap)."""
        d = ev.data
        was_playing = self.playing and self.source == "transport"
        pred = self.now_beat(ev.t)
        self.bpm = max(20.0, d["bpm"])
        self.num, self.den = d.get("num", 4), d.get("den", 4)
        err = d["beat"] - pred
        if was_playing and d["playing"] and abs(err) < 0.25:
            self.beat = pred + 0.15 * err
        else:
            self.beat = d["beat"]
        self.playing = d["playing"]
        self._last_update = ev.t
        self.source = "transport"

    def on_clock(self, ev: LiveEvent) -> None:
        ppqn = ev.data.get("ppqn", 24)
        self._clock_times.append(ev.t)
        self._clock_times = self._clock_times[-(ppqn * 2):]
        if len(self._clock_times) >= ppqn // 2:
            span = self._clock_times[-1] - self._clock_times[0]
            if span > 0:
                bpm = 60.0 * (len(self._clock_times) - 1) / (span * ppqn)
                self.bpm += (bpm - self.bpm) * 0.2
        self.beat = ev.data["tick"] / float(ppqn)
        self._last_update = ev.t
        self.source = "midi_clock"

    def now_beat(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if not self.playing:
            return self.beat
        return self.beat + (now - self._last_update) * self.bpm / 60.0


def send_osc(messages: list[OscMessage], host: str = "127.0.0.1", port: int = DEFAULT_PORT,
             sock: socket.socket | None = None) -> None:
    s = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for m in messages:
        s.sendto(m.encode(), (host, port))
    if sock is None:
        s.close()


def pack_notes(notes: np.ndarray) -> bytes:
    """(N, 4) float array [beat, dur, pitch, vel] -> big-endian float32 blob."""
    return np.asarray(notes, dtype=">f4").tobytes()


__all__ = ["LiveEvent", "OscInput", "MidiInput", "BeatClock", "send_osc", "pack_notes", "DEFAULT_PORT"]
