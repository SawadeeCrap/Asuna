"""Live inputs -> musical onsets, macro controls and triggers.

Everything that can drive the creature ends up in three places:

* **onsets** (:class:`NoteEvent` with a semantic group: kick, snare, hats, perc, bass,
  melody, harmony, texture, fx) - what the body reacts to;
* **controls** (name -> 0..1) - macro knobs: ``energy``, ``stride``, ``sway``, ``style``,
  ``hold`` ...;
* **triggers** (one-shots) - ``camera`` (next shot), ``pose``, ``flourish`` ...

Routes (any combination):

* OSC on UDP 9100: the Myrmex Remote Script in Ableton (transport + clip notes ahead of
  time), Max for Live, VCV Rack (cvOSCcv), TouchOSC, Pd ...  Non-Myrmex addresses are
  mapped by the ``osc`` table of the config;
* MIDI (macOS IAC Driver or any port): notes by channel / GM drum map, CC -> controls,
  MIDI clock -> beat;
* audio (loopback input) -> kick / snare / hats onsets (see ``audio_in``).
"""
from __future__ import annotations

import json
import queue
import time
from dataclasses import dataclass, field

import numpy as np

from ..bus.transport import DEFAULT_PORT, LiveEvent, OscInput
from ..music.timeline import NoteEvent
from .clock import ClockHub, ClockState
from .midimap import MidiMapper, MidiMonitor

GROUPS = ("kick", "snare", "hats", "perc", "bass", "melody", "harmony", "texture", "fx")

# General MIDI percussion (also Ableton Drum Rack's default C1 = 36 = kick layout).
GM_DRUMS = {
    35: "kick", 36: "kick", 37: "perc", 38: "snare", 39: "snare", 40: "snare", 41: "perc", 42: "hats",
    43: "perc", 44: "hats", 45: "perc", 46: "hats", 47: "perc", 48: "perc", 49: "fx", 50: "perc", 51: "hats",
    52: "fx", 53: "hats", 54: "perc", 55: "fx", 56: "perc", 57: "fx", 58: "perc", 59: "hats", 60: "perc",
    61: "perc", 62: "perc", 63: "perc", 64: "perc", 69: "hats", 70: "perc", 75: "perc", 76: "perc", 77: "perc",
}

DEFAULT_CONFIG = {
    # MIDI channel (1-16) -> group, or "gm" (GM / Drum Rack map), or "pitch" (low = bass, high = melody)
    "midi_channels": {"1": "gm", "2": "bass", "3": "melody", "4": "harmony", "5": "fx", "6": "texture",
                      "7": "kick", "8": "snare", "9": "hats", "10": "gm", "11": "perc", "12": "perc",
                      "13": "melody", "14": "harmony", "15": "texture", "16": "control"},
    # MIDI CC number -> control name (any channel)
    "midi_cc": {"1": "energy", "2": "stride", "3": "sway", "4": "style", "5": "hold", "6": "camera",
                "7": "none", "16": "energy", "17": "stride", "18": "sway", "19": "style", "20": "hold",
                "21": "camera", "22": "pose", "23": "flourish", "64": "hold"},
    # Generic OSC address -> group (value > 0.5 on a rising edge = hit) or control / trigger name
    "osc": {"/ch/1": "kick", "/ch/2": "snare", "/ch/3": "hats", "/ch/4": "perc", "/ch/5": "bass",
            "/ch/6": "energy", "/ch/7": "stride", "/ch/8": "camera"},
    "styles": ["catwalk", "swagger", "heels", "natural"],
}

CONTROLS = ("energy", "stride", "sway", "style", "hold", "latency")
TRIGGERS = ("camera", "pose", "flourish", "reset")


@dataclass
class InputConfig:
    osc_port: int = DEFAULT_PORT
    osc_host: str = "0.0.0.0"
    midi: list[str] = field(default_factory=list)        # port names ("IAC" matches "IAC Driver Bus 1"), "auto"
    audio: str | None = None                              # input device name for the loopback analyser
    mapping: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_CONFIG)))
    score_lookahead: float = 0.0                          # beats: emit scored notes this much early

    @classmethod
    def from_file(cls, path: str | None, **kw) -> "InputConfig":
        cfg = cls(**kw)
        if path:
            with open(path) as f:
                user = json.load(f)
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(cfg.mapping.get(k), dict):
                    cfg.mapping[k].update(v)
                else:
                    cfg.mapping[k] = v
        return cfg


class ScoreFollower:
    """Notes of playing clips, sent ahead of time by the Remote Script (song beats)."""

    def __init__(self):
        self.tracks: dict[str, dict] = {}
        self.notes: dict[str, np.ndarray] = {}       # id -> (N, 4) [beat, dur, pitch, vel] sorted by beat
        self.prev: float | None = None

    def set_track(self, tid: str, name: str, group: str) -> None:
        self.tracks[tid] = {"name": name, "group": group}

    def set_notes(self, tid: str, arr: np.ndarray, window: tuple[float, float] | None = None) -> None:
        """A blob replaces the notes of its window (``[from, to)`` song beats; default: from its first note on)."""
        arr = np.asarray(arr, float).reshape(-1, 4)
        if window is None:
            if len(arr) == 0:
                self.notes.pop(tid, None)
                return
            window = (float(arr[:, 0].min()), float("inf"))
        a, b = window
        old = self.notes.get(tid)
        if old is not None and len(old):
            keep = (old[:, 0] < a - 1e-6) | (old[:, 0] >= b - 1e-6)
            arr = np.concatenate([old[keep], arr])
        if len(arr) == 0:
            self.notes.pop(tid, None)
            return
        self.notes[tid] = arr[np.argsort(arr[:, 0], kind="stable")]

    def clear(self) -> None:
        self.notes.clear()

    def due(self, song_beat: float | None, bpm: float, t: float) -> list[NoteEvent]:
        if song_beat is None:
            self.prev = None
            return []
        prev, self.prev = self.prev, song_beat
        if prev is None or not (0.0 < song_beat - prev < 4.0):
            return []                                    # first call, stop, relocation or loop jump
        out = []
        spb = 60.0 / max(bpm, 1.0)
        for tid, arr in self.notes.items():
            i0 = int(np.searchsorted(arr[:, 0], prev, side="right"))
            i1 = int(np.searchsorted(arr[:, 0], song_beat, side="right"))
            tr = self.tracks.get(tid, {})
            hint = tr.get("group") or ""
            for b, dur, pitch, vel in arr[i0:i1]:
                if hint == "control":
                    g = "control"
                elif hint == "gm" or not hint:
                    g = GM_DRUMS.get(int(round(pitch))) if hint == "gm" else None
                    g = g or ("perc" if hint == "gm" else ("bass" if pitch < 48 else "melody"))
                else:
                    g = hint
                out.append(NoteEvent(t, float(dur) * spb, float(pitch), float(vel), tid, beat=float(b), group=g))
            if i0 > 64:                                  # forget the past
                self.notes[tid] = arr[i0:]
        return out


class InputHub:
    def __init__(self, cfg: InputConfig | None = None, clock: ClockHub | None = None, start: bool = True):
        self.cfg = cfg or InputConfig()
        self.clock = clock
        self.q: queue.Queue = queue.Queue()
        self.osc: OscInput | None = None
        self.midi: list = []
        self.audio = None
        self.controls: dict[str, float] = {}
        self.triggers: list[tuple[str, float]] = []
        self.score = ScoreFollower()
        self._open: dict[tuple[int, float], NoteEvent] = {}
        self._osc_state: dict[str, float] = {}
        self.midimap = MidiMapper(self.cfg.mapping.get("midi_bindings", []))
        self.monitor = MidiMonitor()
        from .glove import GloveDecoder
        self.glove = GloveDecoder(self.cfg.mapping.get("glove_profile"))    # Hand Glove, read in parallel
        self.glove_owns = False                    # True while a glove link drives the creature directly
        self.td = {"seen": -1e9, "fps": 0.0}       # TouchDesigner heartbeat (/myrmex/td/alive f fps)
        self.stats = {"notes": 0, "osc_packets": 0, "midi_events": 0, "last_note": -1e9, "errors": []}
        if start:
            self.start()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        c = self.cfg
        if c.osc_port:
            try:
                self.osc = OscInput(c.osc_port, c.osc_host, out=self.q)
            except OSError as e:
                self.stats["errors"].append(f"OSC port {c.osc_port}: {e}")
        for name in c.midi:
            try:
                from ..bus.transport import MidiInput
                self.midi.append(MidiInput(None if name == "auto" else name, out=self.q))
            except Exception as e:           # mido / rtmidi missing, or no such port
                self.stats["errors"].append(f"MIDI {name}: {e}")
        if c.audio:
            try:
                from .audio_in import AudioInput
                self.audio = AudioInput(c.audio, out=self.q)
            except Exception as e:
                self.stats["errors"].append(f"audio {c.audio}: {e}")

    def close(self) -> None:
        if self.osc:
            self.osc.close()
        for m in self.midi:
            m.close()
        if self.audio:
            self.audio.close()

    def push(self, ev: LiveEvent) -> None:
        """Inject an event directly (tests, embedded senders)."""
        self.q.put(ev)

    # ------------------------------------------------------------------ mapping
    def group_for_midi(self, channel: int, pitch: float) -> str:
        m = self.cfg.mapping["midi_channels"].get(str(channel), "gm")
        if m == "gm":
            return GM_DRUMS.get(int(round(pitch)), "perc" if pitch < 36 else ("bass" if pitch < 48 else "melody"))
        if m == "pitch":
            return "bass" if pitch < 52 else "melody"
        return m

    def _set_control(self, name: str, value: float, t: float) -> None:
        if name in (None, "", "none"):
            return
        if name in GROUPS:
            return
        if name in TRIGGERS or name.startswith("pose:") or name.startswith("flourish:") or name.startswith("camera:"):
            value = max(0.0, value)
            prev = self.controls.get("_trig_" + name, 0.0)
            if value > 0.5 >= prev:
                self.triggers.append((name, t))
            self.controls["_trig_" + name] = value
            return
        if value < 0.0:
            self.controls.pop(name, None)          # negative = back to automatic
            return
        self.controls[name] = float(np.clip(value, 0.0, 1.0))

    # ------------------------------------------------------------------ per tick
    def score_due(self, st: ClockState, t: float) -> list[NoteEvent]:
        """Scored notes (Remote Script) whose song beat has been reached."""
        notes = self.score.due(st.song_beat, st.bpm, t)
        if notes:
            self.stats["notes"] += len(notes)
            self.stats["last_note"] = time.perf_counter()
        return notes

    def poll(self, now: float, t: float, clock_state: ClockState | None = None) -> list[NoteEvent]:
        """Drain all inputs; returns the onsets heard since the last call (session time ``t``)."""
        if self.osc is not None:
            self.stats["osc_packets"] += self.osc.poll()
        notes: list[NoteEvent] = []
        while True:
            try:
                ev = self.q.get_nowait()
            except queue.Empty:
                break
            n = self._handle(ev, t)
            if n is not None:
                notes.append(n)
        if clock_state is not None:
            notes.extend(self.score.due(clock_state.song_beat, clock_state.bpm, t))
        if notes:
            self.stats["notes"] += len(notes)
            self.stats["last_note"] = now
            if self.clock is not None:
                for n in notes:
                    g = n.group or ""
                    if g in ("kick", "snare") or (g == "perc" and n.velocity > 0.6):
                        self.clock.hit(now, n.velocity * (1.0 if g == "kick" else 0.6))
        return notes

    def _handle(self, ev: LiveEvent, t: float) -> NoteEvent | None:
        k, d = ev.kind, ev.data
        if k in ("transport", "clock", "start", "stop"):
            if self.clock is not None:
                self.clock.feed(ev)
            if k == "clock":
                self.stats["midi_events"] += 1
            return None
        if k == "note" and d.get("channel"):
            ch, num, vel = int(d["channel"]), float(d["pitch"]), float(d["velocity"])
            if self.midimap.learn("note", ch, int(num)):
                self.monitor.add("note", ch, int(num), vel, "learned")
                return None
            acts = self.midimap.note(ch, int(num), vel, True)
            group_override = None
            for act, target, v in acts:
                if act == "trigger":
                    self.triggers.append((target, t))
                elif act == "control":
                    self._set_control(target, v, t)
                elif act == "group":
                    group_override = target
            mapped = ", ".join(tg for _, tg, _ in acts)
            if acts and group_override is None:
                self.monitor.add("note", ch, int(num), vel, "→ " + mapped)
                return None                                   # a command, not music
            group = group_override or self.group_for_midi(ch, num)
            self.monitor.add("note", ch, int(num), vel, f"→ {mapped or 'music: ' + group}")
            d = dict(d, group=group)
        if k == "note":
            ch = int(d.get("channel", 0))
            group = d.get("group") or (self.group_for_midi(ch, d["pitch"]) if ch else "perc")
            dur = float(d.get("duration", -1.0))
            n = NoteEvent(t, dur if dur > 0 else 30.0, float(d["pitch"]), float(d["velocity"]),
                          str(d.get("track", f"midi{ch}")), group=group)
            if dur <= 0:
                self._open[(ch, float(d["pitch"]))] = n        # closed by note_off
            self.stats["midi_events"] += 1
            return n
        if k == "note_off":
            for act, target, v in self.midimap.note(int(d.get("channel", 0)), int(float(d["pitch"])), 0.0, False):
                if act == "control":
                    self._set_control(target, v, t)
            n = self._open.pop((int(d.get("channel", 0)), float(d["pitch"])), None)
            if n is not None:
                n.duration = max(0.02, t - n.time)
            return None
        if k == "hit":
            g = str(d["group"])
            if g == "kick" and d.get("source") == "audio" and self.clock is not None and \
                    self.clock.current in ("osc", "link", "midi"):
                # Audio can't tell a kick from an off-beat bass note; the beat grid can.
                st = self.clock.sources[self.clock.current].state(ev.t)    # raw source: no side effects
                ph = st.beat - round(st.beat)
                if abs(ph) > 0.15:
                    g = "bass"
            return NoteEvent(t, 0.1, float(d.get("pitch", 60.0)), float(d["velocity"]), "hit",
                             group=g, sharpness=float(d.get("sharpness", 0.8)))
        if k == "pb":
            ch = int(d.get("channel", 0))
            self.glove.feed_pb(ch, float(d["value"]), ev.t)
            self.monitor.add("pb", ch, 0, float(d["value"]), "→ glove" if self.glove.owns_pb(ch) else "pitch bend")
            return None
        if k == "cc":
            ch, num, val = int(d.get("channel", 0)), int(d["control"]), float(d["value"])
            self.glove.feed_cc(ch, num, int(d.get("raw", round(val * 127))), ev.t)
            if self.glove_owns and self.glove.owns(ch, num):     # the glove moves the body, not parameters
                return None
            if self.midimap.learn("cc", ch, num):
                self.monitor.add("cc", ch, num, val, "learned")
                return None
            acts = self.midimap.cc(ch, num, val)
            for _, target, v in acts:
                self._set_control(target, v, t)
            if not acts:                                      # default table (see DEFAULT_CONFIG["midi_cc"])
                name = self.cfg.mapping["midi_cc"].get(str(num))
                if name:
                    self._set_control(name, val, t)
                self.monitor.add("cc", ch, num, val, f"→ {name} (default)" if name else "unmapped")
            else:
                self.monitor.add("cc", ch, num, val, "→ " + ", ".join(f"{tg}={v:.2f}" for _, tg, v in acts))
            return None
        if k in ("cv", "control"):
            self._set_control(str(d["name"]), float(d["value"]), t)
            return None
        if k == "trigger":
            self.triggers.append((str(d["name"]), t))
            return None
        if k == "osc":
            addr, v = d["address"], float(d["value"])
            if addr == "/myrmex/td/alive":                       # TouchDesigner is there (and how fast)
                self.td = {"seen": time.perf_counter(), "fps": v}
                return None
            if self.glove.feed_osc(addr, v, ev.t):
                return None
            target = self.cfg.mapping["osc"].get(addr)
            if target in GROUPS:
                prev = self._osc_state.get(addr, 0.0)
                self._osc_state[addr] = v
                if v > 0.5 >= prev:                              # gate rising edge
                    return NoteEvent(t, 0.1, 60.0, min(1.0, max(0.3, v)), addr, group=target)
                return None
            if target:
                self._set_control(target, v, t)
            else:
                self.controls[addr] = v
            return None
        if k == "score_track":
            self.score.set_track(d["id"], d["name"], d.get("group", ""))
            return None
        if k == "score_notes":
            self.score.set_notes(d["id"], d["notes"], d.get("window"))
            return None
        if k == "score_clear":
            self.score.clear()
            return None
        if k == "reset":
            self.triggers.append(("reset", t))
        return None

    def take_triggers(self) -> list[tuple[str, float]]:
        out, self.triggers = self.triggers, []
        return out


__all__ = ["InputConfig", "InputHub", "ScoreFollower", "GM_DRUMS", "DEFAULT_CONFIG", "GROUPS"]
