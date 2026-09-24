"""Fine MIDI mapping + monitor: what arrives (CC / notes) and what it drives.

A binding maps one CC or note (optionally on one channel) to a *target*:
a control / creature parameter (``energy``, ``aggression``, ``cam_distance`` ...), a trigger
(``camera``, ``pose``, ``creature:collapse`` ...) or a musical group for notes (``kick`` ...).

CC:   value -> curve (linear / exp / log / s) -> invert -> [min, max] range -> smoothing
Note: mode trigger (on note-on), gate (1 while held), toggle, velocity (value = velocity), group.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field

CURVES = ("linear", "exp", "log", "s")
NOTE_MODES = ("trigger", "gate", "toggle", "velocity", "group")


@dataclass
class MidiBinding:
    kind: str = "cc"                 # cc | note
    number: int = 1
    channel: int = 0                 # 0 = any channel
    target: str = ""
    lo: float = 0.0
    hi: float = 1.0
    curve: str = "linear"
    invert: bool = False
    mode: str = "trigger"            # notes only
    smooth: float = 0.0              # s (CC only)

    def matches(self, kind: str, channel: int, number: int) -> bool:
        return self.kind == kind and self.number == number and self.channel in (0, channel)

    def shape(self, v: float) -> float:
        v = min(1.0, max(0.0, v))
        if self.curve == "exp":
            v = v * v
        elif self.curve == "log":
            v = math.sqrt(v)
        elif self.curve == "s":
            v = v * v * (3 - 2 * v)
        if self.invert:
            v = 1.0 - v
        return self.lo + (self.hi - self.lo) * v


class MidiMonitor:
    def __init__(self, size: int = 300):
        self.rows: deque = deque(maxlen=size)
        self.seq = 0

    def add(self, kind: str, channel: int, number, value: float, targets: str) -> None:
        self.seq += 1
        self.rows.append((self.seq, time.strftime("%H:%M:%S"), kind, channel, number, round(float(value), 3), targets))


class MidiMapper:
    def __init__(self, bindings: list | None = None):
        self.bindings: list[MidiBinding] = []
        self.set_bindings(bindings or [])
        self.learning: int | None = None          # index of the binding that learns the next message
        self._toggle: dict[int, bool] = {}
        self._smooth: dict[int, float] = {}
        self._t = time.perf_counter()

    def set_bindings(self, rows: list) -> None:
        self.bindings = [b if isinstance(b, MidiBinding) else MidiBinding(**{k: v for k, v in dict(b).items()
                                                                                if k in MidiBinding.__dataclass_fields__})
                         for b in rows]

    def to_list(self) -> list[dict]:
        return [asdict(b) for b in self.bindings]

    def learn(self, kind: str, channel: int, number: int) -> bool:
        i = self.learning
        if i is None or not (0 <= i < len(self.bindings)):
            return False
        b = self.bindings[i]
        b.kind, b.channel, b.number = kind, channel, int(number)
        self.learning = None
        return True

    def cc(self, channel: int, number: int, value: float) -> list[tuple[str, str, float]]:
        """-> [(action, target, value)] with action 'control'."""
        out = []
        now = time.perf_counter()
        for i, b in enumerate(self.bindings):
            if b.target and b.matches("cc", channel, number):
                v = b.shape(value)
                if b.smooth > 0:
                    prev = self._smooth.get(i, v)
                    a = 1.0 - math.exp(-max(now - self._t, 1e-3) / b.smooth)
                    v = prev + (v - prev) * a
                    self._smooth[i] = v
                out.append(("control", b.target, v))
        self._t = now
        return out

    def note(self, channel: int, number: int, velocity: float, on: bool) -> list[tuple[str, str, float]]:
        """-> [(action, target, value)]: action in trigger / control / group."""
        out = []
        for i, b in enumerate(self.bindings):
            if not (b.target and b.matches("note", channel, number)):
                continue
            if b.mode == "trigger" and on:
                out.append(("trigger", b.target, 1.0))
            elif b.mode == "gate":
                out.append(("control", b.target, b.hi if on else b.lo))
            elif b.mode == "toggle" and on:
                st = not self._toggle.get(i, False)
                self._toggle[i] = st
                out.append(("control", b.target, b.hi if st else b.lo))
            elif b.mode == "velocity" and on:
                out.append(("control", b.target, b.shape(velocity)))
            elif b.mode == "group" and on:
                out.append(("group", b.target, velocity))
        return out
