"""Where the beat comes from, live.

Sources, best first (``clock="auto"`` picks the best one that is alive):

* ``osc``       /myrmex/transport from the Myrmex Remote Script in Ableton (song position,
                tempo, play state, time signature);
* ``link``      Ableton Link (zero set-up: press "Link" in Live, VCV/other apps can join);
* ``midi``      MIDI clock + start/stop/song position (Ableton "Sync" output, VCV CV-MIDI);
* ``onsets``    a phase-locked loop on incoming kick/percussion hits (VCV without a clock,
                live audio input);
* ``internal``  free-running at a fixed tempo.

Every source reports the same :class:`ClockState`.  The beat value is kept
monotonic across source switches so the step grid never jumps backwards.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

from ..bus.transport import BeatClock, LiveEvent


@dataclass
class ClockState:
    beat: float
    bpm: float
    playing: bool
    beats_per_bar: float = 4.0
    source: str = "internal"
    peers: int = 0
    song_beat: float | None = None   # the source's own song position (osc / midi) - for score following


class InternalClock:
    name = "internal"

    def __init__(self, bpm: float = 120.0, t0: float | None = None):
        self.bpm = bpm
        self.t0 = time.perf_counter() if t0 is None else t0
        self.playing = True

    def alive(self, now: float) -> bool:
        return True

    def state(self, now: float) -> ClockState:
        return ClockState((now - self.t0) * self.bpm / 60.0, self.bpm, self.playing, 4.0, self.name)


class LinkClock:
    """Ableton Link via ``aalink`` (runs its own asyncio loop in a daemon thread)."""
    name = "link"

    def __init__(self, bpm: float = 120.0, quantum: float = 4.0):
        import asyncio

        import aalink  # optional dependency: pip install aalink

        self.quantum = quantum
        self.link = None
        self._loop = asyncio.new_event_loop()
        ready = threading.Event()
        err: list[BaseException] = []

        async def _make():
            try:
                self.link = aalink.Link(bpm)
                self.link.quantum = quantum
                self.link.start_stop_sync_enabled = True
                self.link.enabled = True
            except BaseException as e:           # pragma: no cover - platform specific
                err.append(e)
            ready.set()

        def _run():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(_make())
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, name="myrmex-link", daemon=True)
        self._thread.start()
        ready.wait(5.0)
        if err or self.link is None:
            raise RuntimeError(f"Ableton Link unavailable: {err[0] if err else 'timeout'}")
        self._seen_playing = False

    def alive(self, now: float) -> bool:
        return self.link is not None and self.link.num_peers > 0

    def state(self, now: float) -> ClockState:
        L = self.link
        playing = bool(L.playing)
        self._seen_playing |= playing
        # Without Start/Stop Sync in Live, "playing" never turns on: then being linked is enough.
        if not self._seen_playing:
            playing = L.num_peers > 0
        return ClockState(float(L.beat), float(L.tempo), playing, self.quantum, self.name, int(L.num_peers))

    def close(self) -> None:
        if self.link is not None:
            try:
                self.link.enabled = False
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)


class EventClock:
    """MIDI clock or OSC transport, fed with :class:`LiveEvent` s (wraps :class:`BeatClock`)."""

    def __init__(self, name: str, timeout: float):
        self.name = name
        self.timeout = timeout
        self.bc = BeatClock()
        self.last = -1e9

    def feed(self, ev: LiveEvent) -> None:
        if ev.kind == "transport":
            self.bc.on_transport(ev)
        elif ev.kind == "clock":
            self.bc.on_clock(ev)
            self.bc.playing = True
        elif ev.kind == "start":
            self.bc.playing = True
            if not ev.data.get("continue"):
                self.bc.beat = 0.0
                self.bc._last_update = ev.t
        elif ev.kind == "stop":
            self.bc.playing = False
        self.last = ev.t

    def alive(self, now: float) -> bool:
        # A stopped transport stays authoritative (the song is paused, not gone).
        return now - self.last < self.timeout or (self.last > 0 and not self.bc.playing)

    def state(self, now: float) -> ClockState:
        bpb = self.bc.num * 4.0 / max(self.bc.den, 1)
        return ClockState(self.bc.now_beat(now), self.bc.bpm, self.bc.playing, bpb, self.name)


class OnsetClock:
    """Phase-locked loop on percussive hits: tempo and phase from the music itself.

    Inter-onset intervals are folded into one beat range (75-180 BPM), the median
    gives the period, and every hit nudges the phase toward the nearest grid line.
    """
    name = "onsets"

    def __init__(self, bpm: float = 120.0, lo: float = 75.0, hi: float = 180.0):
        self.period = 60.0 / bpm
        self.lo, self.hi = lo, hi
        self.ref = None           # time of a grid line
        self.beat_ref = 0.0       # beat number at ``ref``
        self.hits: deque[float] = deque(maxlen=24)
        self.last = -1e9
        self.locked = 0.0

    def hit(self, t: float, strength: float = 1.0) -> None:
        if self.hits and t - self.hits[-1] < 0.09:
            return
        self.hits.append(t)
        self.last = t
        if len(self.hits) >= 4:
            iois = []
            hs = list(self.hits)
            for k in (1, 2):
                for a, b in zip(hs, hs[k:]):
                    iois.append(self._fold(b - a))
            iois.sort()
            med = iois[len(iois) // 2]
            self.period += (med - self.period) * 0.25
        if self.ref is None:
            self.ref = t
            return
        # Phase error to the nearest grid line; move the grid (and the beat count with it).
        n = (t - self.ref) / self.period
        k = round(n)
        err = (n - k) * self.period
        gain = 0.35 * min(1.0, strength)
        self.ref += gain * err
        self.locked = min(1.0, self.locked + 0.1) if abs(err) < 0.06 else max(0.0, self.locked - 0.2)

    def _fold(self, d: float) -> float:
        lo, hi = 60.0 / self.hi, 60.0 / self.lo
        if d <= 0:
            return self.period
        while d < lo:
            d *= 2.0
        while d > hi:
            d *= 0.5
        return d

    def alive(self, now: float) -> bool:
        return self.ref is not None and now - self.last < 4.0 * self.period + 1.0

    def state(self, now: float) -> ClockState:
        if self.ref is None:
            return ClockState(0.0, 60.0 / self.period, False, 4.0, self.name)
        return ClockState(self.beat_ref + (now - self.ref) / self.period, 60.0 / self.period,
                          now - self.last < 2.0, 4.0, self.name)


class ClockHub:
    """Chooses the live clock and keeps the reported beat monotonic across switches."""

    ORDER = ("osc", "link", "midi", "onsets", "internal")

    def __init__(self, mode: str = "auto", bpm: float = 120.0, link: bool = True, now: float | None = None):
        now = time.perf_counter() if now is None else now
        self.mode = mode
        self.sources: dict[str, object] = {
            "osc": EventClock("osc", timeout=1.5),
            "midi": EventClock("midi", timeout=0.6),
            "onsets": OnsetClock(bpm),
            "internal": InternalClock(bpm, t0=now),
        }
        self.link_error = None
        if link and mode in ("auto", "link"):
            try:
                self.sources["link"] = LinkClock(bpm)
            except Exception as e:           # aalink missing or no network
                self.link_error = str(e)
        self.current = "internal"
        self._offset = 0.0
        self._last_beat = None
        self._last_src = None        # (time, beat, bpm) of the current source at the last call

    # ------------------------------------------------------------------ feeding
    def feed(self, ev: LiveEvent) -> None:
        if ev.kind == "transport":
            self.sources["osc"].feed(ev)
        elif ev.kind in ("clock", "start", "stop"):
            self.sources["midi"].feed(ev)

    def hit(self, t: float, strength: float = 1.0) -> None:
        self.sources["onsets"].hit(t, strength)

    # ------------------------------------------------------------------ state
    def _pick(self, now: float) -> str:
        if self.mode != "auto":
            return self.mode if self.mode in self.sources else "internal"
        for name in self.ORDER:
            src = self.sources.get(name)
            if src is not None and src.alive(now):
                return name
        return "internal"

    def state(self, now: float) -> ClockState:
        name = self._pick(now)
        st = self.sources[name].state(now)
        jump = False
        if self._last_src is not None and name == self.current:
            t_prev, b_prev, bpm_prev = self._last_src
            expected = b_prev + (now - t_prev) * bpm_prev / 60.0 if st.playing else b_prev
            jump = abs(st.beat - expected) > 1.0          # relocation (restart, locate, loop jump)
        if (name != self.current or jump) and self._last_beat is not None:
            # Keep our beat continuous; whole bars only, so the source's bar phase survives
            # and the fractional beat (where footfalls land) follows the new source at once.
            bpb = st.beats_per_bar
            self._offset = bpb * round((self._last_beat - st.beat) / bpb)
        self.current = name
        self._last_src = (now, st.beat, st.bpm)
        if name in ("osc", "midi"):
            st.song_beat = st.beat
        beat = st.beat + self._offset
        if self._last_beat is not None and self._last_beat - 0.1 < beat < self._last_beat:
            beat = self._last_beat                      # swallow tiny backward jitter
        self._last_beat = beat
        st.beat = beat
        st.source = name
        return st

    def close(self) -> None:
        link = self.sources.get("link")
        if link is not None:
            link.close()


__all__ = ["ClockState", "ClockHub", "InternalClock", "LinkClock", "EventClock", "OnsetClock"]
