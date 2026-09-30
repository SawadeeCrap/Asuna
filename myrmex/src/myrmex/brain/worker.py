"""The brain beside the engine, never in its way.

``BrainLink`` lives in the live session's thread and costs microseconds per 120 Hz tick: at the decision
rate it copies a few arrays (the lead body's blend, material and node positions) and sends them to a
separate process; whenever an answer is there it applies it through the adapter.  It never waits.

The worker process (its own interpreter: no Qt, no audio, no engine in it) builds the fingerprint, keeps the memory,
generates and scores candidates and - when asked - talks to a local Kev server over HTTP.  Why a process
and not a thread: the engine thread already spends 2-7 ms of every 8.3 ms tick in Python (measured,
tools: ``morphology_brain_test.py --ipc-bench``); a thread doing candidate geometry and JSON would take the
same GIL.  A process takes none, and a crash, a hang or a slow Kev cannot touch the organism.

Failure is the normal case it is built for: no answer within ``timeout`` (or Kev's timeout + 0.7 s) -> the
request is dropped and the engine carries on with its own state machine; an answer older than that + 0.5 s,
or one that arrives while the organism is busy, is dropped too; a dead worker -> restarted after a back-off (2, 5, 15 s),
given up after 5 failures in 2 minutes; malformed answers -> ignored.  ``enabled=False`` -> nothing runs.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from multiprocessing.connection import Connection

import numpy as np

from .adapter import make_adapter, user_cue
from .core import BrainConfig, BrainCore, Decision, Snapshot


def _engine_ref(engine) -> tuple[str, str]:
    cls = type(engine)
    return cls.__module__, cls.__qualname__


def _load_class(ref: tuple[str, str]):
    import importlib
    mod = importlib.import_module(ref[0])
    obj = mod
    for part in ref[1].split("."):
        obj = getattr(obj, part)
    return obj


class DecisionLog:
    """``<log_dir>/<organism>-<date>.jsonl``: every decision point (what it was, the candidates, what was chosen
    and by whom) and the performer's Good / Bad marks - the material a later Kev fine-tune learns from.  Written
    by the worker, never by the engine thread; a full disk or a missing folder only stops the log."""

    def __init__(self, folder: str, organism: str):
        self.path = os.path.join(folder, f"{organism or 'organism'}-{time.strftime('%Y%m%d')}.jsonl") if folder else ""
        self.f = None
        self.seen = 0

    def write(self, rec: dict) -> None:
        if not self.path:
            return
        try:
            if self.f is None:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                self.f = open(self.path, "a", encoding="utf-8")
            self.f.write(json.dumps({"wall": round(time.time(), 2), **rec}, default=float) + "\n")
            self.f.flush()
        except (OSError, TypeError, ValueError):
            self.path = ""

    def decisions(self, core, summary: dict) -> None:
        new = min(core.logged - self.seen, len(core.log))
        for e in core.log[len(core.log) - new:] if new > 0 else ():
            self.write({"kind": "decision", "organism": core.vocab.organism, "mode": core.cfg.mode, **e,
                        "memory": summary})
        self.seen = core.logged


def _worker_main(conn, ref: tuple[str, str], organism: str, cfg: dict) -> None:
    """The worker process: snapshots in, decisions out."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)                   # the app stops it, not Ctrl+C
    from .fingerprint import Fingerprint, geometry
    from .vocab import vocabulary
    try:
        vocab = vocabulary(_load_class(ref), organism)
        core = BrainCore(vocab, BrainConfig.from_dict(cfg))
        log = DecisionLog(core.cfg.log_dir, organism)
    except Exception as e:                                          # pragma: no cover - reported to the app
        try:
            conn.send(("error", f"{type(e).__name__}: {e}"))
        except OSError:
            pass
        return
    try:
        conn.send(("ready", {"forms": len(vocab.forms), "free": list(vocab.free), "family": vocab.family}))
    except OSError:                                                 # the app already let it go
        return
    while True:
        try:
            if not conn.poll(2.0):
                continue
            kind, payload = conn.recv()
        except (EOFError, OSError):
            return                                                  # the app went away
        try:
            if kind == "snap":
                t0 = time.perf_counter()
                X = payload.pop("X")
                geo = geometry(X) if len(X) >= 4 else np.zeros(8)
                fp = Fingerprint(np.asarray(payload.pop("w")), geo, np.asarray(payload.pop("mat")),
                                 int(payload.pop("bodies")))
                snap = Snapshot(fp=fp, **payload)
                d = core.step(snap)
                ms = (time.perf_counter() - t0) * 1000.0
                summary = core.memory.summary(snap.t, vocab.forms)
                conn.send(("decision", {"d": d.to_dict() if d is not None else None, "sent": payload["t"],
                                        "ms": ms,
                                        "stats": {k: v for k, v in core.stats.items() if not isinstance(v, list)},
                                        "kev_ms": core.stats["kev_ms"][-1] if core.stats["kev_ms"] else None,
                                        "summary": summary}))
                if core.logged != log.seen:
                    log.decisions(core, summary)
            elif kind == "config":
                from .core import _client
                core.cfg = BrainConfig.from_dict(payload)
                core.kev = _client(core.cfg)
                core.load_taste()                                   # (verdicts from the Train page since)
                if core.cfg.log_dir != os.path.dirname(log.path or ""):
                    log = DecisionLog(core.cfg.log_dir, organism)
                    log.seen = core.logged
            elif kind == "mark":
                core.mark(bool(payload))
                log.write({"kind": "mark", "organism": organism, "good": bool(payload),
                           "t": core.memory._t, "form": core.memory.summary(core.memory._t or 0.0, vocab.forms)})
            elif kind == "stop":
                return
        except OSError:                                             # the app went away mid-answer
            return
        except Exception:                                           # a bad snapshot never kills the worker
            try:
                conn.send(("error", traceback.format_exc(limit=3)))
            except OSError:
                return


HOUSEKEEPING_S = 1.0 / 30.0


class BrainLink:
    """The live session's side of the brain (see the module docstring)."""

    def __init__(self, cfg: BrainConfig, engine, organism: str = "", process: bool = True, timeout: float = 1.5):
        self.cfg, self.engine, self.organism = cfg, engine, organism
        self.adapter = make_adapter(engine, organism)
        self.process, self.timeout = process, timeout
        self.proc = self.conn = None
        self.core: BrainCore | None = None
        self.inflight: float | None = None
        self.next_t = 0.0
        self.fails: list[float] = []
        self.retry_at = 0.0
        self.gave_up = False
        self.ready = False
        self.last: dict = {}
        self.stats = {"sent": 0, "answered": 0, "applied": 0, "timeouts": 0, "stale": 0, "restarts": 0,
                      "errors": 0, "rtt_ms": [], "worker_ms": [], "tick_us": []}
        self.error = ""
        self._house = -1e9
        self._boot = None
        if cfg.enabled:
            self._start()

    # ------------------------------------------------------------------ worker lifecycle
    def _start(self) -> None:
        """A separate interpreter (``python -m myrmex.brain.worker``) on one end of a socket pair - what
        ``multiprocessing.Pipe`` is, without multiprocessing's spawn, which would re-import the parent's main
        module (in the app: the Qt window).  Only the child inherits the other end; when either side dies the
        other reads EOF."""
        self.ready, self.inflight, self._boot = False, None, None
        if not self.process:
            self.core = BrainCore(self.adapter.vocab, self.cfg)
            self.ready = True
            return
        a, b = socket.socketpair()
        path = os.pathsep.join(dict.fromkeys(p for p in sys.path if p and os.path.exists(p)))
        env = dict(os.environ, MYRMEX_BRAIN_FD=str(b.fileno()), PYTHONPATH=path)
        try:
            self.proc = subprocess.Popen([sys.executable, "-m", "myrmex.brain.worker"], env=env,
                                         pass_fds=(b.fileno(),), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        except OSError as e:
            a.close()
            self.error = f"the brain worker could not start: {e}"
            self.proc = None
            return
        finally:
            b.close()
        self.conn = Connection(a.detach())
        self.conn.send(("init", _engine_ref(self.engine), self.organism, self.cfg.to_dict()))

    def _fail(self, now: float, why: str) -> None:
        self.error = why
        self.stats["errors"] += 1
        self.fails = [f for f in self.fails if now - f < 120.0] + [now]
        self.close()
        if len(self.fails) >= 5:
            self.gave_up = True                                    # the organism is fine on its own
            return
        self.retry_at = now + (2.0, 5.0, 15.0, 30.0)[min(3, len(self.fails) - 1)]

    def close(self) -> None:
        """Stop the worker without waiting for it (this may be the engine thread): a helper thread reaps it."""
        conn, proc = self.conn, self.proc
        self.proc = self.conn = None
        self.ready, self.inflight, self._boot = False, None, None
        if conn is not None:
            try:
                conn.send(("stop", None))
            except (OSError, ValueError):
                pass
            conn.close()
        if proc is not None:
            threading.Thread(target=_reap, args=(proc,), name="myrmex-brain-reap", daemon=True).start()
        self.adapter.release()

    def configure(self, cfg: BrainConfig) -> None:
        was = self.cfg.enabled
        self.cfg = cfg
        if not cfg.enabled:
            if was:
                self.close()
            return
        self.gave_up, self.fails = False, []
        if self.process and self.conn is not None:
            try:
                self.conn.send(("config", cfg.to_dict()))
            except (OSError, ValueError):
                self._fail(time.perf_counter(), "worker gone")
        elif self.core is not None:
            from .core import _client
            self.core.cfg, self.core.kev = cfg, _client(cfg)
            self.core.load_taste()
        elif not was or (self.process and self.conn is None):
            self._start()

    def _wait(self) -> float:
        """How long an answer may take: ``timeout``, or longer when Kev is asked and its own timeout is longer."""
        from .core import KEV_MODES
        return max(self.timeout, self.cfg.kev_timeout + 0.7) if self.cfg.mode in KEV_MODES else self.timeout

    def mark(self, good: bool) -> None:
        if self.core is not None:
            self.core.mark(good)
        elif self.conn is not None:
            try:
                self.conn.send(("mark", bool(good)))
            except (OSError, ValueError):
                pass

    # ------------------------------------------------------------------ the tick
    def tick(self, now: float, t: float, state=None, glove_state=None, glove_ctrl=None, controls: dict | None = None,
             music: str = "") -> Decision | None:
        """Every engine tick; microseconds unless a snapshot is due. -> the decision applied, if any."""
        if not self.cfg.enabled or self.gave_up:
            return None
        if now - self._house < HOUSEKEEPING_S and t < self.next_t and self.inflight is None:
            return None                     # (restores, liveness: 30 times a second; an awaited answer: every tick)
        self._house = now
        t0 = time.perf_counter()
        applied = None
        try:
            if self.process and self.proc is None:
                if now >= self.retry_at and not self.error.startswith("the brain worker could not start"):
                    self.stats["restarts"] += 1
                    self._start()
                return None
            self.adapter.maintain(t)
            if self.process:
                applied = self._receive(now, t)
                if self.proc is not None and self.proc.poll() is not None:
                    self._fail(now, f"the brain worker stopped (exit {self.proc.returncode})")
                    return None
                if self.inflight is not None and now - self.inflight > self._wait():
                    self.stats["timeouts"] += 1                    # dropped: the engine goes on by itself
                    self.inflight = None
                if not self.ready:                                 # booting (imports: ~1 s)
                    self._boot = now if self._boot is None else self._boot
                    if now - self._boot > 20.0:
                        self._fail(now, "the brain worker did not start")
                        return None
            if t >= self.next_t and self.inflight is None and self.ready:
                rate = max(0.1, self.cfg.controls.merged(controls).rate_hz)
                self.next_t = t + 1.0 / rate
                payload = self._payload(t, state, glove_state, glove_ctrl, controls, music, 1.0 / rate)
                if self.process:
                    try:
                        self.conn.send(("snap", payload))
                        self.inflight = now
                        self.stats["sent"] += 1
                    except (OSError, ValueError):
                        self._fail(now, "the brain worker is gone")
                else:
                    applied = self._inline(t, payload)
        except Exception as e:                                      # a bug in the brain never stops the organism
            self._fail(now, f"{type(e).__name__}: {e}")
            applied = None
        finally:
            self.stats["tick_us"].append((time.perf_counter() - t0) * 1e6)
            del self.stats["tick_us"][:-600]
        return applied

    def _payload(self, t, state, glove_state, glove_ctrl, controls, music, lookahead) -> dict:
        ad = self.adapter
        b = ad._body()
        X = ad._points(state)
        if len(X) > 160:                                            # geometry needs the shape, not every node
            X = X[:: int(np.ceil(len(X) / 160))]
        from .adapter import blend_of, softmax3
        inp = self.engine.inp
        user = user_cue(glove_state, glove_ctrl, time.perf_counter())
        if hasattr(b, "z"):
            w = softmax3(np.asarray(b.z, float))
            blend = blend_of(b.z_goal, ad.vocab.forms)
        else:                                                       # v1: from its morphology vector
            snap = ad.snapshot(t, state, user, controls or {}, music, lookahead)
            w, blend = snap.fp.w, snap.blend
        return {"t": float(t), "w": np.asarray(w, float), "mat": np.asarray(getattr(b, "mat", np.zeros(0)), float),
                "X": np.asarray(X, np.float32), "bodies": ad._bodies(), "blend": blend, "free": ad.free(),
                "due": ad.due(lookahead) if hasattr(b, "z") else False, "energy": float(inp.energy),
                "playing": bool(inp.playing), "music": music, "user": user,
                "controls": {k: v for k, v in (controls or {}).items() if k.startswith("brain_")}}

    def _inline(self, t: float, payload: dict) -> Decision | None:
        from .fingerprint import Fingerprint, geometry
        X = payload.pop("X")
        fp = Fingerprint(payload.pop("w"), geometry(X) if len(X) >= 4 else np.zeros(8), payload.pop("mat"),
                         payload.pop("bodies"))
        d = self.core.step(Snapshot(fp=fp, **payload))
        self.last_summary = self.core.memory.summary(t, self.adapter.vocab.forms)
        self.worker_stats = {k: v for k, v in self.core.stats.items() if not isinstance(v, list)}
        if d is not None and self.adapter.free():
            self.adapter.apply(d, t)
            self.stats["applied"] += 1
            self.last = d.to_dict()
        return d

    def _receive(self, now: float, t: float) -> Decision | None:
        applied = None
        while self.conn is not None and self.conn.poll():
            try:
                kind, payload = self.conn.recv()
            except (EOFError, OSError):
                self._fail(now, "the brain worker stopped")
                return None
            if kind == "ready":
                self.ready = True
                continue
            if kind == "error":
                self.error = str(payload)[-300:]
                self.stats["errors"] += 1
                self.inflight = None
                continue
            if kind != "decision":
                continue
            self.stats["answered"] += 1
            if self.inflight is not None:
                self.stats["rtt_ms"].append((now - self.inflight) * 1000.0)
                del self.stats["rtt_ms"][:-300]
            self.inflight = None
            self.stats["worker_ms"].append(float(payload.get("ms", 0.0)))
            del self.stats["worker_ms"][:-300]
            self.last_summary = payload.get("summary", {})
            self.worker_stats = payload.get("stats", {})
            d = payload.get("d")
            if d is None:
                continue
            if t - float(payload.get("sent", t)) > self._wait() + 0.5 or not self.adapter.free():
                self.stats["stale"] += 1                           # too late, or the organism is busy now
                continue
            try:
                dec = Decision(**d)
            except TypeError:
                self.stats["errors"] += 1
                continue
            self.adapter.apply(dec, t)
            self.stats["applied"] += 1
            self.last = d
            applied = dec
        return applied

    # ------------------------------------------------------------------ reading
    def status(self) -> dict:
        st = self.stats
        def med(x):
            return round(float(np.median(x)), 2) if x else None
        return {"enabled": self.cfg.enabled, "mode": self.cfg.mode, "running": self.ready and not self.gave_up,
                "gave_up": self.gave_up, "error": self.error, "sent": st["sent"], "applied": st["applied"],
                "timeouts": st["timeouts"], "stale": st["stale"], "restarts": st["restarts"],
                "rtt_ms": med(st["rtt_ms"]), "worker_ms": med(st["worker_ms"]),
                "tick_us_p99": round(float(np.percentile(st["tick_us"], 99)), 1) if st["tick_us"] else None,
                "last": {k: self.last.get(k) for k in ("op", "label", "source", "confidence")} if self.last else {},
                "memory": getattr(self, "last_summary", {}),
                "kev": {k: v for k, v in getattr(self, "worker_stats", {}).items() if k.startswith("kev")}}


__all__ = ["BrainLink"]


def _reap(proc: subprocess.Popen) -> None:
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _main() -> None:
    """``python -m myrmex.brain.worker`` (started by BrainLink): its end of the socket pair, then decisions."""
    conn = Connection(int(os.environ.pop("MYRMEX_BRAIN_FD")))
    _kind, ref, organism, cfg = conn.recv()
    _worker_main(conn, tuple(ref), organism, cfg)


if __name__ == "__main__":
    _main()
