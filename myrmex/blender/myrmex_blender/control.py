"""Commands from the Myrmex app to the Blender it started: save / load looks from the app's Character page.

The app writes one JSON object per line to Blender's standard input; a reader thread queues them and a
timer runs them on Blender's main thread.  Replies go to standard output (which the app reads anyway)
as ``MYRMEX_REPLY {json}``.  Only switched on when the app starts Blender (``MYRMEX_CONTROL=stdin``).

    {"cmd": "ping"}
    {"cmd": "save_look", "path": ".../looks/cyber_hive/Neon.blend", "kind": "cyber_hive", "id": 3}
    {"cmd": "load_look", "path": ".../looks/cyber_hive/Neon.blend" | "", "kind": "cyber_hive", "keep": true}
"""
from __future__ import annotations

import json
import os
import queue
import threading

import bpy

REPLY = "MYRMEX_REPLY "
_Q: queue.Queue = queue.Queue()
_STATE = {"thread": None}


def enabled() -> bool:
    return os.environ.get("MYRMEX_CONTROL") == "stdin"


def start() -> None:
    """Listen to the app (idempotent)."""
    if _STATE["thread"] is None:
        t = threading.Thread(target=_reader, name="myrmex-control", daemon=True)
        t.start()
        _STATE["thread"] = t
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=0.25, persistent=True)
    reply({"cmd": "hello", "ok": True})


def _reader(fd: int = 0) -> None:
    buf = b""
    while True:
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            return
        if not chunk:                                  # the app went away
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(cmd, dict):
                _Q.put(cmd)


def _poll():
    while True:
        try:
            cmd = _Q.get_nowait()
        except queue.Empty:
            break
        reply(handle(cmd))
    return 0.25


def reply(d: dict) -> None:
    print(REPLY + json.dumps(d), flush=True)


def handle(cmd: dict) -> dict:
    """Run one command on the main thread -> the reply."""
    from . import looks
    name = cmd.get("cmd", "")
    out = {"cmd": name, "id": cmd.get("id")}
    try:
        if name == "ping":
            out.update(ok=True, kind=looks.look_kind(), file=bpy.data.filepath)
        elif name == "save_look":
            kind = looks.look_kind()
            want = cmd.get("kind")
            if want and kind != want and not (want == "creature" and kind == "nanomaterial"):
                raise ValueError(f"Blender shows {kind}, not {want}")
            path = looks.save_look(bpy.context, path=cmd.get("path") or None, name=cmd.get("name") or None)
            out.update(ok=True, kind=kind, path=path)
        elif name == "load_look":
            kind = cmd.get("kind") or looks.look_kind()
            kind = {"creature": "nanomaterial"}.get(kind, kind)
            if kind not in looks.CREATURES:
                raise ValueError("looks are for organisms (the humanoid keeps its look in its .blend)")
            out.update(ok=True, **looks.load_look(cmd.get("path") or "", kind, bool(cmd.get("keep", True))))
        else:
            raise ValueError(f"unknown command {name!r}")
    except Exception as e:                             # the app shows it; Blender keeps running
        out.update(ok=False, error=f"{type(e).__name__}: {e}")
    return out


__all__ = ["start", "enabled", "handle", "reply", "REPLY"]
