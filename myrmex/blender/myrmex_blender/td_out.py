"""Recorded takes -> TouchDesigner: while Blender plays a take, TD gets the same channels it saw live.

The app sets ``MYRMEX_TD='{"host": "127.0.0.1", "port": 7000}'`` when it opens a take with the
TouchDesigner link on; the take player then sends the recorded ``td`` channels (realtime/touch.py) of the
current frame as one OSC bundle - the effects follow the take exactly as they followed the live show.
"""
from __future__ import annotations

import json
import math
import os
import socket
import struct

_S: dict = {"sock": None, "addr": None, "names": None, "prefix": None}


def target() -> tuple[str, int] | None:
    raw = os.environ.get("MYRMEX_TD")
    if not raw:
        return None
    try:
        d = json.loads(raw)
        return str(d.get("host", "127.0.0.1")), int(d.get("port", 7000))
    except (ValueError, TypeError):
        return None


def send(names, values) -> bool:
    addr = target()
    if addr is None or names is None or values is None:
        return False
    from myrmex.bus.osc import _osc_string
    names = [str(n) for n in names]
    if _S["names"] != names:
        _S["names"] = names
        _S["prefix"] = [_osc_string("/myrmex/" + n) + _osc_string(",f") for n in names]
    if _S["sock"] is None:
        _S["sock"] = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    parts = [b"#bundle\x00", struct.pack(">Q", 1)]
    for pre, x in zip(_S["prefix"], values):
        x = float(x)
        m = pre + struct.pack(">f", x if math.isfinite(x) else 0.0)
        parts += [struct.pack(">i", len(m)), m]
    try:
        _S["sock"].sendto(b"".join(parts), addr)
    except OSError:
        return False
    return True


__all__ = ["send", "target"]
