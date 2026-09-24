"""Minimal OSC 1.0 codec (messages + bundles), pure Python – no dependencies.

Supported argument types: i (int32), f (float32), d (float64), h (int64),
s (string), b (blob), T/F (bools), N (nil).  Enough for every message the
Myrmex bridges send, and compatible with python-osc, Max [udpsend], VCV
modules and AbletonOSC.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


def _pad(b: bytes) -> bytes:
    return b + b"\x00" * ((4 - len(b) % 4) % 4)


def _osc_string(s: str) -> bytes:
    return _pad(s.encode("utf-8") + b"\x00")


@dataclass
class OscMessage:
    address: str
    args: list = field(default_factory=list)

    def encode(self) -> bytes:
        tags = ","
        payload = b""
        for a in self.args:
            if isinstance(a, bool):
                tags += "T" if a else "F"
            elif a is None:
                tags += "N"
            elif isinstance(a, int):
                if -2 ** 31 <= a < 2 ** 31:
                    tags += "i"
                    payload += struct.pack(">i", a)
                else:
                    tags += "h"
                    payload += struct.pack(">q", a)
            elif isinstance(a, float):
                tags += "f"
                payload += struct.pack(">f", a)
            elif isinstance(a, str):
                tags += "s"
                payload += _osc_string(a)
            elif isinstance(a, (bytes, bytearray)):
                tags += "b"
                payload += struct.pack(">i", len(a)) + _pad(bytes(a))
            else:
                raise TypeError(f"unsupported OSC argument {type(a)}")
        return _osc_string(self.address) + _osc_string(tags) + payload


def _read_string(data: bytes, pos: int) -> tuple[str, int]:
    end = data.index(b"\x00", pos)
    s = data[pos:end].decode("utf-8", "replace")
    return s, (end + 4) & ~3


def decode_message(data: bytes) -> OscMessage:
    addr, pos = _read_string(data, 0)
    if pos >= len(data):
        return OscMessage(addr, [])
    tags, pos = _read_string(data, pos)
    args = []
    for t in tags[1:]:
        if t == "i":
            args.append(struct.unpack(">i", data[pos:pos + 4])[0])
            pos += 4
        elif t == "f":
            args.append(struct.unpack(">f", data[pos:pos + 4])[0])
            pos += 4
        elif t == "d":
            args.append(struct.unpack(">d", data[pos:pos + 8])[0])
            pos += 8
        elif t == "h":
            args.append(struct.unpack(">q", data[pos:pos + 8])[0])
            pos += 8
        elif t == "s":
            s, pos = _read_string(data, pos)
            args.append(s)
        elif t == "b":
            n = struct.unpack(">i", data[pos:pos + 4])[0]
            args.append(data[pos + 4:pos + 4 + n])
            pos += 4 + ((n + 3) & ~3)
        elif t == "T":
            args.append(True)
        elif t == "F":
            args.append(False)
        elif t == "N":
            args.append(None)
        else:
            raise ValueError(f"unsupported OSC type tag {t!r}")
    return OscMessage(addr, args)


def decode(data: bytes) -> list[OscMessage]:
    """Decode a packet (message or bundle, recursively) into a flat message list."""
    if data.startswith(b"#bundle\x00"):
        out: list[OscMessage] = []
        pos = 16
        while pos + 4 <= len(data):
            n = struct.unpack(">i", data[pos:pos + 4])[0]
            out.extend(decode(data[pos + 4:pos + 4 + n]))
            pos += 4 + n
        return out
    return [decode_message(data)]


def encode_bundle(messages: list[OscMessage], timetag: int = 1) -> bytes:
    body = b"#bundle\x00" + struct.pack(">Q", timetag)
    for m in messages:
        e = m.encode()
        body += struct.pack(">i", len(e)) + e
    return body
