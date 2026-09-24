"""Minimal audio file reading (WAV PCM/float, AIFF/AIFC PCM) with numpy only."""
from __future__ import annotations

import struct

import numpy as np


def _pcm_to_float(raw: bytes, width: int, big_endian: bool = False) -> np.ndarray:
    if width == 1:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (a - 128.0) / 128.0
    if width == 2:
        a = np.frombuffer(raw, dtype=">i2" if big_endian else "<i2").astype(np.float32)
        return a / 32768.0
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        if big_endian:
            v = (b[:, 0] << 16) | (b[:, 1] << 8) | b[:, 2]
        else:
            v = (b[:, 2] << 16) | (b[:, 1] << 8) | b[:, 0]
        v = np.where(v & 0x800000, v - 0x1000000, v)
        return v.astype(np.float32) / 8388608.0
    if width == 4:
        a = np.frombuffer(raw, dtype=">i4" if big_endian else "<i4").astype(np.float64)
        return (a / 2147483648.0).astype(np.float32)
    raise ValueError(f"unsupported sample width {width}")


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    pos = 12
    fmt = None
    frames = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, sr, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, sr, bits)
        elif cid == b"data":
            frames = body
        pos += 8 + size + (size & 1)
    if fmt is None or frames is None:
        raise ValueError("WAV without fmt/data")
    tag, ch, sr, bits = fmt
    if tag == 3:
        x = np.frombuffer(frames, dtype="<f4" if bits == 32 else "<f8").astype(np.float32)
    else:
        x = _pcm_to_float(frames, bits // 8)
    n = len(x) // ch
    return x[: n * ch].reshape(n, ch), sr


def _ieee_extended(b: bytes) -> float:
    exp = ((b[0] & 0x7F) << 8) | b[1]
    mant = int.from_bytes(b[2:10], "big")
    if exp == 0 and mant == 0:
        return 0.0
    return mant * 2.0 ** (exp - 16383 - 63) * (-1 if b[0] & 0x80 else 1)


def read_aiff(path: str) -> tuple[np.ndarray, int]:
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"FORM" or data[8:12] not in (b"AIFF", b"AIFC"):
        raise ValueError("not an AIFF file")
    pos = 12
    ch = sr = bits = None
    frames = None
    comp = b"NONE"
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"COMM":
            ch, _, bits = struct.unpack(">hIh", body[:8])
            sr = int(_ieee_extended(body[8:18]))
            if data[8:12] == b"AIFC" and len(body) >= 22:
                comp = body[18:22]
        elif cid == b"SSND":
            offset = struct.unpack(">I", body[:4])[0]
            frames = body[8 + offset:]
        pos += 8 + size + (size & 1)
    if frames is None or ch is None:
        raise ValueError("AIFF without COMM/SSND")
    if comp in (b"sowt",):
        x = _pcm_to_float(frames, bits // 8, big_endian=False)
    elif comp in (b"fl32", b"FL32"):
        x = np.frombuffer(frames, dtype=">f4").astype(np.float32)
    else:
        x = _pcm_to_float(frames, bits // 8, big_endian=True)
    n = len(x) // ch
    return x[: n * ch].reshape(n, ch), sr


def read_audio(path: str) -> tuple[np.ndarray, int]:
    low = path.lower()
    if low.endswith((".aif", ".aiff", ".aifc")):
        return read_aiff(path)
    return read_wav(path)
