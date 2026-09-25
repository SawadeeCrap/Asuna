"""Creature state stream (UDP, same port as poses): high-level state only - Blender builds the surface.

    b"MYRC" <B ver> <B flags> <H n> <I seq> <d t> <d beat> <f bpm> <B behavior> <B morph> <H pad>
    <f surface> <f glow> <f arousal> <f instability> <3f com> <f heading>
    pos n*3f, radius n*f, stretch n*3f, kind n*B, anchor n*h, [camera block as in realtime.protocol]
    [FLAG_POLY: <B material> <B fragments> <H links> <H obstacles> <H pad> dispersion n*f,
                links L*(2H + f strength), obstacles O*4f (x y z radius)]
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ..realtime.protocol import _CAM, SHOT_KINDS, CameraState
from .behavior import STATES
from .morphology import MORPHS
from .polyalloy import ATTRACTORS, INTENTS, MATERIAL

MATERIAL_NAMES = tuple(MATERIAL)

MAGIC = b"MYRC"
VERSION = 1
FLAG_PLAYING, FLAG_CAMERA, FLAG_DEBUG, FLAG_POLY = 1, 4, 16, 32
_POLY = struct.Struct("<BBHHH")
_LINK = np.dtype([("i", "<u2"), ("j", "<u2"), ("s", "<f4")])
_HDR = struct.Struct("<4sBBHIddfBBHffff3ff")
MORPH_NAMES = tuple(MORPHS) + tuple(a for a in ATTRACTORS if a not in MORPHS)
BEHAVIOR_NAMES = tuple(STATES) + tuple(i for i in INTENTS if i not in STATES)


@dataclass
class CreatureFrame:
    seq: int
    t: float
    beat: float
    bpm: float
    behavior: str
    morphology: str
    surface: float
    glow: float
    arousal: float
    instability: float
    com: np.ndarray
    heading: float
    pos: np.ndarray
    radius: np.ndarray
    stretch: np.ndarray
    kind: np.ndarray
    anchor: np.ndarray
    flags: int = 0
    camera: CameraState | None = None
    material: str = ""
    fragments: int = 1
    dispersion: np.ndarray | None = None
    links: np.ndarray | None = None          # (L, 3) i, j, strength
    obstacles: np.ndarray | None = None      # (O, 4) x, y, z, radius


def encode_creature(st, seq: int, beat: float, bpm: float, flags: int = 0, camera: CameraState | None = None) -> bytes:
    n = len(st.pos)
    poly = getattr(st, "links", None) is not None
    fl = flags | (FLAG_CAMERA if camera is not None else 0) | (FLAG_POLY if poly else 0)
    out = [_HDR.pack(MAGIC, VERSION, fl, n, seq & 0xFFFFFFFF, st.t, beat, bpm,
                     BEHAVIOR_NAMES.index(st.behavior) if st.behavior in BEHAVIOR_NAMES else 0,
                     MORPH_NAMES.index(st.morphology) if st.morphology in MORPH_NAMES else 0, 0, st.surface, st.glow,
                     st.arousal, st.instability, *map(float, st.com), float(st.heading)),
           np.ascontiguousarray(st.pos, "<f4").tobytes(), np.ascontiguousarray(st.radius, "<f4").tobytes(),
           np.ascontiguousarray(st.stretch, "<f4").tobytes(), np.ascontiguousarray(st.kind, "u1").tobytes(),
           np.ascontiguousarray(st.anchor, "<i2").tobytes()]
    if camera is not None:
        c = camera
        k = SHOT_KINDS.index(c.kind) if c.kind in SHOT_KINDS else len(SHOT_KINDS) - 1
        out.append(_CAM.pack(*map(float, c.position), *map(float, c.target), float(c.lens), float(c.focus),
                             float(c.fstop), c.shot_id & 0xFFFF, k))
    if poly:
        lk = st.links[st.links[:, 0] >= 0]
        ob = st.obstacles                                   # fixed slots (radius 0 = empty)
        out.append(_POLY.pack(MATERIAL_NAMES.index(st.material) if st.material in MATERIAL_NAMES else 0,
                              min(255, int(st.fragments)), len(lk), len(ob), 0))
        out.append(np.ascontiguousarray(st.dispersion, "<f4").tobytes())
        la = np.zeros(len(lk), _LINK)
        la["i"], la["j"], la["s"] = lk[:, 0], lk[:, 1], lk[:, 2]
        out.append(la.tobytes())
        out.append(np.ascontiguousarray(ob, "<f4").tobytes())
    return b"".join(out)


def decode_creature(data: bytes) -> CreatureFrame | None:
    if len(data) < _HDR.size or data[:4] != MAGIC:
        return None
    v = _HDR.unpack_from(data, 0)
    if v[1] != VERSION:
        return None
    flags, n = v[2], v[3]
    off = _HDR.size

    def take(dtype, count, shape=None):
        nonlocal off
        a = np.frombuffer(data, dtype=dtype, count=count, offset=off)
        off += a.nbytes
        return a.reshape(shape).astype(float) if shape else a.copy()
    try:
        pos, rad = take("<f4", 3 * n, (n, 3)), take("<f4", n, (n,))
        stretch, kind, anchor = take("<f4", 3 * n, (n, 3)), take("u1", n), take("<i2", n)
    except ValueError:
        return None
    cam = None
    if flags & FLAG_CAMERA and len(data) >= off + _CAM.size:
        c = _CAM.unpack_from(data, off)
        off += _CAM.size
        cam = CameraState(np.array(c[0:3]), np.array(c[3:6]), c[6], c[7], c[8], c[9],
                          SHOT_KINDS[c[10]] if c[10] < len(SHOT_KINDS) else "free")
    fr = CreatureFrame(v[4], v[5], v[6], v[7], BEHAVIOR_NAMES[v[8]] if v[8] < len(BEHAVIOR_NAMES) else "REST",
                       MORPH_NAMES[v[9]] if v[9] < len(MORPH_NAMES) else "COMPACT", v[11], v[12], v[13], v[14],
                       np.array(v[15:18]), v[18], pos, rad, stretch, kind, anchor, flags, cam)
    if flags & FLAG_POLY and len(data) >= off + _POLY.size:
        m, frag, nl, no, _ = _POLY.unpack_from(data, off)
        off += _POLY.size
        try:
            fr.dispersion = take("<f4", n, (n,))
            la = np.frombuffer(data, _LINK, count=nl, offset=off)
            off += la.nbytes
            fr.links = np.stack([la["i"], la["j"], la["s"]], 1).astype(float)
            fr.obstacles = take("<f4", 4 * no, (no, 4)) if no else np.zeros((0, 4))
        except ValueError:
            return fr
        fr.material, fr.fragments = MATERIAL_NAMES[m] if m < len(MATERIAL_NAMES) else "", frag
    return fr
