"""Pose stream: the realtime engine -> any renderer (Blender live link, game engines, monitors).

One UDP datagram per rendered frame (~1.2 KB for a 22-bone biped), little endian::

    header  <4s B B H I d d f I>   magic b"MYRP", version, flags, n_bones, seq,
                                   engine time (s), song beat, bpm, rig id (crc32 of bone names)
    deltas  n_bones x 12 float32   rest-world deltas D (rows 0..2 of the 4x4):
                                   posed_world(bone) = D @ rest_world(bone)
    camera  <3f 3f f f f H H>      position, look-at target, lens (mm), focus (m), f-stop,
                                   shot id, shot kind index            (flag CAMERA)
    subject <3f f f f>             pelvis position, heading (rad), speed (m/s), energy (0..1)

Bone names travel separately (b"MYRN" packets, sent once per second) so a
receiver can map indices to its own skeleton and verify the rig id.  Deltas
are renderer-neutral: a receiver converts them to local transforms with its
own rest matrices (see ``myrmex_blender.live.BasisSolver``).
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

POSE_MAGIC = b"MYRP"
NAMES_MAGIC = b"MYRN"
VERSION = 1

FLAG_PLAYING = 1
FLAG_RECORDING = 2
FLAG_CAMERA = 4
FLAG_HOLD = 8

_HDR = struct.Struct("<4sBBHIddfI")
_CAM = struct.Struct("<3f3ffffHH")
_SUBJ = struct.Struct("<3ffff")

# Large frames (creatures with many nodes, particle swarms) are cut into fragments: macOS refuses UDP
# datagrams above net.inet.udp.maxdgram (9216 bytes by default) - a bigger frame would silently vanish.
FRAG_MAGIC = b"MYRK"
_FRAG = struct.Struct("<4sBBHHI")          # magic, version, pad, index, count, message id
MAX_DGRAM = 8192


def fragment(data: bytes, msg_id: int, max_size: int = MAX_DGRAM) -> list[bytes]:
    if len(data) <= max_size:
        return [data]
    step = max_size - _FRAG.size
    parts = [data[i:i + step] for i in range(0, len(data), step)]
    return [_FRAG.pack(FRAG_MAGIC, 1, 0, k, len(parts), msg_id & 0xFFFFFFFF) + p for k, p in enumerate(parts)]


class Reassembler:
    """Receiver side of :func:`fragment`: whole packets pass through, fragments are joined."""

    def __init__(self):
        self.msg: int | None = None
        self.parts: dict[int, bytes] = {}
        self.count = 0

    def feed(self, data: bytes) -> bytes | None:
        if data[:4] != FRAG_MAGIC:
            return data
        if len(data) < _FRAG.size:
            return None
        _, _ver, _, idx, count, mid = _FRAG.unpack_from(data, 0)
        if mid != self.msg:                   # a newer frame started: an unfinished older one is dropped
            self.msg, self.parts, self.count = mid, {}, count
        self.parts[idx] = data[_FRAG.size:]
        if len(self.parts) < self.count:
            return None
        out = b"".join(self.parts[i] for i in range(self.count) if i in self.parts)
        self.msg, self.parts = None, {}
        return out


SHOT_KINDS = ("front_dolly", "front_low", "side_track", "three_quarter", "rear_follow", "feet_close",
              "hips_close", "face_close", "wide_orbit", "free",
              # aerial camera (Mimetic Polyalloy)
              "observe", "follow", "approach", "retreat", "orbit", "lock", "track", "impact", "recovery")


def rig_id(names: list[str]) -> int:
    return zlib.crc32("\n".join(names).encode("utf-8")) & 0xFFFFFFFF


@dataclass
class CameraState:
    position: np.ndarray
    target: np.ndarray
    lens: float = 50.0
    focus: float = 4.0
    fstop: float = 4.0
    shot_id: int = 0
    kind: str = "front_dolly"


@dataclass
class PoseFrame:
    seq: int
    time: float
    beat: float
    bpm: float
    rig: int
    deltas: np.ndarray                   # (B, 4, 4) float
    flags: int = 0
    camera: CameraState | None = None
    subject_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading: float = 0.0
    speed: float = 0.0
    energy: float = 0.0

    @property
    def playing(self) -> bool:
        return bool(self.flags & FLAG_PLAYING)


def encode_pose(fr: PoseFrame) -> bytes:
    flags = fr.flags | (FLAG_CAMERA if fr.camera is not None else 0)
    B = fr.deltas.shape[0]
    out = [_HDR.pack(POSE_MAGIC, VERSION, flags & 0xFF, B, fr.seq & 0xFFFFFFFF, fr.time, fr.beat, fr.bpm,
                     fr.rig & 0xFFFFFFFF)]
    out.append(np.ascontiguousarray(fr.deltas[:, :3, :4], dtype="<f4").tobytes())
    if fr.camera is not None:
        c = fr.camera
        kind = SHOT_KINDS.index(c.kind) if c.kind in SHOT_KINDS else len(SHOT_KINDS) - 1
        out.append(_CAM.pack(*map(float, c.position), *map(float, c.target), float(c.lens), float(c.focus),
                             float(c.fstop), c.shot_id & 0xFFFF, kind))
    out.append(_SUBJ.pack(*map(float, fr.subject_pos), float(fr.heading), float(fr.speed), float(fr.energy)))
    return b"".join(out)


def decode_pose(data: bytes) -> PoseFrame | None:
    if len(data) < _HDR.size or data[:4] != POSE_MAGIC:
        return None
    magic, ver, flags, B, seq, t, beat, bpm, rid = _HDR.unpack_from(data, 0)
    if ver != VERSION:
        return None
    off = _HDR.size
    n = B * 12 * 4
    if len(data) < off + n:
        return None
    m = np.frombuffer(data, dtype="<f4", count=B * 12, offset=off).reshape(B, 3, 4).astype(float)
    deltas = np.zeros((B, 4, 4))
    deltas[:, :3, :] = m
    deltas[:, 3, 3] = 1.0
    off += n
    cam = None
    if flags & FLAG_CAMERA and len(data) >= off + _CAM.size:
        v = _CAM.unpack_from(data, off)
        off += _CAM.size
        cam = CameraState(np.array(v[0:3]), np.array(v[3:6]), v[6], v[7], v[8], v[9],
                          SHOT_KINDS[v[10]] if v[10] < len(SHOT_KINDS) else "free")
    pos, heading, speed, energy = np.zeros(3), 0.0, 0.0, 0.0
    if len(data) >= off + _SUBJ.size:
        v = _SUBJ.unpack_from(data, off)
        pos, heading, speed, energy = np.array(v[0:3]), v[3], v[4], v[5]
    return PoseFrame(seq, t, beat, bpm, rid, deltas, flags, cam, pos, heading, speed, energy)


def encode_names(names: list[str]) -> bytes:
    body = "\n".join(names).encode("utf-8")
    return NAMES_MAGIC + struct.pack("<BI", VERSION, rig_id(names)) + body


def decode_names(data: bytes) -> tuple[int, list[str]] | None:
    if len(data) < 9 or data[:4] != NAMES_MAGIC:
        return None
    ver, rid = struct.unpack_from("<BI", data, 4)
    if ver != VERSION:
        return None
    return rid, data[9:].decode("utf-8").split("\n")


__all__ = ["PoseFrame", "CameraState", "encode_pose", "decode_pose", "encode_names", "decode_names", "rig_id",
           "FLAG_PLAYING", "FLAG_RECORDING", "FLAG_CAMERA", "FLAG_HOLD", "SHOT_KINDS"]
