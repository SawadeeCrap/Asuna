"""Standard MIDI File reader (format 0/1, pure Python) -> MusicTimeline.

Useful for MIDI exported from Ableton ("Export MIDI Clip"), recorded from VCV
Rack (CV-MIDI -> any recorder) or any DAW.  Tempo and time-signature meta
events build the tempo map; tracks are named from track-name events; channel
10 (index 9) is treated as drums with General-MIDI note groups.
"""
from __future__ import annotations

import os
import struct

from ..music.classify import apply_classification, classify_drum_note
from ..music.timeline import Marker, MusicTimeline, NoteEvent, TempoMap, Track


def _read_varlen(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos


def parse_smf(data: bytes) -> tuple[int, list[list[tuple[int, str, dict]]]]:
    if data[:4] != b"MThd":
        raise ValueError("not a Standard MIDI File")
    hlen = struct.unpack(">I", data[4:8])[0]
    fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    pos = 8 + hlen
    tracks = []
    for _ in range(ntracks):
        if data[pos:pos + 4] != b"MTrk":
            break
        tlen = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        p = pos + 8
        end = p + tlen
        tick = 0
        status = 0
        events = []
        while p < end:
            delta, p = _read_varlen(data, p)
            tick += delta
            b = data[p]
            if b == 0xFF:
                mtype = data[p + 1]
                length, p = _read_varlen(data, p + 2)
                body = data[p:p + length]
                p += length
                if mtype == 0x51 and length == 3:
                    events.append((tick, "tempo", {"uspq": int.from_bytes(body, "big")}))
                elif mtype == 0x58 and length >= 2:
                    events.append((tick, "timesig", {"num": body[0], "den": 2 ** body[1]}))
                elif mtype in (0x03, 0x04):
                    events.append((tick, "name", {"text": body.decode("latin-1", "replace")}))
                elif mtype == 0x06:
                    events.append((tick, "marker", {"text": body.decode("latin-1", "replace")}))
                continue
            if b in (0xF0, 0xF7):
                length, p = _read_varlen(data, p + 1)
                p += length
                continue
            if b & 0x80:
                status = b
                p += 1
            kind = status & 0xF0
            ch = status & 0x0F
            if kind in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                d1, d2 = data[p], data[p + 1]
                p += 2
                if kind == 0x90 and d2 > 0:
                    events.append((tick, "on", {"ch": ch, "note": d1, "vel": d2}))
                elif kind == 0x80 or (kind == 0x90 and d2 == 0):
                    events.append((tick, "off", {"ch": ch, "note": d1}))
                elif kind == 0xB0:
                    events.append((tick, "cc", {"ch": ch, "cc": d1, "value": d2}))
            elif kind in (0xC0, 0xD0):
                p += 1
            else:
                p += 1
        tracks.append(events)
        pos = end
    return division, tracks


def load_midi(path: str, overrides: dict[str, str] | None = None) -> MusicTimeline:
    with open(path, "rb") as fh:
        division, tracks = parse_smf(fh.read())
    tempo_ev = sorted((t, e["uspq"]) for tr in tracks for (t, k, e) in tr if k == "tempo")
    ts_ev = sorted((t, e["num"], e["den"]) for tr in tracks for (t, k, e) in tr if k == "timesig")
    points = [(t / division, 60_000_000.0 / u) for t, u in tempo_ev] or [(0.0, 120.0)]
    sigs = [(t / division, n, d) for t, n, d in ts_ev] or [(0.0, 4, 4)]
    tl = MusicTimeline(TempoMap(points, sigs), source=f"midi:{os.path.basename(path)}")
    for ti, tr in enumerate(tracks):
        name = next((e["text"] for (_, k, e) in tr if k == "name"), f"track {ti}")
        pending: dict[tuple[int, int], list[tuple[int, int]]] = {}
        channels = set()
        notes_here = []
        for tick, kind, e in tr:
            if kind == "on":
                pending.setdefault((e["ch"], e["note"]), []).append((tick, e["vel"]))
                channels.add(e["ch"])
            elif kind == "off":
                lst = pending.get((e["ch"], e["note"]))
                if lst:
                    t0, vel = lst.pop(0)
                    notes_here.append((t0, tick, e["ch"], e["note"], vel))
            elif kind == "marker":
                tl.markers.append(Marker(tl.tempo.seconds(tick / division), e["text"]))
        for (ch, note), lst in pending.items():
            for t0, vel in lst:
                notes_here.append((t0, t0 + division // 4, ch, note, vel))
        if not notes_here:
            continue
        tid = f"t{ti}"
        drums = channels == {9}
        tl.tracks[tid] = Track(tid, name, "midi", devices=["drums"] if drums else [])
        for t0, t1, ch, note, vel in notes_here:
            b0, b1 = t0 / division, t1 / division
            s0 = tl.tempo.seconds(b0)
            grp = classify_drum_note(note, None) if ch == 9 else None
            tl.notes.append(NoteEvent(s0, max(1e-3, tl.tempo.seconds(b1) - s0), float(note), vel / 127.0, tid,
                                      beat=b0, group=grp, sharpness=0.8 if grp else 0.5))
    apply_classification(tl, overrides)
    return tl.finalize()


def write_midi(tl: MusicTimeline, path: str, division: int = 480) -> None:
    """Write a timeline back to a format-1 SMF (handy for tests and exporting synthetic material)."""
    def varlen(v: int) -> bytes:
        out = [v & 0x7F]
        v >>= 7
        while v:
            out.append((v & 0x7F) | 0x80)
            v >>= 7
        return bytes(reversed(out))

    chunks = []
    meta = bytearray()
    last = 0
    for b, bpm in tl.tempo.points:
        tick = int(round(b * division))
        meta += varlen(tick - last) + b"\xFF\x51\x03" + int(60_000_000 / bpm).to_bytes(3, "big")
        last = tick
    meta += varlen(0) + b"\xFF\x2F\x00"
    chunks.append(bytes(meta))
    for tid, tr in tl.tracks.items():
        evs = []
        for n in tl.notes:
            if n.track != tid:
                continue
            b0 = tl.tempo.beats(n.time)
            b1 = tl.tempo.beats(n.time + n.duration)
            ch = 9 if tr.group in ("kick", "snare", "hats", "perc") else 0
            evs.append((int(round(b0 * division)), 1, bytes([0x90 | ch, int(n.pitch) & 0x7F, max(1, int(n.velocity * 127))])))
            evs.append((int(round(b1 * division)), 0, bytes([0x80 | ch, int(n.pitch) & 0x7F, 0])))
        evs.sort(key=lambda e: (e[0], e[1]))
        body = bytearray()
        name = tr.name.encode("latin-1", "replace")
        body += varlen(0) + b"\xFF\x03" + varlen(len(name)) + name
        last = 0
        for tick, _, msg in evs:
            body += varlen(tick - last) + msg
            last = tick
        body += varlen(0) + b"\xFF\x2F\x00"
        chunks.append(bytes(body))
    with open(path, "wb") as fh:
        fh.write(b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), division))
        for c in chunks:
            fh.write(b"MTrk" + struct.pack(">I", len(c)) + c)
