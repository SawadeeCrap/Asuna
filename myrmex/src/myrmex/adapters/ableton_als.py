"""Offline Ableton Live set (.als) reader – Live 8 … 12.

A ``.als`` file is gzip-compressed XML.  We extract, without Live running:

* tempo (+ tempo automation, ramps subdivided), time signature, locators;
* tracks (MIDI / audio / group / return) with names, colours, devices,
  Drum Rack pads (``ReceivingNote = 128 - note``) and group membership;
* arrangement MIDI clips with loop unrolling, disabled notes and note
  probability (deterministically sampled);
* optionally a session scene (for sets that live in Session View);
* arrangement audio clips (file, position, warp markers) – analysed when the
  sample is a readable WAV/AIFF, otherwise listed so the user can export stems;
* automation envelopes of mixer and device parameters as control curves.

Paths verified against real sets from Live 8.1 up to Live 12.3
(``MasterTrack`` became ``MainTrack`` in Live 12; audio arrangement clips live
under ``MainSequencer/Sample/ArrangerAutomation``; MIDI ones under
``MainSequencer/ClipTimeable/ArrangerAutomation``; ``GroovePool`` clips are
ignored).
"""
from __future__ import annotations

import gzip
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from ..music.classify import apply_classification
from ..music.timeline import ControlCurve, Marker, MusicTimeline, NoteEvent, TempoMap, Track
from ..util.rng import hash_unit, stable_hash64

PRE_SONG = -63072000.0          # Ableton's "before the song" automation time


def _v(el, path: str | None = None, default=None):
    e = el.find(path) if path else el
    if e is None:
        return default
    return e.get("Value", default)


def _f(el, path: str | None = None, default: float = 0.0) -> float:
    v = _v(el, path, None)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _b(el, path: str, default: bool = False) -> bool:
    v = _v(el, path, None)
    return default if v is None else str(v).lower() == "true"


def open_als(path: str) -> ET.Element:
    with open(path, "rb") as fh:
        head = fh.read(2)
    opener = gzip.open if head == b"\x1f\x8b" else open
    with opener(path, "rb") as fh:
        data = fh.read()
    if not data.strip():
        raise ValueError(f"{path}: empty file")
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise ValueError(f"{path}: not a Live set ({exc})") from exc


def decode_time_signature(value: float) -> tuple[int, int]:
    v = int(round(value))
    return (v % 99) + 1, 2 ** (v // 99)


def track_name(tr: ET.Element) -> str:
    for p in ("Name/EffectiveName", "Name/UserName", "Name"):
        v = _v(tr, p)
        if v:
            return str(v)
    return tr.tag


def _envelope_events(owner: ET.Element, target_id: str) -> list[tuple[float, float]]:
    for env in owner.iter("AutomationEnvelope"):
        if _v(env, "EnvelopeTarget/PointeeId") == target_id:
            out = []
            for ev in env.iter():
                if ev.tag in ("FloatEvent", "EnumEvent", "BoolEvent"):
                    t = float(ev.get("Time", "0"))
                    val = ev.get("Value", "0")
                    val = 1.0 if val == "true" else 0.0 if val == "false" else float(val)
                    out.append((t, val))
            return sorted(out, key=lambda e: e[0])      # stable: keeps vertical-step order
    return []


@dataclass
class ClipInfo:
    track: str
    kind: str
    start: float          # arrangement beats
    end: float
    name: str = ""
    file: str | None = None
    warped: bool = True
    warp_markers: list[tuple[float, float]] = field(default_factory=list)   # (sec, beat)
    loop: dict = field(default_factory=dict)
    notes: int = 0


@dataclass
class AlsReport:
    creator: str
    version: str
    tempo: float
    time_signature: tuple[int, int]
    tracks: list[dict]
    clips: list[ClipInfo]
    locators: list[tuple[float, str]]
    automation: int
    warnings: list[str]

    def as_dict(self) -> dict:
        return {"creator": self.creator, "version": self.version, "tempo": self.tempo,
                "time_signature": list(self.time_signature), "tracks": self.tracks,
                "clips": [c.__dict__ for c in self.clips[:200]], "clip_count": len(self.clips),
                "locators": self.locators, "automation_envelopes": self.automation, "warnings": self.warnings}


# ============================================================================ tempo
def _automation_value_fn(events: list[tuple[float, float]]):
    """Piecewise-linear automation value at a beat (vertical steps take the later value)."""
    evs = sorted(((max(0.0, t), v) for t, v in events), key=lambda e: e[0])
    beats = [b for b, _ in evs]

    def f(b: float) -> float:
        import bisect
        i = bisect.bisect_right(beats, b) - 1
        if i < 0:
            return evs[0][1]
        if i >= len(evs) - 1:
            return evs[-1][1]
        b0, v0 = evs[i]
        b1, v1 = evs[i + 1]
        if b1 - b0 < 1e-12:
            return v1
        return v0 + (v1 - v0) * (b - b0) / (b1 - b0)
    return f, (beats[-1] if beats else 0.0)


def read_tempo(ls: ET.Element, quant: int = 16) -> tuple[TempoMap, float, tuple[int, int]]:
    """Tempo map as Live plays it: automation is sampled at every 1/16 boundary
    and held constant within the cell (matches Live's actual timing)."""
    main = ls.find("MainTrack")
    if main is None:
        main = ls.find("MasterTrack")
    bpm, sig = 120.0, (4, 4)
    points: list[tuple[float, float]] = []
    ev: list[tuple[float, float]] = []
    if main is not None:
        mixer = main.find("DeviceChain/Mixer")
        if mixer is None:
            mixer = main.find("MasterChain/Mixer")          # Live 8
        tempo = mixer.find("Tempo") if mixer is not None else None
        if tempo is not None:
            bpm = _f(tempo, "Manual", bpm)
            inline = [(float(e.get("Time", "0")), float(e.get("Value", bpm)))
                      for e in tempo.findall("ArrangerAutomation/Events/FloatEvent")]
            if inline:                                          # Live 8 stores automation inline
                ev = inline
                if tempo.find("Manual") is None:
                    bpm = inline[0][1]
            tid = tempo.find("AutomationTarget")
            if tid is not None and not ev:
                ev = _envelope_events(main, tid.get("Id", ""))
        tsig = mixer.find("TimeSignature") if mixer is not None else None
        if tsig is not None:
            tsv = _v(tsig, "Manual")
            if tsv is None:
                evs = tsig.findall("ArrangerAutomation/Events/EnumEvent")
                tsv = evs[0].get("Value") if evs else None
            if tsv is not None:
                sig = decode_time_signature(float(tsv))
    if main is not None:
        if len(ev) >= 2 and len({v for _, v in ev}) > 1:
            f, last = _automation_value_fn(ev)
            step = 4.0 / quant
            n = int(np.ceil(last / step)) + 1
            prev = None
            for k in range(n + 1):
                b = k * step
                v = f(b)
                if prev is None or abs(v - prev) > 1e-9:
                    points.append((b, v))
                    prev = v
        elif ev:
            bpm = ev[0][1]
    if not points:
        points = [(0.0, bpm)]
    tm = TempoMap(points, [(0.0, sig[0], sig[1])])
    return tm, bpm, sig


# ============================================================================ clips
def _loop(clip: ET.Element) -> dict:
    return {"start": _f(clip, "Loop/LoopStart"), "end": _f(clip, "Loop/LoopEnd"),
            "rel": _f(clip, "Loop/StartRelative"), "on": _b(clip, "Loop/LoopOn"),
            "out": _f(clip, "Loop/OutMarker", _f(clip, "Loop/LoopEnd"))}


def clip_segments(arr_start: float, arr_end: float, lp: dict) -> list[tuple[float, float, float]]:
    """Content ranges [c0, c1) and their arrangement start a0, covering [arr_start, arr_end)."""
    segs = []
    if lp["on"] and lp["end"] > lp["start"]:
        L = lp["end"] - lp["start"]
        c0 = lp["start"] + lp["rel"]
        if c0 >= lp["end"]:
            c0 = lp["start"] + ((c0 - lp["start"]) % L)
        a = arr_start
        segs.append((c0, lp["end"], a))
        a += lp["end"] - c0
        guard = 0
        while a < arr_end - 1e-9 and guard < 100000:
            segs.append((lp["start"], lp["end"], a))
            a += L
            guard += 1
    else:
        c0 = lp["start"] + lp["rel"]
        c1 = lp["out"] if lp["out"] > c0 else c0 + (arr_end - arr_start)
        segs.append((c0, c1, arr_start))
    return segs


def midi_clip_notes(clip: ET.Element, arr_start: float, arr_end: float, track_id: str,
                    seed: int = 0) -> list[tuple[float, float, int, float, float]]:
    """(beat, dur_beats, key, velocity 0..1, probability) in arrangement beats."""
    raw = []
    for kt in clip.findall("Notes/KeyTracks/KeyTrack"):
        key = int(_f(kt, "MidiKey", 60))
        for ev in kt.findall("Notes/MidiNoteEvent"):
            if ev.get("IsEnabled", "true").lower() == "false":
                continue
            raw.append((float(ev.get("Time", "0")), float(ev.get("Duration", "0.25")), key,
                        float(ev.get("Velocity", "100")) / 127.0, float(ev.get("Probability", "1"))))
    out = []
    lp = _loop(clip)
    for it, (c0, c1, a0) in enumerate(clip_segments(arr_start, arr_end, lp)):
        for (t, d, k, v, p) in raw:
            if c0 - 1e-9 <= t < c1 - 1e-9:
                a = a0 + (t - c0)
                if a >= arr_end - 1e-9:
                    continue
                if p < 1.0 and hash_unit(stable_hash64(seed, track_id, t, k), it) > p:
                    continue
                dur = min(d, c1 - t, arr_end - a)
                out.append((a, max(dur, 1e-3), k, v, p))
    return out


def _drum_pads(track: ET.Element) -> dict[int, str]:
    pads: dict[int, str] = {}
    for br in track.iter("DrumBranch"):
        recv = None
        for e in br.iter("ReceivingNote"):
            recv = e.get("Value")
            break
        if recv is None:
            continue
        note = 128 - int(float(recv))
        name = _v(br, "Name/EffectiveName") or _v(br, "Name/UserName") or ""
        if not name:
            for fr in br.iter("FileRef"):
                name = _v(fr, "Name") or os.path.basename(_v(fr, "Path") or _v(fr, "RelativePath") or "")
                if name:
                    break
        pads[note] = name or f"pad {note}"
    return pads


def _devices(track: ET.Element) -> list[str]:
    names = []
    devs = track.find("DeviceChain/DeviceChain/Devices")
    if devs is None:
        return names
    for d in devs:
        names.append(d.tag)
        for p in ("UserName", "PluginDesc/VstPluginInfo/PlugName", "PluginDesc/Vst3PluginInfo/Name",
                  "PluginDesc/AuPluginInfo/Name"):
            v = _v(d, p)
            if v:
                names.append(str(v))
    return names


def _warp_markers(clip: ET.Element) -> list[tuple[float, float]]:
    wm = []
    for m in clip.iter("WarpMarker"):
        try:
            wm.append((float(m.get("SecTime", "0")), float(m.get("BeatTime", "0"))))
        except ValueError:
            continue
    return sorted(set(wm))


def _audio_file(clip: ET.Element, set_dir: str) -> str | None:
    fr = clip.find("SampleRef/FileRef")
    if fr is None:
        return None
    for key in ("Path",):
        p = _v(fr, key)
        if p and os.path.exists(p):
            return p
    rel = _v(fr, "RelativePath")
    if rel:
        cand = os.path.join(set_dir, rel)
        if os.path.exists(cand):
            return cand
    name = _v(fr, "Name")
    if name:
        for root, _, files in os.walk(set_dir):
            if name in files:
                return os.path.join(root, name)
    return _v(fr, "Path") or rel or name


# ============================================================================ public API
def inspect_als(path: str) -> AlsReport:
    root = open_als(path)
    ls = root.find("LiveSet")
    tm, bpm, sig = read_tempo(ls)
    tracks, clips, warnings = [], [], []
    n_auto = 0
    for tr in ls.find("Tracks") if ls.find("Tracks") is not None else []:
        tid = tr.get("Id", str(len(tracks)))
        name = track_name(tr)
        kind = {"MidiTrack": "midi", "AudioTrack": "audio", "GroupTrack": "group", "ReturnTrack": "return"}.get(tr.tag, tr.tag)
        n_auto += sum(1 for _ in tr.iter("AutomationEnvelope"))
        tclips = []
        for c in tr.findall("DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/Events/MidiClip"):
            tclips.append(ClipInfo(tid, "midi", _f(c, None, 0) if c.get("Time") is None else float(c.get("Time")),
                                   _f(c, "CurrentEnd"), _v(c, "Name", "") or "", loop=_loop(c),
                                   notes=sum(1 for _ in c.iter("MidiNoteEvent"))))
        for c in tr.findall("DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events/AudioClip"):
            tclips.append(ClipInfo(tid, "audio", float(c.get("Time", _v(c, "CurrentStart", 0))), _f(c, "CurrentEnd"),
                                   _v(c, "Name", "") or "", file=_audio_file(c, os.path.dirname(path)),
                                   warped=_b(c, "IsWarped", True), warp_markers=_warp_markers(c)[:64], loop=_loop(c)))
        session = sum(1 for _ in tr.findall("DeviceChain/MainSequencer/ClipSlotList/ClipSlot/ClipSlot/Value/*"))
        clips.extend(tclips)
        tracks.append({"id": tid, "name": name, "kind": kind, "arrangement_clips": len(tclips),
                       "session_clips": session, "devices": _devices(tr)[:8], "drum_pads": _drum_pads(tr),
                       "group": _v(tr, "TrackGroupId", "-1")})
    locs = [(float(l.get("Time", _v(l, "Time", 0) or 0)), _v(l, "Name", "") or "")
            for l in ls.iter("Locator") if l.find("Name") is not None]
    if not clips:
        warnings.append("arrangement is empty – use session_scene=… to read Session View clips")
    return AlsReport(root.get("Creator", "?"), root.get("MinorVersion", "?"), bpm, sig, tracks, clips,
                     locs, n_auto, warnings)


def load_als(path: str, *, session_scene: int | None = None, session_bars: int = 16,
             analyze_audio: bool = True, overrides: dict[str, str] | None = None, seed: int = 0,
             include_automation: bool = True, max_audio_seconds: float = 900.0) -> MusicTimeline:
    root = open_als(path)
    ls = root.find("LiveSet")
    if ls is None:
        raise ValueError("no LiveSet element")
    tm, bpm, sig = read_tempo(ls)
    tl = MusicTimeline(tm, source=f"ableton:{os.path.basename(path)}")
    tl.meta.update({"creator": root.get("Creator"), "version": root.get("MinorVersion"), "warnings": []})
    set_dir = os.path.dirname(os.path.abspath(path))
    tracks_el = ls.find("Tracks")
    tracks_el = list(tracks_el) if tracks_el is not None else []
    any_arrangement = False
    end_beat = 0.0
    for tr in tracks_el:
        if tr.tag not in ("MidiTrack", "AudioTrack"):
            continue
        tid = tr.get("Id") or str(len(tl.tracks))
        name = track_name(tr)
        kind = "midi" if tr.tag == "MidiTrack" else "audio"
        pads = _drum_pads(tr)
        track = Track(tid, name, kind, devices=_devices(tr), meta={"drum_pads": pads, "clip_names": ""})
        tl.tracks[tid] = track
        clip_names = []
        # ---------------- MIDI (arrangement or chosen session scene)
        mclips = tr.findall("DeviceChain/MainSequencer/ClipTimeable/ArrangerAutomation/Events/MidiClip")
        placed = []
        for c in mclips:
            if _b(c, "Disabled"):
                continue
            a0 = float(c.get("Time", _v(c, "CurrentStart", 0)))
            placed.append((c, a0, _f(c, "CurrentEnd", a0)))
        if session_scene is not None:
            slots = tr.findall("DeviceChain/MainSequencer/ClipSlotList/ClipSlot")
            if session_scene < len(slots):
                c = slots[session_scene].find("ClipSlot/Value/MidiClip")
                if c is not None:
                    placed = [(c, 0.0, session_bars * sig[0] * 4.0 / sig[1])]
        for c, a0, a1 in placed:
            any_arrangement = True
            clip_names.append(_v(c, "Name", "") or "")
            for (b, d, k, v, p) in midi_clip_notes(c, a0, a1, tid, seed):
                t0 = tm.seconds(b)
                grp = None
                if pads and k in pads:
                    from ..music.classify import classify_drum_note
                    grp = classify_drum_note(k, pads[k])
                tl.notes.append(NoteEvent(t0, max(1e-3, tm.seconds(b + d) - t0), float(k), v, tid, beat=b,
                                          group=grp, sharpness=0.8 if grp in ("kick", "snare", "hats", "perc") else 0.5))
            end_beat = max(end_beat, a1)
        # ---------------- audio clips
        aclips = tr.findall("DeviceChain/MainSequencer/Sample/ArrangerAutomation/Events/AudioClip")
        for c in aclips:
            if _b(c, "Disabled"):
                continue
            a0 = float(c.get("Time", _v(c, "CurrentStart", 0)))
            a1 = _f(c, "CurrentEnd", a0)
            end_beat = max(end_beat, a1)
            any_arrangement = True
            clip_names.append(_v(c, "Name", "") or "")
            f = _audio_file(c, set_dir)
            if not analyze_audio:
                continue
            if not f or not os.path.exists(f) or not f.lower().endswith((".wav", ".wave", ".aif", ".aiff", ".aifc")):
                tl.meta["warnings"].append(f"audio clip on '{name}' not analysed ({os.path.basename(str(f))}); "
                                           "export stems or flatten to WAV")
                continue
            try:
                _add_audio_clip(tl, tid, c, f, a0, a1, bpm, max_audio_seconds)
            except Exception as exc:  # pragma: no cover - defensive
                tl.meta["warnings"].append(f"audio analysis failed for {f}: {exc}")
        track.meta["clip_names"] = " ".join(n for n in clip_names if n)
        # ---------------- automation
        if include_automation:
            _add_automation(tl, tr, tid, tm)
    if not any_arrangement:
        tl.meta["warnings"].append("no arrangement clips found (Session View set?) – pass session_scene")
    for l in ls.iter("Locator"):
        if l.find("Name") is not None or l.get("Time") is not None:
            t = float(l.get("Time") if l.get("Time") is not None else _f(l, "Time"))
            tl.markers.append(Marker(tm.seconds(max(0.0, t)), str(_v(l, "Name", "") or "")))
    tl.duration = max(tm.seconds(end_beat), max((m.time for m in tl.markers), default=0.0))
    apply_classification(tl, overrides)
    return tl.finalize()


def _add_automation(tl: MusicTimeline, tr: ET.Element, tid: str, tm: TempoMap) -> None:
    targets: dict[str, str] = {}
    for parent in tr.iter():
        for child in parent:
            if child.tag == "AutomationTarget" and child.get("Id"):
                targets[child.get("Id")] = parent.tag
    for env in tr.iter("AutomationEnvelope"):
        pid = _v(env, "EnvelopeTarget/PointeeId")
        if pid is None:
            continue
        pts = []
        for ev in env.iter("FloatEvent"):
            t = float(ev.get("Time", "0"))
            pts.append((max(0.0, t), float(ev.get("Value", "0"))))
        if len(pts) < 2:
            continue
        pts.sort()
        vals = np.array([v for _, v in pts])
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-9:
            continue
        times = np.array([tm.seconds(b) for b, _ in pts])
        tl.curves.append(ControlCurve(tid, targets.get(pid, f"param_{pid}"), times, (vals - lo) / (hi - lo)))


def _add_audio_clip(tl: MusicTimeline, tid: str, clip: ET.Element, path: str, a0: float, a1: float,
                    bpm: float, max_seconds: float) -> None:
    from .audio_io import read_audio
    from .audio_stems import analyze_signal
    x, sr = read_audio(path)
    x = x[: int(max_seconds * sr)]
    an = analyze_signal(x, sr)
    lp = _loop(clip)
    wm = _warp_markers(clip)
    warped = _b(clip, "IsWarped", True)
    if warped and len(wm) >= 2:
        secs = np.array([s for s, _ in wm])
        beats = np.array([b for _, b in wm])
        slope = (beats[-1] - beats[-2]) / max(secs[-1] - secs[-2], 1e-9)

        def sec_to_beat(s: float) -> float:
            if s <= secs[-1]:
                return float(np.interp(s, secs, beats))
            return float(beats[-1] + (s - secs[-1]) * slope)
    else:
        def sec_to_beat(s: float) -> float:
            return s * bpm / 60.0
    segs = clip_segments(a0, a1, lp)
    from ..music.timeline import NoteEvent as _NE
    from .audio_stems import centroid_to_pitch
    on_beats = np.array([sec_to_beat(float(t)) for t in an["onset_times"]])
    pitches = centroid_to_pitch(an["onset_centroid"])
    env_t = an["times"][::2]
    env = an["rms"][::2] / (np.percentile(an["rms"], 98) + 1e-9)
    env_b = np.array([sec_to_beat(float(t)) for t in env_t])
    ct, cv = [], []
    for c0, c1, arr in segs:
        m = (on_beats >= c0 - 1e-6) & (on_beats < c1)
        for b, st, h, pch in zip(on_beats[m], an["onset_strength"][m], an["onset_hfc"][m], pitches[m]):
            a = arr + (b - c0)
            if a >= a1:
                continue
            tl.notes.append(_NE(tl.tempo.seconds(a), 0.08, float(pch), float(st), tid,
                                sharpness=float(np.clip(h * 3.0, 0.0, 1.0))))
        me = (env_b >= c0) & (env_b < c1)
        for b, v in zip(env_b[me], env[me]):
            a = arr + (b - c0)
            if a < a1:
                ct.append(tl.tempo.seconds(a))
                cv.append(float(min(1.0, v)))
    if ct:
        order = np.argsort(ct)
        tl.curves.append(ControlCurve(tid, "envelope", np.asarray(ct)[order], np.asarray(cv)[order]))
