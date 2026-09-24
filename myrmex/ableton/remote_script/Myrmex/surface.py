"""Myrmex control surface: Live's transport and scores -> OSC (no dependencies, runs inside Live).

Sent to 127.0.0.1:9100 (change HOST / PORT below):

    /myrmex/transport   f song_beat  f bpm  i playing  i sig_num  i sig_den     as the song position moves
                                                                                 (<= 50 Hz) + on play/stop/tempo
    /myrmex/score/track s id  s name  s group_hint                               when tracks change
    /myrmex/score/notes s id  b float32[beat, dur, pitch, vel]*  f from  f to   every half second:
                        the notes of every playing MIDI clip in the next WINDOW beats, in song beats

    /myrmex/control     s name  f value                                        knobs of the "Myrmex" track

Because notes arrive *before* they sound, the creature can anticipate hits (and the
engine schedules them on its own clock with sub-millisecond precision).

Work with the character inside Live: name a track "Myrmex", put any Rack on it (an empty
Audio Effect Rack is enough).  Its macros drive the character - by name (Energy, Stride,
Sway, Style, Hold, Camera, Pose, Flourish) or by position (Macro 1..8 in that order);
a knob at zero means "automatic".  Automate them in the arrangement.  MIDI notes in the
track's clips are choreography: C3 pose, D3 gesture, E3 camera cut, F3 look back,
G3 hair touch, A3 hand on hip, B3 shoulder roll, C4 hold while the note lasts.
"""
from __future__ import absolute_import

import socket
import struct
import time

try:                                                 # Live 11 / 12
    from ableton.v2.control_surface import ControlSurface
except ImportError:                                  # pragma: no cover - older Live / outside Live
    try:
        from _Framework.ControlSurface import ControlSurface
    except ImportError:
        ControlSurface = object

HOST = "127.0.0.1"
PORT = 9100
WINDOW = 8.0            # beats of notes sent ahead
SCORE_EVERY = 5         # 100 ms ticks between score updates (~0.5 s)
TRANSPORT_MIN_DT = 0.02 # s: song-time listener rate limit

GROUP_WORDS = (
    ("kick", ("kick", "bd", "bassdrum", "808")),
    ("snare", ("snare", "sd", "clap", "rim")),
    ("hats", ("hat", "hh", "hihat", "cymbal", "ride", "shaker")),
    ("perc", ("perc", "tom", "conga", "bongo", "tamb", "cowbell", "drum")),
    ("bass", ("bass", "sub", "reese")),
    ("melody", ("lead", "melody", "arp", "synth", "pluck", "vox", "vocal", "piano", "keys", "guitar", "bell")),
    ("harmony", ("pad", "chord", "string", "organ", "choir")),
    ("fx", ("fx", "riser", "sweep", "impact", "noise", "hit")),
)


# ---------------------------------------------------------------------------- tiny OSC encoder
def _pad(b):
    return b + b"\0" * (4 - len(b) % 4)


def _osc(address, args):
    tags = ","
    body = b""
    for a in args:
        if isinstance(a, bool) or isinstance(a, int):
            tags += "i"
            body += struct.pack(">i", int(a))
        elif isinstance(a, float):
            tags += "f"
            body += struct.pack(">f", a)
        elif isinstance(a, (bytes, bytearray)):
            tags += "b"
            body += struct.pack(">i", len(a)) + bytes(a) + b"\0" * ((4 - len(a) % 4) % 4)
        else:
            tags += "s"
            body += _pad(str(a).encode("utf-8"))
    return _pad(address.encode("utf-8")) + _pad(tags.encode("utf-8")) + body


# The character's own track: any track whose name starts with "Myrmex".
#   * its first Rack's macros are the character's knobs (automate them in the arrangement);
#   * notes in its MIDI clips are choreography commands (C3 pose, D3 gesture, E3 camera cut, ...).
CONTROL_PREFIX = "myrmex"
MACRO_NAMES = {
    "energy": ("energy", "энергия", "intensity"),
    "stride": ("stride", "step", "шаг"),
    "sway": ("sway", "hips", "бёдра", "бедра"),
    "style": ("style", "стиль"),
    "hold": ("hold", "stop", "стоп", "пауза"),
    "camera": ("camera", "cam", "камера"),
    "pose": ("pose", "поза"),
    "flourish": ("flourish", "gesture", "жест"),
}
MACRO_ORDER = ("energy", "stride", "sway", "style", "hold", "camera", "pose", "flourish")
TRIGGERS = ("camera", "pose", "flourish")


def is_control_track(track):
    return track.name.strip().lower().startswith(CONTROL_PREFIX)


def macro_role(name, index):
    """Map a macro to a character control by its name, else by its position (Macro 1 = energy, ...)."""
    n = name.strip().lower()
    for role, words in MACRO_NAMES.items():
        for w in words:
            if n == w or n.startswith(w):
                return role
    if n.startswith("macro") and 0 <= index < len(MACRO_ORDER):
        return MACRO_ORDER[index]
    # Any other macro name is passed through (creature parameters: Aggression, Fluidity, ...).
    return n.replace(" ", "_") if n and n[0].isalpha() else None


def macro_value(role, param):
    """Rack macro -> control value.  Knobs at 0 mean "automatic" (-1); triggers stay 0..1."""
    lo, hi = float(param.min), float(param.max)
    v = (float(param.value) - lo) / max(hi - lo, 1e-9)
    if role in TRIGGERS or role == "hold":
        return v
    return -1.0 if v <= 0.5 / 127.0 else v


def group_hint(track):
    if is_control_track(track):
        return "control"
    try:
        for dev in track.devices:
            if getattr(dev, "can_have_drum_pads", False):
                return "gm"                          # Drum Rack: per-pad General MIDI layout
    except Exception:
        pass
    name = track.name.lower()
    for group, words in GROUP_WORDS:
        for w in words:
            if w in name:
                return group
    return ""


def clip_notes(clip):
    """[(start, duration, pitch, velocity)] in clip time, unmuted notes only."""
    out = []
    try:                                              # Live 11+
        for n in clip.get_notes_extended(0, 128, -8192.0, 16384.0):
            if not getattr(n, "mute", False):
                out.append((n.start_time, n.duration, n.pitch, n.velocity))
    except AttributeError:                            # Live 10
        for pitch, start, dur, vel, mute in clip.get_notes(0.0, 0, max(clip.length, clip.loop_end) + 1.0, 128):
            if not mute:
                out.append((start, dur, pitch, vel))
    return out


def upcoming(notes, clip, clip_pos, song_now, window):
    """Map clip-time notes to song beats for the next ``window`` beats."""
    out = []
    if getattr(clip, "looping", True):
        ls, le = clip.loop_start, clip.loop_end
        L = le - ls
        if L <= 1e-6:
            return out
        for (s, d, p, v) in notes:
            if s < ls or s >= le:
                continue
            ahead = (s - clip_pos) % L
            k = 0
            while ahead + k * L < window:
                out.append((song_now + ahead + k * L, d, p, v))
                k += 1
    else:
        for (s, d, p, v) in notes:
            ahead = s - clip_pos
            if 0.0 <= ahead < window:
                out.append((song_now + ahead, d, p, v))
    return out


class MyrmexSurface(ControlSurface):
    def __init__(self, c_instance):
        ControlSurface.__init__(self, c_instance)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._ticks = 0
        self._known = {}
        self._last_tx = 0.0
        self._macros = {}
        s = self.song()
        s.add_is_playing_listener(self._on_transport)
        s.add_tempo_listener(self._on_transport)
        s.add_current_song_time_listener(self._on_song_time)
        self.schedule_message(1, self._tick)          # Live's 100 ms tick (no threads inside Live)
        self.show_message("Myrmex: streaming to %s:%d" % (HOST, PORT))
        self.log_message("Myrmex remote script loaded")

    # ------------------------------------------------------------------ io
    def _send(self, address, args):
        try:
            self._sock.sendto(_osc(address, args), (HOST, PORT))
        except Exception:
            pass

    def _on_transport(self):
        self._send_transport()

    def _on_song_time(self):
        now = time.time()
        if now - self._last_tx >= TRANSPORT_MIN_DT:
            self._send_transport()

    def _send_transport(self):
        self._last_tx = time.time()
        s = self.song()
        self._send("/myrmex/transport", [float(s.current_song_time), float(s.tempo), 1 if s.is_playing else 0,
                                         int(s.signature_numerator), int(s.signature_denominator)])

    # ------------------------------------------------------------------ score
    def _send_score(self):
        s = self.song()
        now = float(s.current_song_time)
        for ti, track in enumerate(s.tracks):
            if not getattr(track, "has_midi_input", False):
                continue
            tid = "t%d" % ti
            key = (track.name, group_hint(track))
            if self._known.get(tid) != key:
                self._known[tid] = key
                self._send("/myrmex/score/track", [tid, key[0], key[1]])
            events = []
            if s.is_playing and not track.mute:
                idx = track.playing_slot_index
                if idx is not None and idx >= 0:
                    clip = track.clip_slots[idx].clip
                    if clip is not None and clip.is_midi_clip and clip.is_playing:
                        events = upcoming(clip_notes(clip), clip, clip.playing_position, now, WINDOW)
                else:
                    for clip in getattr(track, "arrangement_clips", ()):
                        if not clip.is_midi_clip or clip.end_time <= now or clip.start_time >= now + WINDOW:
                            continue
                        pos = clip.start_marker + (now - clip.start_time)
                        if getattr(clip, "looping", False):
                            L = clip.loop_end - clip.loop_start
                            if pos >= clip.loop_end and L > 0:
                                pos = clip.loop_start + (pos - clip.loop_start) % L
                        for ev in upcoming(clip_notes(clip), clip, pos, now, WINDOW):
                            if clip.start_time <= ev[0] < clip.end_time:
                                events.append(ev)
            flat = []
            for (b, d, p, v) in sorted(events):
                flat.extend((b, d, float(p), v / 127.0))
            blob = struct.pack(">%df" % len(flat), *flat) if flat else b""
            self._send("/myrmex/score/notes", [tid, blob, now, now + WINDOW])

    # ------------------------------------------------------------------ the character's knobs
    def _send_macros(self):
        for track in self.song().tracks:
            if not is_control_track(track):
                continue
            rack = None
            for dev in track.devices:
                if getattr(dev, "can_have_chains", False):
                    rack = dev
                    break
            if rack is None:
                return
            idx = 0
            for p in rack.parameters:
                name = str(p.name)
                if name in ("Device On", "Chain Selector"):
                    continue
                role = macro_role(name, idx)
                idx += 1
                if role is None:
                    continue
                v = macro_value(role, p)
                last = self._macros.get(role)
                if last is None or abs(v - last) > 1e-3 or self._ticks % 20 == 0:
                    self._macros[role] = v
                    self._send("/myrmex/control", [role, float(v)])
            return

    # ------------------------------------------------------------------ Live callbacks
    def _tick(self):
        try:
            self._send_transport()
            self._ticks += 1
            self._send_macros()
            if self._ticks % SCORE_EVERY == 0:
                self._send_score()
        except Exception as e:
            self.log_message("Myrmex error: %s" % e)
        self.schedule_message(1, self._tick)

    def disconnect(self):
        s = self.song()
        try:
            s.remove_is_playing_listener(self._on_transport)
            s.remove_tempo_listener(self._on_transport)
            s.remove_current_song_time_listener(self._on_song_time)
        except Exception:
            pass
        self._send("/myrmex/transport", [float(s.current_song_time), float(s.tempo), 0, 4, 4])
        self._sock.close()
        ControlSurface.disconnect(self)
