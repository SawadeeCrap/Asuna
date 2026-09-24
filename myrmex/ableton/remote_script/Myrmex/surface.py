"""Myrmex control surface: Live's transport and scores -> OSC (no dependencies, runs inside Live).

Sent to 127.0.0.1:9100 (change HOST / PORT below):

    /myrmex/transport   f song_beat  f bpm  i playing  i sig_num  i sig_den     every display tick (~10 Hz)
                                                                                 + immediately on play/stop
    /myrmex/score/track s id  s name  s group_hint                               when tracks change
    /myrmex/score/notes s id  b float32[beat, dur, pitch, vel]*  f from  f to   every half second:
                        the notes of every playing MIDI clip in the next WINDOW beats, in song beats

Because notes arrive *before* they sound, the creature can anticipate hits (and the
engine schedules them on its own clock with sub-millisecond precision).
"""
from __future__ import absolute_import

import socket
import struct

try:
    from _Framework.ControlSurface import ControlSurface
except ImportError:                                  # pragma: no cover - outside Live
    ControlSurface = object

HOST = "127.0.0.1"
PORT = 9100
WINDOW = 8.0            # beats of notes sent ahead
SCORE_EVERY = 5         # display ticks between score updates (~0.5 s)

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


def group_hint(track):
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
        for n in clip.get_notes_extended(0, 128, 0.0, max(clip.length, clip.loop_end) + 1.0):
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
        self._tick = 0
        self._known = {}
        s = self.song()
        s.add_is_playing_listener(self._on_transport)
        s.add_tempo_listener(self._on_transport)
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

    def _send_transport(self):
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

    # ------------------------------------------------------------------ Live callbacks
    def update_display(self):
        ControlSurface.update_display(self)
        self._send_transport()
        self._tick += 1
        if self._tick % SCORE_EVERY == 0:
            try:
                self._send_score()
            except Exception as e:
                self.log_message("Myrmex score error: %s" % e)

    def disconnect(self):
        s = self.song()
        try:
            s.remove_is_playing_listener(self._on_transport)
            s.remove_tempo_listener(self._on_transport)
        except Exception:
            pass
        self._send("/myrmex/transport", [float(s.current_song_time), float(s.tempo), 0, 4, 4])
        self._sock.close()
        ControlSurface.disconnect(self)
