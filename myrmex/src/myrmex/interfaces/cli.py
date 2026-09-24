"""``myrmex`` command line.

    myrmex live      --rig rig.json            realtime engine: Ableton / VCV in -> poses out (UDP 9101)
    myrmex simulate  [--demo|--midi f|--als f] fake Ableton: plays a song as OSC transport + notes
    myrmex monitor                             print what the engine streams (no Blender needed)
    myrmex ports                               MIDI ports, audio inputs, Ableton Link peers
    myrmex generate  --rig rig.json --music …  offline performance (.npz) for high-quality renders
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time


# ---------------------------------------------------------------------------- live
def cmd_live(a) -> int:
    from ..realtime.inputs import InputConfig
    from ..realtime.session import LiveConfig, LiveSession
    inputs = InputConfig.from_file(a.config, osc_port=a.osc_port, midi=a.midi or [], audio=a.audio)
    engine = json.loads(a.engine) if a.engine else {}
    cfg = LiveConfig(rig=a.rig, seed=a.seed, rate=a.rate, out=a.out, out_rate=a.fps, clock=a.clock, bpm=a.bpm,
                     link=not a.no_link, latency=a.latency / 1000.0, style=a.style, engine=engine,
                     camera=not a.no_camera, record=a.record, inputs=inputs)
    s = LiveSession(cfg)
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    s.start()
    print(f"myrmex live: rig {os.path.basename(a.rig)} ({len(s.names)} bones) | OSC in :{a.osc_port} | "
          f"MIDI {a.midi or '-'} | clock {a.clock} | poses -> {', '.join(a.out)} @ {a.fps:g} fps", flush=True)
    for e in s.status()["errors"]:
        print("  !", e, flush=True)
    last = 0.0
    while not stop["flag"]:
        time.sleep(0.1)
        if a.status_every and time.time() - last >= a.status_every:
            last = time.time()
            st = s.status()
            print(f"[{st['t']:7.1f}s] clock={st['clock']:<8} bpm={st['bpm']} beat={st['beat']} "
                  f"{'PLAY' if st['playing'] else 'stop'} {'HOLD' if st['hold'] else 'walk'} peers={st['peers']} "
                  f"notes={st['notes']} section={st['section']} cam={st['camera']} "
                  f"tick={st['tick_ms']}ms (max {st['max_tick_ms']}) sent={st['sent']}", flush=True)
    path = s.stop()
    if path:
        print("take saved:", path)
    return 0


# ---------------------------------------------------------------------------- simulate
def _load_music(a):
    from ..music import synthetic
    if a.midi_file:
        from ..adapters.midi_file import load_midi
        return load_midi(a.midi_file)
    if a.als:
        from ..adapters.ableton_als import load_als
        return load_als(a.als, analyze_audio=False)
    tl = synthetic.TESTS.get(a.pattern, synthetic.demo_arrangement)()
    return tl


def cmd_simulate(a) -> int:
    """Play a song like Ableton would: transport at ``rate`` Hz + notes as they happen (or scored ahead)."""
    import numpy as np

    from ..bus.osc import OscMessage
    from ..bus.transport import pack_notes
    tl = _load_music(a)
    host, _, port = a.to.rpartition(":")
    addr = (host or "127.0.0.1", int(port))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(msg: OscMessage) -> None:
        sock.sendto(msg.encode(), addr)

    tracks = {tid: tr for tid, tr in tl.tracks.items()}
    notes = sorted(tl.notes, key=lambda n: n.time)
    dur = tl.duration
    print(f"simulate: {tl.source} {dur:.1f}s, {len(notes)} notes, {len(tracks)} tracks -> {addr[0]}:{addr[1]} "
          f"mode={a.mode}", flush=True)
    if a.mode in ("score", "both"):
        for tid, tr in tracks.items():
            send(OscMessage("/myrmex/score/track", [tid, tr.name, tr.group]))
    if a.duration:
        dur = min(dur, a.duration)
    loops = 0
    paused = 0.0
    while True:
        t_start = time.perf_counter() + 0.2
        k = 0
        next_tr = 0.0
        next_score = -1.0
        song_offset = loops * tl.tempo.beats(dur)
        while True:
            now = time.perf_counter()
            t = now - t_start - paused
            if a.stop_at is not None and a.stop_for > 0 and t >= a.stop_at and paused == 0.0:
                # Press Stop, wait, press Play again (continue from the same song position).
                b = tl.tempo.beats(a.stop_at) + song_offset
                t_stop = time.perf_counter()
                while time.perf_counter() - t_stop < a.stop_for:
                    send(OscMessage("/myrmex/transport", [float(b), float(tl.tempo.bpm_at(a.stop_at)), 0, 4, 4]))
                    time.sleep(1.0 / a.rate)
                paused = time.perf_counter() - t_stop
                next_score = -1.0
                continue
            if t > dur:
                break
            beat = tl.tempo.beats(max(t, 0.0)) + song_offset
            bpm = tl.tempo.bpm_at(max(t, 0.0))
            if t >= next_tr:
                send(OscMessage("/myrmex/transport", [float(beat), float(bpm), 1, 4, 4]))
                next_tr = t + 1.0 / a.rate
            if a.mode in ("score", "both") and t >= next_score:
                # Like the Remote Script: every bar, the notes of the next two bars, in song beats.
                b0, b1 = beat, beat + 8.0
                by_track: dict[str, list] = {}
                for n in notes:
                    nb = tl.tempo.beats(n.time) + song_offset
                    if b0 <= nb < b1:
                        by_track.setdefault(n.track, []).append([nb, n.duration * bpm / 60.0, n.pitch, n.velocity])
                for tid in tracks:
                    arr = np.array(by_track.get(tid, []), dtype=float).reshape(-1, 4)
                    send(OscMessage("/myrmex/score/notes", [tid, pack_notes(arr), float(b0), float(b1)]))
                next_score = t + 4.0 * 60.0 / bpm
            if a.mode in ("notes", "both"):
                while k < len(notes) and notes[k].time <= t:
                    n = notes[k]
                    g = n.group or (tracks[n.track].group if n.track in tracks else "other")
                    send(OscMessage("/myrmex/note", [n.track, g, float(n.pitch), float(n.velocity), float(n.duration)]))
                    k += 1
            time.sleep(0.001)
        loops += 1
        if not a.loop:
            break
    for _ in range(3):
        send(OscMessage("/myrmex/transport", [float(tl.tempo.beats(dur)), float(tl.tempo.bpm_at(dur)), 0, 4, 4]))
        time.sleep(0.05)
    print("simulate: done")
    return 0


# ---------------------------------------------------------------------------- monitor
def cmd_monitor(a) -> int:
    from ..realtime.protocol import decode_names, decode_pose
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", a.port))
    sock.settimeout(0.5)
    n, t0, last_seq, lost = 0, time.time(), None, 0
    names = None
    end = time.time() + a.seconds if a.seconds else None
    print(f"monitor: listening on :{a.port}", flush=True)
    while end is None or time.time() < end:
        try:
            data, _ = sock.recvfrom(65536)
        except socket.timeout:
            continue
        if data[:4] == b"MYRN":
            nm = decode_names(data)
            if nm and names != nm[1]:
                names = nm[1]
                print(f"rig {nm[0]:08x}: {len(names)} bones", flush=True)
            continue
        fr = decode_pose(data)
        if fr is None:
            continue
        if last_seq is not None and fr.seq > last_seq + 1:
            lost += fr.seq - last_seq - 1
        last_seq = fr.seq
        n += 1
        if time.time() - t0 >= 1.0:
            cam = fr.camera.kind if fr.camera else "-"
            print(f"{n / (time.time() - t0):5.1f} fps  beat {fr.beat:8.2f}  bpm {fr.bpm:6.1f}  "
                  f"{'PLAY' if fr.playing else 'stop'}  speed {fr.speed:4.2f} m/s  cam {cam:<13} lost {lost}", flush=True)
            n, t0 = 0, time.time()
    return 0


# ---------------------------------------------------------------------------- ports
def cmd_ports(a) -> int:
    try:
        import mido
        print("MIDI inputs:", mido.get_input_names() or "none (macOS: Audio MIDI Setup > IAC Driver > Device is online)")
    except Exception as e:
        print("MIDI: unavailable -", e, "(pip install mido python-rtmidi)")
    try:
        import sounddevice as sd
        ins = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
        print("audio inputs:", ins or "none")
    except Exception as e:
        print("audio: unavailable -", e, "(pip install sounddevice)")
    try:
        from ..realtime.clock import LinkClock
        lc = LinkClock(120.0)
        time.sleep(1.5)
        st = lc.state(time.perf_counter())
        print(f"Ableton Link: ok, peers={st.peers}, tempo={st.bpm:.1f}, playing={st.playing}")
        lc.close()
    except Exception as e:
        print("Ableton Link: unavailable -", e, "(pip install aalink)")
    return 0


# ---------------------------------------------------------------------------- generate
def cmd_generate(a) -> int:
    from ..motion.bodyplan import BipedPlan
    from ..performance.generator import GenerateOptions, generate_biped
    from ..rig.rigdesc import RigDescription
    a.midi_file, a.als = (a.music if a.music.endswith((".mid", ".midi")) else None,
                          a.music if a.music.endswith(".als") else None)
    a.pattern = a.music if not (a.midi_file or a.als) else None
    tl = _load_music(a)
    plan = BipedPlan.from_rig(RigDescription.load(a.rig))
    perf = generate_biped(tl, plan, GenerateOptions(seed=a.seed, fps=a.fps, style={"preset": a.style}))
    perf.save(a.out)
    print(f"performance: {perf.frames} frames @ {perf.fps:g} fps -> {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="myrmex", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("live", help="realtime engine")
    q.add_argument("--rig", required=True, help="rig description JSON from the auto-rig step")
    q.add_argument("--out", nargs="+", default=["127.0.0.1:9101"], help="pose stream targets host:port")
    q.add_argument("--fps", type=float, default=60.0, help="pose packets per second")
    q.add_argument("--rate", type=float, default=120.0, help="simulation rate (Hz)")
    q.add_argument("--osc-port", type=int, default=9100)
    q.add_argument("--midi", nargs="*", help="MIDI input ports ('auto' = IAC Driver)")
    q.add_argument("--audio", default=None, help="audio input device for onset analysis (e.g. BlackHole 2ch)")
    q.add_argument("--clock", default="auto", choices=["auto", "osc", "link", "midi", "onsets", "internal"])
    q.add_argument("--bpm", type=float, default=120.0)
    q.add_argument("--no-link", action="store_true")
    q.add_argument("--latency", type=float, default=45.0, help="visual latency compensation (ms)")
    q.add_argument("--style", default="catwalk", choices=["catwalk", "swagger", "heels", "natural"])
    q.add_argument("--engine", default=None, help="JSON overrides for the behaviour engine")
    q.add_argument("--config", default=None, help="input mapping JSON (MIDI channels / CC / OSC addresses)")
    q.add_argument("--no-camera", action="store_true")
    q.add_argument("--record", default=None, help="directory: save the take (.npz) on exit")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--status-every", type=float, default=2.0)
    q.set_defaults(fn=cmd_live)

    q = sub.add_parser("simulate", help="fake Ableton: send a song as OSC")
    q.add_argument("--to", default="127.0.0.1:9100")
    q.add_argument("--pattern", default="demo", help="synthetic song: demo, A..F")
    q.add_argument("--midi-file", default=None)
    q.add_argument("--als", default=None)
    q.add_argument("--mode", default="notes", choices=["notes", "score", "both"])
    q.add_argument("--rate", type=float, default=20.0, help="transport messages per second")
    q.add_argument("--loop", action="store_true")
    q.add_argument("--duration", type=float, default=None, help="stop the song after this many seconds")
    q.add_argument("--stop-at", type=float, default=None, help="press Stop at this song time (s) ...")
    q.add_argument("--stop-for", type=float, default=0.0, help="... and Play again after this many seconds")
    q.set_defaults(fn=cmd_simulate)

    q = sub.add_parser("monitor", help="print the pose stream")
    q.add_argument("--port", type=int, default=9101)
    q.add_argument("--seconds", type=float, default=0.0)
    q.set_defaults(fn=cmd_monitor)

    q = sub.add_parser("ports", help="list MIDI / audio / Link")
    q.set_defaults(fn=cmd_ports)

    q = sub.add_parser("generate", help="offline performance for rendering")
    q.add_argument("--rig", required=True)
    q.add_argument("--music", default="demo", help="demo | A..F | song.mid | set.als")
    q.add_argument("--out", required=True)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--fps", type=float, default=30.0)
    q.add_argument("--style", default="catwalk")
    q.set_defaults(fn=cmd_generate)

    a = p.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
