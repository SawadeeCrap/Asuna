"""Hand Glove link: decoding (14-bit CC, pitch bend, OSC), linking, gestures, presets, the whole path."""
import math

import numpy as np

from myrmex.bus.transport import LiveEvent
from myrmex.realtime.glove import PARAMS, GloveDecoder, GloveLink

CC0 = 20                                             # a glove sending 14-bit CC 20..30 (+32 LSB), channel 1


def feed(dec, vals, t, ch=1, fourteen=True):
    for i, p in enumerate(PARAMS):
        v = int(round(vals.get(p, 0.5) * 16383))
        if fourteen:
            dec.feed_cc(ch, CC0 + i, v >> 7, t)
            dec.feed_cc(ch, CC0 + i + 32, v & 127, t)
        else:
            dec.feed_cc(ch, CC0 + i, v >> 7, t)


def test_decoder_14bit_autodetect_and_ownership():
    dec = GloveDecoder()
    for k in range(5):
        feed(dec, {"thumb": 0.25, "yaw": 0.75}, 0.01 * k)
    prof = dec.autodetect(0.05)
    assert prof["thumb"] == f"cc14:1:{CC0}" and prof["z"] == f"cc14:1:{CC0 + 10}"
    assert abs(dec.value("thumb", 0.05) - 0.25) < 1e-3 and abs(dec.value("yaw", 0.05) - 0.75) < 1e-3
    assert dec.owns(1, CC0 + 3) and dec.owns(1, CC0 + 3 + 32) and not dec.owns(1, 7)


def test_learn_and_osc():
    dec = GloveDecoder()
    feed(dec, {}, 0.0)
    dec.learn_param("index", 0.1)
    for k in range(6):                               # MAP pressed: only 'index' (CC 21) moves
        dec.feed_cc(1, CC0 + 1, 40 + k, 0.2 + 0.05 * k)
        dec.feed_cc(1, CC0 + 1 + 32, 0, 0.2 + 0.05 * k)
    assert dec.poll_learn(2.0) == f"cc14:1:{CC0 + 1}" and dec.profile["index"] == f"cc14:1:{CC0 + 1}"
    dec2 = GloveDecoder()
    assert dec2.feed_osc("/glove/roll", 8192.0, 0.0) and abs(dec2.value("roll", 0.0) - 0.5) < 1e-3


def test_gestures_and_presets():
    dec = GloveDecoder()
    link = GloveLink({"preset": "puppet", "smoothing": 0.0, "sensitivity": 0.6})
    t, dt, fired = 0.0, 1 / 60, []
    for k in range(240):                             # open hand, then a fist, then a fast flick
        t += dt
        closed = 0.9 if 60 <= k < 120 else 0.1
        yaw = 0.5 + (0.25 if k >= 180 else 0.0) * min(1.0, (k - 180) / 4.0) if k >= 180 else 0.5
        feed(dec, {**{f: closed for f in PARAMS[:5]}, "yaw": yaw}, t)
        if k == 2:
            dec.autodetect(t)
        ctrl, events, cam = link.tick(dec, t, dt, "polyalloy")
        fired += [g for g, _ in events]
    assert ctrl.active and "CLENCH" in fired and "SPREAD" in fired and "FLICK" in fired
    ang = link.state.ang
    assert abs(ang[2] - math.pi / 2) < 0.15                        # a quarter of the range = 90 degrees
    link.configure(preset="camera")
    ctrl, _, cam = link.tick(dec, t + dt, dt, "polyalloy")
    assert not ctrl.active and abs(cam["orbit"] - ang[2]) < 0.2


def test_session_puppets_the_creature():
    from myrmex.realtime.session import LiveConfig, LiveSession

    class Sink:
        sent = 0

        def send_raw(self, b):
            pass

        def close(self):
            pass
    s = LiveSession(LiveConfig(backend="polyalloy", clock="internal", out=[], glove={"preset": "puppet"}),
                    start_inputs=False, sink=Sink(), now=0.0)
    now = 0.0
    for k in range(360):
        now += 1 / 120
        yaw = 0.5 + 0.25 * min(1.0, max(0.0, (k - 120) / 60))        # the hand turns 90 degrees
        for i, p in enumerate(PARAMS):
            v = int(round((yaw if p == "yaw" else 0.5) * 16383))
            s.inputs.push(LiveEvent("cc", now, {"channel": 1, "control": CC0 + i, "value": (v >> 7) / 127,
                                                "raw": v >> 7}))
            s.inputs.push(LiveEvent("cc", now, {"channel": 1, "control": CC0 + i + 32, "value": (v & 127) / 127,
                                                "raw": v & 127}))
        s.step(now)                                  # no Learn, no button: it links by itself
    st = s.status()
    G = s.creature.engine.glove.G
    turned = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(G) - 1) / 2))))
    assert st["glove"]["present"] and abs(turned - 90) < 12
    assert "stride" not in s.inputs.controls and "energy" not in s.inputs.controls   # glove CCs don't leak into params
