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


def _hand(dec, t, fingers=0.5, **kw):
    """``fingers``: extension 0 (curled) .. 1 (open), one value or five; the glove sends flexion."""
    ext = fingers if isinstance(fingers, (list, tuple)) else [fingers] * 5
    vals = {f: 1.0 - ext[i] for i, f in enumerate(PARAMS[:5])}
    vals.update(kw)
    feed(dec, vals, t)


def _calibrate(dec, link, t, variant="polyalloy"):
    """Open and close the hand a few times: the link learns each finger's range."""
    for k in range(24):
        t += 1 / 60
        _hand(dec, t, 1.0 if (k // 6) % 2 else 0.0)
        if k == 1:
            dec.autodetect(t)
        link.tick(dec, t, 1 / 60, variant)
    return t


def _drive(preset, variant, frames, hand, engine=None, clock=None):
    """Run a preset: ``hand(k) -> dict`` of glove values (0..1) per 1/60 s frame; returns (link, engine, ctrls)."""
    from myrmex.creature.backend import CreatureBackend
    dec = GloveDecoder()
    link = GloveLink({"preset": preset, "smoothing": 0.0})
    eng = engine or CreatureBackend(None, False, variant).engine
    dt, ctrls = 1 / 60, []
    t = _calibrate(dec, link, 0.0, variant)
    for k in range(frames):
        t += dt
        vals = hand(k)
        _hand(dec, t, vals.pop("fingers", 0.5), **vals)
        ctrl, events, _ = link.tick(dec, t, dt, variant, clock=clock(t) if clock else None, view_yaw=0.0)
        eng.set_glove(ctrl, link.sculpt_shapes(variant))
        for _, names in events:
            for n in names:
                if eng.trigger_event(n):
                    break
        eng.update(dt)
        ctrls.append(ctrl)
    return link, eng, ctrls


def test_twenty_presets_each_couple_differently():
    from myrmex.realtime.glove import PRESETS
    assert len([p for p in PRESETS if p != "off"]) == 20
    sigs = {}
    for p in PRESETS:
        dec = GloveDecoder()
        link = GloveLink({"preset": p, "smoothing": 0.0})
        t = _calibrate(dec, link, 0.0, "hive")
        for k in range(40):                           # the same hand for every preset
            t += 1 / 60
            _hand(dec, t, [0.9, 0.2, 0.8, 0.3, 0.7], roll=0.58, pitch=0.46, yaw=0.55, x=0.6, y=0.62, z=0.4)
            ctrl, _, cam = link.tick(dec, t, 1 / 60, "hive", clock=(t * 2.0, 120.0, True), view_yaw=0.0)
        c = ctrl
        sigs[p] = tuple(np.round(np.concatenate([c.rot.ravel(), c.offset, c.stretch, c.waves, c.wind, c.angvel,
                                                 [c.scale, c.grip, c.spin, c.material or 0, c.energy or 0, c.twist,
                                                  c.pulse, c.scatter, c.freeze, c.rays, c.ray_len, c.speed, c.flock,
                                                  c.swarm_release, c.lines or 0, c.orbit, float(c.active),
                                                  cam["distance"], cam["orbit"], len(c.finger_mode)]]), 3))
    same = [(a, b) for i, a in enumerate(PRESETS) for b in PRESETS[i + 1:] if sigs[a] == sigs[b]]
    assert not same                                           # no two presets map the hand the same way


def test_presets_act_on_the_organism():
    open_hand = lambda k: {"fingers": 0.95}
    fist = lambda k: {"fingers": 0.05}
    # stasis: a fist freezes the motion
    _, e_open, _ = _drive("stasis", "polyalloy", 180, open_hand)
    _, e_fist, _ = _drive("stasis", "polyalloy", 180, fist)
    sp = lambda e: float(np.linalg.norm(e.v - e.v.mean(0), axis=1).mean())
    assert sp(e_fist) < 0.5 * sp(e_open)
    # dust: the open hand scatters the material
    _, e_dust, _ = _drive("dust", "polyalloy", 240, open_hand)
    assert e_dust.fragments() > e_fist.fragments() or float(e_dust.state().dispersion.mean()) > \
        float(e_fist.state().dispersion.mean()) + 0.05
    # shepherd: three extended fingers -> a flock of three
    _, e_flock, _ = _drive("shepherd", "colony", 480, lambda k: {"fingers": [0.95, 0.95, 0.95, 0.05, 0.05]})
    assert len(e_flock._alive()) == 3
    # swarm: the open hand lets the nanomachines out
    _, e_sw, _ = _drive("swarm", "hive", 150, open_hand)
    assert (~e_sw.p_bound).mean() > 0.3
    # leash: it flies to where the hand points (10 s)
    _, e_lh, ctrls = _drive("leash", "polyalloy", 600, lambda k: {"x": 0.95, "z": 0.5, "fingers": 0.1})
    goal = ctrls[-1].point
    assert float(np.linalg.norm(e_lh.P[:2] - goal[:2])) < 4.0


def test_rhythm_snaps_on_the_beat_and_echo_answers_later():
    from myrmex.creature.puppet import rotvec
    link, _, ctrls = _drive("rhythm", "polyalloy", 120, lambda k: {"yaw": 0.5 + 0.1 * min(1.0, k / 30)},
                            clock=lambda t: (t * 2.0, 120.0, True))
    ang = [abs(float(rotvec(c.rot)[2])) for c in ctrls]
    steps = sorted(set(round(a / (math.pi / 4), 3) for a in ang))
    assert all(abs(s - round(s)) < 1e-6 for s in steps)          # only multiples of 45 degrees
    _, _, ctrls = _drive("echo", "polyalloy", 90, lambda k: {"yaw": 0.75 if k >= 30 else 0.5},
                         clock=lambda t: (t * 2.0, 120.0, True))  # a beat = 0.5 s = 30 frames
    yaw = [float(rotvec(c.rot)[2]) for c in ctrls]
    assert abs(yaw[45]) < 0.2 and abs(yaw[-1] - math.pi / 2) < 0.2
