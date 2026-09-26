"""Hand Glove -> creature: decoding, calibration, gestures and synchronisation presets.

The Hand Glove app streams 11 continuous values over MIDI (0..16383 in its window: 14-bit CC pairs,
7-bit CC or pitch bend are all understood).  Myrmex listens to the same MIDI port *in parallel* with
the other programs (macOS lets any number of apps read one source) - or, if the glove sends OSC,
to ``/glove/<name>`` on the OSC port (value 0..1 or 0..16383).

Continuous values have no triggers, so the link derives them from motion: a fast flick of the hand,
clenching into a fist, spreading the fingers, pushing towards the screen, a pinch.  Everything is
filtered with a One-Euro filter (smooth when still, instant when moving).

Presets (how the hand and the organism are coupled) - 20 couplings + off, see docs/GLOVE.md:
    puppet · sculpt · conductor · camera · marionette · harp · heartbeat · elastic · dust · stasis · storm ·
    leash · pilot · flywheel · shepherd · swarm · neon · rhythm · echo · mandala · off
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..creature.puppet import GloveControl, euler

PARAMS = ("thumb", "index", "middle", "ring", "pinky", "roll", "pitch", "yaw", "x", "y", "z")
FINGERS = PARAMS[:5]
PRESETS = ("puppet", "sculpt", "conductor", "camera", "marionette", "harp", "heartbeat", "elastic", "dust", "stasis",
           "storm", "leash", "pilot", "flywheel", "shepherd", "swarm", "neon", "rhythm", "echo", "mandala", "off")
HIVES = ("hive", "osseous_hive", "cyber_hive", "swarm", "cloud")
GESTURES = ("FLICK", "CLENCH", "SPREAD", "PUSH", "PINCH")
# gesture -> events tried in order (the first one the organism knows is used)
GESTURE_EVENTS = {"FLICK": ("STRIKE", "IMPULSE", "APPENDAGE_BURST"),
                  "CLENCH": ("OSSIFY", "PRESSURE", "MASS_REBALANCE"),
                  "SPREAD": ("QUILLS", "APPENDAGE_BURST"),
                  "PUSH": ("STRIKE", "IMPULSE"),
                  "PINCH": ("SPLIT", "MORPHOLOGY_SHIFT")}


class OneEuro:
    """One-Euro filter (Casiez et al.): low jitter at rest, low lag in motion."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.4, d_cutoff: float = 1.5):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _a(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def __call__(self, x: float, t: float) -> float:
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(t - self.t, 1e-4)
        self.t = t
        dx = (x - self.x) / dt
        self.dx += self._a(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._a(cutoff, dt) * (x - self.x)
        return self.x


# ============================================================================ decoding
class GloveDecoder:
    """Raw MIDI / OSC -> 11 normalised glove values.  ``profile``: param -> source key.

    Source keys: ``cc:<ch>:<num>`` (7-bit), ``cc14:<ch>:<num>`` (MSB num + LSB num+32), ``pb:<ch>``,
    ``osc:<name>``.
    """

    def __init__(self, profile: dict | None = None):
        self.profile: dict[str, str] = dict(profile or {})
        self.src: dict[str, tuple[float, float]] = {}
        self.count: dict[str, int] = {}
        self.msb: dict[tuple[int, int], int] = {}
        self.lsb: dict[tuple[int, int], int] = {}
        self.pairs: set[tuple[int, int]] = set()
        self.learn: tuple[str, float, dict] | None = None     # (param, until, counts at start)
        self.last_learned: tuple[str, str] | None = None
        self.msgs = 0
        self.rate = 0.0
        self._rate_t, self._rate_n = time.perf_counter(), 0

    # ---------------------------------------------------------------- input
    def feed_cc(self, ch: int, num: int, v7: int, t: float) -> None:
        base = num - 32
        if 32 <= num <= 63 and (ch, base) in self.msb:         # LSB of a 14-bit pair
            self.lsb[(ch, base)] = v7
            self.pairs.add((ch, base))
            self._update(f"cc14:{ch}:{base}", (self.msb[(ch, base)] * 128 + v7) / 16383.0, t)
            return
        self.msb[(ch, num)] = v7
        if (ch, num) in self.pairs:
            self._update(f"cc14:{ch}:{num}", (v7 * 128 + self.lsb.get((ch, num), 0)) / 16383.0, t)
        else:
            self._update(f"cc:{ch}:{num}", v7 / 127.0, t)

    def feed_pb(self, ch: int, value01: float, t: float) -> None:
        self._update(f"pb:{ch}", value01, t)

    def feed_osc(self, address: str, value: float, t: float) -> bool:
        parts = address.strip("/").lower().split("/")
        if len(parts) < 2 or parts[0] not in ("glove", "hand", "myrmex") or parts[-1] not in PARAMS:
            return False
        name = parts[-1]
        v = value if value <= 1.0 else value / 16383.0
        self._update(f"osc:{name}", v, t)
        self.profile.setdefault(name, f"osc:{name}")           # OSC names itself: linked automatically
        return True

    def _update(self, key: str, val: float, t: float) -> None:
        self.src[key] = (float(min(1.0, max(0.0, val))), t)
        self.count[key] = self.count.get(key, 0) + 1
        self.msgs += 1
        self._rate_n += 1
        now = time.perf_counter()
        if now - self._rate_t >= 1.0:
            self.rate, self._rate_t, self._rate_n = self._rate_n / (now - self._rate_t), now, 0

    # ---------------------------------------------------------------- output
    def _key(self, param: str) -> str | None:
        key = self.profile.get(param)
        if key and key.startswith("cc:"):                      # a pair found later: use the precise value
            _, ch, num = key.split(":")
            if f"cc14:{ch}:{num}" in self.src:
                return f"cc14:{ch}:{num}"
        return key

    def value(self, param: str, now: float, fresh: float = 1.0) -> float | None:
        key = self._key(param)
        if key is None or key not in self.src:
            return None
        v, t = self.src[key]
        return v if now - t <= fresh else None

    def owns(self, ch: int, num: int) -> bool:
        """Is this CC part of the glove (then it drives the puppet, not a parameter)?"""
        for key in self.profile.values():
            if key.startswith("cc"):
                _, c, n = key.split(":")
                if int(c) == ch and int(n) in (num, num - 32):
                    return True
        return False

    def owns_pb(self, ch: int) -> bool:
        return f"pb:{ch}" in self.profile.values()

    def active(self, now: float, fresh: float = 0.5) -> bool:
        return any(self.value(p, now, fresh) is not None for p in PARAMS)

    # ---------------------------------------------------------------- linking
    def autodetect(self, now: float, window: float = 2.0) -> dict:
        """Streaming sources, in (channel, number) order, become thumb .. z (the glove's own order)."""
        live = [k for k, (_, t) in self.src.items() if now - t <= window and not k.startswith("osc:")]
        live = [k for k in live if not (k.startswith("cc:") and "cc14:" + k[3:] in self.src)]

        def order(k):
            p = k.split(":")
            return (0 if p[0] != "pb" else 1, int(p[1]), int(p[2]) if len(p) > 2 else 0)
        live.sort(key=order)
        self.profile = {p: k for p, k in zip(PARAMS, live)}
        return dict(self.profile)

    def learn_param(self, param: str, now: float, window: float = 1.5) -> None:
        """Press MAP on that row in Hand Glove within ``window`` seconds."""
        self.learn = (param, now + window, dict(self.count))

    def poll_learn(self, now: float) -> str | None:
        """Finish a learn window: the source that moved most (and clearly most) is the one."""
        if self.learn is None or now < self.learn[1]:
            return None
        param, _, base = self.learn
        self.learn = None
        taken = {k for p, k in self.profile.items() if p != param}
        gains = sorted(((self.count.get(k, 0) - base.get(k, 0), k) for k in self.count if k not in taken), reverse=True)
        gains = [(g, k) for g, k in gains if not (k.startswith("cc:") and "cc14:" + k[3:] in self.count)]
        if not gains or gains[0][0] < 2 or (len(gains) > 1 and gains[0][0] < 1.6 * gains[1][0]):
            self.last_learned = (param, "")
            return ""
        self.profile[param] = gains[0][1]
        self.last_learned = (param, gains[0][1])
        return gains[0][1]


# ============================================================================ state
class GloveState:
    """Calibrated, filtered hand: finger extension, orientation (rad), position (-1..1), gestures."""

    def __init__(self, smoothing: float = 0.5, invert_fingers: bool = False, sensitivity: float = 0.5,
                 neutral: dict | None = None):
        self.smoothing, self.invert, self.sensitivity = smoothing, invert_fingers, sensitivity
        self.filters = {p: OneEuro(*self._filter_args()) for p in PARAMS}
        self.lo = np.full(5, np.inf)
        self.hi = np.full(5, -np.inf)
        self.neutral = dict(neutral or {})
        self.raw: dict[str, float] = {}
        self.ang = np.zeros(3)
        self._ang_prev = None
        self.omega = np.zeros(3)
        self.ext = np.full(5, 0.5)
        self.pos = np.zeros(3)
        self.pos_v = np.zeros(3)
        self.flex_hist: list[tuple[float, float]] = []
        self.last_gesture: dict[str, float] = {}
        self.gestures: list[tuple[float, str]] = []
        self.present = False
        self.t = None

    def _filter_args(self):
        s = self.smoothing                                    # 0 raw .. 1 very smooth
        return 3.0 - 2.4 * s, 0.9 - 0.7 * s

    def set_smoothing(self, s: float) -> None:
        if abs(s - self.smoothing) > 1e-3:
            self.smoothing = s
            for f in self.filters.values():
                f.min_cutoff, f.beta = self._filter_args()

    def calibrate(self) -> dict:
        """The current pose becomes the neutral one; finger ranges are learned again."""
        self.neutral = {p: self.raw[p] for p in ("roll", "pitch", "yaw", "x", "y", "z") if p in self.raw}
        self.lo[:] = np.inf
        self.hi[:] = -np.inf
        self._ang_prev = None
        return dict(self.neutral)

    def update(self, dec: GloveDecoder, now: float, dt: float) -> list[str]:
        """Read the decoder; returns the gestures that fired."""
        vals = {p: dec.value(p, now) for p in PARAMS}
        self.present = any(v is not None for v in vals.values())
        if not self.present:
            return []
        for p, v in vals.items():
            if v is not None:
                self.raw[p] = self.filters[p](v, now)
        # fingers: self-calibrating range (grows at once, forgets slowly)
        for k, p in enumerate(FINGERS):
            if p not in self.raw:
                continue
            v = self.raw[p]
            self.lo[k] = min(v, self.lo[k] + dt * 0.004) if np.isfinite(self.lo[k]) else v
            self.hi[k] = max(v, self.hi[k] - dt * 0.004) if np.isfinite(self.hi[k]) else v
            span = self.hi[k] - self.lo[k]
            flex = (v - self.lo[k]) / span if span > 0.05 else 0.5
            self.ext[k] = flex if self.invert else 1.0 - flex
        # orientation: angles relative to the neutral pose, unwrapped (continuous turning)
        ang = np.array([(self.raw.get(p, 0.5) - self.neutral.get(p, 0.5)) * (2 * math.pi if p != "pitch" else math.pi)
                        for p in ("roll", "pitch", "yaw")])
        if self._ang_prev is not None:
            ang = self._ang_prev + (ang - self._ang_prev + math.pi) % (2 * math.pi) - math.pi
            self.omega += (np.clip((ang - self._ang_prev) / max(dt, 1e-3), -40, 40) - self.omega) * min(1.0, dt / 0.05)
        self._ang_prev = ang
        self.ang = ang
        pos = np.array([(self.raw.get(p, 0.5) - self.neutral.get(p, 0.5)) * 2.0 for p in ("x", "y", "z")])
        self.pos_v += ((pos - self.pos) / max(dt, 1e-3) - self.pos_v) * min(1.0, dt / 0.06)
        self.pos = pos
        return self._gestures(now)

    def _gestures(self, now: float) -> list[str]:
        k = 1.4 - 0.9 * self.sensitivity                       # sensitivity 0..1 -> thresholds x1.4 .. x0.5
        fired = []
        flex = 1.0 - float(self.ext.mean())
        self.flex_hist.append((now, flex))
        while self.flex_hist and now - self.flex_hist[0][0] > 0.25:
            self.flex_hist.pop(0)
        dflex = flex - self.flex_hist[0][1] if self.flex_hist else 0.0
        e = self.ext
        cands = {"FLICK": float(np.abs(self.omega).max()) > 7.0 * k,
                 "CLENCH": dflex > 0.4 * k and flex > 0.6,
                 "SPREAD": dflex < -0.4 * k and flex < 0.4,
                 "PUSH": self.pos_v[2] > 3.0 * k,
                 "PINCH": e[0] < 0.3 and e[1] < 0.3 and min(e[2], e[3], e[4]) > 0.6}
        for g, on in cands.items():
            last = self.last_gesture.get(g, -1e9)
            armed = self.last_gesture.get(g + "_armed", True)
            if on and armed and now - last > 0.35:
                fired.append(g)
                self.last_gesture[g] = now
                self.last_gesture[g + "_armed"] = False
            elif not on:
                self.last_gesture[g + "_armed"] = True
        for g in fired:
            self.gestures.append((now, g))
        self.gestures = self.gestures[-12:]
        return fired


# ============================================================================ link
SCULPT_SHAPES = {   # five forms per organism, one per finger (thumb .. pinky)
    "polyalloy": ("CORE", "SPINDLE", "WINGS", "RING", "SHIELD"),
    "osseous": ("CARAPACE", "SPINE", "SCYTHE", "CLAW", "THORN"),
    "colony": ("CORE", "TENDRILS", "WINGS", "RING", "CROWN"),
    "hive": ("CORE", "TENDRILS", "WINGS", "RING", "CROWN"),
    "osseous_colony": ("CARAPACE", "SPINE", "SCYTHE", "MANDIBLE", "THORN"),
    "osseous_hive": ("CARAPACE", "SPINE", "SCYTHE", "MANDIBLE", "THORN"),
    "cyber_hive": ("PRISM", "HALO", "ARRAY", "SCYTHE", "SPINE"),
    "swarm": ("STREAM", "TENDRILS", "CLOUD", "SPINE", "CORE"),
    "spear": ("LANCE", "SPINE", "SCYTHE", "THORN", "CORE"),
    "cloud": ("SHARDS", "CLOUD", "STREAM", "SHIELD", "CORE"),
    "blade": ("SWEEP", "SCYTHE", "WINGS", "CLAW", "CORE"),
    "crawler": ("CRAWL", "SPINE", "CLAW", "THORN", "CORE"),
    "tensor": ("SWIM", "COIL", "HELIX", "SPRING", "REACH"),
    "fold": ("GLIDER", "PLEAT", "TUBE", "SHELL", "BELL"),
    "arbor": ("SPHERE", "FAN", "SPIRAL", "HALO", "COMET"),
    "ferro": ("CROWN", "URCHIN", "TONGUE", "LABYRINTH", "STAR"),
    "truss": ("FUSELAGE", "WING", "ARCH", "RING", "TRIPOD"),
}


class GloveLink:
    """Settings + decoder state -> per-tick GloveControl for the engine, events, camera modulation."""

    DEFAULTS = {"preset": "puppet", "intensity": 1.0, "smoothing": 0.45, "invert_fingers": False,
                "gestures": True, "sensitivity": 0.5, "profile": {}, "neutral": {}}

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(self.DEFAULTS, **(cfg or {}))
        if self.cfg["preset"] not in PRESETS:
            self.cfg["preset"] = "puppet"
        self.state = GloveState(self.cfg["smoothing"], self.cfg["invert_fingers"], self.cfg["sensitivity"],
                                self.cfg["neutral"])
        self.pinches = 0
        self.last_events: list[tuple[float, str, str]] = []
        # per-preset memory
        self._alt = 0.0                                   # pilot: altitude integrated from climb
        self._wheel = np.zeros(3)                         # flywheel: angular velocity thrown in
        self._hist: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []   # echo: (t, angles, ext, pos)
        self._held = (np.zeros(3), -1)                    # rhythm: quantized angles, beat index
        self._pluck = np.zeros(5)                         # harp: ringing strings
        self._ext_prev = None
        self._flash = 0.0                                 # neon: a light flash from a finger tap
        self._clock_t = 0.0

    def configure(self, **kw) -> None:
        self.cfg.update(kw)
        if self.cfg["preset"] not in PRESETS:
            self.cfg["preset"] = "puppet"
        self.state.set_smoothing(self.cfg["smoothing"])
        self.state.invert = bool(self.cfg["invert_fingers"])
        self.state.sensitivity = float(self.cfg["sensitivity"])

    def tick(self, dec: GloveDecoder, now: float, dt: float, variant: str = "", clock=None, view_yaw=None):
        """-> (GloveControl, [(gesture, [event names to try])], camera modulation dict).

        ``clock`` = (beat, bpm, playing) of the song, ``view_yaw`` = the camera's heading (rad, world):
        presets that lead the organism to a place map the hand into the camera's view."""
        c = self.cfg
        if not dec.profile and c["preset"] != "off":         # link by itself: the glove streams 11 controls
            live = [k for k, (_, ts) in dec.src.items() if now - ts <= 1.0]
            if len(live) >= 8:
                self._seen = getattr(self, "_seen", now)
                if now - self._seen > 1.0:
                    dec.autodetect(now)
            else:
                self._seen = now
        gestures = self.state.update(dec, now, dt)
        st = self.state
        preset = c["preset"]
        cam = {"distance": 1.0, "orbit": 0.0, "height": 0.0}
        if not st.present or preset == "off":
            self._ext_prev = None
            return GloveControl(), [], cam
        self._clock_t += dt
        if clock is not None and clock[2]:
            beat, bpm = float(clock[0]), max(40.0, float(clock[1] or 120.0))
        else:
            bpm = float(clock[1]) if clock is not None and clock[1] else 120.0
            beat = self._clock_t * bpm / 60.0
        k = float(c["intensity"])
        amt = min(1.5, k)
        ang = st.ang
        ext = st.ext.copy()
        pos = st.pos
        if preset == "echo":                                  # the organism answers one beat later
            self._hist.append((now, ang.copy(), ext.copy(), pos.copy()))
            delay = 60.0 / bpm
            while len(self._hist) > 2 and self._hist[1][0] <= now - delay:
                self._hist.pop(0)
            if self._hist[0][0] <= now - delay:
                _, ang, ext, pos = self._hist[0]
            else:
                ang, ext, pos = np.zeros(3), np.full(5, 0.5), np.zeros(3)
        else:
            self._hist.clear()
        roll, pitch, yaw = ang * amt
        opn = float(ext.mean())                               # 0 fist .. 1 open hand
        flex = 1.0 - opn
        ctrl = GloveControl(active=True, fingers=ext.copy(), amount=amt, grip=min(1.0, 0.55 + 0.4 * k))
        x, y, z = np.clip(pos, -1.5, 1.5)
        d_ext = (ext - self._ext_prev) / max(dt, 1e-3) if self._ext_prev is not None else np.zeros(5)
        self._ext_prev = ext.copy()

        if preset in ("puppet", "echo"):
            ctrl.rot = euler(roll, pitch, yaw)
            ctrl.offset = np.array([x, y, z])
            ctrl.scale = float(np.clip(1.0 + 0.35 * z * k, 0.6, 1.6))
            cam["distance"] = float(np.clip(1.0 - 0.35 * z, 0.55, 1.5))
        elif preset == "sculpt":
            ctrl.rot = euler(roll, pitch, yaw)
            ctrl.finger_mode = "morph"
            ctrl.scale = float(np.clip(0.85 + 0.35 * opn * k + 0.2 * z, 0.6, 1.6))
            ctrl.grip = 1.0
        elif preset == "conductor":
            ctrl.spin = float(np.clip(roll * 2.5, -8.0, 8.0))
            ctrl.rot = euler(0.0, 0.3 * pitch, 0.0)
            ctrl.material = float(np.clip(-pitch / 0.9, -1.0, 1.0))
            ctrl.energy = float(np.clip(opn * k, 0.0, 1.0))
            ctrl.offset = np.array([float(np.clip(yaw / 1.2, -1, 1)), y, 0.0])
            ctrl.finger_mode = "none"
            cam["distance"] = float(np.clip(1.0 - 0.35 * z, 0.55, 1.5))
        elif preset == "camera":
            ctrl = GloveControl()                               # the organism stays free
            cam = {"distance": float(np.clip(1.0 - 0.45 * z, 0.45, 1.8)), "orbit": float(yaw),
                   "height": float(np.clip(y * 1.5 + pitch, -1.0, 2.5))}
        elif preset == "marionette":                           # fingers are strings along the body
            ctrl.rot = euler(0.6 * roll, 0.6 * pitch, 0.0)
            ctrl.finger_mode = "strings"
            ctrl.offset = np.array([x, y, 0.0])
            ctrl.grip = 0.8
        elif preset == "harp":                                 # every finger plucks its own wave
            self._pluck = np.maximum(self._pluck * math.exp(-dt / 1.4), np.clip(np.abs(d_ext) * 0.3, 0, 1.2))
            ctrl.waves = np.clip(0.25 * ext + self._pluck, 0.0, 1.3) * amt
            ctrl.wave_speed = float(np.clip(1.0 + roll / 1.5, 0.2, 3.0))
            ctrl.finger_mode = "none"
            ctrl.offset = np.array([x, y, 0.0])
            ctrl.energy = float(np.clip(self._pluck.max(), 0.0, 1.0))
        elif preset == "heartbeat":                            # it breathes on the beat, the open hand = depth
            ctrl.pulse = float(np.clip(opn * amt, 0.0, 1.5))
            ctrl.material = float(np.clip((flex - 0.5) * 1.8, -0.6, 1.0))
            ctrl.energy = float(np.clip(0.3 + 0.7 * opn, 0.0, 1.0))
            ctrl.rot = euler(0.5 * roll, 0.5 * pitch, 0.5 * yaw)
            ctrl.grip = 0.35
            ctrl.finger_mode = "none"
            ctrl.scale = float(np.clip(1.0 + 0.25 * z, 0.7, 1.4))
        elif preset == "elastic":                              # the hand's position stretches it, a twist screws it
            ctrl.stretch = np.clip(1.0 + 0.9 * np.array([z, x, y]) * k, 0.4, 2.2)
            ctrl.twist = float(np.clip(roll * 1.5, -5.0, 5.0))
            ctrl.rot = euler(0.0, 0.5 * pitch, 0.5 * yaw)
            ctrl.grip = 0.7
            ctrl.material = float(np.clip(0.8 - 1.6 * opn, -0.8, 0.8))
            ctrl.finger_mode = "none"
        elif preset == "dust":                                 # open hand scatters it, the fist gathers it
            sc = float(np.clip((opn - 0.35) / 0.5, 0.0, 1.0)) ** 1.5 * min(1.0, k)
            ctrl.scatter = sc
            ctrl.material = float(np.clip(-0.8 * sc + (0.6 if flex > 0.7 else 0.0), -1.0, 1.0))
            ctrl.spin = float(np.clip(roll * 2.0, -6.0, 6.0))
            ctrl.grip = 0.95 if flex > 0.7 else 0.4
            ctrl.energy = float(np.clip(0.2 + 0.6 * sc, 0.0, 1.0))
            ctrl.offset = np.array([x, y, 0.0])
            ctrl.finger_mode = "none"
        elif preset == "stasis":                               # the fist stops time, the hand turns the still form
            ctrl.freeze = float(np.clip((flex - 0.35) / 0.45, 0.0, 1.0))
            ctrl.rot = euler(roll, pitch, yaw)
            ctrl.grip = 1.0
            ctrl.energy = float(np.clip(opn - 0.3, 0.0, 1.0))
            ctrl.lines = 0.25 + 0.6 * ctrl.freeze
            ctrl.finger_mode = "none"
        elif preset == "storm":                                # the hand is the wind: direction and strength
            d = euler(0.0, pitch, yaw) @ np.array([-1.0, 0.0, 0.0])
            ctrl.wind = d * 14.0 * opn * k
            ctrl.scatter = 0.2 * opn
            ctrl.offset = np.array([x, y, 0.0])
            ctrl.energy = float(np.clip(opn, 0.0, 1.0))
            ctrl.finger_mode = "none"
        elif preset == "leash":                                # it flies where the hand points, circles there
            fy = view_yaw if view_yaw is not None else 0.0
            fwd, right = np.array([math.cos(fy), math.sin(fy)]), np.array([math.sin(fy), -math.cos(fy)])
            xy = right * x * 8.0 + fwd * z * 8.0
            ctrl.point = np.array([xy[0], xy[1], float(np.clip(3.5 + 2.5 * y, 1.0, 9.0))])
            ctrl.orbit = float(np.clip(roll * 1.2, -2.5, 2.5))
            ctrl.orbit_radius = 1.5 + 4.5 * opn
            ctrl.amount = 0.6 * amt
        elif preset == "pilot":                                # bank to turn, pitch to climb, open = throttle
            ctrl.rot = euler(0.8 * roll, 0.8 * pitch, 0.0)
            ctrl.grip = 0.8
            self._alt = float(np.clip(self._alt - pitch * dt * 0.9, -1.0, 2.5))
            ctrl.offset = np.array([float(np.clip(-roll / 0.7, -1.0, 1.0)), self._alt, 0.0])
            ctrl.speed = 0.4 + 1.6 * opn
            ctrl.amount = 0.5 * amt
            cam["distance"] = float(np.clip(1.15 - 0.3 * opn, 0.7, 1.3))
        elif preset == "flywheel":                             # throw a spin into it, the fist brakes
            tau = 0.25 if flex > 0.75 else 4.0
            self._wheel = self._wheel * math.exp(-dt / tau) + st.omega * amt * dt * 6.0
            self._wheel = np.clip(self._wheel, -12.0, 12.0)
            ctrl.angvel = self._wheel.copy()
            ctrl.amount = 0.7 * amt
            ctrl.energy = float(np.clip(np.abs(self._wheel).max() / 8.0, 0.0, 1.0))
        elif preset == "shepherd":                             # extended fingers = bodies, the hand turns the flock
            n_up = int((ext > 0.6).sum())
            ctrl.flock = int(np.clip(n_up, 1, 4))
            ctrl.formation_turn = float(yaw)
            ctrl.formation_spread = 0.4 + 1.4 * opn
            ctrl.offset = np.array([x, y, 0.0])
            ctrl.rot = euler(0.0, 0.0, 0.4 * yaw)
            ctrl.amount = 0.5 * amt
        elif preset == "swarm":                                # open = the nanomachines fly out to your hand
            ctrl.swarm_release = float(np.clip((opn - 0.4) / 0.4, 0.0, 1.0))
            ctrl.swarm_pull = float(np.clip((flex - 0.5) / 0.3, 0.0, 1.0))
            ctrl.cloud = np.array([2.0 + 3.0 * z, -3.0 * x, 1.2 + 2.5 * y])
            ctrl.cloud_swirl = float(np.clip(roll * 2.0, -6.0, 6.0))
            ctrl.scatter = 0.5 * ctrl.swarm_release if variant not in HIVES else 0.0
            ctrl.finger_mode = "none"
            ctrl.energy = float(np.clip(0.2 + 0.8 * ctrl.swarm_release, 0.0, 1.0))
        elif preset == "neon":                                 # the hand plays the light, taps flash it
            if np.any(d_ext < -2.5):
                self._flash = 1.0
            self._flash *= math.exp(-dt / 0.3)
            ctrl.lines = float(np.clip((0.15 + 0.85 * opn) * k + self._flash, 0.0, 1.5))
            ctrl.energy = float(np.clip(0.3 + y, 0.0, 1.0))
            ctrl.rot = euler(0.5 * roll, 0.5 * pitch, 0.5 * yaw)
            ctrl.grip = 0.25
            ctrl.finger_mode = "none"
        elif preset == "rhythm":                               # robotic: it snaps to the hand in steps, on the beat
            b_idx = int(math.floor(beat))
            if b_idx != self._held[1]:
                step = math.pi / 4
                self._held = (np.round(ang * amt / step) * step, b_idx)
            ctrl.rot = euler(*self._held[0])
            ctrl.grip = 1.0
            ctrl.pulse = float(np.clip(opn, 0.0, 1.0))
            ctrl.finger_mode = "none"
            ctrl.lines = 0.3 + 0.7 * math.exp(-(beat % 1.0) / 0.15)
        elif preset == "mandala":                              # extended fingers = rays, the twist turns them
            n_up = int((ext > 0.55).sum())
            ctrl.rays = 2 + n_up
            ctrl.ray_len = float(np.clip((0.2 + 0.8 * opn) * amt, 0.0, 1.4))
            ctrl.ray_phase = float(roll * 2.0)
            ctrl.spin = float(np.clip(yaw * 2.0, -6.0, 6.0))
            ctrl.rot = euler(0.0, 0.5 * pitch, 0.0)
            ctrl.finger_mode = "none"
        events = []
        if c["gestures"] and preset != "off":
            for g in gestures:
                if g == "PINCH":
                    self.pinches += 1
                    events.append((g, ("SPLIT", "MORPHOLOGY_SHIFT") if self.pinches % 2 else ("MERGE", "MORPHOLOGY_SHIFT")))
                elif preset == "camera":
                    events.append((g, ("camera",)))
                else:
                    events.append((g, GESTURE_EVENTS[g]))
        return ctrl, events, cam

    def sculpt_shapes(self, variant: str) -> tuple:
        return SCULPT_SHAPES.get(variant, SCULPT_SHAPES["polyalloy"])

    def snapshot(self, dec: GloveDecoder, now: float) -> dict:
        st = self.state
        return {"present": st.present, "rate": round(dec.rate, 1), "preset": self.cfg["preset"],
                "values": {p: dec.value(p, now) for p in PARAMS}, "profile": dict(dec.profile),
                "ext": st.ext.round(3).tolist(), "angles": np.degrees(st.ang).round(1).tolist(),
                "pos": st.pos.round(3).tolist(), "gestures": [g for _, g in st.gestures[-4:]],
                "learning": dec.learn[0] if dec.learn else None, "learned": dec.last_learned}


__all__ = ["GloveDecoder", "GloveState", "GloveLink", "OneEuro", "PARAMS", "PRESETS", "GESTURES", "SCULPT_SHAPES"]
