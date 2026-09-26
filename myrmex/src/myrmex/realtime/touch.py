"""TouchDesigner link: Myrmex's music, organism and camera -> TD every frame, TD -> Myrmex back.

    Myrmex ──OSC 7000 (data) + 7001 (text)──> TouchDesigner <──Syphon "Myrmex"── Blender
           <──OSC 9100 (/myrmex/td/alive, /myrmex/trigger, /myrmex/control)──

Data: one OSC bundle per frame, one float per address ``/myrmex/<name>``; in TD an OSC In CHOP with
*Strip Prefix Segments = 1* turns it into channels ``energy``, ``kick``, ``sx`` ... (``CHANNELS`` below
is the whole list, in a fixed order - takes record it too).  Four kinds of channels:

* music - beat, phase, bar, bpm, playing, energy, bass, mid, high, flux, kick / snare / hats (envelopes);
* the organism - glow, arousal, speed, size, position, **where it is on screen** (sx, sy, ssize - the
  effects start from it), form (regime), intent, a per-organism ``tension``, impact / morph envelopes and
  an event counter;
* the camera - cut envelope, shot id, lens, focus, f-stop;
* ready-made effect drives (``fx_*`` 0..1: bloom, trails, glitch, chroma, warp, flash, shake, shockwave,
  hue, strobe) and the effect rack of the app (``c_*``: the sliders / presets / MIDI knobs ``td_<fx>``),
  recording and the output window.

Text (``/myrmex/text/<key> s value``, a few times per second): organism, regime, intent, shot, preset,
recfile, take.  TD answers with ``/myrmex/td/alive f fps`` (the app shows it) and can trigger events and
set parameters with the usual ``/myrmex/trigger`` / ``/myrmex/control`` messages.
"""
from __future__ import annotations

import math
import os
import socket
import struct
import time

import numpy as np

from ..bus.osc import _osc_string

MUSIC = ("beat", "phase", "bar", "bpm", "playing", "energy", "bass", "mid", "high", "flux", "kick", "snare", "hats")
BODY = ("glow", "arousal", "surface", "instab", "speed", "size", "x", "y", "z", "heading", "sx", "sy", "ssize",
        "visible", "regime", "variant", "intent", "tension", "impact", "morph", "event")
CAMERA = ("cut", "shot", "lens", "focus", "fstop")
FX_DRIVES = ("fx_bloom", "fx_trails", "fx_glitch", "fx_chroma", "fx_warp", "fx_flash", "fx_shake", "fx_shock",
             "fx_hue", "fx_strobe", "shock_x", "shock_y")
FX = ("bloom", "trails", "chroma", "glitch", "warp", "shock", "kaleido", "edges", "grain", "vignette", "hud",
      "react", "exposure", "contrast", "saturation", "hue", "mix")
CONTROLS = tuple("c_" + k for k in FX) + ("c_preset", "rec", "window", "monitor")
CHANNELS = MUSIC + BODY + CAMERA + FX_DRIVES + CONTROLS

FX_DEFAULTS = {"bloom": 0.6, "trails": 0.45, "chroma": 0.3, "glitch": 0.25, "warp": 0.15, "shock": 0.5,
               "kaleido": 0.0, "edges": 0.0, "grain": 0.25, "vignette": 0.4, "hud": 0.0, "react": 0.8,
               "exposure": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.0, "mix": 1.0}
PRESETS = {
    "Clean": {"bloom": 0.45, "trails": 0.0, "chroma": 0.1, "glitch": 0.0, "warp": 0.0, "shock": 0.2, "kaleido": 0.0,
              "edges": 0.0, "grain": 0.15, "vignette": 0.35, "hud": 0.0, "saturation": 0.5, "hue": 0.0},
    "Neon Trails": {"bloom": 0.85, "trails": 0.75, "chroma": 0.3, "glitch": 0.1, "warp": 0.1, "shock": 0.5,
                    "kaleido": 0.0, "edges": 0.0, "grain": 0.2, "vignette": 0.4, "hud": 0.0, "saturation": 0.6,
                    "hue": 0.3},
    "Glitch Storm": {"bloom": 0.5, "trails": 0.3, "chroma": 0.65, "glitch": 0.9, "warp": 0.3, "shock": 0.7,
                     "kaleido": 0.0, "edges": 0.1, "grain": 0.45, "vignette": 0.45, "hud": 0.3, "saturation": 0.45,
                     "hue": 0.1},
    "Dream": {"bloom": 0.95, "trails": 0.9, "chroma": 0.2, "glitch": 0.0, "warp": 0.45, "shock": 0.3,
              "kaleido": 0.0, "edges": 0.0, "grain": 0.2, "vignette": 0.5, "hud": 0.0, "saturation": 0.7, "hue": 0.5},
    "Kaleido": {"bloom": 0.65, "trails": 0.55, "chroma": 0.3, "glitch": 0.1, "warp": 0.1, "shock": 0.5,
                "kaleido": 0.6, "edges": 0.0, "grain": 0.2, "vignette": 0.45, "hud": 0.0, "saturation": 0.6,
                "hue": 0.25},
    "Scanner": {"bloom": 0.5, "trails": 0.35, "chroma": 0.25, "glitch": 0.2, "warp": 0.05, "shock": 0.6,
                "kaleido": 0.0, "edges": 0.85, "grain": 0.35, "vignette": 0.5, "hud": 1.0, "saturation": 0.25,
                "hue": 0.0},
    "Liquid": {"bloom": 0.7, "trails": 0.65, "chroma": 0.25, "glitch": 0.05, "warp": 0.7, "shock": 0.6,
               "kaleido": 0.0, "edges": 0.0, "grain": 0.2, "vignette": 0.4, "hud": 0.0, "saturation": 0.55,
               "hue": 0.15},
}
PRESET_NAMES = tuple(PRESETS)
DEFAULTS = {"enabled": False, "host": "127.0.0.1", "port": 7000, "text_port": 7001, "rate": 60.0, "syphon": True,
            "syphon_name": "Myrmex", "syphon_size": "1280x720", "syphon_fps": 60, "alpha": False,
            "preset": "Neon Trails", "fx": dict(FX_DEFAULTS), "rec": False, "rec_dir": "~/Myrmex/td_recordings",
            "window": False, "monitor": 1}

# What the organisms do, as an impact on the picture (1 = big, 0.6 = medium).
IMPACT = {**{e: 1.0 for e in ("RESPONSE", "COLLAPSE", "HIT", "STRIKE", "SHED", "SPLIT", "BLOW", "OVERLOAD", "POUNCE",
                               "QUILLS", "GLITCH", "RECONSTRUCTION")},
          **{e: 0.6 for e in ("IMPULSE", "OBSTACLE", "DASH", "SLASH", "LASH", "CLAP", "SURGE", "PRESSURE",
                               "TURBULENCE", "SCATTER", "WAVE", "SCAN", "SPROUT", "PULSE", "BLOOM", "FURL", "COIL",
                               "UNFURL", "CALM", "ANNEAL", "MERGE", "GATHER", "RECONFIGURE", "BUILD", "RECALL",
                               "OSSIFY", "HUNT", "PERCH", "APPENDAGE_BURST", "MASS_REBALANCE")}}


def project(p, cam) -> tuple[float, float, float, float]:
    """World point -> (sx, sy) in 0..1 (TD UV: origin bottom-left), the depth, and 1 if in front of the lens.
    The Blender live camera: 36 mm sensor width, 16:9, no roll."""
    if cam is None:
        return 0.5, 0.5, 1.0, 0.0
    pos, tgt = np.asarray(cam.position, float), np.asarray(cam.target, float)
    f = tgt - pos
    n = float(np.linalg.norm(f))
    if n < 1e-9:
        return 0.5, 0.5, 1.0, 0.0
    f /= n
    r = np.cross(f, [0.0, 0.0, 1.0])
    if float(np.linalg.norm(r)) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = np.asarray(p, float) - pos
    z = float(d @ f)
    if z <= 1e-3:
        return 0.5, 0.5, 1.0, 0.0
    tx = 18.0 / max(float(cam.lens), 1.0)                 # tan(half fov): 36 mm sensor
    ty = tx * 9.0 / 16.0
    sx = 0.5 + 0.5 * float(d @ r) / (z * tx)
    sy = 0.5 + 0.5 * float(d @ u) / (z * ty)
    vis = 1.0 if (-0.1 <= sx <= 1.1 and -0.1 <= sy <= 1.1) else 0.0
    return sx, sy, z, vis


class TouchBridge:
    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(DEFAULTS)
        self.cfg["fx"] = dict(FX_DEFAULTS)
        self.rec_file = ""
        self.configure(**(cfg or {}))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._prefix = {name: _osc_string("/myrmex/" + name) + _osc_string(",f") for name in CHANNELS}
        self.env = {k: 0.0 for k in ("kick", "snare", "hats", "impact", "morph", "cut", "flash")}
        self.shock_t, self.shock_x, self.shock_y = -1e9, 0.5, 0.5
        self.events = 0
        self.hue = 0.0
        self.t = 0.0
        self.next_send = 0.0
        self.next_text = 0.0
        self.texts: dict[str, str] = {}
        self.sent_text: dict[str, str] = {}
        self.last_shot = None
        self.last_morph = None
        self.prev = None
        self.values = np.zeros(len(CHANNELS))
        self.sent = 0
        self.errors = 0
        self.last_error = ""

    # ------------------------------------------------------------------ settings
    def configure(self, **kw) -> None:
        fx = kw.pop("fx", None)
        preset = kw.get("preset")
        for k, v in kw.items():
            if k in DEFAULTS:
                self.cfg[k] = v
        if preset in PRESETS and fx is None:                   # a preset sets the rack
            self.cfg["fx"].update(PRESETS[preset])
        if isinstance(fx, dict):
            for k, v in fx.items():
                if k in FX:
                    self.cfg["fx"][k] = float(min(1.0, max(0.0, float(v))))
        if kw.get("rec") and not self.rec_file:
            folder = os.path.expanduser(self.cfg.get("rec_dir") or "~/Myrmex/td_recordings")
            self.rec_file = os.path.join(folder, time.strftime("Myrmex_TD_%Y%m%d_%H%M%S.mov"))
        elif "rec" in kw and not kw["rec"]:
            self.rec_file = ""

    @property
    def target(self) -> tuple[str, int]:
        return str(self.cfg["host"]), int(self.cfg["port"])

    # ------------------------------------------------------------------ one frame
    def update(self, dt: float, st=None, inp=None, notes=(), state=None, events=(), cam=None,
               controls: dict | None = None, variant: str = "", size: float = 1.8, texts: dict | None = None):
        """Compute every channel (returns them, in CHANNELS order)."""
        self.t += dt
        v = dict.fromkeys(CHANNELS, 0.0)
        env = self.env
        for k, tau in (("kick", 0.12), ("snare", 0.1), ("hats", 0.06), ("impact", 0.35), ("morph", 1.0),
                       ("cut", 0.15), ("flash", 0.1)):
            env[k] *= math.exp(-dt / tau)
        # music
        beat = float(st.beat) if st is not None else self.t * 2.0
        bpm = float(st.bpm) if st is not None else 120.0
        bpb = float(getattr(st, "beats_per_bar", 4.0) or 4.0) if st is not None else 4.0
        v.update(beat=beat, phase=beat % 1.0, bar=(beat % bpb) / bpb, bpm=bpm,
                 playing=float(bool(st.playing)) if st is not None else 0.0)
        if inp is not None:
            v.update(energy=inp.energy, bass=inp.bass, mid=inp.mid, high=inp.high, flux=inp.spectral_flux)
            if inp.transient > 0.3:
                env["kick"] = max(env["kick"], float(inp.transient))
        for n in notes or ():
            g = getattr(n, "group", "")
            vel = float(getattr(n, "velocity", 1.0))
            if g == "kick":
                env["kick"] = max(env["kick"], vel)
            elif g == "snare":
                env["snare"] = max(env["snare"], vel)
            elif g == "hats":
                env["hats"] = max(env["hats"], vel)
        v.update(kick=env["kick"], snare=env["snare"], hats=env["hats"])
        # the organism
        if state is not None:
            com = np.asarray(getattr(state, "com", np.zeros(3)), float)
            v.update(glow=float(getattr(state, "glow", 0.0)), arousal=float(getattr(state, "arousal", 0.0)),
                     surface=float(getattr(state, "surface", 0.0)), instab=float(getattr(state, "instability", 0.0)),
                     x=com[0], y=com[1], z=com[2], heading=float(getattr(state, "heading", 0.0)))
            if self.prev is not None and dt > 0:
                v["speed"] = float(np.linalg.norm(com - self.prev)) / dt
            self.prev = com.copy()
            pos = getattr(state, "pos", None)
            rms = getattr(state, "rms", None)
            if rms is None and pos is not None and len(pos):
                rms = float(np.sqrt(((np.asarray(pos) - com) ** 2).sum(1).mean()))
            v["size"] = float(rms or 0.0)
            sx, sy, depth, vis = project(com, cam)
            ty = 18.0 / max(float(cam.lens), 1.0) * 9.0 / 16.0 if cam is not None else 0.5
            v.update(sx=sx, sy=sy, visible=vis, ssize=min(3.0, v["size"] / max(depth * ty, 1e-3)))
            v["tension"] = _tension(state)
            from ..creature.protocol import BEHAVIOR_NAMES, MORPH_NAMES
            morph = str(getattr(state, "morphology", ""))
            v["regime"] = float(MORPH_NAMES.index(morph)) if morph in MORPH_NAMES else -1.0
            beh = str(getattr(state, "behavior", ""))
            v["intent"] = float(BEHAVIOR_NAMES.index(beh)) if beh in BEHAVIOR_NAMES else -1.0
            if self.last_morph is not None and morph != self.last_morph:
                env["morph"] = 1.0
            self.last_morph = morph
        v["variant"] = float(_variant_index(variant))
        for name in events or ():
            name = str(name)
            self.events += 1
            if name == "MORPHOLOGY_SHIFT":
                env["morph"] = 1.0
                continue
            w = IMPACT.get(name, 0.3)
            env["impact"] = max(env["impact"], w)
            if w >= 0.6:
                env["flash"] = max(env["flash"], 0.6 * w)
                self.shock_t, self.shock_x, self.shock_y = self.t, v["sx"], v["sy"]
        v.update(impact=env["impact"], morph=env["morph"], event=float(self.events % 100000))
        # the camera
        if cam is not None:
            if self.last_shot is not None and cam.shot_id != self.last_shot:
                env["cut"] = 1.0
                env["flash"] = max(env["flash"], 0.5)
            self.last_shot = cam.shot_id
            v.update(cut=env["cut"], shot=float(cam.shot_id), lens=float(cam.lens), focus=float(cam.focus),
                     fstop=float(cam.fstop))
        # ready-made effect drives
        e, k, im = v["energy"], env["kick"], env["impact"]
        v["fx_bloom"] = min(1.5, 0.35 + 0.5 * v["glow"] + 0.4 * e + 0.3 * k)
        v["fx_trails"] = min(1.0, 0.3 + 0.35 * e + 0.25 * min(v["speed"] / 4.0, 1.0) + 0.2 * v["arousal"])
        v["fx_glitch"] = min(1.0, max(0.9 * im, 0.6 * env["cut"], 1.2 * max(v["flux"] - 0.6, 0.0)))
        v["fx_chroma"] = min(1.5, 0.15 + 0.6 * k + 0.5 * im + 0.2 * v["flux"])
        v["fx_warp"] = min(1.0, 0.2 * v["instab"] + 0.4 * v["flux"] * e + 0.3 * env["morph"])
        v["fx_flash"] = min(1.0, max(env["flash"], 0.35 * k if e > 0.75 else 0.0))
        v["fx_shake"] = min(1.0, k * (0.4 + 0.6 * v["arousal"]))
        v["fx_shock"] = min(1.0, (self.t - self.shock_t) / 0.9)
        v["shock_x"], v["shock_y"] = self.shock_x, self.shock_y
        self.hue = (self.hue + dt * bpm / 60.0 / 32.0) % 1.0                # a turn every 8 bars
        v["fx_hue"] = self.hue
        v["fx_strobe"] = 1.0 if (v["phase"] < 0.08 and e > 0.75 and v["playing"] > 0.5) else 0.0
        # the effect rack (app sliders; MIDI knobs td_<fx> win)
        c = controls or {}
        for name in FX:
            val = c.get("td_" + name)
            v["c_" + name] = float(val) if val is not None else float(self.cfg["fx"].get(name, FX_DEFAULTS[name]))
        p = self.cfg.get("preset")
        v["c_preset"] = float(PRESET_NAMES.index(p)) if p in PRESET_NAMES else -1.0
        v["rec"] = 1.0 if self.cfg.get("rec") else 0.0
        v["window"] = 1.0 if self.cfg.get("window") else 0.0
        v["monitor"] = float(self.cfg.get("monitor", 1))
        self.values = np.array([v[n] for n in CHANNELS], float)
        if texts is not None:
            self.texts.update({k: str(x) for k, x in texts.items()})
        self.texts["preset"] = str(p or "")
        self.texts["recfile"] = self.rec_file
        return self.values

    # ------------------------------------------------------------------ out
    def packet(self, values=None) -> bytes:
        vals = self.values if values is None else values
        parts = [b"#bundle\x00", struct.pack(">Q", 1)]
        for name, x in zip(CHANNELS, vals):
            m = self._prefix[name] + struct.pack(">f", float(x) if math.isfinite(x) else 0.0)
            parts.append(struct.pack(">i", len(m)))
            parts.append(m)
        return b"".join(parts)

    def text_packets(self, force: bool = False) -> list[bytes]:
        out = []
        for k, s in self.texts.items():
            if force or self.sent_text.get(k) != s:
                out.append(_osc_string("/myrmex/text/" + k) + _osc_string(",s") + _osc_string(s))
                self.sent_text[k] = s
        return out

    def tick(self, now: float, **kw) -> bool:
        """update() + send at the configured rate.  Returns True when a packet went out."""
        vals = self.update(**kw)
        if not self.cfg.get("enabled"):
            return False
        if now + 1e-9 < self.next_send:
            return False
        self.next_send = max(self.next_send + 1.0 / float(self.cfg["rate"]), now)
        self._tx(self.packet(vals), self.target)
        if now >= self.next_text:
            self.next_text = now + 0.25
            force = int(now) % 2 == 0 and now - int(now) < 0.25             # everything again every 2 s
            for pk in self.text_packets(force):
                self._tx(pk, (str(self.cfg["host"]), int(self.cfg["text_port"])))
        return True

    def _tx(self, data: bytes, addr) -> None:
        try:
            self.sock.sendto(data, addr)
            self.sent += 1
        except (ConnectionRefusedError, BlockingIOError):
            pass
        except OSError as e:
            self.errors += 1
            self.last_error = str(e)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def _variant_index(variant: str) -> int:
    from .session import CREATURES
    order = ("humanoid",) + tuple(CREATURES)
    return order.index(variant) if variant in order else -1


def _tension(state) -> float:
    """How loaded the organism is right now (per organism), about 0..1."""
    M = getattr(state, "members", None)
    kind = getattr(state, "bkind", -1)
    extra = getattr(state, "extra", None)
    try:
        if M is not None and len(M) and kind in (0, 2, 4):
            ok = M[:, 0] >= 0
            if kind == 0:
                ok &= M[:, 2] > 0
            vals = np.abs(M[ok, 4]) if ok.any() else np.zeros(1)
            return float(min(1.5, vals.max() if kind == 2 else vals.mean() * 1.6))
        if kind == 1 and extra is not None and len(extra) > 4:
            return float(min(1.5, extra[3] + extra[4] / 1.45))
        if kind == 3 and extra is not None and len(extra) > 12:
            return float(np.clip(extra[12] - 1.0, 0.0, 1.5))
    except (IndexError, ValueError):
        pass
    return float(min(1.5, getattr(state, "glow", 0.0) + 0.5 * getattr(state, "surface", 0.0)))


__all__ = ["TouchBridge", "CHANNELS", "FX", "FX_DEFAULTS", "PRESETS", "PRESET_NAMES", "DEFAULTS", "project",
           "IMPACT"]
