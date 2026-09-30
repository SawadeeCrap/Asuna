"""Non-GUI logic of the app: the live engine, Blender, Ableton helpers, device discovery."""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import time

from ..bus.transport import LiveEvent
from .settings import REPO, AppSettings, characters_dir

STYLES = ["catwalk", "swagger", "heels", "natural"]
SHOTS = ["auto", "front_dolly", "front_low", "three_quarter", "side_track", "rear_follow", "hips_close",
         "feet_close", "face_close", "wide_orbit"]
REMOTE_SCRIPT_SRC = os.path.join(REPO, "ableton", "remote_script", "Myrmex")
REMOTE_SCRIPTS_DIR = os.path.expanduser("~/Music/Ableton/User Library/Remote Scripts")
AUTOSTART = os.path.join(REPO, "blender", "scripts", "live_autostart.py")
TD_SCRIPT = os.path.join(REPO, "touchdesigner", "myrmex_td.py")              # builds the TD network
TD_HOME = os.path.expanduser("~/Myrmex/touchdesigner")
TD_PROJECT = os.path.join(TD_HOME, "Myrmex_FX.toe")                           # saved by the script
PREPARE = os.path.join(REPO, "blender", "scripts", "prepare_character.py")
OPEN_TAKE = os.path.join(REPO, "blender", "scripts", "open_take.py")
CREATURE_BACKENDS = ("creature", "polyalloy", "colony", "hive", "osseous", "osseous_colony", "osseous_hive",
                     "cyber_hive", "swarm", "spear", "cloud", "blade", "crawler",       # organisms: no .blend,
                     "tensor", "fold", "arbor", "ferro", "truss")                       # the scene is built


def variant_of(backend: str) -> str:
    return {"creature": "nanomaterial"}.get(backend, backend)


def looks_dir() -> str:
    return os.path.join(characters_dir(), "looks")


LOOK_DEFAULT_NAME, LOOK_AUTOSAVE = "My look", "Autosave"     # as in blender/myrmex_blender/looks.py


def list_looks(backend_or_variant: str) -> list[tuple[str, str]]:
    """(name, path) of an organism's saved looks: "My look" (the older single one), then by name, autosave last."""
    v = variant_of(backend_or_variant)
    out = []
    legacy = os.path.join(looks_dir(), v + ".blend")
    if os.path.isfile(legacy):
        out.append((LOOK_DEFAULT_NAME, legacy))
    d = os.path.join(looks_dir(), v)
    if os.path.isdir(d):
        names = sorted((f[:-6] for f in os.listdir(d) if f.endswith(".blend") and not f.startswith(".")),
                       key=lambda n: (n == LOOK_AUTOSAVE, n.lower()))
        out += [(n, os.path.join(d, n + ".blend")) for n in names]
    return out


def look_path(backend_or_variant: str, name: str) -> str:
    """Where a look of this name is kept ("My look" = the older single file)."""
    v = variant_of(backend_or_variant)
    if name == LOOK_DEFAULT_NAME:
        return os.path.join(looks_dir(), v + ".blend")
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", name).strip().strip(".")
    name = re.sub(r"\s+", " ", name)[:60] or "Look"
    return os.path.join(looks_dir(), v, name + ".blend")


def look_file(backend_or_variant: str, chosen: dict | None = None) -> str:
    """The look to open for an organism: the one chosen in the app ("" = the default studio), else the
    older single look, else none."""
    v = variant_of(backend_or_variant)
    if chosen is not None and v in chosen:
        p = chosen[v]
        if not p or os.path.isfile(p):
            return p or ""
    p = os.path.join(looks_dir(), v + ".blend")
    return p if os.path.exists(p) else ""


def td_settings(s: AppSettings) -> dict:
    """The TouchDesigner link settings with every default filled in (the app keeps them in s.td)."""
    from ..realtime.touch import DEFAULTS, FX_DEFAULTS
    d = dict(DEFAULTS)
    d.update({k: v for k, v in (s.td or {}).items() if k in DEFAULTS})
    d["fx"] = {**FX_DEFAULTS, **((s.td or {}).get("fx") or {})}
    return d


def syphon_config(s: AppSettings) -> dict:
    """What Blender's Syphon picture should be (the "syphon" control command / MYRMEX_SYPHON)."""
    d = td_settings(s)
    try:
        w, h = (int(x) for x in str(d.get("syphon_size", "1280x720")).lower().split("x"))
    except ValueError:
        w, h = 1280, 720
    return {"on": bool(d["enabled"] and d["syphon"]), "name": d.get("syphon_name") or "Myrmex", "width": w,
            "height": h, "fps": int(d.get("syphon_fps", 60)), "alpha": bool(d.get("alpha", False))}


def td_env(s: AppSettings) -> dict:
    """Environment for a Blender Myrmex opens: the Syphon picture, and the TD channels of a played take."""
    import json
    d = td_settings(s)
    if not d["enabled"]:
        return {}
    env = {"MYRMEX_TD": json.dumps({"host": d["host"], "port": int(d["port"])})}
    sy = syphon_config(s)
    if sy["on"]:
        env["MYRMEX_SYPHON"] = json.dumps(sy)
    return env


def fx_settings(s: AppSettings) -> dict:
    """Myrmex FX settings with every default filled in (the app keeps them in s.fx)."""
    from ..realtime.fx import DEFAULT_RACK, DEFAULTS, rack_from_preset
    d = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    d["replay"] = True
    d["monitor"] = True
    got = s.fx or {}
    d.update({k: v for k, v in got.items() if k in d and k != "rack"})
    from ..realtime.fx import PRESETS
    if d.get("preset") and d["preset"] not in PRESETS:          # a preset that no longer exists
        d["preset"], got = "Afterimage", {k: v for k, v in got.items() if k != "rack"}
    base = rack_from_preset(d["preset"]) if d.get("preset") else dict(DEFAULT_RACK)
    d["rack"] = {**base, **{k: float(v) for k, v in (got.get("rack") or {}).items() if k in DEFAULT_RACK}}
    return d


def brain_settings(s: AppSettings) -> dict:
    """The morphology brain's settings (s.brain) with every default filled in - off unless switched on; its
    decision log goes to ~/Myrmex/brain."""
    from ..brain.core import BrainConfig
    d = BrainConfig.from_dict(s.brain or {}).to_dict()
    d["log_dir"] = d["log_dir"] or os.path.join(characters_dir(), "brain")
    return d


def train_settings(s: AppSettings) -> dict:
    """The Train page's settings (s.train) with defaults, the brain's log folder and its Kev address."""
    t = dict(s.train or {})
    b = brain_settings(s)
    try:
        spread = min(1.0, max(0.0, float(t.get("spread", 0.8))))
    except (TypeError, ValueError):
        spread = 0.8
    return {"range": "own" if t.get("range") == "own" else "all", "spread": spread, "kev": bool(t.get("kev", False)),
            "fx_off": bool(t.get("fx_off", True)), "log_dir": b["log_dir"], "kev_url": b["kev_url"] or "http://127.0.0.1:8009"}


def fx_engine(s: AppSettings) -> dict:
    """What the engine's FX rack takes (LiveConfig.fx)."""
    d = fx_settings(s)
    return {"enabled": bool(d["enabled"]), "preset": d.get("preset", ""), "rack": dict(d["rack"]),
            "vertical": bool(d["vertical"]), "preview": float(d["preview"])}


def fx_blender(s: AppSettings) -> dict:
    """What a Blender needs (the "fx" control command / MYRMEX_FX): takes have no engine stream."""
    d = fx_settings(s)
    return {"on": bool(d["enabled"]), "preset": d.get("preset", ""), "rack": dict(d["rack"]),
            "vertical": bool(d["vertical"]), "preview": float(d["preview"]), "replay": bool(d.get("replay", True)),
            "monitor": bool(d.get("monitor", True))}


def gfx_settings(s: AppSettings) -> dict:
    """Blender's live graphics (realtime/gfx.py) with every default filled in.  EEVEE's viewport options are
    Myrmex's to set only when "Keep my Blender settings" is off (unless chosen here explicitly)."""
    from ..realtime.gfx import normalize
    got = dict(s.gfx or {})
    d = normalize({k: v for k, v in got.items() if k != "preset"})
    if got.get("preset"):
        d = normalize({"preset": got["preset"]}, d) if got["preset"] != "custom" else {**d, "preset": "custom"}
    if "eevee" not in got:
        d["eevee"] = not s.keep_blender_settings
    return d


def gfx_env(s: AppSettings) -> dict:
    import json
    return {"MYRMEX_GFX": json.dumps(gfx_settings(s))}


def fx_env(s: AppSettings) -> dict:
    import json
    return {"MYRMEX_FX": json.dumps(fx_blender(s))}


# The stage (blender/myrmex_blender/stage.py): the HDRI every organism is lit by and reflects, as in Blender's
# Material Preview - the camera sees black.  "" = the organism's own soft studio panels.
STAGE = {"hdri": "", "strength": 1.0, "rotation": 0.0, "lights": True}
HDRIS = (("", "Studio panels (the organism's own light)"), ("city.exr", "City"), ("courtyard.exr", "Courtyard"),
         ("forest.exr", "Forest"), ("interior.exr", "Interior"), ("night.exr", "Night"), ("studio.exr", "Studio"),
         ("sunrise.exr", "Sunrise"), ("sunset.exr", "Sunset"))


def stage_settings(s: AppSettings) -> dict:
    """The stage with every default filled in (the app keeps it in s.stage)."""
    d = dict(STAGE)
    got = s.stage or {}
    try:
        if "hdri" in got:
            d["hdri"] = str(got["hdri"] or "")
        if "strength" in got:
            d["strength"] = min(20.0, max(0.0, float(got["strength"])))
        if "rotation" in got:
            d["rotation"] = float(got["rotation"]) % 360.0
        if "lights" in got:
            d["lights"] = bool(got["lights"])
    except (TypeError, ValueError):
        pass
    return d


def stage_env(s: AppSettings) -> dict:
    import json
    return {"MYRMEX_STAGE": json.dumps(stage_settings(s))}


def hdri_label(hdri: str) -> str:
    return dict(HDRIS).get(hdri) or os.path.splitext(os.path.basename(hdri))[0].replace("_", " ").title()


def install_td_files() -> str:
    """Copy the TD network builder next to the user's TD project; returns its path."""
    os.makedirs(TD_HOME, exist_ok=True)
    dst = os.path.join(TD_HOME, os.path.basename(TD_SCRIPT))
    shutil.copy2(TD_SCRIPT, dst)
    return dst


def td_build_command(path: str | None = None) -> str:
    """The one line to paste into TouchDesigner's Textport."""
    return f"exec(open({(path or os.path.join(TD_HOME, os.path.basename(TD_SCRIPT)))!r}).read())"


def find_touchdesigner() -> str | None:
    for base in ("/Applications", os.path.expanduser("~/Applications")):
        for app in sorted(glob.glob(os.path.join(base, "TouchDesigner*.app")), reverse=True):
            return app
    return None


def open_touchdesigner_command(project: str | None = None) -> list[str] | None:
    app = find_touchdesigner()
    if app is None and sys.platform != "darwin":
        return None
    cmd = ["open", "-a", app or "TouchDesigner"]
    return cmd + ([project] if project and os.path.exists(project) else [])


class EngineController:
    """Owns the LiveSession (it runs in its own thread inside the app process)."""

    def __init__(self, settings: AppSettings, log):
        self.s = settings
        self.log = log
        self.session = None
        self.rec_last: dict = {}                          # the take recording as it was last seen

    @property
    def running(self) -> bool:
        return self.session is not None

    def start(self) -> bool:
        from ..realtime.inputs import InputConfig
        from ..realtime.session import LiveConfig, LiveSession
        s = self.s
        creature = s.backend in CREATURE_BACKENDS
        if not creature and not os.path.exists(s.rig_json):
            self.log(f"! no rig description next to the character: {s.rig_json}")
            return False
        try:
            ports = list(s.midi_ports)
            if (s.glove or {}).get("preset", "puppet") != "off":      # the Hand Glove joins by itself
                ports += [p for p in midi_inputs()[0] if any(w in p.lower() for w in ("glove", "hand")) and p not in ports]
            inputs = InputConfig.from_file(s.mapping_file or None, osc_port=int(s.osc_port), midi=ports,
                                           audio=s.audio_device or None)
            inputs.mapping["midi_bindings"] = list(s.midi_bindings)
            out = [f"127.0.0.1:{int(s.pose_port)}"] + [t.strip() for t in s.extra_targets.split(",") if t.strip()]
            cfg = LiveConfig(rig=None if creature else s.rig_json, backend=s.backend, seed=int(s.seed), out=out, out_rate=float(s.out_fps), clock=s.clock,
                             bpm=float(s.bpm), link=bool(s.link), latency=float(s.latency_ms) / 1000.0, style=s.style,
                             camera=bool(s.camera), record=takes_dir(s), rec_on_start=False, inputs=inputs,
                             glove=dict(s.glove or {}), touch=td_settings(s), fx=fx_engine(s),
                             brain=brain_settings(s))
            self.session = LiveSession(cfg)
            self.session.start()
        except Exception as e:
            self.session = None
            self.log(f"! engine failed to start: {type(e).__name__}: {e}")
            return False
        who = {"creature": "Black Nanomaterial Creature", "polyalloy": "Mimetic Polyalloy",
               "colony": "Polyalloy Colony", "hive": "Polyalloy Hive", "osseous": "Osseous Polyalloy",
               "osseous_colony": "Osseous Colony", "osseous_hive": "Osseous Hive",
               "cyber_hive": "Cyber Hive", "swarm": "Mimetic Swarm", "spear": "Mimetic Spear",
               "cloud": "Mimetic Cloud", "blade": "Mimetic Blade", "crawler": "Mimetic Crawler",
               "tensor": "Bionic Tensor", "fold": "Bionic Fold", "arbor": "Bionic Arbor", "ferro": "Bionic Ferro",
               "truss": "Bionic Truss"}.get(s.backend) \
            or os.path.basename(s.character)
        self.log(f"engine started: {who} | OSC :{s.osc_port} | -> {', '.join(out)}")
        for e in self.session.status()["errors"]:
            self.log(f"  ! {e}")
        self.push_knobs()
        return True

    def stop(self) -> None:
        if self.session is None:
            return
        path = self.session.stop()                        # (a take still being recorded is saved)
        self.rec_last = dict(self.session.rec_status())
        self.session = None
        if path:
            self.log(f"engine stopped - the take being recorded is saved: {os.path.basename(path)} "
                     f"({fmt_seconds(self.rec_last.get('last_seconds', 0.0))})")
        else:
            self.log("engine stopped")

    def restart(self) -> bool:
        self.stop()
        return self.start()

    # ------------------------------------------------------------------ live controls
    def control(self, name: str, value: float) -> None:
        if self.session is not None:
            self.session.inputs.push(LiveEvent("control", time.perf_counter(), {"name": name, "value": float(value)}))

    def trigger(self, name: str) -> None:
        if self.session is not None:
            self.session.inputs.push(LiveEvent("trigger", time.perf_counter(), {"name": name}))

    def push_knobs(self) -> None:
        s = self.s
        for k, v in s.creature_params.items():
            self.control(k, v)
        for k, v in s.camera_controls.items():
            self.control(k, v)
        for k in ("energy", "stride", "sway"):
            v = getattr(s, k)
            self.control(k, -1.0 if v is None else v)
        self.control("style", (STYLES.index(s.style) + 0.5) / len(STYLES) if s.style in STYLES else 0.1)

    def set_latency(self, ms: float) -> None:
        if self.session is not None:
            self.session.cfg.latency = ms / 1000.0

    # ------------------------------------------------------------------ Hand Glove
    def glove_config(self, **kw) -> None:
        if self.session is not None:
            self.session.glove.configure(**kw)

    def td_config(self, **kw) -> None:
        """Change the TouchDesigner link live (effects, recording, the output window ...)."""
        if self.session is not None:
            self.session.touch.configure(**kw)

    def fx_config(self, **kw) -> None:
        """Change Myrmex FX live (it travels to Blender with every frame)."""
        if self.session is not None:
            self.session.fx.configure(**kw)

    def brain_config(self, d: dict) -> dict:
        """Switch the morphology brain on / off or change it, live (the engine keeps running)."""
        if self.session is None:
            return {}
        return self.session.set_brain(d)

    def training(self, on: bool) -> dict:
        """The Train page on / off (or its settings changed while on)."""
        if self.session is None:
            return {"on": False, "error": "start the engine first"}
        return self.session.set_training({**train_settings(self.s), "on": bool(on)})

    def train_rate(self, good: bool | None) -> dict:
        """Good (True) / Bad (False) / Skip (None) on the change shown."""
        return self.session.train_rate(good) if self.session is not None else {"on": False}

    def glove_calibrate(self) -> dict:
        return self.session.glove.state.calibrate() if self.session is not None else {}

    def glove_learn(self, param: str) -> None:
        if self.session is not None:
            self.session.inputs.glove.learn_param(param, time.perf_counter())

    def glove_autodetect(self) -> dict:
        return self.session.inputs.glove.autodetect(time.perf_counter()) if self.session is not None else {}

    def glove_snapshot(self) -> dict | None:
        if self.session is None:
            return None
        dec = self.session.inputs.glove
        dec.poll_learn(time.perf_counter())
        snap = self.session.glove.snapshot(dec, time.perf_counter())
        snap["events"] = [f"{g} → {e.lower()}" for _, g, e in self.session.glove.last_events[-4:]]
        return snap

    # ------------------------------------------------------------------ takes: REC / STOP
    def rec_start(self) -> bool:
        return self.session.rec_start() if self.session is not None else False

    def rec_stop(self) -> str | None:
        """STOP: the take is written in the background (rec_status: saving, then last)."""
        return self.session.rec_stop(background=True) if self.session is not None else None

    def rec_status(self) -> dict:
        if self.session is not None:
            self.rec_last = dict(self.session.rec_status())
        return self.rec_last

    def status(self) -> dict:
        if self.session is None:
            return {}
        st = self.session.status()
        ls = self.session.last_state
        st["bpb"] = ls.beats_per_bar if ls else 4.0
        st["beat_raw"] = ls.beat if ls else 0.0
        return st


# ---------------------------------------------------------------------------- blender
def find_blender(hint: str = "") -> str | None:
    cands = [hint] if hint else []
    if sys.platform == "darwin":
        cands += ["/Applications/Blender.app/Contents/MacOS/Blender",
                  os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender")]
        try:
            out = subprocess.run(["mdfind", "kMDItemCFBundleIdentifier == 'org.blenderfoundation.blender'"],
                                 capture_output=True, text=True, timeout=5).stdout.split("\n")
            cands += [os.path.join(p, "Contents", "MacOS", "Blender") for p in out if p.strip()]
        except Exception:
            pass
    w = shutil.which("blender")
    if w:
        cands.append(w)
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def blender_live_command(blender: str, character: str, pose_port: int, backend: str = "humanoid",
                         keep_settings: bool = True, looks: dict | None = None,
                         extra_env: dict | None = None) -> tuple[list[str], dict]:
    env = dict(os.environ, MYRMEX_POSE_PORT=str(pose_port), MYRMEX_ENGINE_MODE="EXTERNAL", MYRMEX_MODE=backend,
               MYRMEX_KEEP_SETTINGS="1" if keep_settings else "0", MYRMEX_LOOKS=looks_dir(), MYRMEX_CONTROL="stdin")
    env.update(extra_env or {})
    if backend in CREATURE_BACKENDS:           # the chosen look, or a scene built from scratch
        look = look_file(backend, looks)
        return [blender] + ([look] if look else []) + ["--python", AUTOSTART], env
    return [blender, character, "--python", AUTOSTART], env


def take_variant(take: str) -> str | None:
    base = os.path.basename(take)
    for prefix, variant in (("tensor_take", "tensor"), ("fold_take", "fold"), ("arbor_take", "arbor"),
                            ("ferro_take", "ferro"), ("truss_take", "truss"),
                            ("cyber_hive_take", "cyber_hive"), ("swarm_take", "swarm"), ("spear_take", "spear"),
                            ("cloud_take", "cloud"), ("blade_take", "blade"), ("crawler_take", "crawler"),
                            ("osseous_hive_take", "osseous_hive"), ("osseous_colony_take", "osseous_colony"),
                            ("osseous_take", "osseous"), ("hive_take", "hive"), ("colony_take", "colony"),
                            ("polyalloy_take", "polyalloy"),
                            ("nanomaterial_take", "nanomaterial"), ("creature_take", "nanomaterial")):
        if base.startswith(prefix):
            return variant
    return None


def take_command(blender: str, take: str, character: str = "", audio: str = "", render: bool = False,
                 size: str = "1920x1080", quality: str = "eevee", keep_settings: bool = True,
                 looks: dict | None = None) -> list[str]:
    """Open (or render, headless) a recorded take in Blender - in the chosen saved look when there is one."""
    variant = take_variant(take)
    scene = (look_file(variant, looks) if variant else character) or ""
    cmd = [blender] + (["-b"] if render else []) + ([scene] if scene else [])
    cmd += ["--python", OPEN_TAKE, "--", "--take", take, "--size", size, "--quality", quality]
    if keep_settings:
        cmd.append("--keep-settings")
    if audio:
        cmd += ["--audio", audio]
    if render:
        cmd.append("--render")
    return cmd


def takes_dir(s: AppSettings) -> str:
    return os.path.expanduser(s.record_dir or os.path.join(characters_dir(), "takes"))


def list_takes(folder: str) -> list[str]:
    """Every take in the folder, newest first."""
    files = glob.glob(os.path.join(os.path.expanduser(folder or ""), "*take_*.npz"))
    return sorted(files, key=os.path.getmtime, reverse=True)


def last_take(folder: str) -> str:
    files = list_takes(folder)
    return files[0] if files else ""


def take_videos(take: str) -> list[str]:
    """The videos rendered from a take (next to it), newest first."""
    files = glob.glob(glob.escape(os.path.splitext(take)[0]) + "_*.mp4")
    return sorted(files, key=os.path.getmtime, reverse=True)


_INFO: dict = {}


def take_info(path: str) -> dict:
    """{"frames", "fps", "seconds", "variant"} of a take (read once per file version)."""
    import json

    import numpy as np
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return {}
    if key in _INFO:
        return _INFO[key]
    info = {"frames": 0, "fps": 30.0, "seconds": 0.0, "variant": take_variant(path) or "humanoid"}
    try:
        with np.load(path, allow_pickle=False) as z:
            if "t" in z.files:                              # an organism's take
                info["frames"] = int(len(z["t"]))
                info["fps"] = float(z["fps"]) if "fps" in z.files else 30.0
            else:                                           # a humanoid's: the frame rate is in its .json
                ch = next((k for k in z.files if k.startswith("ch__")), None)
                info["frames"] = int(len(z[ch])) if ch else int(z["deltas"].shape[0])
                with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
                    info["fps"] = float(json.load(f).get("fps", 30.0))
    except (OSError, ValueError, KeyError):
        return {}
    info["seconds"] = info["frames"] / max(1.0, info["fps"])
    _INFO[key] = info
    return info


def fmt_seconds(sec: float) -> str:
    """12.4 -> "0:12.4", 75 -> "1:15.0"."""
    sec = max(0.0, float(sec))
    return f"{int(sec // 60)}:{sec % 60:04.1f}"


def prepare_command(blender: str, glb: str, out: str, height: float, material: str, smooth: int) -> list[str]:
    return [blender, "-b", "--python", PREPARE, "--", "--glb", glb, "--out", out, "--height", f"{height:.3f}",
            "--material", material, "--smooth", str(int(smooth))]


def known_characters(current: str = "") -> list[str]:
    found = []
    for p in [current, os.path.join(REPO, "characters", "humanoid", "character_live.blend")] + \
            sorted(glob.glob(os.path.join(characters_dir(), "*.blend"))):
        if p and os.path.exists(p) and os.path.exists(os.path.splitext(p)[0] + ".rig.json") and p not in found:
            found.append(p)
    return found


# ---------------------------------------------------------------------------- ableton / devices
def remote_script_installed() -> bool:
    return os.path.exists(os.path.join(REMOTE_SCRIPTS_DIR, "Myrmex", "surface.py"))


def install_remote_script() -> str:
    dst = os.path.join(REMOTE_SCRIPTS_DIR, "Myrmex")
    os.makedirs(REMOTE_SCRIPTS_DIR, exist_ok=True)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(REMOTE_SCRIPT_SRC, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def midi_inputs() -> tuple[list[str], str]:
    try:
        import mido
        return list(mido.get_input_names()), ""
    except Exception as e:
        return [], f"MIDI unavailable ({e}); pip install mido python-rtmidi"


def audio_inputs() -> tuple[list[str], str]:
    try:
        import sounddevice as sd
        return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0], ""
    except Exception as e:
        return [], f"audio unavailable ({e}); pip install sounddevice"


def link_available() -> bool:
    try:
        import aalink  # noqa: F401
        return True
    except Exception:
        return False
