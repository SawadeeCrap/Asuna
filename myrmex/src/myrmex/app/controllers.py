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
PREPARE = os.path.join(REPO, "blender", "scripts", "prepare_character.py")
OPEN_TAKE = os.path.join(REPO, "blender", "scripts", "open_take.py")
CREATURE_BACKENDS = ("creature", "polyalloy", "colony", "hive", "osseous", "osseous_colony", "osseous_hive",
                     "cyber_hive", "swarm", "spear", "cloud", "blade", "crawler")        # organisms: no .blend, the scene is built


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


class EngineController:
    """Owns the LiveSession (it runs in its own thread inside the app process)."""

    def __init__(self, settings: AppSettings, log):
        self.s = settings
        self.log = log
        self.session = None

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
                             camera=bool(s.camera), record=s.record_dir if s.record else None, inputs=inputs,
                             glove=dict(s.glove or {}))
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
               "cloud": "Mimetic Cloud", "blade": "Mimetic Blade", "crawler": "Mimetic Crawler"}.get(s.backend) \
            or os.path.basename(s.character)
        self.log(f"engine started: {who} | OSC :{s.osc_port} | -> {', '.join(out)}")
        for e in self.session.status()["errors"]:
            self.log(f"  ! {e}")
        self.push_knobs()
        return True

    def stop(self) -> None:
        if self.session is None:
            return
        path = self.session.stop()
        self.session = None
        self.log("engine stopped" + (f" - take saved: {path}" if path else ""))

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

    def save_take(self) -> str | None:
        return self.session.save_take() if self.session is not None else None

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
                         keep_settings: bool = True, looks: dict | None = None) -> tuple[list[str], dict]:
    env = dict(os.environ, MYRMEX_POSE_PORT=str(pose_port), MYRMEX_ENGINE_MODE="EXTERNAL", MYRMEX_MODE=backend,
               MYRMEX_KEEP_SETTINGS="1" if keep_settings else "0", MYRMEX_LOOKS=looks_dir(), MYRMEX_CONTROL="stdin")
    if backend in CREATURE_BACKENDS:           # the chosen look, or a scene built from scratch
        look = look_file(backend, looks)
        return [blender] + ([look] if look else []) + ["--python", AUTOSTART], env
    return [blender, character, "--python", AUTOSTART], env


def take_variant(take: str) -> str | None:
    base = os.path.basename(take)
    for prefix, variant in (("cyber_hive_take", "cyber_hive"), ("swarm_take", "swarm"), ("spear_take", "spear"),
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


def last_take(folder: str) -> str:
    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder or ""), "*take_*.npz")), key=os.path.getmtime)
    return files[-1] if files else ""


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
