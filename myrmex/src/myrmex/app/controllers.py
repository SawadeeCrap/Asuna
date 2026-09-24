"""Non-GUI logic of the app: the live engine, Blender, Ableton helpers, device discovery."""
from __future__ import annotations

import glob
import os
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
        creature = s.backend == "creature"
        if not creature and not os.path.exists(s.rig_json):
            self.log(f"! no rig description next to the character: {s.rig_json}")
            return False
        try:
            inputs = InputConfig.from_file(s.mapping_file or None, osc_port=int(s.osc_port), midi=list(s.midi_ports),
                                           audio=s.audio_device or None)
            inputs.mapping["midi_bindings"] = list(s.midi_bindings)
            out = [f"127.0.0.1:{int(s.pose_port)}"] + [t.strip() for t in s.extra_targets.split(",") if t.strip()]
            cfg = LiveConfig(rig=None if creature else s.rig_json, backend=s.backend, seed=int(s.seed), out=out, out_rate=float(s.out_fps), clock=s.clock,
                             bpm=float(s.bpm), link=bool(s.link), latency=float(s.latency_ms) / 1000.0, style=s.style,
                             camera=bool(s.camera), record=s.record_dir if s.record else None, inputs=inputs)
            self.session = LiveSession(cfg)
            self.session.start()
        except Exception as e:
            self.session = None
            self.log(f"! engine failed to start: {type(e).__name__}: {e}")
            return False
        who = "Black Nanomaterial Creature" if creature else os.path.basename(s.character)
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


def blender_live_command(blender: str, character: str, pose_port: int, backend: str = "humanoid") -> tuple[list[str], dict]:
    env = dict(os.environ, MYRMEX_POSE_PORT=str(pose_port), MYRMEX_ENGINE_MODE="EXTERNAL", MYRMEX_MODE=backend)
    if backend == "creature":                  # no .blend: the creature scene is built procedurally
        return [blender, "--python", AUTOSTART], env
    return [blender, character, "--python", AUTOSTART], env


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
