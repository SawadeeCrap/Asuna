"""Persistent app settings (JSON in the user's application-support folder)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_CHARACTER = os.path.join(REPO, "characters", "humanoid", "character_live.blend")


def settings_dir() -> str:
    if sys.platform == "darwin":
        d = os.path.expanduser("~/Library/Application Support/Myrmex")
    else:
        d = os.path.join(os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")), "myrmex")
    os.makedirs(d, exist_ok=True)
    return d


def characters_dir() -> str:
    d = os.path.expanduser("~/Myrmex")
    os.makedirs(d, exist_ok=True)
    return d


def _user_copy(blend: str) -> str:
    """Inside Myrmex.app the bundled character is copied to ~/Myrmex once, so edits live outside the app."""
    import shutil
    dst = os.path.join(characters_dir(), "humanoid_live.blend")
    try:
        if not os.path.exists(dst):
            shutil.copy2(blend, dst)
            shutil.copy2(os.path.splitext(blend)[0] + ".rig.json", os.path.splitext(dst)[0] + ".rig.json")
        return dst
    except OSError:
        return blend


@dataclass
class AppSettings:
    # character / blender
    character: str = DEFAULT_CHARACTER
    blender: str = ""
    open_blender_on_start: bool = False
    # inputs
    clock: str = "auto"
    bpm: float = 120.0
    link: bool = True
    osc_port: int = 9100
    midi_ports: list[str] = field(default_factory=list)
    audio_device: str = ""
    mapping_file: str = ""
    latency_ms: float = 50.0
    # behaviour
    style: str = "catwalk"
    seed: int = 0
    energy: float | None = None          # None = automatic
    stride: float | None = None
    sway: float | None = None
    # camera / output
    camera: bool = True
    pose_port: int = 9101
    out_fps: float = 60.0
    extra_targets: str = ""
    record: bool = False
    record_dir: str = ""
    start_engine_on_launch: bool = True
    backend: str = "humanoid"               # humanoid | creature | polyalloy | colony
    keep_blender_settings: bool = True      # Myrmex leaves EEVEE / colour / shadows / samples alone
    take_audio: str = ""                    # song for take renders
    render_size: str = "1920x1080"
    render_quality: str = "eevee"
    creature_params: dict = field(default_factory=dict)   # manual overrides (missing = automatic)
    midi_bindings: list = field(default_factory=list)
    camera_controls: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "AppSettings":
        path = os.path.join(settings_dir(), "settings.json")
        s = cls()
        try:
            with open(path) as f:
                data = json.load(f)
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
        except (OSError, ValueError):
            pass
        if not s.record_dir:
            s.record_dir = os.path.join(characters_dir(), "takes")
        if not os.path.exists(s.character):
            s.character = DEFAULT_CHARACTER
        if os.environ.get("MYRMEX_BUNDLE") and os.path.abspath(s.character) == os.path.abspath(DEFAULT_CHARACTER):
            s.character = _user_copy(DEFAULT_CHARACTER)
        return s

    def save(self) -> None:
        path = os.path.join(settings_dir(), "settings.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    @property
    def rig_json(self) -> str:
        return os.path.splitext(self.character)[0] + ".rig.json"
