"""The stage every organism stands on: black for the camera, lit by the light you like.

Material Preview lights the organism with one of Blender's HDRIs and shows the grey of the viewport behind
it.  The stage takes that HDRI - with its strength and rotation - into the scene as the world the organism
is lit by and reflects, while the camera still sees pure black: in the Rendered view, the FX monitor,
Syphon and every render alike, for every organism.

* The app sets it for all of them (FX page: ``MYRMEX_STAGE`` at start, the ``stage`` command after); in
  Blender, Myrmex > Stage > "Use This View's Lighting" takes it from a 3D view and tells the app.
* ``hdri`` "" = no HDRI: the organism's own world (the soft studio panels, or a saved look's own world).
  A name is one of Blender's world HDRIs (``forest.exr`` - yours installed in Preferences > Lighting too);
  a full path is any .exr / .hdr file.
* The HDRI lives in a world of its own (``MyrmexStage``).  The organism's own world stays underneath,
  untouched, and is what a saved look keeps (``saving``).
* ``lights`` off = Scene Lights off in Material Preview: the HDRI alone lights the organism.  Only the lamps
  the stage switched off come back on; ones you hid yourself stay hidden.
"""
from __future__ import annotations

import contextlib
import json
import math
import os

import bpy

from . import compat

STAGE_WORLD = "MyrmexStage"
DEFAULTS = {"hdri": "", "strength": 1.0, "rotation": 0.0, "lights": True}
STAGE: dict = dict(DEFAULTS)
OFF = "myrmex_stage_off"          # (lamp) switched off by the stage
OWN = "myrmex_own_world"          # (scene) the organism's own world while the stage's HDRI is shown
KEPT = "myrmex_stage_kept"        # (world) kept in the file by the stage (fake user) while it is not shown
SYNC = {"busy": False}            # the Stage panel is being set from here (not by you)


def configure(d: dict | None = None) -> dict:
    """Change the stage (any of hdri / strength / rotation / lights) -> the whole stage."""
    if d:
        if "hdri" in d:
            STAGE["hdri"] = str(d.get("hdri") or "")
        if "strength" in d:
            STAGE["strength"] = round(min(20.0, max(0.0, float(d["strength"]))), 4)
        if "rotation" in d:
            STAGE["rotation"] = round(float(d["rotation"]) % 360.0, 3)
        if "lights" in d:
            STAGE["lights"] = bool(d["lights"])
    return dict(STAGE)


def from_env() -> dict | None:
    """MYRMEX_STAGE='{"hdri": "forest.exr", "strength": 1.0, "rotation": 0, "lights": true}' (set by the app)."""
    raw = os.environ.get("MYRMEX_STAGE")
    if not raw:
        return None
    try:
        d = json.loads(raw)
    except ValueError:
        return None
    return configure(d) if isinstance(d, dict) else None


def env() -> dict:
    """The stage for another Blender (a background render): {"MYRMEX_STAGE": json}."""
    return {"MYRMEX_STAGE": json.dumps(STAGE)}


def _datafiles() -> str:
    return bpy.utils.system_resource("DATAFILES", path="studiolights/world") or ""


def hdri_names() -> list:
    """Blender's world HDRIs - the ones Material Preview lights with, yours installed in Preferences too."""
    names = []
    with contextlib.suppress(AttributeError, RuntimeError):
        names = [sl.name for sl in bpy.context.preferences.studio_lights if sl.type == "WORLD"]
    if not names and os.path.isdir(_datafiles()):
        names = [f for f in os.listdir(_datafiles()) if f.lower().endswith((".exr", ".hdr"))]
    return sorted(set(names), key=str.lower)


def hdri_path(name: str) -> str:
    """The file of a world HDRI by its name ("forest.exr") or a path of its own, or "" (none / missing)."""
    if not name:
        return ""
    if os.path.isabs(name):
        return name if os.path.isfile(name) else ""
    with contextlib.suppress(AttributeError, RuntimeError):
        for sl in bpy.context.preferences.studio_lights:
            if sl.type == "WORLD" and sl.name == name and sl.path and os.path.isfile(sl.path):
                return sl.path
    p = os.path.join(_datafiles(), name)
    return p if os.path.isfile(p) else ""


def hdri_world(world: bpy.types.World, path: str, strength: float = 1.0, rotation: float = 0.0) -> None:
    """The world Material Preview builds for its HDRI - the same nodes, so the light falls the same way:
    Generated -> Vector Rotate (Z, the view's rotation) -> Environment -> Background (its strength).
    Built once; the HDRI, strength and rotation then change in place (no shader rebuild while you turn it)."""
    nt = compat.world_node_tree(world)
    env_n, rot, bg = (nt.nodes.get(n) for n in ("MyrmexStageHDRI", "MyrmexStageRotate", "MyrmexStageLight"))
    if env_n is None or rot is None or bg is None:
        for nd in list(nt.nodes):
            nt.nodes.remove(nd)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        rot = nt.nodes.new("ShaderNodeVectorRotate")
        rot.name = rot.label = "MyrmexStageRotate"
        rot.rotation_type = "Z_AXIS"
        nt.links.new(tc.outputs["Generated"], rot.inputs["Vector"])
        env_n = nt.nodes.new("ShaderNodeTexEnvironment")
        env_n.name = env_n.label = "MyrmexStageHDRI"
        nt.links.new(rot.outputs["Vector"], env_n.inputs["Vector"])
        bg = nt.nodes.new("ShaderNodeBackground")
        bg.name = bg.label = "MyrmexStageLight"
        nt.links.new(env_n.outputs["Color"], bg.inputs["Color"])
        out = nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(bg.outputs[0], out.inputs["Surface"])
        for nd, x in ((tc, -900), (rot, -700), (env_n, -480), (bg, -180), (out, 300)):
            nd.location = (x, 0)
    img = env_n.image
    if img is None or os.path.abspath(bpy.path.abspath(img.filepath)) != os.path.abspath(path):
        env_n.image = bpy.data.images.load(path, check_existing=True)
    ang = math.radians(float(rotation))
    if abs(rot.inputs["Angle"].default_value - ang) > 1e-6:
        rot.inputs["Angle"].default_value = ang
    if abs(bg.inputs["Strength"].default_value - float(strength)) > 1e-6:
        bg.inputs["Strength"].default_value = float(strength)


def _is_stage(world) -> bool:
    return world is not None and world.name == STAGE_WORLD


def own_world(scene: bpy.types.Scene):
    """The organism's own world (what it shows without the stage's HDRI, and what its look keeps)."""
    if _is_stage(scene.world):
        return bpy.data.worlds.get(scene.get(OWN) or "")
    return scene.world


def set_own_world(scene: bpy.types.Scene, world: bpy.types.World) -> None:
    if _is_stage(scene.world):
        _keep(scene, world)
    else:
        scene.world = world


def _keep(scene, world) -> None:
    """The own world waits under the stage: remembered by the scene, kept in the file (a fake user)."""
    scene[OWN] = world.name
    if not world.use_fake_user:
        world.use_fake_user = True
        world[KEPT] = True


def _release(world) -> None:
    if world.get(KEPT):
        world.use_fake_user = False
        del world[KEPT]


def world_for(scene: bpy.types.Scene) -> bpy.types.World:
    """Put the stage's light on the scene -> the world the organism is lit by now: the HDRI, else its own
    (a new organism's own world gets the soft studio panels)."""
    path = hdri_path(STAGE["hdri"])
    own = own_world(scene)
    if path:
        sw = bpy.data.worlds.get(STAGE_WORLD) or bpy.data.worlds.new(STAGE_WORLD)
        hdri_world(sw, path, STAGE["strength"], STAGE["rotation"])
        if scene.world != sw:
            if own is not None:
                _keep(scene, own)
            scene.world = sw
        return sw
    if own is None:
        from .creature import studio_world
        own = bpy.data.worlds.new("MyrmexCreatureWorld")
        studio_world(own)
    if scene.world != own:
        scene.world = own
    _release(own)
    return own


def lamps(scene: bpy.types.Scene) -> None:
    """Lamps on or off as the stage says (off: the HDRI alone lights the organism, as in Material Preview)."""
    off = not STAGE["lights"]
    for ob in scene.objects:
        if ob.type != "LIGHT":
            continue
        if off and not ob.get(OFF) and not (ob.hide_viewport and ob.hide_render):
            ob[OFF] = True
            ob.hide_viewport = ob.hide_render = True
        elif not off and ob.get(OFF):
            del ob[OFF]
            ob.hide_viewport = ob.hide_render = False


def sync_props(scene: bpy.types.Scene) -> None:
    """The Stage panel (Myrmex sidebar) shows the stage as it is now."""
    p = getattr(scene, "myrmex_stage", None)
    if p is None:
        return
    hdri = STAGE["hdri"]
    want = (("hdri", "PATH" if os.path.isabs(hdri) else (hdri or "NONE")), ("strength", STAGE["strength"]),
            ("rotation", math.radians(STAGE["rotation"])), ("lights", STAGE["lights"]))
    SYNC["busy"] = True
    try:
        for k, v in want:
            with contextlib.suppress(Exception):        # (an HDRI Blender does not list; a read-only moment)
                if getattr(p, k) != v:
                    setattr(p, k, v)
    finally:
        SYNC["busy"] = False


def is_organism(scene: bpy.types.Scene) -> bool:
    from .looks import look_kind
    return look_kind(scene) != "humanoid"


def apply(d: dict | None = None, scene: bpy.types.Scene | None = None) -> dict:
    """Change the stage and put it on the organism Blender shows, at once -> the whole stage."""
    st = configure(d)
    scene = scene or bpy.context.scene
    if scene is not None and is_organism(scene):
        from .creature import black_stage
        black_stage(scene)
    return st


def view_light(shading) -> dict | None:
    """The HDRI a 3D view lights with (Material Preview, or Rendered without the scene's world) as a stage,
    or None when the view shows the scene's own world."""
    if shading.type == "MATERIAL" and not shading.use_scene_world:
        lights = shading.use_scene_lights
    elif shading.type == "RENDERED" and not shading.use_scene_world_render:
        lights = shading.use_scene_lights_render
    else:
        return None
    return {"hdri": shading.studio_light, "strength": float(shading.studiolight_intensity),
            "rotation": math.degrees(shading.studiolight_rotate_z) % 360.0, "lights": bool(lights)}


def show_in(screen) -> None:
    """Views that show light (Material Preview, Rendered) switch to Rendered with the scene's world and
    lamps - the stage itself: that light on the organism, black behind it."""
    for area in (screen.areas if screen is not None else ()):
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type == "VIEW_3D" and space.shading.type in ("MATERIAL", "RENDERED"):
                space.shading.type = "RENDERED"
                space.shading.use_scene_world_render = True
                space.shading.use_scene_lights_render = True
        area.tag_redraw()


@contextlib.contextmanager
def saving(scene: bpy.types.Scene):
    """While a look is written: its own world and lamps (the stage is the app's, for every organism)."""
    undo = []
    try:
        own = own_world(scene) if _is_stage(scene.world) else None
        if own is not None:
            sw, kept = scene.world, bool(own.get(KEPT))
            scene.world = own
            _release(own)

            def back(sw=sw, own=own, kept=kept):
                if kept:
                    _keep(scene, own)
                scene.world = sw
            undo.append(back)
        lit = [ob for ob in scene.objects if ob.get(OFF)]
        for ob in lit:
            del ob[OFF]
            ob.hide_viewport = ob.hide_render = False

        def re_off(lit=lit):
            for ob in lit:
                ob[OFF] = True
                ob.hide_viewport = ob.hide_render = True
        undo.append(re_off)
        yield
    finally:
        for f in reversed(undo):
            with contextlib.suppress(Exception):
                f()


__all__ = ["STAGE", "STAGE_WORLD", "DEFAULTS", "configure", "from_env", "env", "hdri_names", "hdri_path",
           "hdri_world", "own_world", "set_own_world", "world_for", "lamps", "sync_props", "apply", "view_light",
           "show_in", "saving", "SYNC"]
