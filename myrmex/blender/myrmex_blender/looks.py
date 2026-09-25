"""Saved looks: your materials, lights, world, colour management and render settings, kept between sessions.

* Organisms: every type has its own list of looks - ``~/Myrmex/looks/<variant>/<name>.blend`` (plus the
  older single ``~/Myrmex/looks/<variant>.blend``, shown as "My look").  The Myrmex app lists them under
  Character type, saves the look Blender shows (Save current look) and loads a look into the running
  Blender without restarting it; live sessions and take renders open the chosen one.
* A look is saved without the take that happens to be open (its keyframes, shot cameras, markers, song).
* Humanoid: the look lives in the character's own .blend (saved in place, next to its rig.json).
"""
from __future__ import annotations

import contextlib
import os
import re

import bpy

CREATURES = ("nanomaterial", "polyalloy", "colony", "hive", "osseous", "osseous_colony", "osseous_hive", "cyber_hive",
             "swarm", "spear", "cloud", "blade", "crawler")
DEFAULT_NAME = "My look"          # the older single look per type (<variant>.blend)
AUTOSAVE = "Autosave"             # what Blender showed before another look was loaded


def looks_dir() -> str:
    return os.path.expanduser(os.environ.get("MYRMEX_LOOKS", "~/Myrmex/looks"))


def look_kind(scene: bpy.types.Scene | None = None) -> str:
    scene = scene or bpy.context.scene
    v = scene.get("myrmex_variant")
    if v in CREATURES:
        return v
    if bpy.data.objects.get("CreatureBody") is not None:
        bone = bpy.data.objects.get("PolyBones") is not None
        if bpy.data.objects.get("CyberRails") is not None:
            return "cyber_hive"
        if bpy.data.objects.get("MimeticTendons") is not None:          # (normally the scene says which)
            if bpy.data.objects.get("CrawlerTerrain") is not None:
                return "crawler"
            return "swarm" if bpy.data.objects.get("HiveSwarm") is not None else "spear"
        if bpy.data.objects.get("HiveSwarm") is not None:
            return "osseous_hive" if bone else "hive"
        if bpy.data.objects.get("ColonyScutes") or bpy.data.objects.get("ColonyPlates"):
            return "osseous_colony" if bone else "colony"
        if bone:
            return "osseous"
        return "polyalloy" if bpy.data.objects.get("PolyLattice") else "nanomaterial"
    return "humanoid"


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', " ", str(name)).strip().strip(".")
    return re.sub(r"\s+", " ", name)[:60] or "Look"


def look_path(kind: str, name: str | None = None) -> str:
    """The look file: the older single one (no name) or a named one in the type's folder."""
    if name is None or name == DEFAULT_NAME:
        return os.path.join(looks_dir(), f"{kind}.blend")
    return os.path.join(looks_dir(), kind, safe_name(name) + ".blend")


def list_looks(kind: str) -> list[tuple[str, str]]:
    """(name, path) of every saved look of a type: "My look" first, then by name, the autosave last."""
    out = []
    legacy = look_path(kind)
    if os.path.isfile(legacy):
        out.append((DEFAULT_NAME, legacy))
    d = os.path.join(looks_dir(), kind)
    if os.path.isdir(d):
        named = sorted((f[:-6] for f in os.listdir(d) if f.endswith(".blend") and not f.startswith(".")),
                       key=lambda n: (n == AUTOSAVE, n.lower()))
        out += [(n, os.path.join(d, n + ".blend")) for n in named]
    return out


def current_look(kind: str) -> str:
    """The look file this Blender has open (a look is being edited), or ""."""
    fp = bpy.data.filepath
    if not fp or kind not in CREATURES:
        return ""
    fp = os.path.abspath(fp)
    return fp if any(os.path.abspath(p) == fp for _, p in list_looks(kind)) else ""


@contextlib.contextmanager
def _without_take(scene: bpy.types.Scene):
    """Hide the open take while the look is written: keyframes, shot cameras + markers, the song, caches."""
    undo = []
    try:
        for k in [k for k in scene.keys() if k.startswith("myrmex_take")]:
            v = scene[k]
            del scene[k]
            undo.append(lambda k=k, v=v: scene.__setitem__(k, v))
        ids = list(bpy.data.objects) + list(bpy.data.metaballs) + \
            [m.node_tree for m in bpy.data.materials if m.node_tree is not None]
        for idb in ids:
            ad = idb.animation_data
            if ad is None or ad.action is None or not ad.action.name.startswith("MyrmexTake"):
                continue
            act, slot = ad.action, getattr(ad, "action_slot", None)

            def back(ad=ad, act=act, slot=slot):
                ad.action = act
                if slot is not None and getattr(ad, "action_slot", None) is None:
                    with contextlib.suppress(Exception):
                        ad.action_slot = slot
            ad.action = None
            undo.append(back)
        coll = bpy.data.collections.get("MyrmexCameras")
        if coll is not None and coll.name in scene.collection.children:
            scene.collection.children.unlink(coll)
            undo.append(lambda: scene.collection.children.link(coll))
        marks = [(m.name, m.frame, m.camera) for m in scene.timeline_markers if m.name.startswith("Shot")]
        for m in [m for m in scene.timeline_markers if m.name.startswith("Shot")]:
            scene.timeline_markers.remove(m)

        def re_mark():
            for name, frame, cam in marks:
                mk = scene.timeline_markers.new(name, frame=frame)
                mk.camera = cam
        undo.append(re_mark)
        se = scene.sequence_editor
        strips = (se.strips if hasattr(se, "strips") else se.sequences) if se is not None else None
        if strips is not None:
            for s in [s for s in strips if s.name == "MyrmexAudio"]:
                song = (s.sound.filepath if getattr(s, "sound", None) else "", s.channel, int(s.frame_start))
                strips.remove(s)
                undo.append(lambda song=song: strips.new_sound("MyrmexAudio", song[0], song[1], song[2]))
        for ob in bpy.data.objects:
            m = ob.modifiers.get("TakeCache")
            if m is None:
                continue
            keep = {k: getattr(m, k) for k in ("cache_format", "filepath", "time_mode", "play_mode", "frame_start",
                                               "frame_scale") if hasattr(m, k)}
            ob.modifiers.remove(m)

            def re_cache(ob=ob, keep=keep):
                mc = ob.modifiers.new("TakeCache", "MESH_CACHE")
                for k, v in keep.items():
                    with contextlib.suppress(Exception):
                        setattr(mc, k, v)
                ob.modifiers.move(ob.modifiers.find(mc.name), 0)
            undo.append(re_cache)
        yield
    finally:
        for f in reversed(undo):
            with contextlib.suppress(Exception):
                f()


def _write_look(path: str, scene: bpy.types.Scene) -> None:
    from . import ui
    link = ui._LINK.get("link")
    hidden = []
    if link is not None:                    # live-only speed-ups must not end up in the saved file
        for ob_name, m_name in link._restore:
            ob = bpy.data.objects.get(ob_name)
            if ob is not None and m_name in ob.modifiers:
                ob.modifiers[m_name].show_viewport = True
                hidden.append(ob.modifiers[m_name])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with _without_take(scene):
            bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
    finally:
        for m in hidden:
            m.show_viewport = False


def save_look(context, path: str | None = None, name: str | None = None) -> str:
    """Save the current scene as a look of its character type.  Returns the file written.

    ``path`` / ``name``: where (the app decides); otherwise the look being edited, or "My look"."""
    scene = context.scene
    kind = look_kind(scene)
    if kind == "humanoid":
        if not bpy.data.filepath:
            raise ValueError("open the character's .blend first (the humanoid look is saved in it)")
        bpy.ops.wm.save_mainfile()
        return bpy.data.filepath
    path = path or (look_path(kind, name) if name else "") or current_look(kind) or look_path(kind)
    _write_look(path, scene)
    return path


def viewport_settings(scene: bpy.types.Scene) -> None:
    """Light EEVEE settings for a smooth live viewport (used unless 'Keep my settings')."""
    ee = scene.eevee
    for attr, val in (("use_raytracing", False), ("use_fast_gi", False), ("taa_samples", 8),
                      ("use_shadows", True), ("shadow_resolution_scale", 0.5)):
        if hasattr(ee, attr):
            with contextlib.suppress(Exception):
                setattr(ee, attr, val)
    studio = bpy.data.collections.get("MyrmexStudio")
    if studio is not None:
        for ob in studio.objects:
            if ob.type == "LIGHT":
                ob.data.use_shadow = ob.name == "Key"


def load_look(path: str, kind: str, keep_settings: bool = True) -> dict:
    """Switch the running Blender to another look (or the default studio for ``path`` ""), live link and
    open take included.  What was on screen is kept first as the type's "Autosave" look."""
    sc = bpy.context.scene
    if path and not os.path.isfile(path):
        raise FileNotFoundError(path)
    take = None
    if sc.get("myrmex_take") and os.path.isfile(sc["myrmex_take"]):
        se = sc.sequence_editor
        strips = (se.strips if hasattr(se, "strips") else se.sequences) if se is not None else []
        song = next((bpy.path.abspath(s.sound.filepath) for s in strips
                     if s.name == "MyrmexAudio" and getattr(s, "sound", None)), "")
        take = (sc["myrmex_take"], song, int(sc.get("myrmex_take_start", sc.frame_start)),
                int(sc.get("myrmex_take_fps", sc.render.fps)))
    autosaved = ""
    was = look_kind(sc)
    if was in CREATURES and bpy.data.objects.get("CreatureBody") is not None:
        autosaved = look_path(was, AUTOSAVE)
        if os.path.abspath(path or "") != os.path.abspath(autosaved):
            _write_look(autosaved, sc)
        else:
            autosaved = ""
    if path:
        bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    else:
        bpy.ops.wm.read_homefile(use_empty=True, load_ui=False)
    sc = bpy.context.scene
    from .creature import setup_creature_scene
    setup_creature_scene(sc, kind, keep_look=bool(path))
    if not keep_settings:
        viewport_settings(sc)
    from . import live
    link = live._ACTIVE.get("link")
    if link is not None:
        link.refresh()
    if take is not None:
        from . import creature_take
        creature_take.import_take(take[0], audio=take[1] or None, frame_start=take[2], fps=take[3], keep_look=True)
    return {"path": path, "kind": kind, "autosaved": autosaved, "take": take is not None}


def forget_look(kind: str, path: str | None = None) -> bool:
    path = path or look_path(kind)
    if kind in CREATURES and os.path.isfile(path) and os.path.abspath(path).startswith(os.path.abspath(looks_dir())):
        os.remove(path)
        return True
    return False


__all__ = ["save_look", "load_look", "forget_look", "list_looks", "look_kind", "look_path", "looks_dir",
           "current_look", "viewport_settings", "CREATURES", "DEFAULT_NAME", "AUTOSAVE"]
