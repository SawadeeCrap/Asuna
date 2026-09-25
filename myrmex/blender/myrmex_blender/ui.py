"""Sidebar UI: View3D > Sidebar (N) > Myrmex.

Live panel - the main workflow:
  1. pick the armature (auto-rigged character),
  2. choose where the engine runs:
       * "Separate process" - you start ``myrmex live --rig ...`` in Terminal (recommended:
         no competition with Blender's UI thread),
       * "Inside Blender"   - one click, the engine runs in a background thread here;
  3. press Start, then play Ableton / VCV.
"""

import json
import os
import tempfile

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty, StringProperty

from . import live

_LINK = {"link": None}

CLOCKS = [("auto", "Auto", "Remote Script transport > Ableton Link > MIDI clock > kicks > internal"),
          ("osc", "Remote Script", "Transport from the Myrmex Remote Script in Ableton"),
          ("link", "Ableton Link", "Tempo and phase from Link (enable Link in Live / VCV)"),
          ("midi", "MIDI clock", "MIDI clock + start/stop (Ableton Sync out, VCV CV-MIDI)"),
          ("onsets", "Follow kicks", "Lock to incoming kick / snare hits (no clock available)"),
          ("internal", "Internal", "Fixed tempo")]
STYLES = [("catwalk", "Catwalk", "Confident runway strut, crossing steps, strong hips"),
          ("swagger", "Swagger", "Loose, wide, bouncy"),
          ("heels", "Heels", "Short precise steps"),
          ("natural", "Natural", "Plain walk")]


def _is_armature(self, obj):
    return obj.type == "ARMATURE"


class MyrmexLiveSettings(bpy.types.PropertyGroup):
    armature: PointerProperty(name="Armature", type=bpy.types.Object, poll=_is_armature)
    port: IntProperty(name="Pose port", default=9101, min=1024, max=65535)
    follow_camera: BoolProperty(name="Live camera", default=True)
    follow_lights: BoolProperty(name="Lights follow", default=True)
    follow_floor: BoolProperty(name="Endless floor", default=True)
    fast_viewport: BoolProperty(name="Fast viewport", default=True,
                                description="Skip Corrective Smooth in the viewport while live (render keeps it)")
    engine_mode: EnumProperty(name="Engine", items=[
        ("EXTERNAL", "Separate process", "Run `myrmex live` in Terminal (recommended)"),
        ("EMBEDDED", "Inside Blender", "Run the engine in a background thread of Blender")], default="EXTERNAL")
    rig_json: StringProperty(name="Rig JSON", subtype="FILE_PATH",
                             description="Rig description (only needed if the armature has none embedded)")
    osc_port: IntProperty(name="OSC in", default=9100, min=1024, max=65535)
    clock: EnumProperty(name="Clock", items=CLOCKS, default="auto")
    bpm: FloatProperty(name="BPM", default=120.0, min=40.0, max=240.0)
    midi_port: StringProperty(name="MIDI in", default="",
                              description="MIDI input port ('auto' = IAC Driver, empty = none)")
    style: EnumProperty(name="Style", items=STYLES, default="catwalk")
    latency_ms: FloatProperty(name="Latency comp. (ms)", default=60.0, min=0.0, max=300.0)
    record_dir: StringProperty(name="Record takes to", subtype="DIR_PATH", default="")
    take_audio: StringProperty(name="Song", subtype="FILE_PATH", default="",
                               description="The track you played (export it from Ableton from bar 1): lined up "
                                           "with the take by the recorded song position")
    render_size: EnumProperty(name="Size", items=[
        ("1920x1080", "1920×1080", "Landscape HD"), ("1080x1920", "1080×1920", "Vertical (reels / stories)"),
        ("1080x1080", "1080×1080", "Square"), ("3840x2160", "3840×2160", "4K"), ("1280x720", "1280×720", "Draft")],
        default="1920x1080")
    render_quality: EnumProperty(name="Quality", items=[
        ("eevee_preview", "Draft (EEVEE fast)", ""), ("eevee", "Final (EEVEE)", ""),
        ("cycles", "Cinema (Cycles, slow)", "")], default="eevee")
    keep_settings: BoolProperty(
        name="Keep my settings", default=True,
        description="Myrmex never changes EEVEE, colour management, shadows or samples: live start leaves them "
                    "alone and Render Video only sets size, frame rate and the output file")
    micro_viewport: BoolProperty(name="Micro-machines in viewport", default=False,
                                 description="Show the micro-machine surface layer in the viewport (always rendered)",
                                 update=lambda self, ctx: _micro_viewport(self.micro_viewport))


def _micro_viewport(on: bool) -> None:
    ob = bpy.data.objects.get("PolyMicro")
    if ob is not None and "MicroMachines" in ob.modifiers:
        ob.modifiers["MicroMachines"].show_viewport = on


def import_any_take(context, path: str, audio: str | None = None, use_camera: bool = True) -> str:
    """Creature (v1 / v2) or humanoid take -> animation, cameras, music.  Returns a summary line."""
    from myrmex.creature.take import is_creature_take
    path = bpy.path.abspath(path)
    if is_creature_take(path):
        from . import creature_take
        info = creature_take.import_take(path, audio=audio or None, use_camera=use_camera)
        return (f"{info['variant']} take: {info['frames']} frames @ {info['fps']} fps, {info['cameras']} shots"
                + (f", song offset {info['audio_offset']:.2f} s" if info["audio_offset"] is not None else ""))
    from myrmex.performance.performance import Performance
    from myrmex.performance.takes import recorded_camera_track, take_audio_offset

    from . import bake, cinema, preview
    s = context.scene.myrmex_live
    arm = s.armature or next((o for o in bpy.data.objects if o.type == "ARMATURE" and o.get("myrmex_rig")), None) \
        or next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise ValueError("a humanoid take needs the character: open its .blend first")
    _stop_link()
    perf = Performance.load(path)
    sc = context.scene
    sc.render.fps, sc.render.fps_base = int(round(perf.fps)), 1.0
    bake.bake(arm, perf, 1)
    shots = 0
    if use_camera:
        track = recorded_camera_track(perf)
        if track is not None:
            shots = len(cinema.apply_camera_track(track, 1))
    off = take_audio_offset(perf)
    if audio:
        preview.add_audio(bpy.path.abspath(audio), int(round(1 - (off or 0.0) * perf.fps)))
    sc["myrmex_take"] = path
    return f"humanoid take: {perf.frames} frames, {shots} shots"


class MYRMEX_OT_import_take(bpy.types.Operator):
    bl_idname = "myrmex.import_take"
    bl_label = "Import Take"
    bl_description = "Turn a recorded take (creature or humanoid .npz) into a Blender animation with its cameras and music"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.npz", options={"HIDDEN"})
    use_camera: BoolProperty(name="Recorded camera", default=True)

    def invoke(self, context, event):
        d = context.scene.myrmex_live.record_dir or os.path.expanduser("~/Myrmex/takes/")
        self.filepath = bpy.path.abspath(d) if os.path.isdir(bpy.path.abspath(d)) else ""
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        s = context.scene.myrmex_live
        try:
            msg = import_any_take(context, self.filepath, s.take_audio, self.use_camera)
        except Exception as e:
            self.report({"ERROR"}, f"Import failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class MYRMEX_OT_save_look(bpy.types.Operator):
    bl_idname = "myrmex.save_look"
    bl_label = "Save Look"
    bl_description = ("Keep this scene's look (materials, lights, world, colour, render settings) for the next "
                      "sessions and for take renders")

    def execute(self, context):
        from . import looks
        try:
            path = looks.save_look(context)
        except Exception as e:
            self.report({"ERROR"}, f"Save Look failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Look saved: {path}")
        return {"FINISHED"}


class MYRMEX_OT_forget_look(bpy.types.Operator):
    bl_idname = "myrmex.forget_look"
    bl_label = "Forget Saved Look"
    bl_description = "Delete the look this Blender has open from the saved looks"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        from . import looks
        kind = looks.look_kind(context.scene)
        cur = looks.current_look(kind)
        self.report({"INFO"}, f"Saved look removed ({kind})" if cur and looks.forget_look(kind, cur) else
                    "This scene is not a saved look")
        return {"FINISHED"}


class MYRMEX_OT_render_take(bpy.types.Operator):
    bl_idname = "myrmex.render_take"
    bl_label = "Render Video"
    bl_description = "Render the imported take to an .mp4 next to it (H.264 + AAC, with the song)"

    def execute(self, context):
        from . import creature_take
        sc = context.scene
        s = sc.myrmex_live
        take = sc.get("myrmex_take")
        if not take:
            self.report({"ERROR"}, "Import a take first")
            return {"CANCELLED"}
        w, h = (int(x) for x in s.render_size.split("x"))
        out = os.path.splitext(take)[0] + f"_{w}x{h}.mp4"
        creature_take.configure_video_output(out, (w, h), s.render_quality, keep=s.keep_settings)
        bpy.ops.render.render("INVOKE_DEFAULT", animation=True)
        self.report({"INFO"}, f"Rendering to {out}")
        return {"FINISHED"}


def _rig_json_for(settings, arm) -> str | None:
    desc = arm.get("myrmex_rig_desc") if arm is not None else None
    if desc:
        path = os.path.join(tempfile.gettempdir(), f"myrmex_rig_{arm.name}.json")
        with open(path, "w") as f:
            f.write(desc if isinstance(desc, str) else json.dumps(desc))
        return path
    if settings.rig_json:
        return bpy.path.abspath(settings.rig_json)
    return None


class MYRMEX_OT_live_start(bpy.types.Operator):
    bl_idname = "myrmex.live_start"
    bl_label = "Start Live"
    bl_description = "Listen for the pose stream and drive the armature in real time"

    def execute(self, context):
        s = context.scene.myrmex_live
        arm = s.armature or (context.object if context.object and context.object.type == "ARMATURE" else None)
        if arm is None:
            # The character prepared by prepare_character.py carries its rig description.
            arm = next((o for o in bpy.data.objects if o.type == "ARMATURE" and o.get("myrmex_rig")), None)
        from . import looks
        kind = looks.look_kind(context.scene)
        creature = kind in looks.CREATURES
        if creature:
            arm = None                                     # creature scenes stream the organism, not a rig
        elif arm is None:
            self.report({"ERROR"}, "Pick the character's armature first")
            return {"CANCELLED"}
        else:
            s.armature = arm
        _stop_link()
        link = live.LiveLink(arm, port=s.port, camera=s.follow_camera, lights=s.follow_lights, floor=s.follow_floor,
                             fast_viewport=s.fast_viewport)
        try:
            link.start()
        except OSError as e:
            self.report({"ERROR"}, f"Port {s.port} busy: {e}")
            return {"CANCELLED"}
        _LINK["link"] = link
        if s.engine_mode == "EMBEDDED":
            rig = None if creature else _rig_json_for(s, arm)
            if not rig and not creature:
                self.report({"ERROR"}, "No rig description: set 'Rig JSON' or re-rig the character")
                return {"CANCELLED"}
            midi = [s.midi_port] if s.midi_port else []
            try:
                live.start_embedded_engine(rig, port=s.port, osc_port=s.osc_port, clock=s.clock, midi=midi,
                                           bpm=s.bpm, style=s.style, latency=s.latency_ms / 1000.0,
                                           record=bpy.path.abspath(s.record_dir) if s.record_dir else None,
                                           backend={"nanomaterial": "creature"}.get(kind, kind) if creature
                                           else "humanoid")
            except Exception as e:
                self.report({"ERROR"}, f"Engine failed: {e}")
                return {"CANCELLED"}
        self.report({"INFO"}, f"Myrmex live on port {s.port}")
        return {"FINISHED"}


def _stop_link():
    link = _LINK.get("link")
    if link is not None:
        link.stop()
        _LINK["link"] = None


class MYRMEX_OT_live_stop(bpy.types.Operator):
    bl_idname = "myrmex.live_stop"
    bl_label = "Stop Live"

    def execute(self, context):
        path = live.stop_embedded_engine()
        _stop_link()
        if path:
            self.report({"INFO"}, f"Take saved: {path}")
        return {"FINISHED"}


class MYRMEX_OT_live_camera_view(bpy.types.Operator):
    bl_idname = "myrmex.live_camera_view"
    bl_label = "Look Through Live Camera"

    def execute(self, context):
        cam = bpy.data.objects.get(live.LIVE_CAMERA)
        if cam is not None:
            context.scene.camera = cam
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.region_3d.view_perspective = "CAMERA"
                        space.shading.type = "MATERIAL" if space.shading.type in ("SOLID", "WIREFRAME") else space.shading.type
        return {"FINISHED"}


class MYRMEX_OT_setup_live_scene(bpy.types.Operator):
    bl_idname = "myrmex.setup_live_scene"
    bl_label = "Studio + Chrome Look"
    bl_description = "Black chrome material, light-grey studio with travelling lights, endless floor"

    material: EnumProperty(name="Material", items=[(k, k.replace("_", " ").title(), "") for k in
                                                    ("black_chrome", "chrome", "liquid_metal", "gunmetal", "ceramic",
                                                     "clay", "iridescent")], default="black_chrome")

    def execute(self, context):
        import numpy as np

        from . import cinema
        s = context.scene.myrmex_live
        arm = s.armature
        if arm is None:
            self.report({"ERROR"}, "Pick the armature first")
            return {"CANCELLED"}
        for ob in bpy.data.objects:
            if ob.type == "MESH" and ob.parent == arm and not ob.name.endswith("_proxy"):
                cinema.apply_material(ob, self.material, keep_textures=True)
        cinema.setup_studio(np.zeros((1, 3)), np.array([[0.0, -1.0, 0.0]]))
        return {"FINISHED"}


class MYRMEX_OT_export_rig(bpy.types.Operator):
    bl_idname = "myrmex.export_rig"
    bl_label = "Export Rig JSON"
    bl_description = "Write the armature's rig description for `myrmex live --rig`"

    filepath: StringProperty(subtype="FILE_PATH")

    def invoke(self, context, event):
        self.filepath = bpy.path.abspath("//myrmex_rig.json")
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        arm = context.scene.myrmex_live.armature
        desc = arm.get("myrmex_rig_desc") if arm else None
        if not desc:
            self.report({"ERROR"}, "This armature has no embedded rig description")
            return {"CANCELLED"}
        with open(self.filepath, "w") as f:
            f.write(desc)
        self.report({"INFO"}, f"Saved {self.filepath}")
        return {"FINISHED"}


class MYRMEX_PT_live(bpy.types.Panel):
    bl_label = "Myrmex Live"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Myrmex"

    def draw(self, context):
        s = context.scene.myrmex_live
        L = self.layout
        L.prop(s, "armature")
        L.prop(s, "engine_mode")
        link = _LINK.get("link")
        col = L.column(align=True)
        if s.engine_mode == "EMBEDDED":
            col.prop(s, "clock")
            if s.clock == "internal":
                col.prop(s, "bpm")
            col.prop(s, "osc_port")
            col.prop(s, "midi_port")
            col.prop(s, "style")
            col.prop(s, "latency_ms")
            col.prop(s, "record_dir")
            if not (s.armature and s.armature.get("myrmex_rig_desc")):
                col.prop(s, "rig_json")
        else:
            box = L.box()
            box.label(text="In Terminal:")
            box.label(text="myrmex live --rig myrmex_rig.json")
            box.operator("myrmex.export_rig", icon="EXPORT")
        row = L.row(align=True)
        row.prop(s, "follow_camera", toggle=True)
        row.prop(s, "follow_lights", toggle=True)
        row.prop(s, "follow_floor", toggle=True)
        row = L.row(align=True)
        row.prop(s, "port")
        row.prop(s, "fast_viewport")
        row = L.row(align=True)
        if link is None:
            row.operator("myrmex.live_start", icon="PLAY")
        else:
            row.operator("myrmex.live_stop", icon="PAUSE")
        row.operator("myrmex.live_camera_view", icon="CAMERA_DATA", text="")
        L.operator("myrmex.setup_live_scene", icon="LIGHT_AREA")
        if link is not None:
            st = link.stats
            box = L.box()
            fr = link.last
            if fr is None:
                box.label(text=f"waiting for poses on :{s.port} …", icon="TIME")
            else:
                box.label(text=f"{st['fps']:.0f} fps  beat {fr.beat:.1f}  {fr.bpm:.1f} BPM", icon="SOUND")
                if hasattr(fr, "morphology"):              # creature stream
                    box.label(text=f"{fr.behavior.lower()} · {fr.morphology.lower()}"
                                   + (f" · {fr.material.lower()}" if getattr(fr, "material", "") else "")
                                   + (f" · {fr.camera.kind}" if fr.camera else ""))
                else:
                    box.label(text=("playing" if fr.playing else "stopped") + (" · pose" if fr.flags & 8 else " · walking")
                                   + (f" · {fr.camera.kind}" if fr.camera else ""))
            if st.get("error"):
                box.label(text=st["error"], icon="ERROR")
        from . import looks
        box = L.box()
        kind = looks.look_kind(context.scene)
        box.label(text=f"Look · {kind}", icon="SHADING_RENDERED")
        row = box.row(align=True)
        row.operator("myrmex.save_look", icon="FILE_TICK")
        if kind in looks.CREATURES:
            cur = looks.current_look(kind)
            if cur:
                row.operator("myrmex.forget_look", icon="X", text="")
                name = next((n for n, p in looks.list_looks(kind) if os.path.abspath(p) == cur), "")
                box.label(text=f"editing: {name}", icon="CHECKMARK")
            n = len(looks.list_looks(kind))
            box.label(text=f"{n} saved look{'s' if n != 1 else ''} · choose / save them in the Myrmex app", icon="INFO")
        box.prop(s, "keep_settings")
        box = L.box()
        box.label(text="Takes → video", icon="RENDER_ANIMATION")
        box.prop(s, "take_audio")
        box.operator("myrmex.import_take", icon="IMPORT")
        row = box.row(align=True)
        row.prop(s, "render_size", text="")
        sub = row.row(align=True)
        sub.enabled = not s.keep_settings                  # with "Keep my settings" the scene decides
        sub.prop(s, "render_quality", text="")
        box.operator("myrmex.render_take", icon="RENDER_ANIMATION")
        if bpy.data.objects.get("PolyMicro") is not None:
            box.prop(s, "micro_viewport")
        if context.scene.get("myrmex_take"):
            box.label(text=os.path.basename(context.scene["myrmex_take"]), icon="FILE_MOVIE")
        eng = live.embedded_session()
        if eng is not None:
            es = eng.status()
            box = L.box()
            box.label(text=f"engine: {es['clock']} · notes {es['notes']} · tick {es['tick_ms']} ms", icon="PHYSICS")
            for e in es["errors"][:3]:
                box.label(text=e, icon="ERROR")


CLASSES = (MyrmexLiveSettings, MYRMEX_OT_live_start, MYRMEX_OT_live_stop, MYRMEX_OT_live_camera_view,
           MYRMEX_OT_setup_live_scene, MYRMEX_OT_export_rig, MYRMEX_OT_import_take, MYRMEX_OT_render_take,
           MYRMEX_OT_save_look, MYRMEX_OT_forget_look, MYRMEX_PT_live)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.myrmex_live = PointerProperty(type=MyrmexLiveSettings)


def unregister():
    live.stop_embedded_engine()
    _stop_link()
    del bpy.types.Scene.myrmex_live
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
