"""Sidebar UI: View3D > Sidebar (N) > Myrmex.

Live panel - the main workflow:
  1. pick the armature (auto-rigged character),
  2. choose where the engine runs:
       * "Separate process" - you start ``myrmex live --rig ...`` in Terminal (recommended:
         no competition with Blender's UI thread),
       * "Inside Blender"   - one click, the engine runs in a background thread here;
  3. press Start, then play Ableton / VCV.
"""
from __future__ import annotations

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
            self.report({"ERROR"}, "Pick the character's armature first")
            return {"CANCELLED"}
        s.armature = arm
        _stop_link()
        link = live.LiveLink(arm, port=s.port, camera=s.follow_camera, lights=s.follow_lights, floor=s.follow_floor)
        try:
            link.start()
        except OSError as e:
            self.report({"ERROR"}, f"Port {s.port} busy: {e}")
            return {"CANCELLED"}
        _LINK["link"] = link
        if s.engine_mode == "EMBEDDED":
            rig = _rig_json_for(s, arm)
            if not rig:
                self.report({"ERROR"}, "No rig description: set 'Rig JSON' or re-rig the character")
                return {"CANCELLED"}
            midi = [s.midi_port] if s.midi_port else []
            try:
                live.start_embedded_engine(rig, port=s.port, osc_port=s.osc_port, clock=s.clock, midi=midi,
                                           bpm=s.bpm, style=s.style, latency=s.latency_ms / 1000.0,
                                           record=bpy.path.abspath(s.record_dir) if s.record_dir else None)
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
        L.prop(s, "port")
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
                box.label(text=("playing" if fr.playing else "stopped") + (" · pose" if fr.flags & 8 else " · walking")
                               + (f" · {fr.camera.kind}" if fr.camera else ""))
            if st.get("error"):
                box.label(text=st["error"], icon="ERROR")
        eng = live.embedded_session()
        if eng is not None:
            es = eng.status()
            box = L.box()
            box.label(text=f"engine: {es['clock']} · notes {es['notes']} · tick {es['tick_ms']} ms", icon="PHYSICS")
            for e in es["errors"][:3]:
                box.label(text=e, icon="ERROR")


CLASSES = (MyrmexLiveSettings, MYRMEX_OT_live_start, MYRMEX_OT_live_stop, MYRMEX_OT_live_camera_view,
           MYRMEX_OT_setup_live_scene, MYRMEX_OT_export_rig, MYRMEX_PT_live)


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
