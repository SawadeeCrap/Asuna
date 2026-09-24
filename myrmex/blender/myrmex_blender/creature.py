"""Presentation of the Black Nanomaterial Creature in Blender.

The engine streams high-level state (node positions, material radii, surface activity);
Blender builds the visible body locally: every node is an element of one metaball field,
so the material stays a single continuous substance that thickens, thins, splits into
appendages and merges back.  The nanomaterial shader reads per-frame values (activity,
glow, time) written on its nodes.  An optional debug overlay shows the control network.
"""
from __future__ import annotations

import math

import bpy
import numpy as np

COLL = "MyrmexCreature"
META = "CreatureBody"
DEBUG = "CreatureDebug"
MAT = "MyrmexNanomaterial"


def nanomaterial() -> bpy.types.Material:
    mat = bpy.data.materials.get(MAT)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(MAT)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    val = nt.nodes.new("ShaderNodeValue")
    val.name = "MyrmexTime"
    act = nt.nodes.new("ShaderNodeValue")
    act.name = "MyrmexActivity"
    glow = nt.nodes.new("ShaderNodeValue")
    glow.name = "MyrmexGlow"
    # Microstructure: flowing 4D noise (W = time) -> micro grooves (bump) and roughness variation.
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = 38.0
    nz.inputs["Detail"].default_value = 6.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    nt.links.new(val.outputs[0], nz.inputs["W"])
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 90.0
    nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMath")
    mix.operation = "MULTIPLY_ADD"
    nt.links.new(vor.outputs["Distance"], mix.inputs[0])
    nt.links.new(act.outputs[0], mix.inputs[1])
    nt.links.new(nz.outputs["Fac"], mix.inputs[2])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    bump.inputs["Distance"].default_value = 0.004
    nt.links.new(mix.outputs[0], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = 0.12
    rr.inputs["To Max"].default_value = 0.34
    nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    for k, v in (("Base Color", (0.006, 0.006, 0.007, 1.0)), ("Metallic", 0.85), ("Coat Weight", 0.45),
                 ("Coat Roughness", 0.06), ("Anisotropic", 0.35), ("Specular IOR Level", 0.6)):
        if k in b.inputs:
            b.inputs[k].default_value = v
    # Internal energy traces: a thin fresnel-masked, noise-gated, very dark amber emission.
    fr = nt.nodes.new("ShaderNodeLayerWeight")
    em = nt.nodes.new("ShaderNodeMath")
    em.operation = "MULTIPLY"
    nt.links.new(fr.outputs["Facing"], em.inputs[0])
    nt.links.new(glow.outputs[0], em.inputs[1])
    gate = nt.nodes.new("ShaderNodeMath")
    gate.operation = "GREATER_THAN"
    gate.inputs[1].default_value = 0.62
    nt.links.new(nz.outputs["Fac"], gate.inputs[0])
    em2 = nt.nodes.new("ShaderNodeMath")
    em2.operation = "MULTIPLY"
    nt.links.new(em.outputs[0], em2.inputs[0])
    nt.links.new(gate.outputs[0], em2.inputs[1])
    if "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = (1.0, 0.32, 0.08, 1.0)
        nt.links.new(em2.outputs[0], b.inputs["Emission Strength"])
    return mat


def dark_studio(scene: bpy.types.Scene) -> None:
    """Dark environment, one large soft source, two rim lights, glossy black floor (all follow the creature)."""
    world = scene.world or bpy.data.worlds.new("MyrmexCreatureWorld")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    bg = next((n for n in world.node_tree.nodes if n.type == "BACKGROUND"), None)
    if bg is not None:
        bg.inputs["Color"].default_value = (0.004, 0.004, 0.005, 1.0)
        bg.inputs["Strength"].default_value = 1.0
    coll = bpy.data.collections.get("MyrmexStudio") or bpy.data.collections.new("MyrmexStudio")
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    for o in list(coll.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    rig = bpy.data.objects.new("MyrmexLightRig", None)
    coll.objects.link(rig)

    def light(name, loc, rot, size, power, color=(1, 1, 1)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.size, ld.energy, ld.color = size, power, color
        ob = bpy.data.objects.new(name, ld)
        coll.objects.link(ob)
        ob.parent = rig
        ob.location, ob.rotation_euler = loc, rot
    light("Key", (1.5, 2.5, 5.0), (math.radians(28), 0, math.radians(150)), 6.0, 1400)
    light("RimL", (-3.2, 2.2, 2.2), (math.radians(70), 0, math.radians(-125)), 1.2, 900, (0.92, 0.95, 1.0))
    light("RimR", (-2.8, -2.6, 1.8), (math.radians(72), 0, math.radians(-50)), 1.0, 700, (1.0, 0.93, 0.86))
    me = bpy.data.meshes.new("MyrmexFloor")
    s = 200.0
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("MyrmexFloor", me)
    coll.objects.link(floor)
    fm = bpy.data.materials.new("MyrmexBlackFloor")
    try:
        fm.use_nodes = True
    except Exception:
        pass
    fb = next(n for n in fm.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    fb.inputs["Base Color"].default_value = (0.012, 0.012, 0.013, 1)
    fb.inputs["Roughness"].default_value = 0.22
    me.materials.append(fm)


class CreatureView:
    def __init__(self, scene: bpy.types.Scene | None = None, resolution: float = 0.03):
        self.scene = scene or bpy.context.scene
        coll = bpy.data.collections.get(COLL) or bpy.data.collections.new(COLL)
        if coll.name not in self.scene.collection.children:
            self.scene.collection.children.link(coll)
        self.coll = coll
        mb = bpy.data.metaballs.get(META) or bpy.data.metaballs.new(META)
        mb.resolution = resolution                       # viewport polygonisation (speed)
        mb.render_resolution = 0.02
        mb.threshold = 0.6
        self.obj = bpy.data.objects.get(META) or bpy.data.objects.new(META, mb)
        if self.obj.name not in coll.objects:
            coll.objects.link(self.obj)
        if not mb.materials:
            mb.materials.append(nanomaterial())
        self.mb = mb
        self.debug_obj = None
        self.t0 = None

    def _ensure(self, n: int) -> None:
        els = self.mb.elements
        while len(els) < n:
            e = els.new()
            e.type = "ELLIPSOID"
        while len(els) > n:
            els.remove(els[-1])

    def _debug(self, fr, show: bool) -> None:
        if not show:
            if self.debug_obj is not None:
                self.debug_obj.hide_viewport = True
            return
        n = len(fr.pos)
        ob = bpy.data.objects.get(DEBUG)
        if ob is None or len(ob.data.vertices) != n:
            me = bpy.data.meshes.new(DEBUG)
            edges = [(i, int(a)) for i, a in enumerate(fr.anchor) if 0 <= a < n]
            me.from_pydata(fr.pos.tolist(), edges, [])
            if ob is None:
                ob = bpy.data.objects.new(DEBUG, me)
                self.coll.objects.link(ob)
            else:
                ob.data = me
            ob.show_in_front = True
            ob.display_type = "WIRE"
        ob.hide_viewport = False
        ob.data.vertices.foreach_set("co", fr.pos.astype(np.float32).ravel())
        ob.data.update()
        self.debug_obj = ob

    def apply(self, fr) -> None:
        n = len(fr.pos)
        self._ensure(n)
        els = self.mb.elements
        for i, e in enumerate(els):
            r = float(fr.radius[i])
            if r < 0.01:
                e.hide = True
                continue
            e.hide = False
            e.co = fr.pos[i]
            e.radius = r
            sx, sy, sz = fr.stretch[i]
            e.size_x, e.size_y, e.size_z = 0.55 * sx, 0.55 * sy, 0.55 * sz
            e.stiffness = 2.0 if fr.kind[i] == 0 else 1.6
        nt = self.mb.materials[0].node_tree if self.mb.materials and self.mb.materials[0] else None
        if nt is not None:
            if "MyrmexTime" in nt.nodes:
                nt.nodes["MyrmexTime"].outputs[0].default_value = float(fr.t) * (0.05 + 0.25 * fr.surface)
            if "MyrmexActivity" in nt.nodes:
                nt.nodes["MyrmexActivity"].outputs[0].default_value = 0.2 + 0.8 * float(fr.surface)
            if "MyrmexGlow" in nt.nodes:
                nt.nodes["MyrmexGlow"].outputs[0].default_value = 4.0 * float(fr.glow)
        self._debug(fr, bool(fr.flags & 16))


def setup_creature_scene(scene: bpy.types.Scene | None = None) -> CreatureView:
    scene = scene or bpy.context.scene
    for name in ("Cube", "Light", "Camera"):                  # Blender's default startup objects
        ob = bpy.data.objects.get(name)
        if ob is not None:
            bpy.data.objects.remove(ob, do_unlink=True)
    view = CreatureView(scene)
    dark_studio(scene)
    return view
