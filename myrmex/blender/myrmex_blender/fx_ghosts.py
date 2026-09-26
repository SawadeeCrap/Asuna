"""Afterimages: copies of the organism left behind as it moves - the "Sandevistan" look.

A ring of ghost slots.  A slot holds one copy of every visible part of the organism: the metaball body as a
metaball of its own (Blender polygonises it once), meshes as meshes (only their vertices are copied while
the topology stays), Geometry Nodes parts with their node trees - all drawn with one holographic material:
a neon tint per copy, a bright rim, a faint fill, scanlines, and a noise dissolve as the copy dies.

Copies are real geometry, so they are in the viewport, in Syphon and in renders.  Per frame only the
material's clock moves; a copy's geometry is written once, when it is left behind.  A copy is left when the
organism has moved far enough since the last one (evenly spaced along its path, none while it stands
still), and on impacts and shape changes; dense settings make a continuous echo smear.
"""
from __future__ import annotations

import math

import bpy
import numpy as np

from . import compat

COLL = "MyrmexFX"
PREFIX = "MyrmexGhost"
MAT = "MyrmexGhost"
SETMAT = "MyrmexGhostMaterial"
MAX_SLOTS = 16
SKIP = {"CreatureDebug", "ColonyPrey", "CrawlerTerrain", "PolyMicro", "BionicPlumes", "MyrmexFloor"}
DEFORM = {"ARMATURE", "CORRECTIVE_SMOOTH", "LATTICE", "CURVE", "HOOK", "SHRINKWRAP", "SIMPLE_DEFORM", "CAST",
          "SMOOTH", "LAPLACIANSMOOTH", "SURFACE_DEFORM", "MESH_DEFORM", "WARP", "WAVE", "DISPLACE",
          "LAPLACIANDEFORM", "MESH_CACHE", "MESH_SEQUENCE_CACHE"}

# Linear RGB, one colour per copy in turn (the Sandevistan cycle runs through the spectrum).
PALETTES = {
    "Sandevistan": [(0.25, 1.0, 0.18), (0.0, 1.0, 0.7), (0.05, 0.6, 1.0), (0.45, 0.2, 1.0), (1.0, 0.12, 0.75),
                    (1.0, 0.3, 0.08), (1.0, 0.85, 0.08)],
    "Neon": [(0.0, 0.9, 1.0), (1.0, 0.08, 0.8)],
    "Ice": [(0.7, 0.88, 1.0), (0.45, 0.7, 1.0), (1.0, 1.0, 1.0)],
    "Blood": [(1.0, 0.04, 0.02), (0.85, 0.0, 0.12), (1.0, 0.22, 0.04)],
    "Gold": [(1.0, 0.66, 0.18), (1.0, 0.84, 0.42), (1.0, 0.45, 0.08)],
    "Toxic": [(0.45, 1.0, 0.0), (0.08, 1.0, 0.3), (0.8, 1.0, 0.08)],
}
PALETTE_ORDER = ("Sandevistan", "Neon", "Ice", "Blood", "Gold", "Toxic")


def palette(x: float) -> list:
    i = int(round(min(1.0, max(0.0, float(x))) * (len(PALETTE_ORDER) - 1)))
    return PALETTES[PALETTE_ORDER[i]]


def fx_collection(scene: bpy.types.Scene | None = None) -> bpy.types.Collection:
    scene = scene or bpy.context.scene
    coll = bpy.data.collections.get(COLL)
    if coll is None:
        coll = bpy.data.collections.new(COLL)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


# ---------------------------------------------------------------------- the material
def _n(nt, kind, x=0, y=0):
    nd = nt.nodes.new(kind)
    nd.location = (x, y)
    return nd


def _m(nt, op, a, b=None, clamp=False, x=0, y=0, c=None):
    nd = _n(nt, "ShaderNodeMath", x, y)
    nd.operation = op
    nd.use_clamp = clamp
    for k, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            nd.inputs[k].default_value = float(v)
        else:
            nt.links.new(v, nd.inputs[k])
    return nd.outputs[0]


def _value(nt, name: str, v: float, x=0, y=0):
    nd = _n(nt, "ShaderNodeValue", x, y)
    nd.name = nd.label = name
    nd.outputs[0].default_value = v
    return nd.outputs[0]


def ghost_material() -> bpy.types.Material:
    """Holographic copy: tint (object colour) x (rim + fill), fading with age, dissolving into noise."""
    mat = bpy.data.materials.get(MAT)
    if mat is not None and mat.node_tree is not None and "MyrmexNow" in mat.node_tree.nodes:
        return mat
    mat = mat or bpy.data.materials.new(MAT)
    nt = compat.material_node_tree(mat)
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    now = _value(nt, "MyrmexNow", 0.0, -1400, 300)
    life = _value(nt, "MyrmexLife", 0.8, -1400, 200)
    gain = _value(nt, "MyrmexGain", 1.5, -1400, 100)
    fill = _value(nt, "MyrmexFill", 0.3, -1400, 0)
    birth = _n(nt, "ShaderNodeAttribute", -1400, 450)
    birth.attribute_type = "OBJECT"
    birth.attribute_name = "myrmex_birth"
    info = _n(nt, "ShaderNodeObjectInfo", -1400, -250)
    age = _m(nt, "SUBTRACT", now, birth.outputs["Fac"], x=-1200, y=400)
    x = _m(nt, "DIVIDE", age, life, clamp=True, x=-1000, y=400)
    alive = _m(nt, "GREATER_THAN", age, -0.001, x=-1000, y=250)
    fade = _m(nt, "MULTIPLY", _m(nt, "POWER", _m(nt, "SUBTRACT", 1.0, x, x=-850, y=400), 1.6, x=-700, y=400),
              alive, x=-550, y=400)
    # dissolve: noise on the copy's own surface; the threshold rises as it fades
    tc = _n(nt, "ShaderNodeTexCoord", -1200, -500)
    seed = _m(nt, "MULTIPLY", info.outputs["Random"], 37.0, x=-1200, y=-700)
    off = _n(nt, "ShaderNodeCombineXYZ", -1050, -700)
    nt.links.new(seed, off.inputs[0])
    nt.links.new(seed, off.inputs[1])
    vec = _n(nt, "ShaderNodeVectorMath", -900, -550)
    vec.operation = "ADD"
    nt.links.new(tc.outputs["Object"], vec.inputs[0])
    nt.links.new(off.outputs[0], vec.inputs[1])
    noise = _n(nt, "ShaderNodeTexNoise", -750, -550)
    noise.inputs["Scale"].default_value = 7.0
    noise.inputs["Detail"].default_value = 2.0
    nt.links.new(vec.outputs[0], noise.inputs["Vector"])
    thr = _m(nt, "SUBTRACT", _m(nt, "MULTIPLY", _m(nt, "SUBTRACT", 1.0, fade, x=-750, y=-250), 1.15, x=-600,
                                                   y=-250), 0.1, x=-450, y=-250)
    cut = _n(nt, "ShaderNodeMapRange", -300, -400)
    cut.clamp = True
    nt.links.new(noise.outputs["Fac"], cut.inputs["Value"])
    nt.links.new(_m(nt, "SUBTRACT", thr, 0.06, x=-450, y=-400), cut.inputs["From Min"])
    nt.links.new(_m(nt, "ADD", thr, 0.06, x=-450, y=-500), cut.inputs["From Max"])
    live = _m(nt, "MULTIPLY", fade, cut.outputs["Result"], x=-150, y=300)
    # scanlines (screen space) and the rim
    sep = _n(nt, "ShaderNodeSeparateXYZ", -900, -900)
    nt.links.new(tc.outputs["Window"], sep.inputs[0])
    wave = _m(nt, "SINE", _m(nt, "MULTIPLY", sep.outputs["Y"], 1600.0, x=-750, y=-900), x=-600, y=-900)
    scan = _m(nt, "MULTIPLY_ADD", wave, 0.18, x=-450, y=-900, c=0.82)
    lw = _n(nt, "ShaderNodeLayerWeight", -900, 100)
    lw.inputs["Blend"].default_value = 0.35
    rim = _m(nt, "POWER", lw.outputs["Facing"], 1.8, x=-700, y=100)
    r1 = _m(nt, "MULTIPLY_ADD", rim, 4.0, x=-550, y=100, c=0.15)
    r2 = _m(nt, "MULTIPLY", r1, gain, x=-400, y=100)
    r3 = _m(nt, "MULTIPLY", live, scan, x=-400, y=-100)
    rim_s = _m(nt, "MULTIPLY", r2, r3, x=-200, y=100)
    em_rim = _n(nt, "ShaderNodeEmission", 50, 150)
    nt.links.new(info.outputs["Color"], em_rim.inputs["Color"])
    nt.links.new(rim_s, em_rim.inputs["Strength"])
    em_fill = _n(nt, "ShaderNodeEmission", 50, -50)
    nt.links.new(info.outputs["Color"], em_fill.inputs["Color"])
    em_fill.inputs["Strength"].default_value = 1.4
    tr = _n(nt, "ShaderNodeBsdfTransparent", 50, -200)
    mix = _n(nt, "ShaderNodeMixShader", 250, -100)
    nt.links.new(_m(nt, "MULTIPLY", live, fill, clamp=True, x=50, y=-350), mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em_fill.outputs[0], mix.inputs[2])
    add = _n(nt, "ShaderNodeAddShader", 450, 0)
    nt.links.new(mix.outputs[0], add.inputs[0])
    nt.links.new(em_rim.outputs[0], add.inputs[1])
    out = _n(nt, "ShaderNodeOutputMaterial", 650, 0)
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    _transparent(mat)
    return mat


def _transparent(mat: bpy.types.Material) -> None:
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND"), ("use_backface_culling", True),
                      ("use_transparency_overlap", False), ("use_transparent_shadow", True),
                      ("shadow_method", "NONE")):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, val)
            except (TypeError, ValueError, AttributeError):
                pass
    mat.diffuse_color = (0.3, 1.0, 0.6, 0.4)                 # (the solid viewport)


def set_material_group() -> bpy.types.NodeTree:
    """Geometry Nodes parts carry their own materials: realise, then everything gets the ghost material."""
    ng = bpy.data.node_groups.get(SETMAT)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(SETMAT, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    rl = ng.nodes.new("GeometryNodeRealizeInstances")
    sm = ng.nodes.new("GeometryNodeSetMaterial")
    sm.inputs["Material"].default_value = ghost_material()
    ng.links.new(gi.outputs[0], rl.inputs[0])
    ng.links.new(rl.outputs[0], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], go.inputs[0])
    return ng


# ---------------------------------------------------------------------- what the organism is made of
def sources(scene: bpy.types.Scene | None = None) -> list:
    """The organism's visible parts (creature scenes), or the character's skinned meshes (humanoid)."""
    coll = bpy.data.collections.get("MyrmexCreature")
    out = []
    if coll is not None and len(coll.all_objects):
        for ob in coll.all_objects:
            if ob.name in SKIP or ob.name.startswith("PolyObstacle") or ob.type not in ("MESH", "META"):
                continue
            if ob.hide_render or ob.hide_viewport:
                continue
            if ob.type == "META" and (len(ob.data.elements) == 0 or "." in ob.name):
                continue
            if ob.type == "MESH" and len(ob.data.vertices) == 0:
                continue
            out.append(ob)
        return out
    for ob in bpy.data.objects:                              # humanoid: meshes deformed by an armature
        if ob.type == "MESH" and not ob.hide_render and not ob.hide_viewport and not ob.name.endswith("_proxy") \
                and any(m.type == "ARMATURE" for m in ob.modifiers):
            out.append(ob)
    return out


def _pc2_frame(mod, frame: float) -> np.ndarray | None:
    """Positions of a Point Cache 2 file at a scene frame (the swarm of an imported take)."""
    import struct
    path = bpy.path.abspath(mod.filepath)
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            _, _, n, start, rate, count = struct.unpack("<12siiffi", head)
            k = int(round((frame - float(getattr(mod, "frame_start", 0.0))) * float(getattr(mod, "frame_scale", 1.0))))
            k = min(max(k, 0), count - 1)
            f.seek(32 + k * n * 12)
            return np.frombuffer(f.read(n * 12), "<f4").reshape(n, 3).copy()
    except (OSError, ValueError, struct.error):
        return None


def _deformed(ob) -> bool:
    return any(m.type in DEFORM and m.show_viewport and m.name != "TakeCache" for m in ob.modifiers)


# ---------------------------------------------------------------------- ghost objects
class Ghosts:
    def __init__(self):
        self.birth = [-1e9] * MAX_SLOTS
        self.parts: list[dict] = [dict() for _ in range(MAX_SLOTS)]
        self.next = 0
        self.count = 0
        self.last_t = None
        self.last_spawn_t = -1e9
        self.last_com = None
        self.last_impact = 0.0
        self.last_morph = 0.0
        self.k = 0
        self.spawned = 0
        self.signature = None

    # -- lifecycle
    def clear(self, remove: bool = False) -> None:
        for s in range(MAX_SLOTS):
            self._clear_slot(s)
        self.last_com = None
        if remove:
            drop_all()
            self.parts = [dict() for _ in range(MAX_SLOTS)]

    def _clear_slot(self, s: int) -> None:
        self.birth[s] = -1e9
        for name in list(self.parts[s]):
            ob = bpy.data.objects.get(name)
            if ob is None:
                del self.parts[s][name]
                continue
            ob["myrmex_birth"] = -1e9
            if ob.type == "META":
                if len(ob.data.elements):
                    ob.data.elements.clear()
            elif len(ob.data.vertices):
                ob.data.clear_geometry()

    # -- one frame
    def update(self, scene, t: float, com, size: float, rack: dict, drives: dict, frame_ok: bool = True) -> None:
        amt = float(rack.get("ghosts", 0.0))
        mat = ghost_material()
        nt = mat.node_tree
        if amt < 0.01:
            if any(b > -1e8 for b in self.birth):
                self.clear()
            self.last_t = t
            return
        if self.last_t is None or t < self.last_t - 1e-6 or t - self.last_t > 1.0 or not frame_ok:
            self.clear()                                      # a jump (scrub, new take): start over
            self.last_spawn_t = -1e9
        self.last_t = t
        density = float(rack.get("ghost_density", 0.35))
        life = 0.15 + 1.6 * float(rack.get("ghost_life", 0.45))
        n = int(round(4 + 12 * density))
        min_dt = 0.16 - 0.13 * density
        spacing = max(0.02, float(size) * (0.9 - 0.8 * density))
        com = np.asarray(com, float)
        moved = float(np.linalg.norm(com - self.last_com)) if self.last_com is not None else 1e9
        impact, morph = float(drives.get("impact", 0.0)), float(drives.get("morph", 0.0))
        burst = (impact - self.last_impact > 0.25) or (morph - self.last_morph > 0.4)
        self.last_impact, self.last_morph = impact, morph
        due = t - self.last_spawn_t >= min_dt and (moved >= spacing or burst or
                                                   (moved > 0.35 * spacing and t - self.last_spawn_t > 4 * min_dt))
        if due:
            self.spawn(scene, t, n, rack)
            self.last_spawn_t = t
            self.last_com = com.copy()
        nd = nt.nodes
        nd["MyrmexNow"].outputs[0].default_value = float(t)
        nd["MyrmexLife"].outputs[0].default_value = life
        nd["MyrmexGain"].outputs[0].default_value = 0.35 + 1.6 * amt
        nd["MyrmexFill"].outputs[0].default_value = 0.1 + 0.35 * amt
        for s in range(MAX_SLOTS):                              # expired copies: no geometry, no cost
            if self.birth[s] > -1e8 and (t - self.birth[s] > life or s >= n):
                self._clear_slot(s)
        self.count = n

    def spawn(self, scene, t: float, n: int, rack: dict) -> None:
        srcs = sources(scene)
        sig = tuple(sorted((o.name, o.type) for o in srcs))
        if sig != self.signature:                             # another organism: other parts
            self.clear(remove=True)
            self.signature = sig
        s = self.next % max(1, n)
        self.next = s + 1
        pal = palette(rack.get("palette", 0.0))
        tint = pal[self.k % len(pal)]
        self.k += 1
        coll = fx_collection(scene)
        written = set()
        depsgraph = None
        for src in srcs:
            name = f"{PREFIX}{s:02d}_{src.name}"
            try:
                if src.type == "META":
                    ob = self._meta_ghost(name, src, coll)
                    _copy_elements(src.data, ob.data)
                else:
                    if _deformed(src):
                        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
                        ob = self._eval_ghost(name, src, coll, depsgraph)
                    else:
                        ob = self._mesh_ghost(name, src, coll)
                        co = None
                        mc = src.modifiers.get("TakeCache")
                        if mc is not None and mc.type == "MESH_CACHE":
                            co = _pc2_frame(mc, scene.frame_current)
                        if co is None:
                            co = np.empty(3 * len(src.data.vertices), np.float32)
                            src.data.vertices.foreach_get("co", co)
                        if co.size == 3 * len(ob.data.vertices):
                            ob.data.vertices.foreach_set("co", np.ascontiguousarray(co, np.float32).ravel())
                            ob.data.update()
                ob.matrix_world = src.matrix_world
            except (ReferenceError, RuntimeError, ValueError) as e:
                print("Myrmex FX ghost:", src.name, e)
                continue
            ob["myrmex_birth"] = float(t)
            ob.color = (*tint, 1.0)
            written.add(ob.name)
            self.parts[s][ob.name] = True
        for name in list(self.parts[s]):                      # a part the organism no longer has
            if name not in written:
                ob = bpy.data.objects.get(name)
                if ob is not None:
                    bpy.data.objects.remove(ob, do_unlink=True)
                del self.parts[s][name]
        self.birth[s] = float(t)
        self.spawned += 1

    # -- the three kinds of copy
    @staticmethod
    def _link(ob, coll) -> None:
        if ob.name not in coll.objects:
            coll.objects.link(ob)
        ob.visible_shadow = False
        if hasattr(ob, "visible_volume_scatter"):
            ob.visible_volume_scatter = False

    def _meta_ghost(self, name: str, src, coll):
        ob = bpy.data.objects.get(name)
        if ob is None or ob.type != "META":
            if ob is not None:
                bpy.data.objects.remove(ob, do_unlink=True)
            mb = bpy.data.metaballs.new(name)
            ob = bpy.data.objects.new(name, mb)
            self._link(ob, coll)
        mb, sm = ob.data, src.data
        mb.resolution = max(0.02, sm.resolution * 1.6)              # a copy may be coarser: it is a glow
        mb.render_resolution = max(0.012, sm.render_resolution * 1.5)
        mb.threshold = sm.threshold
        gm = ghost_material()
        if len(mb.materials) != 1 or mb.materials[0] is not gm:
            mb.materials.clear()
            mb.materials.append(gm)
        return ob

    def _mesh_ghost(self, name: str, src, coll):
        ob = bpy.data.objects.get(name)
        me = src.data
        same = ob is not None and ob.type == "MESH" and len(ob.data.vertices) == len(me.vertices) and \
            len(ob.data.polygons) == len(me.polygons) and len(ob.data.edges) == len(me.edges)
        if same:
            return ob
        new = me.copy()                                         # the topology (once per topology)
        new.name = name
        gm = ghost_material()
        for i in range(len(new.materials)):
            new.materials[i] = gm
        if not len(new.materials):
            new.materials.append(gm)
        if ob is None or ob.type != "MESH":
            if ob is not None:
                bpy.data.objects.remove(ob, do_unlink=True)
            ob = bpy.data.objects.new(name, new)
            self._link(ob, coll)
            _copy_modifiers(src, ob)
        else:
            old = ob.data
            ob.data = new
            if old.users == 0:
                bpy.data.meshes.remove(old)
        return ob

    def _eval_ghost(self, name: str, src, coll, depsgraph):
        """A deformed mesh (a skinned character): the evaluated surface, frozen."""
        ev = src.evaluated_get(depsgraph)
        ob = bpy.data.objects.get(name)
        n = len(ev.data.vertices)
        if ob is None or ob.type != "MESH" or len(ob.data.vertices) != n:
            new = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=False, depsgraph=depsgraph)
            new.name = name
            gm = ghost_material()
            new.materials.clear()
            new.materials.append(gm)
            if ob is None or ob.type != "MESH":
                if ob is not None:
                    bpy.data.objects.remove(ob, do_unlink=True)
                ob = bpy.data.objects.new(name, new)
                self._link(ob, coll)
            else:
                old = ob.data
                ob.data = new
                if old.users == 0:
                    bpy.data.meshes.remove(old)
            return ob
        co = np.empty(3 * n, np.float32)
        ev.data.vertices.foreach_get("co", co)
        ob.data.vertices.foreach_set("co", co)
        ob.data.update()
        return ob


def _copy_elements(src, dst) -> None:
    n = len(src.elements)
    els = dst.elements
    if len(els) != n:
        els.clear()
        for _ in range(n):
            e = els.new()
            e.type = "ELLIPSOID"
    if n == 0:
        return
    for attr, width, dt in (("co", 3, np.float32), ("radius", 1, np.float32), ("size_x", 1, np.float32),
                            ("size_y", 1, np.float32), ("size_z", 1, np.float32), ("stiffness", 1, np.float32),
                            ("hide", 1, bool)):
        a = np.empty(n * width, dt)
        src.elements.foreach_get(attr, a)
        els.foreach_set(attr, a)


def _copy_modifiers(src, dst) -> None:
    """The parts' own modifiers (tubes from edges, instanced flakes, panel thickness) - not the deformers:
    the copy is frozen.  Geometry Nodes parts then get the ghost material."""
    nodes = False
    for m in src.modifiers:
        if m.type in DEFORM or not m.show_viewport:
            continue
        try:
            new = dst.modifiers.new(m.name, m.type)
        except (TypeError, RuntimeError):
            continue
        for p in m.bl_rna.properties:
            if p.is_readonly or p.identifier in ("name", "type", "show_expanded", "is_override_data", "rna_type"):
                continue
            try:
                setattr(new, p.identifier, getattr(m, p.identifier))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
        if m.type == "NODES":
            nodes = True
            for k in m.keys():
                try:
                    new[k] = m[k]
                except (TypeError, ValueError, KeyError):
                    pass
    if nodes:
        mod = dst.modifiers.new(SETMAT, "NODES")
        mod.node_group = set_material_group()


def drop_all() -> None:
    """Remove every copy (and their data)."""
    for ob in [o for o in bpy.data.objects if o.name.startswith(PREFIX)]:
        data = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.MetaBall):
                bpy.data.metaballs.remove(data)


__all__ = ["Ghosts", "ghost_material", "sources", "palette", "PALETTES", "PALETTE_ORDER", "drop_all",
           "fx_collection", "COLL"]
