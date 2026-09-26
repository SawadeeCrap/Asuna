"""Afterimages: copies of the organism left behind as it moves - made of what the organism is made of.

A ring of ghost slots.  A slot holds one copy of every visible part of the organism: the metaball body as a
metaball of its own (Blender polygonises it once), meshes as meshes (only their vertices and attributes
are copied while the topology stays), Geometry Nodes parts with their node trees.  Every copy wears a
*ghost version of the organism's own material*: the same shader - its metal, its liquid, its light lines,
its reflections - fading out and dissolving (object-space noise) with age.  No colours of their own: the
copies are in the organism's tones, on the black stage.

Copies are real geometry, so they are in the viewport, in Syphon and in renders.  Per frame only the
ghost materials' clock moves; a copy's geometry is written once, when it is left behind.  A copy is left
when the organism has moved far enough since the last one (evenly spaced along its path, none while it
stands still), and on impacts and shape changes; dense settings make a continuous echo of the motion.
"""
from __future__ import annotations

import bpy
import numpy as np

from . import compat

COLL = "MyrmexFX"
PREFIX = "MyrmexGhost"
MAT_PREFIX = "MyrmexGhost·"
MAX_SLOTS = 16
SKIP = {"CreatureDebug", "ColonyPrey", "CrawlerTerrain", "PolyMicro", "BionicPlumes", "MyrmexFloor"}
DEFORM = {"ARMATURE", "CORRECTIVE_SMOOTH", "LATTICE", "CURVE", "HOOK", "SHRINKWRAP", "SIMPLE_DEFORM", "CAST",
          "SMOOTH", "LAPLACIANSMOOTH", "SURFACE_DEFORM", "MESH_DEFORM", "WARP", "WAVE", "DISPLACE",
          "LAPLACIANDEFORM", "MESH_CACHE", "MESH_SEQUENCE_CACHE"}
_MATS: set = set()                             # the ghost materials whose clock moves every frame


def fx_collection(scene: bpy.types.Scene | None = None) -> bpy.types.Collection:
    scene = scene or bpy.context.scene
    coll = bpy.data.collections.get(COLL)
    if coll is None:
        coll = bpy.data.collections.new(COLL)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


# ---------------------------------------------------------------------- node helpers
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


def _transparent(mat: bpy.types.Material) -> None:
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND"), ("use_backface_culling", True),
                      ("use_transparency_overlap", False), ("use_transparent_shadow", True),
                      ("shadow_method", "NONE")):
        if hasattr(mat, attr):
            try:
                setattr(mat, attr, val)
            except (TypeError, ValueError, AttributeError):
                pass


def _fade_alpha(nt, x0: float = -1600.0, y0: float = -400.0):
    """alpha = gain x fade(age / life) x dissolve - the clock and the life are value nodes, the birth is the
    copy's own custom property (myrmex_birth)."""
    now = _value(nt, "MyrmexNow", 0.0, x0, y0 + 300)
    life = _value(nt, "MyrmexLife", 0.8, x0, y0 + 200)
    gain = _value(nt, "MyrmexGain", 0.5, x0, y0 + 100)
    birth = _n(nt, "ShaderNodeAttribute", x0, y0 + 450)
    birth.attribute_type = "OBJECT"
    birth.attribute_name = "myrmex_birth"
    age = _m(nt, "SUBTRACT", now, birth.outputs["Fac"], x=x0 + 200, y=y0 + 400)
    x = _m(nt, "DIVIDE", age, life, clamp=True, x=x0 + 400, y=y0 + 400)
    alive = _m(nt, "GREATER_THAN", age, -0.001, x=x0 + 400, y=y0 + 250)
    fade = _m(nt, "MULTIPLY", _m(nt, "POWER", _m(nt, "SUBTRACT", 1.0, x, x=x0 + 550, y=y0 + 400), 1.4,
                                 x=x0 + 700, y=y0 + 400), alive, x=x0 + 850, y=y0 + 400)
    # dissolve: noise on the copy's own surface, the threshold rises as it fades
    info = _n(nt, "ShaderNodeObjectInfo", x0, y0 - 100)
    tc = _n(nt, "ShaderNodeTexCoord", x0, y0 - 300)
    seed = _m(nt, "MULTIPLY", info.outputs["Random"], 37.0, x=x0 + 200, y=y0 - 100)
    off = _n(nt, "ShaderNodeCombineXYZ", x0 + 350, y0 - 100)
    nt.links.new(seed, off.inputs[0])
    nt.links.new(seed, off.inputs[1])
    vec = _n(nt, "ShaderNodeVectorMath", x0 + 500, y0 - 250)
    vec.operation = "ADD"
    nt.links.new(tc.outputs["Object"], vec.inputs[0])
    nt.links.new(off.outputs[0], vec.inputs[1])
    noise = _n(nt, "ShaderNodeTexNoise", x0 + 650, y0 - 250)
    noise.inputs["Scale"].default_value = 6.0
    noise.inputs["Detail"].default_value = 3.0
    nt.links.new(vec.outputs[0], noise.inputs["Vector"])
    thr = _m(nt, "SUBTRACT", _m(nt, "MULTIPLY", _m(nt, "SUBTRACT", 1.0, fade, x=x0 + 650, y=y0),
                                1.15, x=x0 + 800, y=y0), 0.1, x=x0 + 950, y=y0)
    cut = _n(nt, "ShaderNodeMapRange", x0 + 1100, y0 - 150)
    cut.clamp = True
    nt.links.new(noise.outputs["Fac"], cut.inputs["Value"])
    nt.links.new(_m(nt, "SUBTRACT", thr, 0.05, x=x0 + 950, y=y0 - 150), cut.inputs["From Min"])
    nt.links.new(_m(nt, "ADD", thr, 0.05, x=x0 + 950, y=y0 - 250), cut.inputs["From Max"])
    return _m(nt, "MULTIPLY", _m(nt, "MULTIPLY", fade, cut.outputs["Result"], x=x0 + 1250, y=y0 + 200), gain,
              clamp=True, x=x0 + 1400, y=y0 + 200)


def _fallback_material() -> bpy.types.Material:
    """(A part without a material: dark glossy metal, like the rest of the stage.)"""
    mat = bpy.data.materials.new("MyrmexGhostBase")
    nt = compat.material_node_tree(mat)
    b = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if b is not None:
        b.inputs["Base Color"].default_value = (0.02, 0.02, 0.022, 1.0)
        b.inputs["Metallic"].default_value = 0.9
        b.inputs["Roughness"].default_value = 0.2
    return mat


def ghost_variant(src: bpy.types.Material | None) -> bpy.types.Material:
    """The ghost version of one of the organism's materials: its own shader, fading and dissolving."""
    name = MAT_PREFIX + (src.name if src is not None else "")
    mat = bpy.data.materials.get(name)
    if mat is not None and mat.node_tree is not None and "MyrmexNow" in mat.node_tree.nodes:
        _MATS.add(mat.name)
        return mat
    if mat is not None:
        bpy.data.materials.remove(mat)
    base = src if src is not None else _fallback_material()
    mat = base.copy()
    if base is not src:
        bpy.data.materials.remove(base)
    mat.name = name
    nt = compat.material_node_tree(mat)
    for idb in (mat, nt):                                   # a copy is frozen: no shader animation
        if idb.animation_data is not None:
            idb.animation_data_clear()
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) or \
        next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or nt.nodes.new("ShaderNodeOutputMaterial")
    for k in ("Volume",):                                   # (a glass of haze would fill the copy)
        if k in out.inputs:
            for lk in list(out.inputs[k].links):
                nt.links.remove(lk)
    surf = out.inputs["Surface"]
    shader = surf.links[0].from_socket if surf.is_linked else None
    if shader is None:
        b = _n(nt, "ShaderNodeBsdfPrincipled", out.location.x - 400, out.location.y)
        b.inputs["Base Color"].default_value = (0.02, 0.02, 0.022, 1.0)
        shader = b.outputs[0]
    alpha = _fade_alpha(nt, out.location.x - 2200, out.location.y - 600)
    tr = _n(nt, "ShaderNodeBsdfTransparent", out.location.x - 400, out.location.y - 250)
    mix = _n(nt, "ShaderNodeMixShader", out.location.x - 200, out.location.y - 100)
    mix.name = "MyrmexGhostMix"
    nt.links.new(alpha, mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(shader, mix.inputs[2])
    nt.links.new(mix.outputs[0], surf)
    _transparent(mat)
    _MATS.add(mat.name)
    return mat


def set_clock(now: float, life: float, gain: float) -> None:
    """Every ghost material's clock (the only per-frame change of the afterimages)."""
    for name in list(_MATS):
        mat = bpy.data.materials.get(name)
        if mat is None or mat.node_tree is None:
            _MATS.discard(name)
            continue
        nd = mat.node_tree.nodes
        if "MyrmexNow" in nd:
            nd["MyrmexNow"].outputs[0].default_value = float(now)
            nd["MyrmexLife"].outputs[0].default_value = float(life)
            nd["MyrmexGain"].outputs[0].default_value = float(gain)


def _group_materials(ng, seen=None) -> list:
    """The materials a Geometry Nodes tree sets (nested groups too)."""
    seen = seen if seen is not None else set()
    out = []
    if ng is None or ng.name in seen:
        return out
    seen.add(ng.name)
    for nd in ng.nodes:
        if nd.bl_idname == "GeometryNodeSetMaterial":
            m = nd.inputs["Material"].default_value
            if m is not None and m not in out:
                out.append(m)
        elif nd.bl_idname == "GeometryNodeGroup":
            out += [m for m in _group_materials(nd.node_tree, seen) if m not in out]
    return out


def gn_ghost_group(src_ob) -> bpy.types.NodeTree:
    """Trailing modifier of a Geometry Nodes copy: realise, then every material -> its ghost version."""
    mats = []
    for m in src_ob.modifiers:
        if m.type == "NODES":
            mats += [x for x in _group_materials(m.node_group) if x not in mats]
    if not mats and src_ob.material_slots:
        mats = [s.material for s in src_ob.material_slots if s.material is not None]
    name = "MyrmexGhostMat·" + src_ob.name
    ng = bpy.data.node_groups.get(name)
    if ng is not None:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi = ng.nodes.new("NodeGroupInput")
    go = ng.nodes.new("NodeGroupOutput")
    last = ng.nodes.new("GeometryNodeRealizeInstances")
    ng.links.new(gi.outputs[0], last.inputs[0])
    if mats:
        for m in mats:
            rp = ng.nodes.new("GeometryNodeReplaceMaterial")
            rp.inputs["Old"].default_value = m
            rp.inputs["New"].default_value = ghost_variant(m)
            ng.links.new(last.outputs[0], rp.inputs["Geometry"])
            last = rp
    else:
        sm = ng.nodes.new("GeometryNodeSetMaterial")
        sm.inputs["Material"].default_value = ghost_variant(None)
        ng.links.new(last.outputs[0], sm.inputs["Geometry"])
        last = sm
    ng.links.new(last.outputs[0], go.inputs[0])
    return ng


def ghost_materials_for(sources_list) -> list:
    """Every ghost material the organism's parts need (the warm-up compiles them before the first copy)."""
    out = []
    for ob in sources_list:
        mats = list(ob.data.materials) if ob.type == "META" else [s.material for s in ob.material_slots]
        for m in mats:
            g = ghost_variant(m)
            if g not in out:
                out.append(g)
        if any(md.type == "NODES" for md in ob.modifiers):
            for md in ob.modifiers:
                if md.type == "NODES":
                    for m in _group_materials(md.node_group):
                        g = ghost_variant(m)
                        if g not in out:
                            out.append(g)
    return out


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


def accent(sources_list) -> tuple:
    """The organism's own accent colour (for the light traces): the strongest coloured light its materials
    give off (light lines, cables, glints), else a cool silver for dark metal / liquid skins."""
    best, score = None, 0.0
    for ob in sources_list:
        mats = list(ob.data.materials) if ob.type == "META" else [s.material for s in ob.material_slots]
        for md in ob.modifiers:
            if md.type == "NODES":
                mats += _group_materials(md.node_group)
        for mat in mats:
            nt = mat.node_tree if mat is not None else None
            if nt is None:
                continue
            for nd in nt.nodes:
                c = None
                if nd.type == "BSDF_PRINCIPLED" and "Emission Color" in nd.inputs:
                    st = nd.inputs.get("Emission Strength")
                    if st is not None and (st.is_linked or st.default_value > 0.0):
                        c = nd.inputs["Emission Color"].default_value
                elif nd.type == "EMISSION":
                    c = nd.inputs["Color"].default_value
                if c is None:
                    continue
                r, g, b = float(c[0]), float(c[1]), float(c[2])
                mx, mn = max(r, g, b), min(r, g, b)
                sc = mx * (0.35 + (mx - mn) / max(mx, 1e-6))          # bright and coloured wins
                if mx > 1e-4 and sc > score:
                    best, score = (r / mx, g / mx, b / mx), sc
    return best or (0.78, 0.84, 0.95)


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
            self.spawn(scene, t, n)
            self.last_spawn_t = t
            self.last_com = com.copy()
        # a copy starts as a translucent double of the organism and fades / dissolves away
        set_clock(t, life, 0.18 + 0.62 * amt)
        for s in range(MAX_SLOTS):                              # expired copies: no geometry, no cost
            if self.birth[s] > -1e8 and (t - self.birth[s] > life or s >= n):
                self._clear_slot(s)
        self.count = n

    def spawn(self, scene, t: float, n: int) -> None:
        srcs = sources(scene)
        sig = tuple(sorted((o.name, o.type) for o in srcs))
        if sig != self.signature:                             # another organism: other parts
            self.clear(remove=True)
            self.signature = sig
        s = self.next % max(1, n)
        self.next = s + 1
        coll = fx_collection(scene)
        written = set()
        depsgraph = None
        for src in srcs:
            name = f"{PREFIX}{s:02d}_{src.name}"
            try:
                if src.type == "META":
                    ob = self._meta_ghost(name, src, coll)
                    _copy_elements(src.data, ob.data)
                elif _deformed(src):
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
                        _copy_attributes(src.data, ob.data)
                        ob.data.update()
                ob.matrix_world = src.matrix_world
            except (ReferenceError, RuntimeError, ValueError) as e:
                print("Myrmex FX ghost:", src.name, e)
                continue
            ob["myrmex_birth"] = float(t)
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
        mb.resolution = max(0.02, sm.resolution * 1.6)              # a copy may be a little coarser
        mb.render_resolution = max(0.012, sm.render_resolution * 1.5)
        mb.threshold = sm.threshold
        want = [ghost_variant(m) for m in sm.materials] or [ghost_variant(None)]
        if list(mb.materials) != want:
            mb.materials.clear()
            for m in want:
                mb.materials.append(m)
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
        _ghost_slots(new, [s.material for s in src.material_slots])
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
            _ghost_slots(new, [s.material for s in src.material_slots])
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


def _ghost_slots(me, mats) -> None:
    """Every material slot of a copy -> the ghost version of the organism's material in that slot."""
    mats = list(mats) or [None]
    while len(me.materials) < len(mats):
        me.materials.append(None)
    for i in range(len(me.materials)):
        me.materials[i] = ghost_variant(mats[i] if i < len(mats) else mats[-1])


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


def _copy_attributes(src, dst) -> None:
    """The per-vertex values the organism's shaders read (light lines, stress, glints) at this moment."""
    for at in src.attributes:
        if at.domain != "POINT" or at.data_type != "FLOAT" or at.name.startswith(".") or at.name == "position":
            continue
        d = dst.attributes.get(at.name)
        if d is None or len(d.data) != len(at.data):
            continue
        a = np.empty(len(at.data), np.float32)
        at.data.foreach_get("value", a)
        d.data.foreach_set("value", a)


def _copy_modifiers(src, dst) -> None:
    """The parts' own modifiers (tubes from edges, instanced flakes, panel thickness) - not the deformers:
    the copy is frozen.  Geometry Nodes parts then get the ghost versions of their materials."""
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
        mod = dst.modifiers.new("MyrmexGhostMaterial", "NODES")
        mod.node_group = gn_ghost_group(src)


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


__all__ = ["Ghosts", "ghost_variant", "ghost_materials_for", "set_clock", "sources", "accent", "drop_all",
           "fx_collection", "COLL"]
