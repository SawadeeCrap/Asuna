"""The Bionic line (v14-v18) in Blender: structure you can see - no metaballs.

Every organism is drawn from what its physics really is (myrmex.creature.bionic.meshes builds the
geometry, pure numpy; here it only becomes meshes and materials):

* Tensor  - machined struts in anisotropic black metal, cables that light up with the tension they
  carry (the force network made visible), chrome joints;
* Fold    - rigid panels with a thickness: carbon-black outside with light in the creases, structural
  colour inside (thin-film interference, like a beetle's shell) - folding shows one side, then the other;
* Arbor   - vessels in a dark translucent skin; heartbeats run through them as light from the root to
  the tips, growth cones glow, dropped branches go dark;
* Ferro   - a black mirror liquid with an oily sheen, needle-sharp spikes, droplets, and iron filings
  lying along the invisible magnet's field lines;
* Truss   - bone-metal struts whose temper colour follows their stress (thin-film oxide: straw, bronze,
  purple, blue), buckled struts bow, ion plumes at the thrusters, blows flash red on the hubs.
"""
from __future__ import annotations

import bpy
import numpy as np

from myrmex.creature.bionic import meshes

from .creature import OBSTACLE, _attr, _math, _principled, _simple_material, _value

KIND_OF = {"tensor": 0, "fold": 1, "arbor": 2, "ferro": 3, "truss": 4}
OBJECTS = {0: {"struts": "BionicStruts", "cables": "BionicCables", "hubs": "BionicHubs"},
           1: {"panels": "BionicPanels"},
           2: {"vessels": "BionicVessels", "nodes": "BionicNodes"},
           3: {"ferro": "BionicFerro", "droplets": "BionicDroplets", "filings": "BionicFilings"},
           4: {"struts": "BionicTruss", "hubs": "BionicTrussHubs", "plumes": "BionicPlumes"}}
ALL_OBJECTS = tuple(n for d in OBJECTS.values() for n in d.values())
SMOOTH = {"BionicPanels": False}
AMBER = (1.0, 0.42, 0.1, 1.0)
DEEP_RED = (0.55, 0.02, 0.008, 1.0)
ION = (0.35, 0.55, 1.0, 1.0)
N_FILINGS = 520


# ---------------------------------------------------------------------- materials
def _mat(name: str):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat, False
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    return mat, True


def _bsdf(mat):
    return next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")


def _link(nt, out, b, name: str) -> None:
    if name in b.inputs:
        nt.links.new(out, b.inputs[name])


def _uv_tangent(nt, b) -> None:
    """Brushed metal runs along the member (UV v)."""
    tg = nt.nodes.new("ShaderNodeTangent")
    tg.direction_type = "UV_MAP"
    tg.uv_map = "UVMap"
    _link(nt, tg.outputs["Tangent"], b, "Tangent")


def _brushed_roughness(nt, b, lo: float, hi: float) -> None:
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (3.0, 90.0, 1.0)
    nt.links.new(uv.outputs["UV"], mp.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 6.0
    nz.inputs["Detail"].default_value = 3.0
    nt.links.new(mp.outputs["Vector"], nz.inputs["Vector"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = lo, hi
    nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
    _link(nt, rr.outputs["Result"], b, "Roughness")


def strut_metal() -> bpy.types.Material:
    """Tensor struts: machined black metal, brushed along the strut, a faint temper sheen."""
    mat, new = _mat("MyrmexBionicStrut")
    if new:
        b = _principled(mat, Base_Color=(0.02, 0.02, 0.022, 1.0), Metallic=1.0, Roughness=0.3, Anisotropic=0.8,
                        Anisotropic_Rotation=0.25, Coat_Weight=0.3, Coat_Roughness=0.08, Thin_Film_Thickness=210.0,
                        Thin_Film_IOR=2.1)
        _uv_tangent(mat.node_tree, b)
        _brushed_roughness(mat.node_tree, b, 0.22, 0.36)
    return mat


def cable_material() -> bpy.types.Material:
    """Tension cables: dark, and they glow warm with the load they carry (slack ones stay dark)."""
    mat, new = _mat("MyrmexTensionCable")
    if new:
        nt = mat.node_tree
        b = _principled(mat, Base_Color=(0.03, 0.03, 0.032, 1.0), Metallic=1.0, Roughness=0.35)
        glow = _value(nt, "MyrmexGlow")
        load = _math(nt, "MAXIMUM", _math(nt, "SUBTRACT", _attr(nt, "stress"), 0.22), 0.0)
        if "Emission Color" in b.inputs:
            b.inputs["Emission Color"].default_value = AMBER
            s = _math(nt, "MULTIPLY", _math(nt, "POWER", load, 1.4), 7.0)
            nt.links.new(_math(nt, "MULTIPLY_ADD", glow.outputs[0], 1.5, s), b.inputs["Emission Strength"])
    return mat


def chrome_material(name: str = "MyrmexBionicJoint") -> bpy.types.Material:
    mat, new = _mat(name)
    if new:
        _principled(mat, Base_Color=(0.03, 0.03, 0.033, 1.0), Metallic=1.0, Roughness=0.1, Coat_Weight=0.6,
                    Coat_Roughness=0.03, Thin_Film_Thickness=170.0, Thin_Film_IOR=1.8)
    return mat


def fold_materials() -> list:
    """Outside: carbon black, light in the creases.  Inside: structural colour.  Edges: a thin light."""
    out, new = _mat("MyrmexFoldOuter")
    if new:
        nt = out.node_tree
        b = _principled(out, Base_Color=(0.01, 0.01, 0.011, 1.0), Metallic=0.25, Roughness=0.38, Coat_Weight=0.7,
                        Coat_Roughness=0.12)
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = "UVMap"
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(uv.outputs["UV"], sep.inputs[0])
        u, v = sep.outputs["X"], sep.outputs["Y"]
        d = _math(nt, "MINIMUM", _math(nt, "MINIMUM", u, _math(nt, "SUBTRACT", 1.0, u)),
                  _math(nt, "MINIMUM", v, _math(nt, "SUBTRACT", 1.0, v)))
        seam = nt.nodes.new("ShaderNodeMapRange")
        seam.inputs["From Min"].default_value, seam.inputs["From Max"].default_value = 0.0, 0.045
        seam.inputs["To Min"].default_value, seam.inputs["To Max"].default_value = 1.0, 0.0
        nt.links.new(d, seam.inputs["Value"])
        glow = _value(nt, "MyrmexGlow")
        k = _math(nt, "MULTIPLY_ADD", _attr(nt, "clap"), 3.0, _math(nt, "MULTIPLY_ADD", glow.outputs[0], 1.5, 0.5))
        if "Emission Color" in b.inputs:
            b.inputs["Emission Color"].default_value = (0.78, 0.9, 1.0, 1.0)
            nt.links.new(_math(nt, "MULTIPLY", _math(nt, "POWER", seam.outputs["Result"], 2.0), k),
                         b.inputs["Emission Strength"])
    inner, new = _mat("MyrmexFoldInner")
    if new:
        nt = inner.node_tree
        b = _principled(inner, Base_Color=(0.22, 0.22, 0.25, 1.0), Metallic=1.0, Roughness=0.14, Coat_Weight=0.8,
                        Coat_Roughness=0.04, Thin_Film_IOR=1.5)
        tc = nt.nodes.new("ShaderNodeTexCoord")
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 2.5
        nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
        th = _math(nt, "MULTIPLY_ADD", _math(nt, "ADD", nz.outputs["Fac"], _math(nt, "MULTIPLY", _attr(nt, "row"), 0.6)),
                   300.0, 330.0)
        _link(nt, th, b, "Thin Film Thickness")
    rim, new = _mat("MyrmexFoldRim")
    if new:
        b = _principled(rim, Base_Color=(0.05, 0.05, 0.06, 1.0), Metallic=0.5, Roughness=0.3)
        if "Emission Color" in b.inputs:
            b.inputs["Emission Color"].default_value = (0.85, 0.95, 1.0, 1.0)
            b.inputs["Emission Strength"].default_value = 1.6
    return [out, inner, rim]


def vessel_material() -> bpy.types.Material:
    """Arbor: a dark translucent skin, light running inside it (pulses), glowing growth cones."""
    mat, new = _mat("MyrmexVessel")
    if new:
        nt = mat.node_tree
        b = _principled(mat, Base_Color=(0.02, 0.012, 0.012, 1.0), Metallic=0.0, Roughness=0.3,
                        Subsurface_Weight=0.35, Subsurface_Radius=(1.0, 0.3, 0.15), Subsurface_Scale=0.02,
                        Coat_Weight=0.9, Coat_Roughness=0.05, Specular_IOR_Level=0.6, Thin_Film_IOR=1.45)
        glow, dist, shed = _attr(nt, "glow"), _attr(nt, "dist"), _attr(nt, "shed")
        lw = nt.nodes.new("ShaderNodeLayerWeight")
        lw.inputs["Blend"].default_value = 0.35
        inside = _math(nt, "POWER", _math(nt, "SUBTRACT", 1.0, lw.outputs["Facing"]), 2.0)
        col = nt.nodes.new("ShaderNodeMix")
        col.data_type = "RGBA"
        col.inputs[6].default_value = (0.6, 0.05, 0.012, 1.0)
        col.inputs[7].default_value = AMBER
        nt.links.new(_math(nt, "MINIMUM", glow, 1.0), col.inputs[0])
        alive = _math(nt, "SUBTRACT", 1.0, shed)
        heart = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, dist), 0.25)
        s = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY_ADD", _math(nt, "MULTIPLY", glow, 9.0), inside, heart), alive)
        if "Emission Color" in b.inputs:
            nt.links.new(col.outputs[2], b.inputs["Emission Color"])
            nt.links.new(s, b.inputs["Emission Strength"])
        _link(nt, _math(nt, "MULTIPLY", dist, 320.0), b, "Thin Film Thickness")     # iridescent tips
    return mat


def ferro_material() -> bpy.types.Material:
    """Ferrofluid: a black mirror with an oily sheen; the spike tips flush dark red on hits."""
    mat, new = _mat("MyrmexFerrofluid")
    if new:
        nt = mat.node_tree
        b = _principled(mat, Base_Color=(0.004, 0.004, 0.005, 1.0), Metallic=1.0, Roughness=0.045, Coat_Weight=1.0,
                        Coat_Roughness=0.015, Thin_Film_Thickness=150.0, Thin_Film_IOR=1.4)
        kick, glow = _value(nt, "MyrmexKick"), _value(nt, "MyrmexGlow")
        tip = _math(nt, "POWER", _attr(nt, "spike"), 6.0)
        if "Emission Color" in b.inputs:
            b.inputs["Emission Color"].default_value = DEEP_RED
            k = _math(nt, "MULTIPLY_ADD", kick.outputs[0], 3.0, _math(nt, "MULTIPLY", glow.outputs[0], 1.5))
            nt.links.new(_math(nt, "MULTIPLY", tip, k), b.inputs["Emission Strength"])
    return mat


def filings_material() -> bpy.types.Material:
    return _simple_material("MyrmexIronFilings", (0.06, 0.06, 0.065), 1.0, 0.5)


def truss_material() -> bpy.types.Material:
    """Bone metal: temper colours follow the stress (thin-film oxide); overloaded struts glow."""
    mat, new = _mat("MyrmexBoneMetal")
    if new:
        nt = mat.node_tree
        b = _principled(mat, Base_Color=(0.025, 0.024, 0.024, 1.0), Metallic=1.0, Roughness=0.28, Anisotropic=0.7,
                        Anisotropic_Rotation=0.25, Coat_Weight=0.2, Thin_Film_IOR=2.3)
        _uv_tangent(nt, b)
        _brushed_roughness(nt, b, 0.2, 0.34)
        st = _attr(nt, "stress")
        mag = _math(nt, "ABSOLUTE", st)
        _link(nt, _math(nt, "MULTIPLY_ADD", _math(nt, "MINIMUM", mag, 1.5), 230.0, 150.0), b, "Thin Film Thickness")
        over = nt.nodes.new("ShaderNodeMapRange")
        over.inputs["From Min"].default_value, over.inputs["From Max"].default_value = 0.8, 1.5
        nt.links.new(mag, over.inputs["Value"])
        col = nt.nodes.new("ShaderNodeMix")                        # tension amber, compression deep red
        col.data_type = "RGBA"
        col.inputs[6].default_value = DEEP_RED
        col.inputs[7].default_value = AMBER
        nt.links.new(_math(nt, "GREATER_THAN", st, 0.0), col.inputs[0])
        alive = _math(nt, "SUBTRACT", 1.0, _attr(nt, "dying"))
        glow = _value(nt, "MyrmexGlow")
        if "Emission Color" in b.inputs:
            nt.links.new(col.outputs[2], b.inputs["Emission Color"])
            nt.links.new(_math(nt, "MULTIPLY", _math(nt, "MULTIPLY_ADD", over.outputs["Result"], 2.5,
                                                     _math(nt, "MULTIPLY", glow.outputs[0], 0.5)), alive),
                         b.inputs["Emission Strength"])
    return mat


def truss_hub_material() -> bpy.types.Material:
    mat, new = _mat("MyrmexTrussHub")
    if new:
        nt = mat.node_tree
        b = _principled(mat, Base_Color=(0.03, 0.03, 0.033, 1.0), Metallic=1.0, Roughness=0.12, Coat_Weight=0.5)
        thrust, hit = _attr(nt, "thrust"), _attr(nt, "hit")
        col = nt.nodes.new("ShaderNodeMix")
        col.data_type = "RGBA"
        col.inputs[6].default_value = ION
        col.inputs[7].default_value = DEEP_RED
        nt.links.new(_math(nt, "MINIMUM", _math(nt, "MULTIPLY", hit, 3.0), 1.0), col.inputs[0])
        if "Emission Color" in b.inputs:
            nt.links.new(col.outputs[2], b.inputs["Emission Color"])
            nt.links.new(_math(nt, "ADD", _math(nt, "MULTIPLY", thrust, 1.2), _math(nt, "MULTIPLY", hit, 8.0)),
                         b.inputs["Emission Strength"])
    return mat


def plume_material() -> bpy.types.Material:
    """Ion plumes at the thrusters: blue light fading along the cone."""
    mat, new = _mat("MyrmexIonPlume")
    if new:
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.type == "BSDF_PRINCIPLED":
                nt.nodes.remove(n)
        outn = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = ION
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        uv = nt.nodes.new("ShaderNodeUVMap")
        uv.uv_map = "UVMap"
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(uv.outputs["UV"], sep.inputs[0])
        fade = _math(nt, "POWER", _math(nt, "SUBTRACT", 1.0, sep.outputs["Y"]), 1.6)
        thrust = _attr(nt, "thrust")
        nt.links.new(_math(nt, "MULTIPLY", _math(nt, "MULTIPLY", fade, thrust), 14.0), em.inputs["Strength"])
        nt.links.new(_math(nt, "MINIMUM", _math(nt, "MULTIPLY", fade, 1.3), 1.0), mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], outn.inputs["Surface"])
        for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
            if hasattr(mat, attr):
                try:
                    setattr(mat, attr, val)
                except Exception:
                    pass
    return mat


MATERIALS = {"BionicStruts": strut_metal, "BionicCables": cable_material, "BionicHubs": chrome_material,
             "BionicPanels": fold_materials, "BionicVessels": vessel_material, "BionicNodes": vessel_material,
             "BionicFerro": ferro_material, "BionicDroplets": ferro_material, "BionicFilings": filings_material,
             "BionicTruss": truss_material, "BionicTrussHubs": truss_hub_material, "BionicPlumes": plume_material}


# ---------------------------------------------------------------------- meshes
def _build(name: str, part: meshes.Part, coll) -> bpy.types.Object:
    """A mesh with the part's topology (made once per topology; frames only move the vertices)."""
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(part.verts, float).tolist(), [], np.asarray(part.faces).tolist())
    me.polygons.foreach_set("use_smooth", np.full(len(me.polygons), SMOOTH.get(name, True)))
    if part.uv is not None:
        uvl = me.uv_layers.new(name="UVMap")
        vi = np.zeros(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", vi)
        uvl.data.foreach_set("uv", np.asarray(part.uv, np.float32)[vi].ravel())
    for a in part.attrs:
        me.attributes.new(a, "FLOAT", "POINT")
    mats = MATERIALS[name]()
    for m in (mats if isinstance(mats, list) else [mats]):
        me.materials.append(m)
    ob = bpy.data.objects.get(name)
    if ob is None:
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
    else:
        old = ob.data
        keep = [m for m in old.materials]                         # a look keeps the materials you chose
        ob.data = me
        if keep and len(keep) == len(me.materials) and all(m is not None for m in keep):
            for k, m in enumerate(keep):
                me.materials[k] = m
    if name == "BionicPanels" and "Thickness" not in ob.modifiers:
        mod = ob.modifiers.new("Thickness", "SOLIDIFY")
        mod.thickness, mod.offset = 0.012, 0.0
        mod.use_rim = True
        mod.material_offset, mod.material_offset_rim = 1, 2
    ob["myrmex_faces"] = len(part.faces)
    return ob


def _set(ob: bpy.types.Object, part: meshes.Part) -> None:
    me = ob.data
    me.vertices.foreach_set("co", np.ascontiguousarray(part.verts, np.float32).ravel())
    for a, vals in part.attrs.items():
        at = me.attributes.get(a)
        if at is not None:
            at.data.foreach_set("value", np.ascontiguousarray(vals, np.float32))
    me.update()


class BionicView:
    def __init__(self, coll):
        self.coll = coll
        self.kind = -1
        self.objects: dict = {}
        self.seeds = _filing_seeds(N_FILINGS)
        self.obstacles: list = []

    def make(self, kind: int) -> None:
        for k, names in OBJECTS.items():                           # another Bionic organism's parts go
            if k != kind:
                for n in names.values():
                    ob = bpy.data.objects.get(n)
                    if ob is not None:
                        bpy.data.objects.remove(ob, do_unlink=True)
        self.kind = kind
        self.objects = {}
        self._make_obstacles(4)

    def _make_obstacles(self, n: int) -> None:
        self.obstacles = []
        for k in range(n):
            name = f"{OBSTACLE}_{k}"
            o = bpy.data.objects.get(name)
            if o is None:
                import bmesh
                me = bpy.data.meshes.new(name)
                bm = bmesh.new()
                bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
                bm.to_mesh(me)
                bm.free()
                me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
                me.materials.append(_simple_material("MyrmexObstacle", (0.03, 0.03, 0.032), 0.0, 0.08, 1.0))
                o = bpy.data.objects.new(name, me)
                self.coll.objects.link(o)
            o.scale = (0.0, 0.0, 0.0)
            self.obstacles.append(o)

    def apply_arrays(self, kind: int, pos, radius, members, extra, obstacles, t: float, glow: float) -> None:
        if kind != self.kind:
            self.make(kind)
        kw = {"seeds": self.seeds, "t": float(t)} if kind == 3 else {}
        P = meshes.parts(kind, pos, radius, members, extra, **kw)
        for part_name, part in P.items():
            name = OBJECTS[kind][part_name]
            ob = self.objects.get(name) or bpy.data.objects.get(name)
            if ob is None or len(ob.data.vertices) != len(part.verts) or ob.get("myrmex_faces") != len(part.faces):
                ob = _build(name, part, self.coll)
            self.objects[name] = ob
            _set(ob, part)
        ob_arr = np.asarray(obstacles, float) if obstacles is not None else np.zeros((0, 4))
        for k, o in enumerate(self.obstacles):
            if k < len(ob_arr) and ob_arr[k, 3] > 0:
                o.location = ob_arr[k, :3]
                o.scale = (float(ob_arr[k, 3]),) * 3
            else:
                o.scale = (0.0, 0.0, 0.0)
        kick = float(extra[13]) if kind == 3 and extra is not None and len(extra) > 13 else 0.0
        for mname in ("MyrmexTensionCable", "MyrmexFoldOuter", "MyrmexFerrofluid", "MyrmexBoneMetal"):
            m = bpy.data.materials.get(mname)
            if m is not None and m.node_tree is not None:
                nd = m.node_tree.nodes
                if "MyrmexGlow" in nd:
                    nd["MyrmexGlow"].outputs[0].default_value = float(glow)
                if "MyrmexKick" in nd:
                    nd["MyrmexKick"].outputs[0].default_value = kick

    def apply(self, fr) -> None:
        self.apply_arrays(int(fr.bkind), fr.pos, fr.radius, fr.members, fr.extra, fr.obstacles, float(fr.t),
                          float(fr.glow))


def _filing_seeds(n: int) -> np.ndarray:
    """Fixed places for the iron filings round the body (in body radii): a shell, denser near the poles."""
    rng = np.random.default_rng(1729)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] *= 1.25
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d * rng.uniform(1.25, 3.2, (n, 1))


def drop_all() -> None:
    for n in ALL_OBJECTS:
        ob = bpy.data.objects.get(n)
        if ob is not None:
            bpy.data.objects.remove(ob, do_unlink=True)


__all__ = ["BionicView", "KIND_OF", "OBJECTS", "ALL_OBJECTS", "drop_all"]
