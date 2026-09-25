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
POLY_MAT = "MyrmexPolyalloy"
PLATES = "ColonyPlates"          # (old armour plates - removed when found)
SCUTES = "ColonyScutes"
BONES = "PolyBones"
SWARM = "HiveSwarm"
LURE = "ColonyPrey"
LATTICE = "PolyLattice"          # (old strut tubes - removed when found)
MICRO = "PolyMicro"
OBSTACLE = "PolyObstacle"
RAILS = "CyberRails"            # Cyber Hive (v8): hexagonal rails with light lines
PANELS = "CyberPanels"          # Cyber Hive (v8): hex panels with a light ring
CYBER_MAT = "MyrmexCyberWhite"
CYBER_GREEN = (0.55, 1.0, 0.3, 1.0)   # light acid green (linear), soft: low strength, never glaring
TENDONS = "MimeticTendons"      # Mimetic line (v9-v13): glossy tendons along the skeleton
FINS = "MimeticFins"            # ... filaments / shards / flakes / blades / claws that follow the motion
TERRAIN = "CrawlerTerrain"      # the Crawler's rough ground
LIQUID_MAT = "MyrmexLiquidBlack"
DEEP_RED = (0.5, 0.012, 0.006, 1.0)     # very dark red internal glints, never dominant
MIMETIC = {"swarm": (3, "hive"), "spear": (4, "colony"), "cloud": (5, "hive"), "blade": (6, "colony"),
           "crawler": (7, "colony")}


def variant_style(variant: str) -> tuple[int, str]:
    """-> (look: 0 classic · 1 osseous · 2 cyber · 3-7 mimetic, base organism: nanomaterial / polyalloy / colony /
    hive)."""
    if variant in MIMETIC:
        return MIMETIC[variant]
    if variant.startswith("cyber"):
        return 2, variant.replace("cyber_", "") or "hive"
    if variant.startswith("osseous"):
        return 1, variant.replace("osseous_", "").replace("osseous", "polyalloy")
    return 0, variant


def nanomaterial(name: str = MAT, poly: bool = False) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
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
    if poly:                                   # microscopic segmentation: cell borders become fine grooves
        vor.feature = "DISTANCE_TO_EDGE"
        vor.inputs["Scale"].default_value = 160.0
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
    if poly:
        rr.inputs["To Min"].default_value = 0.16
        rr.inputs["To Max"].default_value = 0.42
        bump.inputs["Strength"].default_value = 0.5
    look = ((("Base Color", (0.004, 0.004, 0.0045, 1.0)), ("Metallic", 0.7), ("Coat Weight", 0.25),
             ("Coat Roughness", 0.1), ("Anisotropic", 0.5), ("Specular IOR Level", 0.55)) if poly else
            (("Base Color", (0.006, 0.006, 0.007, 1.0)), ("Metallic", 0.85), ("Coat Weight", 0.45),
             ("Coat Roughness", 0.06), ("Anisotropic", 0.35), ("Specular IOR Level", 0.6)))
    for k, v in look:
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


def _simple_material(name: str, color, metallic: float, rough: float, coat: float = 0.0) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    b = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    for k, v in (("Base Color", (*color, 1.0)), ("Metallic", metallic), ("Roughness", rough), ("Coat Weight", coat)):
        if k in b.inputs:
            b.inputs[k].default_value = v
    return mat


def _node(ng, kind, loc=(0, 0)):
    n = ng.nodes.new(kind)
    n.location = loc
    return n


def tube_nodes(radius: float = 0.016) -> bpy.types.NodeTree:
    """Edges -> dark mechanical struts (zero-length edges = retracted links are dropped)."""
    ng = bpy.data.node_groups.get("MyrmexStruts")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new("MyrmexStruts", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = _node(ng, "NodeGroupInput", (-800, 0)), _node(ng, "NodeGroupOutput", (600, 0))
    ev = _node(ng, "GeometryNodeInputMeshEdgeVertices", (-800, -200))
    dist = _node(ng, "ShaderNodeVectorMath", (-600, -200))
    dist.operation = "DISTANCE"
    ng.links.new(ev.outputs["Position 1"], dist.inputs[0])
    ng.links.new(ev.outputs["Position 2"], dist.inputs[1])
    cmp = _node(ng, "FunctionNodeCompare", (-400, -200))
    cmp.data_type, cmp.operation = "FLOAT", "LESS_THAN"
    cmp.inputs[1].default_value = 1e-3
    ng.links.new(dist.outputs["Value"], cmp.inputs[0])
    dele = _node(ng, "GeometryNodeDeleteGeometry", (-400, 0))
    dele.domain = "EDGE"
    ng.links.new(gi.outputs[0], dele.inputs["Geometry"])
    ng.links.new(cmp.outputs["Result"], dele.inputs["Selection"])
    m2c = _node(ng, "GeometryNodeMeshToCurve", (-200, 0))
    ng.links.new(dele.outputs["Geometry"], m2c.inputs["Mesh"])
    circ = _node(ng, "GeometryNodeCurvePrimitiveCircle", (-200, -200))
    circ.inputs["Resolution"].default_value = 6
    circ.inputs["Radius"].default_value = radius
    c2m = _node(ng, "GeometryNodeCurveToMesh", (0, 0))
    ng.links.new(m2c.outputs["Curve"], c2m.inputs["Curve"])
    ng.links.new(circ.outputs["Curve"], c2m.inputs["Profile Curve"])
    sm = _node(ng, "GeometryNodeSetMaterial", (300, 0))
    sm.inputs["Material"].default_value = _simple_material("MyrmexPolyStrut", (0.012, 0.012, 0.013), 0.9, 0.3, 0.3)
    ng.links.new(c2m.outputs["Mesh"], sm.inputs["Geometry"])
    ng.links.new(sm.outputs["Geometry"], go.inputs[0])
    return ng


def micro_nodes(body: bpy.types.Object, density: float = 2500.0) -> bpy.types.NodeTree:
    """Millions-of-machines look: tiny hexagonal plates scattered over the body surface (render detail)."""
    ng = bpy.data.node_groups.get("MyrmexMicroMachines")
    if ng is None:
        ng = bpy.data.node_groups.new("MyrmexMicroMachines", "GeometryNodeTree")
        ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        go = _node(ng, "NodeGroupOutput", (800, 0))
        oi = _node(ng, "GeometryNodeObjectInfo", (-600, 0))
        oi.name = "Body"
        oi.transform_space = "RELATIVE"
        dp = _node(ng, "GeometryNodeDistributePointsOnFaces", (-350, 0))
        dp.inputs["Density"].default_value = density
        ng.links.new(oi.outputs["Geometry"], dp.inputs["Mesh"])
        cyl = _node(ng, "GeometryNodeMeshCylinder", (-350, -250))
        cyl.inputs["Vertices"].default_value = 6
        cyl.inputs["Radius"].default_value = 0.007
        cyl.inputs["Depth"].default_value = 0.0018
        sm = _node(ng, "GeometryNodeSetMaterial", (-100, -250))
        sm.inputs["Material"].default_value = _simple_material("MyrmexMicroPlate", (0.02, 0.02, 0.022), 1.0, 0.22, 0.4)
        ng.links.new(cyl.outputs["Mesh"], sm.inputs["Geometry"])
        rnd = _node(ng, "FunctionNodeRandomValue", (-100, -450))
        rnd.data_type = "FLOAT"
        rnd.inputs["Min"].default_value = 0.45
        rnd.inputs["Max"].default_value = 1.35
        iop = _node(ng, "GeometryNodeInstanceOnPoints", (200, 0))
        ng.links.new(dp.outputs["Points"], iop.inputs["Points"])
        ng.links.new(sm.outputs["Geometry"], iop.inputs["Instance"])
        ng.links.new(dp.outputs["Rotation"], iop.inputs["Rotation"])
        ng.links.new(rnd.outputs["Value"], iop.inputs["Scale"])
        ng.links.new(iop.outputs["Instances"], go.inputs[0])
    ng.nodes["Body"].inputs["Object"].default_value = body
    return ng


# ---------------------------------------------------------------------------- bone links and scutes
def bone_material() -> bpy.types.Material:
    """Dark bionic bone: warm satin black, porous micro relief, a thin coat."""
    mat = bpy.data.materials.get("MyrmexBone")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("MyrmexBone")
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    b = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    for k, v in (("Base Color", (0.052, 0.048, 0.043, 1.0)), ("Metallic", 0.3), ("Roughness", 0.42),
                 ("Coat Weight", 0.35), ("Coat Roughness", 0.22), ("Specular IOR Level", 0.5)):
        if k in b.inputs:
            b.inputs[k].default_value = v
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 95.0
    nz.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.3
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = 0.3
    rr.inputs["To Max"].default_value = 0.52
    nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    return mat


_RING_S = np.array([0.0, 0.1, 0.32, 0.68, 0.9, 1.0])
_SEC = 7
_BONE_V = len(_RING_S) * _SEC + 2


def bone_faces(K: int) -> list[tuple]:
    R, S = len(_RING_S), _SEC
    faces = []
    for k in range(K):
        b = k * _BONE_V
        for r in range(R - 1):
            for q in range(S):
                faces.append((b + r * S + q, b + r * S + (q + 1) % S, b + (r + 1) * S + (q + 1) % S, b + (r + 1) * S + q))
        for q in range(S):
            faces.append((b + R * S, b + (q + 1) % S, b + q))
            faces.append((b + R * S + 1, b + (R - 1) * S + q, b + (R - 1) * S + (q + 1) % S))
    return faces


def bone_points(pos: np.ndarray, links: np.ndarray, up: np.ndarray, t: float, arousal: float = 0.5,
                scale: float = 1.0) -> np.ndarray:
    """Articulated bone links along the skeleton, one per slot; they mutate continuously.

    Knuckled ends and a thin waist, a sharp dorsal crest facing out, a twist, a hooked tip - all
    breathing with time and arousal, each link with its own phase.  Slots without a link collapse.
    """
    K = len(links)
    R, S = len(_RING_S), _SEC
    ok = links[:, 0] >= 0
    i = np.where(ok, links[:, 0], 0).astype(int)
    j = np.where(ok, links[:, 1], 0).astype(int)
    st = np.where(ok, np.clip(links[:, 2], 0.0, 1.0), 0.0)
    A, B = pos[i], pos[j]
    d = B - A
    L = np.linalg.norm(d, axis=1)
    X = d / np.maximum(L, 1e-6)[:, None]
    uh = up[i] + up[j]
    uh = uh - (uh * X).sum(1, keepdims=True) * X
    alt = np.cross(X, np.array([0.0, 0.0, 1.0]))
    alt[np.linalg.norm(alt, axis=1) < 1e-5] = (0.0, 1.0, 0.0)
    bad = np.linalg.norm(uh, axis=1) < 1e-5
    uh[bad] = alt[bad]
    Z = uh / np.maximum(np.linalg.norm(uh, axis=1, keepdims=True), 1e-6)
    Y = np.cross(Z, X)
    A2, L2 = A + d * 0.1, L * 0.8                            # gaps between the links: articulated, not a tube
    ph = (np.arange(K) * 0.6180339887) % 1.0 * 2 * np.pi
    w = 0.6 + 1.2 * float(arousal)
    flare = 0.35 + 0.25 * np.sin(w * t + ph)
    waist = 0.45 + 0.15 * np.sin(1.3 * w * t + ph)
    crest = (0.5 + 0.5 * np.sin(0.7 * w * t + 2 * ph)) * (0.6 + 0.8 * float(arousal))
    twist = 0.6 * np.sin(0.5 * w * t + 3 * ph)
    hook = 0.3 * np.sin(0.9 * w * t + 1.7 * ph)
    r0 = st * (0.03 + 0.05 * np.minimum(1.0, L / 0.45)) * scale
    ones = np.ones(K)
    ringmul = np.stack([0.3 * ones, 1 + flare, waist, waist, 1 + 0.8 * flare, 0.25 * ones], 1)          # (K, R)
    sR = _RING_S
    ridge = np.sin(np.pi * sR) ** 0.8
    th = 2 * np.pi * np.arange(S) / S + np.pi / 2                                                       # q=0 -> dorsal
    theta = th[None, None, :] + twist[:, None, None] * (sR - 0.5)[None, :, None]                       # (K, R, S)
    dors = np.maximum(0.0, np.sin(theta)) ** 6
    mul = (1.0 + 2.2 * crest[:, None, None] * ridge[None, :, None] * dors) * np.where(np.sin(theta) < 0, 0.78, 1.0)
    rad = r0[:, None, None] * ringmul[:, :, None] * mul
    zoff = hook[:, None] * L2[:, None] * sR[None, :] ** 2 * 0.35
    center = A2[:, None, :] + X[:, None, :] * (sR[None, :] * L2[:, None])[..., None] + Z[:, None, :] * zoff[..., None]
    offs = (Y[:, None, None, :] * np.cos(theta)[..., None] + Z[:, None, None, :] * np.sin(theta)[..., None]) * rad[..., None]
    ring = center[:, :, None, :] + offs
    out = np.empty((K, _BONE_V, 3))
    out[:, :R * S] = ring.reshape(K, R * S, 3)
    out[:, R * S] = A2 - X * (0.06 * L2)[:, None]
    out[:, R * S + 1] = A2 + X * (1.14 * L2)[:, None] + Z * (hook * 0.12 * L2)[:, None]
    hide = (~ok) | (st < 0.02) | (L < 1e-4)
    if hide.any():
        out[hide] = pos.mean(0)
    return out.reshape(-1, 3)


def scute_points(pos: np.ndarray, nrm: np.ndarray, plate: np.ndarray, radius: np.ndarray, heading: float,
                 scale: float = 1.0) -> np.ndarray:
    """Bony scutes: raised, swept back into a spike (against the direction of travel)."""
    n = len(pos)
    nr = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    back = -np.array([math.cos(heading), math.sin(heading), 0.0])
    T = back[None, :] - (nr @ back)[:, None] * nr
    e1 = np.cross(nr, np.array([0.0, 0.0, 1.0]))
    e1[np.linalg.norm(e1, axis=1) < 1e-5] = (1.0, 0.0, 0.0)
    weak = np.linalg.norm(T, axis=1) < 1e-4
    T[weak] = e1[weak]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-6)
    Bv = np.cross(nr, T)
    c = pos + nr * (0.92 * radius)[:, None]
    h = (np.clip(plate, 0.0, 1.0) * 0.1 * scale)[:, None]
    out = np.empty((n, 7, 3))
    out[:, 0] = c + nr * 0.35 * h
    out[:, 1] = c - T * 0.6 * h
    out[:, 2] = c - T * 0.15 * h + Bv * 0.55 * h
    out[:, 3] = c + T * 0.5 * h + Bv * 0.4 * h
    out[:, 4] = c + T * 1.8 * h + nr * 0.3 * h
    out[:, 5] = c + T * 0.5 * h - Bv * 0.4 * h
    out[:, 6] = c - T * 0.15 * h - Bv * 0.55 * h
    return out.reshape(-1, 3)


def strut_points(pos: np.ndarray, links: np.ndarray, n_links: int) -> np.ndarray:
    """Segment endpoints for every link slot; weak links retract into their midpoint (invisible)."""
    out = np.zeros((n_links, 2, 3))
    m = min(n_links, len(links))
    lk = links[:m]
    ok = lk[:, 0] >= 0
    i = np.where(ok, lk[:, 0], 0).astype(int)
    j = np.where(ok, lk[:, 1], 0).astype(int)
    s = np.where(ok, np.clip(lk[:, 2], 0, 1), 0.0)[:, None]
    mid = 0.5 * (pos[i] + pos[j])
    half = 0.5 * (pos[j] - pos[i]) * s
    out[:m, 0], out[:m, 1] = mid - half, mid + half
    if m < n_links:
        out[m:] = pos.mean(0)
    return out.reshape(-1, 3)


def plate_points(pos: np.ndarray, nrm: np.ndarray, plate: np.ndarray, radius: np.ndarray,
                 scale: float = 0.075) -> np.ndarray:
    """Hexagonal plates on the surface, facing out; size 0 collapses a plate to a point."""
    n = len(pos)
    nr = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    up = np.where(np.abs(nr[:, 2:3]) > 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 1.0]]))
    e1 = np.cross(nr, up)
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-6)
    e2 = np.cross(nr, e1)
    c = pos + nr * (0.92 * radius)[:, None]
    h = (np.clip(plate, 0, 1) * scale)[:, None]
    out = np.empty((n, 7, 3))
    out[:, 0] = c
    for j in range(6):
        a = math.pi / 3 * j
        out[:, 1 + j] = c + (math.cos(a) * e1 + math.sin(a) * e2) * h
    return out.reshape(-1, 3)


# ---------------------------------------------------------------------------- Cyber Hive (v8)
def _value(nt, name: str, value: float = 0.0):
    n = nt.nodes.new("ShaderNodeValue")
    n.name = n.label = name
    n.outputs[0].default_value = value
    return n


def _math(nt, op: str, a, b=None, c=None, clamp: bool = False):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    n.use_clamp = clamp
    for k, v in enumerate((a, b, c)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[k].default_value = float(v)
        else:
            nt.links.new(v, n.inputs[k])
    return n.outputs[0]


def _attr(nt, name: str):
    n = nt.nodes.new("ShaderNodeAttribute")
    n.attribute_type = "GEOMETRY"
    n.attribute_name = name
    return n.outputs["Fac"]


def _principled(mat, **vals):
    try:
        mat.use_nodes = True
    except Exception:
        pass
    b = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    for k, v in vals.items():
        k = k.replace("_", " ")
        if k in b.inputs:
            b.inputs[k].default_value = v
    return b


def cyber_body_material() -> bpy.types.Material:
    """White nanomaterial: ceramic gloss over microscopic cells; faint circuit seams and a scan band glow in
    soft acid green.  The pattern rides with the body (centre + heading written every frame)."""
    mat = bpy.data.materials.get(CYBER_MAT)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(CYBER_MAT)
    b = _principled(mat, Base_Color=(0.84, 0.86, 0.85, 1.0), Metallic=0.0, Coat_Weight=0.6, Coat_Roughness=0.08,
                    Specular_IOR_Level=0.5)
    nt = mat.node_tree
    tm, act, glow = _value(nt, "MyrmexTime"), _value(nt, "MyrmexActivity"), _value(nt, "MyrmexGlow")
    scan, hd = _value(nt, "MyrmexScan", -1000.0), _value(nt, "MyrmexHeading")
    com = nt.nodes.new("ShaderNodeCombineXYZ")
    for a in "XYZ":
        nt.links.new(_value(nt, "MyrmexC" + a).outputs[0], com.inputs[a])
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    nt.links.new(tc.outputs["Object"], sub.inputs[0])
    nt.links.new(com.outputs[0], sub.inputs[1])
    rot = nt.nodes.new("ShaderNodeVectorRotate")               # into the body frame (x = forward)
    rot.rotation_type = "Z_AXIS"
    rot.invert = True
    nt.links.new(sub.outputs[0], rot.inputs["Vector"])
    nt.links.new(hd.outputs[0], rot.inputs["Angle"])
    P = rot.outputs[0]
    cells = nt.nodes.new("ShaderNodeTexVoronoi")                # nano cells: fine seams (relief)
    cells.feature = "DISTANCE_TO_EDGE"
    cells.inputs["Scale"].default_value = 55.0
    nt.links.new(P, cells.inputs["Vector"])
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = 2.5
    nz.inputs["Detail"].default_value = 3.0
    nt.links.new(P, nz.inputs["Vector"])
    nt.links.new(_math(nt, "MULTIPLY", tm.outputs[0], 0.35), nz.inputs["W"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.003
    nt.links.new(cells.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = 0.14, 0.3
    nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    # circuit seams: cell borders of a coarse Voronoi, only a part of them lit (a slow gate)
    circ = nt.nodes.new("ShaderNodeTexVoronoi")
    circ.feature = "DISTANCE_TO_EDGE"
    circ.inputs["Scale"].default_value = 4.5
    nt.links.new(P, circ.inputs["Vector"])
    ln = nt.nodes.new("ShaderNodeMapRange")
    ln.inputs["From Min"].default_value, ln.inputs["From Max"].default_value = 0.0, 0.03
    ln.inputs["To Min"].default_value, ln.inputs["To Max"].default_value = 1.0, 0.0
    nt.links.new(circ.outputs["Distance"], ln.inputs["Value"])
    gate = nt.nodes.new("ShaderNodeMapRange")
    gate.inputs["From Min"].default_value, gate.inputs["From Max"].default_value = 0.47, 0.56
    nt.links.new(nz.outputs["Fac"], gate.inputs["Value"])
    lines = _math(nt, "MULTIPLY", ln.outputs["Result"], gate.outputs["Result"])
    lit = _math(nt, "MULTIPLY", lines, _math(nt, "MULTIPLY_ADD", glow.outputs[0], 0.2, 0.45))
    # the scan band: a thin ring of light sweeping the body tail -> head
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(P, sep.inputs[0])
    q = _math(nt, "DIVIDE", _math(nt, "SUBTRACT", sep.outputs["X"], scan.outputs[0]), 0.07)
    band = _math(nt, "EXPONENT", _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", q, q), -1.0))
    em = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY_ADD", band, 1.4, lit), 1.8)
    if "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = CYBER_GREEN
        nt.links.new(em, b.inputs["Emission Strength"])
    return mat


def cyber_hull_material() -> bpy.types.Material:
    """The rails' and panels' shell: the same white, a touch harder and smoother."""
    mat = bpy.data.materials.get("MyrmexCyberHull")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("MyrmexCyberHull")
    _principled(mat, Base_Color=(0.8, 0.82, 0.81, 1.0), Metallic=0.1, Roughness=0.24, Coat_Weight=0.5,
                Coat_Roughness=0.06, Specular_IOR_Level=0.5)
    return mat


def cyber_line_material() -> bpy.types.Material:
    """Light lines: pale green-white, glowing by the node's light; pulses run along the body."""
    mat = bpy.data.materials.get("MyrmexCyberLine")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("MyrmexCyberLine")
    b = _principled(mat, Base_Color=(0.72, 0.86, 0.66, 1.0), Metallic=0.0, Roughness=0.3, Coat_Weight=0.3)
    nt = mat.node_tree
    tm, glow = _value(nt, "MyrmexTime"), _value(nt, "MyrmexGlow")
    light, flow = _attr(nt, "light"), _attr(nt, "flow")
    pulse = _math(nt, "POWER", _math(nt, "FRACT", _math(nt, "MULTIPLY_ADD", flow, 1.6, tm.outputs[0])), 10.0)
    s = _math(nt, "MULTIPLY", light, _math(nt, "MULTIPLY_ADD", pulse, 2.6, 1.0))
    em = _math(nt, "MULTIPLY", _math(nt, "ADD", s, glow.outputs[0]), 2.2)
    if "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = CYBER_GREEN
        nt.links.new(em, b.inputs["Emission Strength"])
    return mat


def cyber_mote_material() -> bpy.types.Material:
    """The nanomachines: white chips, one in five a small green light."""
    mat = bpy.data.materials.get("MyrmexCyberMote")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("MyrmexCyberMote")
    b = _principled(mat, Base_Color=(0.82, 0.84, 0.83, 1.0), Metallic=0.1, Roughness=0.3)
    nt = mat.node_tree
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    em = _math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", oi.outputs["Random"], 0.8), 3.0)
    if "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = CYBER_GREEN
        nt.links.new(em, b.inputs["Emission Strength"])
    return mat


_RAIL_A = np.array([-0.16, 0.16, 1.2, 1.95, np.pi - 0.16, np.pi + 0.16, np.pi + 1.2, np.pi + 1.95]) + np.pi / 2
_RAIL_LINE = (0, 4)             # faces from these columns to the next are the light lines (outside, inside)
_RAIL_R = 7                     # rings: cap, shoulder, collar (3), shoulder, cap
_RAIL_C = len(_RAIL_A)
_RAIL_V = _RAIL_R * _RAIL_C + 2
_RAIL_MUL = np.array([0.55, 1.0, 1.0, 1.35, 1.0, 1.0, 0.55])


def rail_faces(K: int) -> tuple[list, list]:
    R, C = _RAIL_R, _RAIL_C
    faces, mats = [], []
    for k in range(K):
        b = k * _RAIL_V
        for r in range(R - 1):
            for q in range(C):
                faces.append((b + r * C + q, b + r * C + (q + 1) % C, b + (r + 1) * C + (q + 1) % C, b + (r + 1) * C + q))
                mats.append(1 if (q in _RAIL_LINE or r in (2, 3)) else 0)        # lines + the collar glow
        for q in range(C):
            faces.append((b + R * C, b + (q + 1) % C, b + q))
            faces.append((b + R * C + 1, b + (R - 1) * C + q, b + (R - 1) * C + (q + 1) % C))
            mats += [0, 0]
    return faces, mats


def rail_points(pos: np.ndarray, links: np.ndarray, up: np.ndarray, t: float, arousal: float = 0.5,
                scale: float = 1.0) -> np.ndarray:
    """Hexagonal rail modules along the skeleton (one per slot): two light lines along them, a glowing
    collar that slides like a piston; separate modules with gaps.  Slots without a link collapse."""
    K = len(links)
    R, C = _RAIL_R, _RAIL_C
    ok = links[:, 0] >= 0
    i = np.where(ok, links[:, 0], 0).astype(int)
    j = np.where(ok, links[:, 1], 0).astype(int)
    st = np.where(ok, np.clip(links[:, 2], 0.0, 1.0), 0.0)
    A, B = pos[i], pos[j]
    d = B - A
    L = np.linalg.norm(d, axis=1)
    X = d / np.maximum(L, 1e-6)[:, None]
    uh = up[i] + up[j]
    uh = uh - (uh * X).sum(1, keepdims=True) * X
    alt = np.cross(X, np.array([0.0, 0.0, 1.0]))
    alt[np.linalg.norm(alt, axis=1) < 1e-5] = (0.0, 1.0, 0.0)
    bad = np.linalg.norm(uh, axis=1) < 1e-5
    uh[bad] = alt[bad]
    Z = uh / np.maximum(np.linalg.norm(uh, axis=1, keepdims=True), 1e-6)
    Y = np.cross(Z, X)
    A2, L2 = A + d * 0.07, L * 0.86
    ph = (np.arange(K) * 0.6180339887) % 1.0 * 2 * np.pi
    c = 0.5 + 0.26 * np.sin((0.5 + 1.0 * float(arousal)) * t + ph)             # the collar slides
    zero = np.zeros(K)
    S = np.stack([zero, zero + 0.06, c - 0.05, c, c + 0.05, zero + 0.94, zero + 1.0], 1)          # (K, R)
    r0 = st * (0.02 + 0.028 * np.minimum(1.0, L / 0.45)) * scale
    rad = r0[:, None] * _RAIL_MUL[None, :]
    center = A2[:, None, :] + X[:, None, :] * (S * L2[:, None])[..., None]
    cs, sn = np.cos(_RAIL_A), np.sin(_RAIL_A)
    offs = (Y[:, None, None, :] * cs[None, None, :, None] + Z[:, None, None, :] * sn[None, None, :, None]) * \
        rad[:, :, None, None]
    out = np.empty((K, _RAIL_V, 3))
    out[:, :R * C] = (center[:, :, None, :] + offs).reshape(K, R * C, 3)
    out[:, R * C] = A2 - X * (0.02 * L2)[:, None]
    out[:, R * C + 1] = A2 + X * (1.02 * L2)[:, None]
    hide = (~ok) | (st < 0.02) | (L < 1e-4)
    if hide.any():
        out[hide] = pos.mean(0)
    return out.reshape(-1, 3)


def _flow(pts: np.ndarray, com, heading: float) -> np.ndarray:
    """Metres along the direction of travel (light pulses run along it)."""
    h = np.array([math.cos(heading), math.sin(heading), 0.0])
    return ((pts - np.asarray(com, float)) @ h).astype(np.float32)


def rail_light(links: np.ndarray, light) -> np.ndarray:
    ok = links[:, 0] >= 0
    if light is None:
        li = np.full(len(links), 0.35)
    else:
        light = np.asarray(light, float)
        i = np.where(ok, links[:, 0], 0).astype(int)
        j = np.where(ok, links[:, 1], 0).astype(int)
        li = 0.5 * (light[i] + light[j])
    return np.repeat(li, _RAIL_V).astype(np.float32)


_PANEL_V = 25


def panel_faces(n: int) -> tuple[list, list]:
    faces, mats = [], []
    for i in range(n):
        b = _PANEL_V * i
        for j in range(6):
            j2 = (j + 1) % 6
            faces += [(b, b + 1 + j, b + 1 + j2), (b + 1 + j, b + 7 + j, b + 7 + j2, b + 1 + j2),
                      (b + 7 + j, b + 13 + j, b + 13 + j2, b + 7 + j2), (b + 13 + j, b + 19 + j, b + 19 + j2, b + 13 + j2)]
            mats += [0, 0, 1, 0]                          # the ring between 0.7 and 0.8 of the radius glows
    return faces, mats


def panel_points(pos: np.ndarray, nrm: np.ndarray, plate: np.ndarray, radius: np.ndarray, heading: float,
                 scale: float = 1.0) -> np.ndarray:
    """Hexagonal panels tiling the surface, aligned with the direction of travel, gently domed."""
    n = len(pos)
    nr = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
    T = fwd[None, :] - (nr @ fwd)[:, None] * nr
    e1 = np.cross(nr, np.array([0.0, 0.0, 1.0]))
    e1[np.linalg.norm(e1, axis=1) < 1e-5] = (1.0, 0.0, 0.0)
    weak = np.linalg.norm(T, axis=1) < 1e-4
    T[weak] = e1[weak]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-6)
    Bv = np.cross(nr, T)
    c = pos + nr * (0.93 * radius)[:, None]
    h = (np.clip(plate, 0.0, 1.0) * 0.085 * scale)[:, None]
    a = np.pi / 3 * np.arange(6) + np.pi / 6
    unit = np.cos(a)[None, :, None] * T[:, None, :] + np.sin(a)[None, :, None] * Bv[:, None, :]    # (n, 6, 3)
    out = np.empty((n, _PANEL_V, 3))
    out[:, 0] = c + nr * 0.1 * h
    for k, (rr, lift) in enumerate(((0.62, 0.07), (0.7, 0.05), (0.8, 0.04), (1.0, -0.02))):
        out[:, 1 + 6 * k:7 + 6 * k] = c[:, None, :] + unit * (rr * h)[:, :, None] + (nr * lift * h)[:, None, :]
    return out.reshape(-1, 3)


def panel_light(light, n: int) -> np.ndarray:
    li = np.full(n, 0.35) if light is None else np.asarray(light, float)
    return np.repeat(li, _PANEL_V).astype(np.float32)


def _cyber_mesh(name: str, verts: int, faces: list, mats: list, coll) -> bpy.types.Object:
    """A mesh with the hull + line materials and the per-vertex 'flow' / 'light' attributes."""
    ob = bpy.data.objects.get(name)
    if ob is not None and len(ob.data.vertices) == verts and "light" in ob.data.attributes:
        return ob
    me = bpy.data.meshes.new(name)
    me.from_pydata([(0.0, 0.0, 0.0)] * verts, [], faces)
    me.materials.append(cyber_hull_material())
    me.materials.append(cyber_line_material())
    me.polygons.foreach_set("material_index", np.asarray(mats, np.int32))
    for poly in me.polygons:
        poly.use_smooth = True
    for a in ("flow", "light"):
        me.attributes.new(a, "FLOAT", "POINT")
    if ob is None:
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
    else:
        ob.data = me
    return ob


def set_cyber_mesh(ob: bpy.types.Object, pts: np.ndarray, flow: np.ndarray, light: np.ndarray) -> None:
    me = ob.data
    if len(pts) != len(me.vertices):
        return
    me.vertices.foreach_set("co", pts.astype(np.float32).ravel())
    if "flow" in me.attributes:
        me.attributes["flow"].data.foreach_set("value", flow)
        me.attributes["light"].data.foreach_set("value", light)
    me.update()


def set_cyber_shading(fr_t: float, glow: float, arousal: float, scan: float, com, heading: float,
                      body: bpy.types.Material | None) -> None:
    """Per-frame values of the Cyber Hive's shaders (live, and keyed by the take importer)."""
    lm = bpy.data.materials.get("MyrmexCyberLine")
    if lm is not None and lm.node_tree is not None:
        nd = lm.node_tree.nodes
        if "MyrmexTime" in nd:
            nd["MyrmexTime"].outputs[0].default_value = float(fr_t) * (0.35 + 0.5 * float(arousal))
            nd["MyrmexGlow"].outputs[0].default_value = 0.8 * float(glow)
    if body is not None and body.node_tree is not None:
        nd = body.node_tree.nodes
        if "MyrmexScan" in nd:
            nd["MyrmexScan"].outputs[0].default_value = float(scan) if math.isfinite(scan) else -1000.0
            nd["MyrmexHeading"].outputs[0].default_value = float(heading)
            for k, a in enumerate("XYZ"):
                nd["MyrmexC" + a].outputs[0].default_value = float(com[k])


# ---------------------------------------------------------------------------- Mimetic line (v9-v13)
def liquid_material(fin: bool = False) -> bpy.types.Material:
    """Glossy black liquid metal: mirror-dark, coated, fine ripples; very dark red internal glints."""
    name = "MyrmexLiquidFin" if fin else LIQUID_MAT
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    b = _principled(mat, Base_Color=(0.006, 0.006, 0.007, 1.0), Metallic=0.92, Roughness=0.1, Coat_Weight=0.85,
                    Coat_Roughness=0.03, Specular_IOR_Level=0.7, Anisotropic=0.25)
    nt = mat.node_tree
    tm, _act, glow = _value(nt, "MyrmexTime"), _value(nt, "MyrmexActivity"), _value(nt, "MyrmexGlow")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.noise_dimensions = "4D"
    nz.inputs["Scale"].default_value = 9.0
    nz.inputs["Detail"].default_value = 5.0
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    nt.links.new(_math(nt, "MULTIPLY", tm.outputs[0], 0.3), nz.inputs["W"])
    rr = nt.nodes.new("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value, rr.inputs["To Max"].default_value = 0.05, 0.18
    nt.links.new(nz.outputs["Fac"], rr.inputs["Value"])
    nt.links.new(rr.outputs["Result"], b.inputs["Roughness"])
    rip = nt.nodes.new("ShaderNodeTexNoise")                     # liquid ripples
    rip.inputs["Scale"].default_value = 60.0
    rip.inputs["Detail"].default_value = 2.0
    nt.links.new(tc.outputs["Object"], rip.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.004
    nt.links.new(rip.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    if fin:                                                        # some fins carry a red edge
        red = _math(nt, "MULTIPLY", _attr(nt, "glint"), _math(nt, "MULTIPLY_ADD", glow.outputs[0], 0.2, 0.5))
    else:                                                          # scattered internal sparks, a slow gate
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = 18.0
        nt.links.new(tc.outputs["Object"], vor.inputs["Vector"])
        dots = nt.nodes.new("ShaderNodeMapRange")
        dots.inputs["From Min"].default_value, dots.inputs["From Max"].default_value = 0.0, 0.07
        dots.inputs["To Min"].default_value, dots.inputs["To Max"].default_value = 1.0, 0.0
        nt.links.new(vor.outputs["Distance"], dots.inputs["Value"])
        gate = nt.nodes.new("ShaderNodeMapRange")
        gate.inputs["From Min"].default_value, gate.inputs["From Max"].default_value = 0.58, 0.66
        nt.links.new(nz.outputs["Fac"], gate.inputs["Value"])
        red = _math(nt, "MULTIPLY", _math(nt, "MULTIPLY", dots.outputs["Result"], gate.outputs["Result"]),
                    _math(nt, "MULTIPLY_ADD", glow.outputs[0], 0.25, 0.6))
    if "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = DEEP_RED
        nt.links.new(_math(nt, "MULTIPLY", red, 2.2), b.inputs["Emission Strength"])
    return mat


def set_liquid_shading(t: float, glow: float) -> None:
    fm = bpy.data.materials.get("MyrmexLiquidFin")
    if fm is not None and fm.node_tree is not None and "MyrmexTime" in fm.node_tree.nodes:
        fm.node_tree.nodes["MyrmexTime"].outputs[0].default_value = float(t) * 0.2
        fm.node_tree.nodes["MyrmexGlow"].outputs[0].default_value = 4.0 * float(glow)


def tube_faces(K: int, rings: int, cols: int, per: int) -> list[tuple]:
    """Closed tubes (``rings`` x ``cols`` + two cap centres per slot)."""
    faces = []
    for k in range(K):
        b = k * per
        for r in range(rings - 1):
            for q in range(cols):
                faces.append((b + r * cols + q, b + r * cols + (q + 1) % cols, b + (r + 1) * cols + (q + 1) % cols,
                              b + (r + 1) * cols + q))
        for q in range(cols):
            faces.append((b + rings * cols, b + (q + 1) % cols, b + q))
            faces.append((b + rings * cols + 1, b + (rings - 1) * cols + q, b + (rings - 1) * cols + (q + 1) % cols))
    return faces


_TEN_S = np.array([0.0, 0.12, 0.3, 0.5, 0.7, 0.88, 1.0])
_TEN_MUL = np.array([0.35, 0.8, 1.0, 0.95, 0.8, 0.5, 0.15])
_TEN_C = 8
_TEN_V = len(_TEN_S) * _TEN_C + 2
TENDON_THICK = {3: 0.8, 4: 1.0, 5: 0.7, 6: 0.9, 7: 1.7}


def tendon_points(pos: np.ndarray, links: np.ndarray, up: np.ndarray, t: float, arousal: float = 0.5,
                  thick: float = 1.0) -> np.ndarray:
    """Smooth tapered strands along the skeleton, gently bending (liquid, not bone).  Empty slots collapse."""
    K = len(links)
    R, C = len(_TEN_S), _TEN_C
    ok = links[:, 0] >= 0
    i = np.where(ok, links[:, 0], 0).astype(int)
    j = np.where(ok, links[:, 1], 0).astype(int)
    st = np.where(ok, np.clip(links[:, 2], 0.0, 1.0), 0.0)
    A, B = pos[i], pos[j]
    d = B - A
    L = np.linalg.norm(d, axis=1)
    X = d / np.maximum(L, 1e-6)[:, None]
    uh = up[i] + up[j]
    uh = uh - (uh * X).sum(1, keepdims=True) * X
    alt = np.cross(X, np.array([0.0, 0.0, 1.0]))
    alt[np.linalg.norm(alt, axis=1) < 1e-5] = (0.0, 1.0, 0.0)
    bad = np.linalg.norm(uh, axis=1) < 1e-5
    uh[bad] = alt[bad]
    Z = uh / np.maximum(np.linalg.norm(uh, axis=1, keepdims=True), 1e-6)
    Y = np.cross(Z, X)
    A2, L2 = A + d * 0.04, L * 0.92
    ph = (np.arange(K) * 0.6180339887) % 1.0 * 2 * np.pi
    w = 0.6 + 1.0 * float(arousal)
    bend = 0.07 * np.sin(w * t + ph)
    wig = 0.04 * np.cos(1.3 * w * t + 2 * ph)
    S = _TEN_S
    center = A2[:, None, :] + X[:, None, :] * (S[None, :] * L2[:, None])[..., None] + \
        Z[:, None, :] * (bend[:, None] * L2[:, None] * np.sin(np.pi * S)[None, :])[..., None] + \
        Y[:, None, :] * (wig[:, None] * L2[:, None] * np.sin(2 * np.pi * S)[None, :])[..., None]
    r0 = st * thick * (0.014 + 0.02 * np.minimum(1.0, L / 0.45))
    rad = r0[:, None] * _TEN_MUL[None, :]
    a = 2 * np.pi * np.arange(C) / C
    offs = (Y[:, None, None, :] * np.cos(a)[None, None, :, None] + Z[:, None, None, :] * np.sin(a)[None, None, :, None]) * \
        rad[:, :, None, None]
    out = np.empty((K, _TEN_V, 3))
    out[:, :R * C] = (center[:, :, None, :] + offs).reshape(K, R * C, 3)
    out[:, R * C] = A2
    out[:, R * C + 1] = A2 + X * L2[:, None]
    hide = (~ok) | (st < 0.02) | (L < 1e-4)
    if hide.any():
        out[hide] = pos.mean(0)
    return out.reshape(-1, 3)


_FIN_N = 5                                      # points along a fin
_FIN_V = 2 * _FIN_N
_FIN_TAUS = {3: (0.0, 0.035, 0.08, 0.15, 0.24), 6: (0.0, 0.05, 0.12, 0.2, 0.3)}     # trail samples (s)


def fin_faces(n: int) -> list[tuple]:
    return [(b + 2 * k, b + 2 * k + 1, b + 2 * k + 3, b + 2 * k + 2) for b in range(0, _FIN_V * n, _FIN_V)
            for k in range(_FIN_N - 1)]


def trail(hist: list, taus) -> np.ndarray:
    """(t, pos) history, newest first -> positions at the given ages (n, len(taus), 3)."""
    t0 = hist[0][0]
    ts = np.array([h[0] for h in hist])
    return np.stack([hist[int(np.argmin(np.abs(t0 - tau - ts)))][1] for tau in taus], 1)


def fin_points(style: int, hist: list, nrm: np.ndarray, com, heading: float, t: float) -> np.ndarray:
    """One fin per node: swarm filaments / spear shards / cloud flakes / blade trails / crawler claws + spikes."""
    pos = np.asarray(hist[0][1], float)
    n = len(pos)
    k = np.arange(n)
    h1, h2 = (k * 0.6180339887) % 1.0, (k * 0.4142135624 + 0.3) % 1.0
    nr = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
    sN = np.linspace(0.0, 1.0, _FIN_N)
    if style in _FIN_TAUS:                                        # trails of the motion itself
        spine = trail(hist, _FIN_TAUS[style])
        off = spine - spine[:, :1]
        ln = np.linalg.norm(off[:, -1], axis=1)
        top = {3: 1.7, 6: 1.1}[style] * (0.7 + 0.6 * h2)            # long streaks, never longer than this
        spine = spine[:, :1] + off * np.minimum(1.0, top / np.maximum(ln, 1e-6))[:, None, None]
        d = spine[:, -1] - spine[:, 0]
        dn = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        if style == 3:
            wdir = np.cross(dn, nr)
            width = 0.004 + 0.012 * h1
        else:
            wdir = nr - (nr * dn).sum(1, keepdims=True) * dn
            width = 0.03 + 0.07 * h1
        prof = np.array([1.0, 0.85, 0.6, 0.35, 0.02])
    elif style == 4:                                              # straight shards swept back, tilted out
        L = (0.12 + 0.42 * h1)
        spine = pos[:, None] + (-fwd[None, :] * L[:, None])[:, None, :] * sN[None, :, None] + \
            (nr * 0.3 * L[:, None])[:, None, :] * (sN ** 2)[None, :, None]
        wdir = nr - (nr @ fwd)[:, None] * fwd[None, :]
        width = 0.02 + 0.04 * h2
        prof = np.array([1.0, 0.8, 0.55, 0.3, 0.02])
    elif style == 5:                                              # tumbling flakes
        a, b = 2 * np.pi * (h1 + 0.07 * t), 2 * np.pi * (h2 + 0.05 * t)
        dirs = np.stack([np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)], 1)
        L = 0.06 + 0.12 * h2
        spine = pos[:, None] + (dirs * L[:, None])[:, None, :] * (sN - 0.5)[None, :, None]
        wdir = np.cross(dirs, nr + np.array([0.0, 0.0, 0.3]))
        width = 0.03 + 0.05 * h1
        prof = np.array([0.3, 0.9, 1.0, 0.8, 0.2])
    else:                                                         # crawler: claws on the feet, spikes on the back
        from myrmex.creature.mimetic import terrain_height
        hz = pos[:, 2] - terrain_height(pos[:, 0], pos[:, 1])
        c = np.asarray(com, float)
        feet = hz < 0.2
        dorsal = ~feet & ((pos[:, 2] - c[2]) > 0.12) & (h1 < 0.6)
        claw = pos[:, None] + (fwd * 0.16)[None, None, :] * sN[None, :, None] + \
            np.array([0.0, 0.0, -0.12])[None, None, :] * (sN ** 2)[None, :, None]
        spike = pos[:, None] + (np.array([0.0, 0.0, 0.2]) - fwd * 0.1)[None, None, :] * sN[None, :, None] * \
            (0.6 + 0.8 * h2)[:, None, None]
        spine = np.where(feet[:, None, None], claw, np.where(dorsal[:, None, None], spike, pos[:, None].repeat(_FIN_N, 1)))
        wdir = np.cross(np.broadcast_to(fwd, (n, 3)), np.array([0.0, 0.0, 1.0]))
        width = np.where(feet, 0.028, np.where(dorsal, 0.02, 0.0))
        prof = np.array([1.0, 0.8, 0.55, 0.3, 0.02])
    wdir = wdir / np.maximum(np.linalg.norm(wdir, axis=1, keepdims=True), 1e-6)
    half = (wdir * np.asarray(width)[:, None])[:, None, :] * prof[None, :, None]
    out = np.empty((n, _FIN_N, 2, 3))
    out[:, :, 0] = spine + half
    out[:, :, 1] = spine - half
    return out.reshape(-1, 3)


def fin_glints(n: int) -> np.ndarray:
    """Which fins carry a red edge (about one in eight)."""
    k = np.arange(n)
    return np.repeat((((k * 0.7548776662) % 1.0) > 0.875).astype(np.float32), _FIN_V)


def flakes_nodes() -> bpy.types.NodeTree:
    """The Mimetic swarm particles: glossy black shards, a few with a red glint."""
    ng = bpy.data.node_groups.get("MyrmexLiquidFlakes")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new("MyrmexLiquidFlakes", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = _node(ng, "NodeGroupInput", (-600, 0)), _node(ng, "NodeGroupOutput", (500, 0))
    ico = _node(ng, "GeometryNodeMeshIcoSphere", (-800, -250))
    ico.inputs["Radius"].default_value = 0.012
    ico.inputs["Subdivisions"].default_value = 1
    tr = _node(ng, "GeometryNodeTransform", (-600, -250))
    tr.inputs["Scale"].default_value = (2.4, 0.7, 0.18)            # a thin shard
    ng.links.new(ico.outputs["Mesh"], tr.inputs["Geometry"])
    sm = _node(ng, "GeometryNodeSetMaterial", (-350, -250))
    sm.inputs["Material"].default_value = _mote_red_material()
    ng.links.new(tr.outputs["Geometry"], sm.inputs["Geometry"])
    rot = _node(ng, "FunctionNodeRandomValue", (-350, -450))
    rot.data_type = "FLOAT_VECTOR"
    rot.inputs["Max"].default_value = (6.2832, 6.2832, 6.2832)
    size = _node(ng, "FunctionNodeRandomValue", (-350, -650))
    size.data_type = "FLOAT"
    size.inputs["Min"].default_value = 0.5
    size.inputs["Max"].default_value = 1.8
    iop = _node(ng, "GeometryNodeInstanceOnPoints", (100, 0))
    ng.links.new(gi.outputs[0], iop.inputs["Points"])
    ng.links.new(sm.outputs["Geometry"], iop.inputs["Instance"])
    ng.links.new(rot.outputs["Value"], iop.inputs["Rotation"])
    ng.links.new(size.outputs["Value"], iop.inputs["Scale"])
    ng.links.new(iop.outputs["Instances"], go.inputs[0])
    return ng


def _mote_red_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("MyrmexLiquidMote")
    if mat is not None:
        return mat
    mat = bpy.data.materials.new("MyrmexLiquidMote")
    b = _principled(mat, Base_Color=(0.006, 0.006, 0.007, 1.0), Metallic=0.9, Roughness=0.08, Coat_Weight=0.6)
    nt = mat.node_tree
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    if "Emission Color" in b.inputs:
        b.inputs["Emission Color"].default_value = DEEP_RED
        nt.links.new(_math(nt, "MULTIPLY", _math(nt, "GREATER_THAN", oi.outputs["Random"], 0.92), 2.5),
                     b.inputs["Emission Strength"])
    return mat


def crawler_terrain(on: bool, coll) -> None:
    """The Crawler walks on rough ground (the same heights as the engine); others keep the flat floor."""
    floor = bpy.data.objects.get("MyrmexFloor")
    if not on:
        _drop(TERRAIN)
        if floor is not None:
            floor.hide_viewport = floor.hide_render = False
        return
    if floor is not None:
        floor.hide_viewport = floor.hide_render = True
    if bpy.data.objects.get(TERRAIN) is not None:
        return
    from myrmex.creature.mimetic import terrain_height
    N, half = 241, 40.0
    xs = np.linspace(-half, half, N)
    X, Y = np.meshgrid(xs, xs)
    Z = terrain_height(X, Y)
    idx = np.arange(N * N).reshape(N, N)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    me = bpy.data.meshes.new(TERRAIN)
    me.from_pydata(np.stack([X, Y, Z], -1).reshape(-1, 3).tolist(), [], quads.tolist())
    for poly in me.polygons:
        poly.use_smooth = True
    mat = bpy.data.materials.get("MyrmexTerrain")
    if mat is None:
        mat = bpy.data.materials.new("MyrmexTerrain")
        b = _principled(mat, Base_Color=(0.011, 0.011, 0.012, 1.0), Metallic=0.2, Roughness=0.45)
        nt = mat.node_tree
        nz = nt.nodes.new("ShaderNodeTexNoise")
        nz.inputs["Scale"].default_value = 3.0
        nz.inputs["Detail"].default_value = 8.0
        bump = nt.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.4
        nt.links.new(nz.outputs["Fac"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    me.materials.append(mat)
    ob = bpy.data.objects.new(TERRAIN, me)
    coll.objects.link(ob)


def _replace_mesh_object(name: str, verts: int, faces: list, material, coll) -> bpy.types.Object:
    ob = bpy.data.objects.get(name)
    if ob is not None and len(ob.data.vertices) == verts:
        return ob
    me = bpy.data.meshes.new(name)
    me.from_pydata([(0.0, 0.0, 0.0)] * verts, [], faces)
    for poly in me.polygons:
        poly.use_smooth = True
    me.materials.append(material)
    if ob is None:
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
    else:
        ob.data = me
    return ob


def _drop(name: str) -> None:
    ob = bpy.data.objects.get(name)
    if ob is not None:
        bpy.data.objects.remove(ob, do_unlink=True)


def swarm_nodes(cyber: bool = False) -> bpy.types.NodeTree:
    """Loose points -> tiny hexagonal machine plates, randomly turned and sized (the nanomachine swarm).
    Cyber Hive: white chips, some of them small green lights."""
    name = "MyrmexCyberSwarm" if cyber else "MyrmexNanoSwarm"
    ng = bpy.data.node_groups.get(name)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = _node(ng, "NodeGroupInput", (-600, 0)), _node(ng, "NodeGroupOutput", (500, 0))
    cyl = _node(ng, "GeometryNodeMeshCylinder", (-600, -250))
    cyl.inputs["Vertices"].default_value = 6
    cyl.inputs["Radius"].default_value = 0.012
    cyl.inputs["Depth"].default_value = 0.003
    sm = _node(ng, "GeometryNodeSetMaterial", (-350, -250))
    sm.inputs["Material"].default_value = cyber_mote_material() if cyber else \
        _simple_material("MyrmexMicroPlate", (0.02, 0.02, 0.022), 1.0, 0.22, 0.4)
    ng.links.new(cyl.outputs["Mesh"], sm.inputs["Geometry"])
    rot = _node(ng, "FunctionNodeRandomValue", (-350, -450))
    rot.data_type = "FLOAT_VECTOR"
    rot.inputs["Max"].default_value = (6.2832, 6.2832, 6.2832)
    size = _node(ng, "FunctionNodeRandomValue", (-350, -650))
    size.data_type = "FLOAT"
    size.inputs["Min"].default_value = 0.6
    size.inputs["Max"].default_value = 1.4
    iop = _node(ng, "GeometryNodeInstanceOnPoints", (100, 0))
    ng.links.new(gi.outputs[0], iop.inputs["Points"])
    ng.links.new(sm.outputs["Geometry"], iop.inputs["Instance"])
    ng.links.new(rot.outputs["Value"], iop.inputs["Rotation"])
    ng.links.new(size.outputs["Value"], iop.inputs["Scale"])
    ng.links.new(iop.outputs["Instances"], go.inputs[0])
    return ng


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
        mb = bpy.data.metaballs.get(META)
        if mb is None:                                   # a saved look keeps its own settings
            mb = bpy.data.metaballs.new(META)
            mb.resolution = resolution                   # viewport polygonisation (speed)
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
        self.poly = False
        self.lattice = self.micro = None
        self.plates = self.lure = self.swarm = None
        self.obstacles: list = []

    def make_polyalloy(self, n_links: int, n_obstacles: int = 4, style: int = 0) -> None:
        """Polyalloy family: skin, frame (struts, or bone links for the Osseous line), obstacles, micro-machines."""
        self.style = style
        self.poly = True
        pm = cyber_body_material() if style == 2 else liquid_material() if style >= 3 else \
            nanomaterial(POLY_MAT, poly=True)
        if not self.mb.materials:
            self.mb.materials.append(pm)
        elif self.mb.materials[0] is None or self.mb.materials[0].name in (MAT, POLY_MAT, CYBER_MAT, LIQUID_MAT):
            self.mb.materials[0] = pm                      # (keep a material you chose yourself)
        if style < 3:
            _drop(TENDONS)
            _drop(FINS)
        crawler_terrain(style == 7, self.coll)
        if style >= 3:                                     # Mimetic line: glossy tendons (+ fins, per frame)
            _drop(LATTICE)
            _drop(BONES)
            _drop(RAILS)
            self.lattice = _replace_mesh_object(TENDONS, n_links * _TEN_V, tube_faces(n_links, len(_TEN_S), _TEN_C,
                                                                                      _TEN_V), liquid_material(), self.coll)
        elif style == 2:                                   # Cyber: hexagonal rails with light lines
            _drop(LATTICE)
            _drop(BONES)
            faces, mats = rail_faces(n_links)
            self.lattice = _cyber_mesh(RAILS, n_links * _RAIL_V, faces, mats, self.coll)
        elif style == 1:                                   # Osseous: articulated bone links
            _drop(LATTICE)
            _drop(RAILS)
            self.lattice = _replace_mesh_object(BONES, n_links * _BONE_V, bone_faces(n_links), bone_material(),
                                                self.coll)
        else:                                              # classic: strut tubes (Geometry Nodes)
            _drop(BONES)
            _drop(RAILS)
            ob = bpy.data.objects.get(LATTICE)
            if ob is None or len(ob.data.vertices) != 2 * n_links:
                me = bpy.data.meshes.new(LATTICE)
                me.from_pydata([(0.0, 0.0, 0.0)] * (2 * n_links), [(2 * k, 2 * k + 1) for k in range(n_links)], [])
                if ob is None:
                    ob = bpy.data.objects.new(LATTICE, me)
                    self.coll.objects.link(ob)
                else:
                    ob.data = me
            if "Struts" not in ob.modifiers:
                ob.modifiers.new("Struts", "NODES").node_group = tube_nodes()
            self.lattice = ob
        mi = bpy.data.objects.get(MICRO)
        if mi is None:
            mi = bpy.data.objects.new(MICRO, bpy.data.meshes.new(MICRO))
            self.coll.objects.link(mi)
            mod = mi.modifiers.new("MicroMachines", "NODES")
            mod.node_group = micro_nodes(self.obj)
            mod.show_viewport = False                      # render detail; enable in the panel to preview
        mod = mi.modifiers.get("MicroMachines")
        sm = next((nd for nd in mod.node_group.nodes if nd.bl_idname == "GeometryNodeSetMaterial"), None) \
            if mod is not None and mod.node_group is not None else None
        if sm is not None:                                 # micro-machines: white on the Cyber Hive
            cur = sm.inputs["Material"].default_value
            if cur is None or cur.name in ("MyrmexMicroPlate", "MyrmexCyberHull"):
                sm.inputs["Material"].default_value = cyber_hull_material() if style == 2 else \
                    _simple_material("MyrmexMicroPlate", (0.02, 0.02, 0.022), 1.0, 0.22, 0.4)
        self.micro = mi
        self.obstacles = []
        for k in range(n_obstacles):
            name = f"{OBSTACLE}_{k}"
            o = bpy.data.objects.get(name)
            if o is None:
                import bmesh
                me = bpy.data.meshes.new(name)
                bm = bmesh.new()
                bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
                bm.to_mesh(me)
                bm.free()
                for poly in me.polygons:
                    poly.use_smooth = True
                me.materials.append(_simple_material("MyrmexObstacle", (0.03, 0.03, 0.032), 0.0, 0.08, 1.0))
                o = bpy.data.objects.new(name, me)
                self.coll.objects.link(o)
            o.scale = (0.0, 0.0, 0.0)
            self.obstacles.append(o)

    def _fins(self, n: int) -> bpy.types.Object:
        ob = bpy.data.objects.get(FINS)
        if ob is not None and len(ob.data.vertices) == _FIN_V * n:
            return ob
        me = bpy.data.meshes.new(FINS)
        me.from_pydata([(0.0, 0.0, 0.0)] * (_FIN_V * n), [], fin_faces(n))
        for poly in me.polygons:
            poly.use_smooth = True
        me.materials.append(liquid_material(fin=True))
        me.attributes.new("glint", "FLOAT", "POINT").data.foreach_set("value", fin_glints(n))
        if ob is None:
            ob = bpy.data.objects.new(FINS, me)
            self.coll.objects.link(ob)
        else:
            ob.data = me
        return ob

    def _push_hist(self, fr) -> None:
        """Recent positions (newest first, ~0.5 s): the fins trail the motion."""
        h = getattr(self, "hist", None)
        t, pos = float(fr.t), np.asarray(fr.pos, float)
        if h is None or not h or len(h[0][1]) != len(pos) or t < h[0][0]:
            h = []
        h.insert(0, (t, pos.copy()))
        while len(h) > 2 and t - h[-1][0] > 0.5:
            h.pop()
        self.hist = h[:96]

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

    def make_colony(self, n_nodes: int, style: int = 0) -> None:
        """Colony family: armour (hexagonal plates, or bony scutes for the Osseous line) and the prey."""
        faces = [(7 * i, 7 * i + 1 + j, 7 * i + 1 + (j + 1) % 6) for i in range(n_nodes) for j in range(6)]
        if style >= 3:                                     # Mimetic: no armour, the fins do it
            for name in (PLATES, SCUTES, PANELS):
                _drop(name)
            self.plates = None
        elif style == 2:                                   # Cyber: hex panels with a light ring
            _drop(PLATES)
            _drop(SCUTES)
            pf, pm = panel_faces(n_nodes)
            self.plates = _cyber_mesh(PANELS, _PANEL_V * n_nodes, pf, pm, self.coll)
        elif style == 1:
            _drop(PLATES)
            _drop(PANELS)
            self.plates = _replace_mesh_object(SCUTES, 7 * n_nodes, faces, bone_material(), self.coll)
        else:
            _drop(SCUTES)
            _drop(PANELS)
            self.plates = _replace_mesh_object(PLATES, 7 * n_nodes, faces, _simple_material(
                "MyrmexArmor", (0.028, 0.028, 0.032), 1.0, 0.28, 0.5), self.coll)
        self.armour_style = style
        lure = bpy.data.objects.get(LURE)
        if lure is None:
            import bmesh
            me = bpy.data.meshes.new(LURE)
            bm = bmesh.new()
            bmesh.ops.create_icosphere(bm, subdivisions=3, radius=1.0)
            bm.to_mesh(me)
            bm.free()
            for poly in me.polygons:
                poly.use_smooth = True
            mat = _simple_material("MyrmexPrey", (0.02, 0.02, 0.022), 0.0, 0.06, 1.0)
            b = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            if "Emission Color" in b.inputs:                  # a faint warm core: something alive to hunt
                b.inputs["Emission Color"].default_value = (1.0, 0.36, 0.12, 1.0)
                b.inputs["Emission Strength"].default_value = 0.6
            me.materials.append(mat)
            lure = bpy.data.objects.new(LURE, me)
            self.coll.objects.link(lure)
        lure.scale = (0.0, 0.0, 0.0)
        self.lure = lure

    def make_hive(self, n_particles: int, style: int | None = None) -> None:
        """Fourth organism: the nanomachine swarm (points instanced as tiny plates)."""
        style = getattr(self, "style", 0) if style is None else style
        ob = bpy.data.objects.get(SWARM)
        if ob is None or len(ob.data.vertices) != n_particles:
            me = bpy.data.meshes.new(SWARM)
            me.from_pydata([(0.0, 0.0, 0.0)] * n_particles, [], [])
            if ob is None:
                ob = bpy.data.objects.new(SWARM, me)
                self.coll.objects.link(ob)
            else:
                ob.data = me
        mod = ob.modifiers.get("Swarm") or ob.modifiers.new("Swarm", "NODES")
        want = flakes_nodes() if style in (3, 5) else swarm_nodes(cyber=style == 2)
        if mod.node_group is None or mod.node_group.name in ("MyrmexNanoSwarm", "MyrmexCyberSwarm",
                                                             "MyrmexLiquidFlakes"):
            mod.node_group = want                          # (keep a node group you chose yourself)
        self.swarm = ob

    def _apply_hive(self, fr) -> None:
        n = len(fr.particles)
        if getattr(self, "swarm", None) is None or len(self.swarm.data.vertices) != n:
            self.make_hive(n, int(getattr(fr, "style", 0)))
        me = self.swarm.data
        me.vertices.foreach_set("co", np.asarray(fr.particles, np.float32).ravel())
        me.update()

    def _apply_colony(self, fr) -> None:
        n = len(fr.pos)
        style = int(getattr(fr, "style", 0))
        per = _PANEL_V if style == 2 else 7
        if style >= 3:
            if getattr(self, "armour_style", -1) != style:
                self.make_colony(n, style)
        elif getattr(self, "plates", None) is None or len(self.plates.data.vertices) != per * n or \
                getattr(self, "armour_style", -1) != style:
            self.make_colony(n, style)
        if style >= 3:
            pass
        elif style == 2:
            pts = panel_points(fr.pos, fr.nrm, fr.plate, fr.radius, float(fr.heading))
            set_cyber_mesh(self.plates, pts, _flow(pts, fr.com, float(fr.heading)),
                           panel_light(getattr(fr, "light", None), n))
        else:
            pts = scute_points(fr.pos, fr.nrm, fr.plate, fr.radius, float(fr.heading)) if style == 1 else \
                plate_points(fr.pos, fr.nrm, fr.plate, fr.radius)
            me = self.plates.data
            me.vertices.foreach_set("co", pts.astype(np.float32).ravel())
            me.update()
        lx, ly, lz, lr = (float(v) for v in fr.lure)
        self.lure.location = (lx, ly, lz)
        self.lure.scale = (lr, lr, lr)

    def _apply_poly(self, fr) -> None:
        style = int(getattr(fr, "style", 0))
        per = {1: _BONE_V, 2: _RAIL_V}.get(style, _TEN_V if style >= 3 else 2)
        if self.lattice is None or getattr(self, "style", -1) != style or \
                len(self.lattice.data.vertices) != per * len(fr.links):
            self.make_polyalloy(len(fr.links), max(4, len(fr.obstacles)), style)
        up = fr.nrm if getattr(fr, "nrm", None) is not None else fr.pos - fr.pos.mean(0)
        if style >= 3:
            pts = tendon_points(fr.pos, fr.links, up, float(fr.t), float(fr.arousal), TENDON_THICK.get(style, 1.0))
            me = self.lattice.data
            me.vertices.foreach_set("co", pts.astype(np.float32).ravel())
            me.update()
            fins = self._fins(len(fr.pos))
            fp = fin_points(style, self.hist, up, fr.com, float(fr.heading), float(fr.t))
            fins.data.vertices.foreach_set("co", fp.astype(np.float32).ravel())
            fins.data.update()
        elif style == 2:
            pts = rail_points(fr.pos, fr.links, up, float(fr.t), float(fr.arousal))
            set_cyber_mesh(self.lattice, pts, _flow(pts, fr.com, float(fr.heading)),
                           rail_light(fr.links, getattr(fr, "light", None)))
        else:
            pts = bone_points(fr.pos, fr.links, up, float(fr.t), float(fr.arousal)) if style == 1 else \
                strut_points(fr.pos, fr.links, len(fr.links))
            me = self.lattice.data
            me.vertices.foreach_set("co", pts.astype(np.float32).ravel())
            me.update()
        for k, o in enumerate(self.obstacles):
            if k < len(fr.obstacles) and fr.obstacles[k, 3] > 0:
                o.location = fr.obstacles[k, :3]
                o.scale = (float(fr.obstacles[k, 3]),) * 3
            else:
                o.scale = (0.0, 0.0, 0.0)

    def apply(self, fr) -> None:
        n = len(fr.pos)
        self._ensure(n)
        self._push_hist(fr)
        if getattr(fr, "links", None) is not None:
            self._apply_poly(fr)
        if getattr(fr, "plate", None) is not None:
            self._apply_colony(fr)
        if getattr(fr, "particles", None) is not None:
            self._apply_hive(fr)
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
        if int(getattr(fr, "style", 0)) >= 3:
            set_liquid_shading(float(fr.t), float(fr.glow))
        if int(getattr(fr, "style", 0)) == 2:
            set_cyber_shading(float(fr.t), float(fr.glow), float(fr.arousal), float(getattr(fr, "scan", float("nan"))),
                              fr.com, float(fr.heading), self.mb.materials[0] if self.mb.materials else None)
        self._debug(fr, bool(fr.flags & 16))


def setup_creature_scene(scene: bpy.types.Scene | None = None, variant: str = "nanomaterial",
                         keep_look: bool = False) -> CreatureView:
    """Build (or reuse) the creature scene.  ``keep_look`` keeps an existing studio / materials as they are."""
    scene = scene or bpy.context.scene
    if not (keep_look and bpy.data.filepath):
        for name in ("Cube", "Light", "Camera"):              # Blender's default startup objects
            ob = bpy.data.objects.get(name)
            if ob is not None:
                bpy.data.objects.remove(ob, do_unlink=True)
    scene["myrmex_variant"] = variant
    view = CreatureView(scene)
    style, base = variant_style(variant)
    if base in ("polyalloy", "colony", "hive"):
        view.make_polyalloy((128 if base == "polyalloy" else 192) if style else (480 if base == "polyalloy" else 640),
                            4, style)
    if base in ("colony", "hive"):
        view.make_colony(128, style)
    if base == "hive":
        view.make_hive(1536, style)
    if not (keep_look and bpy.data.objects.get("MyrmexLightRig")):
        dark_studio(scene)
    if base in ("polyalloy", "colony", "hive"):
        crawler_terrain(style == 7, view.coll)             # (the studio floor comes back for the others)
    return view
