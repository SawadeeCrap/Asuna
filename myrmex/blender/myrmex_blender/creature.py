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


def swarm_nodes() -> bpy.types.NodeTree:
    """Loose points -> tiny hexagonal machine plates, randomly turned and sized (the nanomachine swarm)."""
    ng = bpy.data.node_groups.get("MyrmexNanoSwarm")
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new("MyrmexNanoSwarm", "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    gi, go = _node(ng, "NodeGroupInput", (-600, 0)), _node(ng, "NodeGroupOutput", (500, 0))
    cyl = _node(ng, "GeometryNodeMeshCylinder", (-600, -250))
    cyl.inputs["Vertices"].default_value = 6
    cyl.inputs["Radius"].default_value = 0.012
    cyl.inputs["Depth"].default_value = 0.003
    sm = _node(ng, "GeometryNodeSetMaterial", (-350, -250))
    sm.inputs["Material"].default_value = _simple_material("MyrmexMicroPlate", (0.02, 0.02, 0.022), 1.0, 0.22, 0.4)
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
        pm = nanomaterial(POLY_MAT, poly=True)
        if not self.mb.materials:
            self.mb.materials.append(pm)
        elif self.mb.materials[0] is None or self.mb.materials[0].name == MAT:   # keep a material you chose
            self.mb.materials[0] = pm
        if style == 1:                                     # Osseous: articulated bone links
            _drop(LATTICE)
            self.lattice = _replace_mesh_object(BONES, n_links * _BONE_V, bone_faces(n_links), bone_material(),
                                                self.coll)
        else:                                              # classic: strut tubes (Geometry Nodes)
            _drop(BONES)
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
        if style == 1:
            _drop(PLATES)
            self.plates = _replace_mesh_object(SCUTES, 7 * n_nodes, faces, bone_material(), self.coll)
        else:
            _drop(SCUTES)
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

    def make_hive(self, n_particles: int) -> None:
        """Fourth organism: the nanomachine swarm (points instanced as tiny plates)."""
        ob = bpy.data.objects.get(SWARM)
        if ob is None or len(ob.data.vertices) != n_particles:
            me = bpy.data.meshes.new(SWARM)
            me.from_pydata([(0.0, 0.0, 0.0)] * n_particles, [], [])
            if ob is None:
                ob = bpy.data.objects.new(SWARM, me)
                self.coll.objects.link(ob)
            else:
                ob.data = me
        if "Swarm" not in ob.modifiers:
            ob.modifiers.new("Swarm", "NODES").node_group = swarm_nodes()
        self.swarm = ob

    def _apply_hive(self, fr) -> None:
        n = len(fr.particles)
        if getattr(self, "swarm", None) is None or len(self.swarm.data.vertices) != n:
            self.make_hive(n)
        me = self.swarm.data
        me.vertices.foreach_set("co", np.asarray(fr.particles, np.float32).ravel())
        me.update()

    def _apply_colony(self, fr) -> None:
        n = len(fr.pos)
        style = int(getattr(fr, "style", 0))
        if getattr(self, "plates", None) is None or len(self.plates.data.vertices) != 7 * n or \
                getattr(self, "armour_style", -1) != style:
            self.make_colony(n, style)
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
        per = _BONE_V if style == 1 else 2
        if self.lattice is None or getattr(self, "style", -1) != style or \
                len(self.lattice.data.vertices) != per * len(fr.links):
            self.make_polyalloy(len(fr.links), max(4, len(fr.obstacles)), style)
        if style == 1:
            up = fr.nrm if getattr(fr, "nrm", None) is not None else fr.pos - fr.pos.mean(0)
            pts = bone_points(fr.pos, fr.links, up, float(fr.t), float(fr.arousal))
        else:
            pts = strut_points(fr.pos, fr.links, len(fr.links))
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
    style = 1 if variant.startswith("osseous") else 0
    base = variant.replace("osseous_", "").replace("osseous", "polyalloy")
    if base in ("polyalloy", "colony", "hive"):
        view.make_polyalloy((128 if base == "polyalloy" else 192) if style else (480 if base == "polyalloy" else 640),
                            4, style)
    if base in ("colony", "hive"):
        view.make_colony(128, style)
    if base == "hive":
        view.make_hive(1536)
    if not (keep_look and bpy.data.objects.get("MyrmexLightRig")):
        dark_studio(scene)
    return view
