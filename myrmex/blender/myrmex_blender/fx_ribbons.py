"""Light ribbons: glowing strips trailing from the organism's extremities (slash arcs, light writing).

A few of the organism's nodes are followed (the ones farthest out when the organism is first seen, spread
over the body); their recent path becomes a strip facing the camera, widest at the node and thinning /
fading along its length, one colour per ribbon from the afterimage palette.  One mesh object, fixed
topology: every frame only its vertices move, so it is cheap live and exact in renders.
"""
from __future__ import annotations

import bpy
import numpy as np

from . import compat
from .fx_ghosts import _m, _n, _transparent, fx_collection, palette

NAME = "MyrmexRibbons"
MAT = "MyrmexRibbon"
K_MAX = 10
M = 24                                  # samples along a ribbon


def ribbon_material() -> bpy.types.Material:
    mat = bpy.data.materials.get(MAT)
    if mat is not None and mat.node_tree is not None and "MyrmexRibbonGain" in mat.node_tree.nodes:
        return mat
    mat = mat or bpy.data.materials.new(MAT)
    nt = compat.material_node_tree(mat)
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    gain = _n(nt, "ShaderNodeValue", -900, 300)
    gain.name = gain.label = "MyrmexRibbonGain"
    gain.outputs[0].default_value = 3.0
    age = _n(nt, "ShaderNodeAttribute", -900, 100)
    age.attribute_name = "age"
    side = _n(nt, "ShaderNodeAttribute", -900, -100)
    side.attribute_name = "side"
    col = _n(nt, "ShaderNodeAttribute", -900, -300)
    col.attribute_name = "col"
    along = _m(nt, "POWER", _m(nt, "SUBTRACT", 1.0, age.outputs["Fac"], x=-700, y=100), 1.5, x=-550, y=100)
    s2 = _m(nt, "MULTIPLY_ADD", side.outputs["Fac"], 2.0, x=-700, y=-100, c=-1.0)
    across = _m(nt, "SUBTRACT", 1.0, _m(nt, "POWER", _m(nt, "ABSOLUTE", s2, x=-550, y=-100), 2.0, x=-400, y=-100),
                x=-250, y=-100)
    strength = _m(nt, "MULTIPLY", _m(nt, "MULTIPLY", along, across, x=-250, y=100), gain.outputs[0], x=-100, y=100)
    em = _n(nt, "ShaderNodeEmission", 100, 0)
    nt.links.new(col.outputs["Color"], em.inputs["Color"])
    nt.links.new(strength, em.inputs["Strength"])
    tr = _n(nt, "ShaderNodeBsdfTransparent", 100, -200)
    add = _n(nt, "ShaderNodeAddShader", 300, 0)
    nt.links.new(tr.outputs[0], add.inputs[0])
    nt.links.new(em.outputs[0], add.inputs[1])
    out = _n(nt, "ShaderNodeOutputMaterial", 500, 0)
    nt.links.new(add.outputs[0], out.inputs["Surface"])
    _transparent(mat)
    mat.use_backface_culling = False
    return mat


def _faces(k: int) -> np.ndarray:
    f = []
    for r in range(k):
        b = r * M * 2
        for i in range(M - 1):
            a0, a1 = b + 2 * i, b + 2 * i + 1
            f.append((a0, a1, a1 + 2, a0 + 2))
    return np.array(f, np.int32)


class Ribbons:
    def __init__(self):
        self.idx: np.ndarray | None = None
        self.n = -1
        self.hist: np.ndarray | None = None               # (M, K, 3) newest first
        self.last_t = None
        self.k = 0

    def clear(self) -> None:
        ob = bpy.data.objects.get(NAME)
        if ob is not None and len(ob.data.vertices):
            ob.data.clear_geometry()
        self.hist = None
        self.idx = None
        self.k = 0

    def _pick(self, pos: np.ndarray, vis: np.ndarray, com, k: int) -> np.ndarray:
        """k nodes far out and far apart (farthest-point sampling over the outer half)."""
        cand = np.nonzero(vis)[0]
        if len(cand) == 0:
            return np.zeros(0, int)
        d = np.linalg.norm(pos[cand] - com, axis=1)
        outer = cand[d >= np.median(d)] if len(cand) > 2 * k else cand
        chosen = [int(outer[np.argmax(np.linalg.norm(pos[outer] - com, axis=1))])]
        dmin = np.linalg.norm(pos[outer] - pos[chosen[0]], axis=1)
        while len(chosen) < min(k, len(outer)):
            j = int(outer[np.argmax(dmin)])
            chosen.append(j)
            dmin = np.minimum(dmin, np.linalg.norm(pos[outer] - pos[j], axis=1))
        return np.array(chosen, int)

    def update(self, scene, t: float, pos, radius, com, size: float, rack: dict, cam_pos) -> None:
        amt = float(rack.get("ribbons", 0.0))
        if amt < 0.01 or pos is None or len(pos) == 0:
            if self.hist is not None:
                self.clear()
            self.last_t = t
            return
        pos = np.asarray(pos, float)
        vis = np.asarray(radius, float) > 0.01 if radius is not None else np.ones(len(pos), bool)
        k = int(round(3 + (K_MAX - 3) * min(1.0, amt)))
        jump = self.last_t is None or t < self.last_t - 1e-6 or t - self.last_t > 0.5
        self.last_t = t
        if self.idx is None or len(pos) != self.n or len(self.idx) != k or jump:
            self.n = len(pos)
            self.idx = self._pick(pos, vis, np.asarray(com, float), k)
            self.hist = None
        if len(self.idx) == 0:
            return
        cur = pos[self.idx]
        if self.hist is None or self.hist.shape[1] != len(self.idx):
            self.hist = np.repeat(cur[None], M, axis=0)
        else:
            self.hist = np.concatenate([cur[None], self.hist[:-1]], axis=0)
        self._write(scene, self.hist, size, amt, rack, cam_pos)

    def _write(self, scene, H: np.ndarray, size: float, amt: float, rack: dict, cam_pos) -> None:
        kk = H.shape[1]
        ob = bpy.data.objects.get(NAME)
        nv = kk * M * 2
        if ob is None or ob.type != "MESH" or len(ob.data.vertices) != nv:
            me = bpy.data.meshes.new(NAME)
            me.vertices.add(nv)
            fc = _faces(kk)
            me.loops.add(fc.size)
            me.polygons.add(len(fc))
            me.loops.foreach_set("vertex_index", fc.ravel())
            me.polygons.foreach_set("loop_start", np.arange(0, fc.size, 4, dtype=np.int32))
            me.update(calc_edges=True)
            age = np.repeat(np.tile(np.linspace(0.0, 1.0, M), kk), 2).astype(np.float32)
            side = np.tile(np.array([0.0, 1.0], np.float32), kk * M)
            me.attributes.new("age", "FLOAT", "POINT").data.foreach_set("value", age)
            me.attributes.new("side", "FLOAT", "POINT").data.foreach_set("value", side)
            me.attributes.new("col", "FLOAT_COLOR", "POINT")
            me.materials.append(ribbon_material())
            if ob is None:
                ob = bpy.data.objects.new(NAME, me)
                fx_collection(scene).objects.link(ob)
                ob.visible_shadow = False
            else:
                old = ob.data
                ob.data = me
                if old.users == 0:
                    bpy.data.meshes.remove(old)
            self.k = -1
        me = ob.data
        pal = palette(rack.get("palette", 0.0))
        key = (kk, tuple(pal))
        if self.k != key:                                      # colours only when the palette changes
            cols = np.array([(*pal[r % len(pal)], 1.0) for r in range(kk)], np.float32)
            me.attributes["col"].data.foreach_set("color", np.repeat(cols, M * 2, axis=0).ravel())
            self.k = key
        # strips facing the camera
        tang = np.empty_like(H)
        tang[1:-1] = H[:-2] - H[2:]
        tang[0] = H[0] - H[1]
        tang[-1] = H[-2] - H[-1]
        view = np.asarray(cam_pos, float)[None, None, :] - H
        side = np.cross(tang, view)
        ln = np.linalg.norm(side, axis=2, keepdims=True)
        side = np.where(ln > 1e-9, side / np.maximum(ln, 1e-9), 0.0)
        moving = np.linalg.norm(tang, axis=2, keepdims=True) > 1e-4
        w = max(float(size), 0.2) * (0.02 + 0.06 * amt) * (1.0 - np.linspace(0.0, 1.0, M)) ** 0.6
        half = side * (w[:, None, None] * 0.5) * moving
        V = np.empty((kk, M, 2, 3))
        V[:, :, 0] = (H - half).transpose(1, 0, 2)
        V[:, :, 1] = (H + half).transpose(1, 0, 2)
        me.vertices.foreach_set("co", V.astype(np.float32).ravel())
        me.update()
        nt = ob.active_material.node_tree if ob.active_material is not None else None
        if nt is not None and "MyrmexRibbonGain" in nt.nodes:
            nt.nodes["MyrmexRibbonGain"].outputs[0].default_value = 1.0 + 5.0 * amt


def drop() -> None:
    ob = bpy.data.objects.get(NAME)
    if ob is not None:
        me = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)


__all__ = ["Ribbons", "ribbon_material", "drop", "NAME"]
