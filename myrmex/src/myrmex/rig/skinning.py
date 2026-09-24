"""Robust skin weights for dense generated meshes.

Blender's heat-diffusion "Automatic Weights" frequently fails on Hunyuan3D
meshes (dense, self-touching marching-cubes surfaces).  This module computes
weights on the analysis proxy instead:

1. every bone may only *seed* from vertices of its own limb (arm bones from
   arm arcs, leg bones from leg arcs + the fused region under a skirt on their
   side, trunk bones from the core) – so a hanging arm never grabs the ribs;
2. unambiguous vertices close to a bone become seeds;
3. a geodesic Voronoi sweep over the surface labels every vertex;
4. diffusion smoothing turns labels into blended weights, with a wider blend
   inside fused garment regions (skirts) where cloth-like stretching is wanted.

Weights are then transferred to the working mesh by nearest-neighbour
interpolation (see ``myrmex_blender.skin``).
"""
from __future__ import annotations

import math

import numpy as np

from . import meshgraph as mg
from .morphology import Morphology
from .rigdesc import RigDescription


def segment_distances(P: np.ndarray, heads: np.ndarray, tails: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distances (V, B) from points to bone segments and projection parameters t (V, B)."""
    ab = tails - heads                                   # (B, 3)
    L2 = np.maximum((ab * ab).sum(axis=1), 1e-12)        # (B,)
    ap = P[:, None, :] - heads[None, :, :]               # (V, B, 3)
    t = (ap * ab[None]).sum(axis=2) / L2[None]
    tc = np.clip(t, 0.0, 1.0)
    closest = heads[None] + tc[..., None] * ab[None]
    d = np.linalg.norm(P[:, None, :] - closest, axis=2)
    return d, t


def _limb_membership(m: Morphology) -> dict[str, np.ndarray]:
    out = {}
    for l in m.limbs:
        mask = np.zeros(len(m.verts), bool)
        for a in l.arcs:
            if a in m.arcs:
                mask[m.arcs[a].verts] = True
        out[l.name] = mask
    return out


def compute_weights(m: Morphology, rd: RigDescription, *, blend: float = 0.035,
                    garment_blend: float = 0.09, max_influences: int = 4,
                    rigid: bool = False) -> tuple[list[str], np.ndarray]:
    """Return (deform bone names, weights (V, B)) on the morphology's proxy mesh."""
    V = m.verts
    H = m.height
    bones = [b for b in rd.bones if b.deform]
    names = [b.name for b in bones]
    heads = np.array([b.head for b in bones], dtype=float)
    tails = np.array([b.tail for b in bones], dtype=float)
    d, t = segment_distances(V, heads, tails)

    limb_mask = _limb_membership(m)
    in_limb = np.zeros(len(V), bool)
    for msk in limb_mask.values():
        in_limb |= msk
    fwd = np.asarray(rd.forward, dtype=float)
    left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    hip_z = float(rd.params.get("hip_height", 0.5 * H))

    # Which vertices may each bone seed from?
    allowed = np.zeros((len(V), len(bones)), bool)
    for j, b in enumerate(bones):
        if b.limb and b.limb in limb_mask:
            ok = limb_mask[b.limb].copy()
            if b.role == "thigh":
                # Under skirts the upper thigh lives inside the fused core region.
                side = 1.0 if b.side == "L" else -1.0
                lat = V @ left
                core_low = (~in_limb) & (V[:, 2] < hip_z) & (side * lat > 0.01 * H)
                ok |= core_low
            allowed[:, j] = ok
        elif b.role in ("head", "neck"):
            head_mask = np.zeros(len(V), bool)
            for l in m.limbs:
                if l.role == "head":
                    head_mask |= limb_mask[l.name]
            allowed[:, j] = head_mask | (~in_limb)
        else:
            allowed[:, j] = ~in_limb
    # Every vertex needs at least one allowed bone.
    none = ~allowed.any(axis=1)
    allowed[none] = True

    dm = np.where(allowed, d, np.inf)
    order = np.argsort(dm, axis=1)
    nearest = order[:, 0]
    d1 = dm[np.arange(len(V)), nearest]
    d2 = dm[np.arange(len(V)), order[:, 1]] if len(bones) > 1 else np.full(len(V), np.inf)
    tn = t[np.arange(len(V)), nearest]
    unambiguous = (d1 < 0.72 * d2) & (tn > -0.05) & (tn < 1.08)
    seeds = [np.nonzero(unambiguous & (nearest == j))[0].tolist() for j in range(len(bones))]
    for j, s in enumerate(seeds):
        if not s:
            # Fallback: the closest allowed vertex seeds this bone.
            cand = np.where(allowed[:, j], d[:, j], np.inf)
            seeds[j] = [int(np.argmin(cand))]

    g = m.graph
    labels, _ = mg.dijkstra_labels(g, seeds)
    W = np.zeros((len(V), len(bones)))
    W[np.arange(len(V)), labels] = 1.0

    if not rigid:
        e = max(g.mean_edge, 1e-6)
        k = int(min(400, max(4, 0.5 * (blend / e) ** 2)))
        W = mg.laplacian_smooth_values(g, W, k, alpha=0.5)
        # Extra blending inside fused garments: core vertices below the hips.
        garment = (~in_limb) & (V[:, 2] < hip_z + 0.02 * H) & (V[:, 2] > 0.1 * H)
        if garment.any() and garment_blend > blend:
            k2 = int(min(600, max(4, 0.5 * (garment_blend / e) ** 2)))
            Wg = mg.laplacian_smooth_values(g, W, k2, alpha=0.5)
            W[garment] = Wg[garment]

    W = prune_weights(W, max_influences)
    return names, W


def prune_weights(W: np.ndarray, max_influences: int = 4, min_weight: float = 0.01) -> np.ndarray:
    W = np.array(W, dtype=float, copy=True)
    if W.shape[1] > max_influences:
        idx = np.argsort(-W, axis=1)[:, max_influences:]
        np.put_along_axis(W, idx, 0.0, axis=1)
    W[W < min_weight] = 0.0
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return W / s


def transfer_weights(src_verts: np.ndarray, W: np.ndarray, dst_verts: np.ndarray,
                     k: int = 4, chunk: int = 20000) -> np.ndarray:
    """Pure-numpy k-NN inverse-distance transfer (used when no KD-tree is available)."""
    out = np.zeros((len(dst_verts), W.shape[1]))
    # Coarse grid acceleration.
    cell = max(float(np.ptp(src_verts, axis=0).max()) / 64.0, 1e-6)
    keys = np.floor(src_verts / cell).astype(np.int64)
    table: dict[tuple[int, int, int], list[int]] = {}
    for i, kk in enumerate(map(tuple, keys)):
        table.setdefault(kk, []).append(i)
    for s in range(0, len(dst_verts), chunk):
        P = dst_verts[s:s + chunk]
        for r, p in enumerate(P):
            base = np.floor(p / cell).astype(np.int64)
            cand: list[int] = []
            rad = 1
            while len(cand) < k and rad < 6:
                cand = []
                for dx in range(-rad, rad + 1):
                    for dy in range(-rad, rad + 1):
                        for dz in range(-rad, rad + 1):
                            cand.extend(table.get((base[0] + dx, base[1] + dy, base[2] + dz), ()))
                rad += 1
            cand_a = np.array(cand)
            dd = np.linalg.norm(src_verts[cand_a] - p, axis=1)
            sel = np.argsort(dd)[:k]
            w = 1.0 / (dd[sel] + 1e-6) ** 2
            out[s + r] = (W[cand_a[sel]] * w[:, None]).sum(axis=0) / w.sum()
    return prune_weights(out)


def joint_mask(W: np.ndarray, threshold: float = 0.97) -> np.ndarray:
    """Per-vertex [0,1] mask of blended regions (for Corrective Smooth vertex groups)."""
    mx = W.max(axis=1)
    return np.clip((threshold - mx) / max(threshold - 0.5, 1e-6), 0.0, 1.0) ** 0.5
