"""Geometry for the Bionic line, pure numpy (shared by the engines and the Blender view).

Nothing here is a blob: struts and vessels are real tubes (tapered, sculpted), hubs are small
spheres, the fold is its own panels, the ferrofluid is a displaced sphere whose spikes are sharp
cones.  Every builder returns flat arrays (vertices, faces or a fixed topology) so the Blender side
only uploads coordinates each frame.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


# ---------------------------------------------------------------------- primitives
@lru_cache(maxsize=8)
def icosphere(subdiv: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere: (V, 3) vertices, (F, 3) faces (outward, counter-clockwise)."""
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
                  (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)], float)
    f = np.array([(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
                  (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5),
                  (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdiv):
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e.sort(axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.ravel()
        mid = v[uniq[:, 0]] + v[uniq[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        base = len(v)
        v = np.concatenate([v, mid])
        nf = len(f)
        ab, bc, ca = inv[:nf] + base, inv[nf:2 * nf] + base, inv[2 * nf:] + base
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        f = np.concatenate([np.stack([a, ab, ca], 1), np.stack([b, bc, ab], 1), np.stack([c, ca, bc], 1),
                            np.stack([ab, bc, ca], 1)])
    v.flags.writeable = False
    f.flags.writeable = False
    return v, f


def fib_sphere(n: int) -> np.ndarray:
    """n nearly even directions (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    th = math.pi * (1.0 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def frames(d: np.ndarray, ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each row of d (and to each other)."""
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    if ref is None:
        ref = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0]]))
    u = np.cross(d, ref)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return u, np.cross(d, u)


# ---------------------------------------------------------------------- tubes (struts, cables, vessels)
@lru_cache(maxsize=16)
def tube_topology(m: int, sides: int, rings: int = 2, caps: bool = True) -> np.ndarray:
    """Faces for m tubes of ``rings`` rings x ``sides`` vertices each (+ a centre vertex per end if caps)."""
    per = rings * sides + (2 if caps else 0)
    faces = []
    s = np.arange(sides)
    s1 = (s + 1) % sides
    for r in range(rings - 1):
        a, b = r * sides + s, r * sides + s1
        faces.append(np.stack([a, b, b + sides, a + sides], 1))
    quads = np.concatenate(faces) if faces else np.zeros((0, 4), int)
    tris = []
    if caps:
        c0, c1 = rings * sides, rings * sides + 1
        tris.append(np.stack([np.full(sides, c0), s1, s], 1))
        last = (rings - 1) * sides
        tris.append(np.stack([np.full(sides, c1), last + s, last + s1], 1))
    tri = np.concatenate(tris) if tris else np.zeros((0, 3), int)
    # quads as two triangles (one mesh type: triangles)
    qt = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]) if len(quads) else np.zeros((0, 3), int)
    one = np.concatenate([qt, tri])
    off = (np.arange(m) * per)[:, None, None]
    return (one[None, :, :] + off).reshape(-1, 3)


def tubes(p0: np.ndarray, p1: np.ndarray, r0: np.ndarray, r1: np.ndarray, sides: int = 8, rings: int = 2,
          profile=None, caps: bool = True, bow: np.ndarray | None = None) -> np.ndarray:
    """Vertices of m tapered tubes from p0 to p1 (radius r0 -> r1); ``profile(u)`` (rings,) multiplies the
    radius along the tube (sculpted struts); ``bow`` (m,) bends each tube sideways by that much at its middle
    (a buckled strut).  Topology: tube_topology(m, sides, rings, caps)."""
    m = len(p0)
    d = p1 - p0
    L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
    dn = d / L[:, None]
    u, w = frames(dn)
    ang = 2 * math.pi * np.arange(sides) / sides
    ca, sa = np.cos(ang), np.sin(ang)
    uu = np.linspace(0.0, 1.0, rings)
    prof = np.ones(rings) if profile is None else np.asarray(profile(uu), float)
    rad = (r0[:, None] + (r1 - r0)[:, None] * uu[None, :]) * prof[None, :]                  # (m, rings)
    centre = p0[:, None, :] + d[:, None, :] * uu[None, :, None]                             # (m, rings, 3)
    if bow is not None:
        centre = centre + (np.asarray(bow, float)[:, None] * np.sin(math.pi * uu)[None, :])[:, :, None] * w[:, None, :]
    ring = (ca[None, None, :, None] * u[:, None, None, :] + sa[None, None, :, None] * w[:, None, None, :])
    verts = centre[:, :, None, :] + rad[:, :, None, None] * ring                           # (m, rings, sides, 3)
    verts = verts.reshape(m, rings * sides, 3)
    if caps:
        verts = np.concatenate([verts, p0[:, None, :], p1[:, None, :]], 1)
    return verts.reshape(-1, 3)


@lru_cache(maxsize=16)
def tube_uv(m: int, sides: int, rings: int = 2, caps: bool = True) -> np.ndarray:
    """Per-vertex (u around, v along) of tube_topology's tubes (for a UV map: brushed metal, fades)."""
    uv = np.stack(np.meshgrid(np.arange(sides) / sides, np.linspace(0.0, 1.0, rings)), -1).reshape(-1, 2)
    if caps:
        uv = np.concatenate([uv, [[0.5, 0.0], [0.5, 1.0]]])
    return np.tile(uv, (m, 1))


def strut_profile(u: np.ndarray) -> np.ndarray:
    """A machined strut: swollen belly, necked collars near both ends, flared sockets."""
    return 0.72 + 0.38 * np.sin(np.pi * u) ** 0.8 - 0.22 * np.exp(-((u - 0.12) / 0.05) ** 2) \
        - 0.22 * np.exp(-((u - 0.88) / 0.05) ** 2) + 0.3 * np.exp(-(u / 0.035) ** 2) + 0.3 * np.exp(-((1 - u) / 0.035) ** 2)


def spheres(c: np.ndarray, r: np.ndarray, subdiv: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Vertices of len(c) small icospheres + their faces (fixed topology for a given count)."""
    v, _ = icosphere(subdiv)
    verts = c[:, None, :] + np.asarray(r, float)[:, None, None] * v[None, :, :]
    return verts.reshape(-1, 3), sphere_topology(len(c), subdiv)


@lru_cache(maxsize=16)
def sphere_topology(m: int, subdiv: int = 1) -> np.ndarray:
    v, f = icosphere(subdiv)
    return (f[None, :, :] + (np.arange(m) * len(v))[:, None, None]).reshape(-1, 3)


# ---------------------------------------------------------------------- ferrofluid
# extra layout (see ferro.py)
FERRO_S, FERRO_SAT, FERRO_GRID, FERRO_HEAD = 72, 6, 32, 32
FERRO_LEN = FERRO_HEAD + 8 * FERRO_S + 4 * FERRO_SAT + FERRO_GRID * FERRO_GRID


def ferro_base(n: np.ndarray, R: float, sc, tear: float = 0.0, k: float = 5.0, lobe: float = 0.0,
               phase: float = 0.0) -> np.ndarray:
    """The ferrofluid's base surface for unit directions n (body frame): a volume-true ellipsoid, a tongue
    drawn out towards +x, k rotating arms at the rim (a star in a rotating field)."""
    nx, ny, nz = n[:, 0], n[:, 1], n[:, 2]
    m = 1.0 + tear * np.maximum(nx, 0.0) ** 3
    if lobe > 1e-4:
        phi = np.arctan2(ny, nx)
        m = m * (1.0 + lobe * (1.0 - nz ** 2) ** 2 * np.cos(k * (phi - phase)))
    return (R * m)[:, None] * n * np.asarray(sc, float)[None, :]


def bilinear(grid: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample a periodic grid at u, v in [-1, 1]."""
    n = grid.shape[0]
    x = (u + 1.0) * 0.5 * n
    y = (v + 1.0) * 0.5 * n
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x0, y0 = x0 % n, y0 % n
    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
    return (grid[x0, y0] * (1 - fx) * (1 - fy) + grid[x1, y0] * fx * (1 - fy) + grid[x0, y1] * (1 - fx) * fy +
            grid[x1, y1] * fx * fy)


def ferro_unpack(extra: np.ndarray) -> dict:
    e = np.asarray(extra, float)
    S, NS, G, H = FERRO_S, FERRO_SAT, FERRO_GRID, FERRO_HEAD
    sp = e[H:H + 8 * S].reshape(S, 8)
    sat = e[H + 8 * S:H + 8 * S + 4 * NS].reshape(NS, 4)
    grid = e[H + 8 * S + 4 * NS:H + 8 * S + 4 * NS + G * G].reshape(G, G)
    return {"R": e[0], "sc": e[1:4], "tear": e[4], "k": e[5], "lobe": e[6], "phase": e[7], "ridge": e[8],
            "ripple": e[9], "ripple_phase": e[10], "sharp": e[11], "B": e[12], "kick": e[13], "n_spikes": int(e[14]),
            "n_sats": int(e[15]), "rot": e[16:25].reshape(3, 3), "magnet": e[25:28], "dipole": e[28:31],
            "strength": e[31], "dirs": sp[:, 0:3], "h": sp[:, 3], "w": sp[:, 4], "axis": sp[:, 5:8], "sats": sat,
            "grid": grid}


@lru_cache(maxsize=4)
def _n32(subdiv: int) -> np.ndarray:
    return icosphere(subdiv)[0].astype(np.float32)


def ferro_surface(extra: np.ndarray, centre: np.ndarray, subdiv: int = 5, world: bool = True):
    """Vertices (and the per-vertex spike mask, 0..1) of the ferrofluid body; faces: icosphere(subdiv)[1]."""
    F = ferro_unpack(extra)
    n = icosphere(subdiv)[0]
    R, sc = max(F["R"], 1e-3), F["sc"]
    base = ferro_base(n, R, sc, F["tear"], F["k"], F["lobe"], F["phase"])
    nb = n / np.maximum(np.asarray(sc), 1e-3)[None, :]
    nb /= np.linalg.norm(nb, axis=1, keepdims=True)
    if F["ridge"] > 1e-4:                                                       # the labyrinth on the disc
        u = base[:, 0] / (1.1 * R * max(sc[0], 1e-3))
        v = base[:, 1] / (1.1 * R * max(sc[1], 1e-3))
        top = np.clip((nb[:, 2] + 0.15) / 0.5, 0.0, 1.0)
        rg = np.maximum(bilinear(F["grid"], np.clip(u, -1, 1), np.clip(v, -1, 1)), 0.0) ** 1.4
        base = base + nb * (F["ridge"] * rg * top)[:, None]
    if F["ripple"] > 1e-5:                                                      # capillary ripples (the highs)
        rp = np.sin(11.0 * n[:, 0] + 7.0 * n[:, 1] - F["ripple_phase"]) * np.sin(9.0 * n[:, 2] + 0.7 * F["ripple_phase"])
        base = base + nb * (F["ripple"] * R * rp)[:, None]
    mask = np.zeros(len(n))
    live = F["h"] > 1e-4
    if live.any():
        d, h, w, ax = F["dirs"][live], F["h"][live], np.maximum(F["w"][live], 1e-3), F["axis"][live]
        C = _n32(subdiv) @ d.T.astype(np.float32)                               # (V, S) cosines
        vi, si = np.nonzero(C > np.float32(math.cos(min(float(w.max()), 3.0))))  # only the caps under a spike
        t = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - C[vi, si].astype(float)))) / w[si]
        k = t < 1.0
        vi, si, t = vi[k], si[k], t[k]
        prof = (1.0 - t) ** max(F["sharp"], 1.0)                                 # a cone with concave flanks
        disp = np.zeros((len(n), 3))
        np.add.at(disp, vi, (prof * h[si])[:, None] * ax[si])
        base = base + disp
        np.maximum.at(mask, vi, prof)
        tip = np.argmax(C, axis=0)                                              # needle points: the nearest vertex
        base[tip] = ferro_base(d, R, sc, F["tear"], F["k"], F["lobe"], F["phase"]) + h[:, None] * ax  # is the tip
        mask[tip] = 1.0
    if not world:
        return base, mask
    return np.asarray(centre, float)[None, :] + base @ F["rot"].T, mask


def filings(extra: np.ndarray, centre: np.ndarray, seeds: np.ndarray, t: float, length: float = 0.07):
    """Iron filings round the body: short needles on fixed seeds (body frame, in body radii), each lying
    along the local field of the invisible dipole and drifting along its field line.
    Returns (2 * len(seeds), 3) needle end points (world) and a (len(seeds),) field strength 0..1."""
    F = ferro_unpack(extra)
    R = max(F["R"], 1e-3)
    rot = F["rot"]
    m_pos = F["magnet"]
    m = F["dipole"] / max(float(np.linalg.norm(F["dipole"])), 1e-9)
    p = np.asarray(centre, float)[None, :] + (seeds * R) @ rot.T
    r = p - m_pos
    d = np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 0.25 * R)
    rh = r / d
    B = (3.0 * (rh @ m)[:, None] * rh - m[None, :]) / (d / R) ** 3                 # dipole field (units of R)
    Bm = np.linalg.norm(B, axis=1, keepdims=True)
    b = B / np.maximum(Bm, 1e-9)
    s = np.clip(Bm[:, 0] / 3.0, 0.0, 1.0) * min(1.5, max(F["B"], 0.0))
    drift = ((t * 0.35 + seeds[:, 0] * 3.1) % 1.0 - 0.5)[:, None] * 0.6 * R        # sliding along the line
    c = p + b * drift
    half = (length * (0.4 + 0.6 * s))[:, None] * b * 0.5
    return np.stack([c - half, c + half], 1).reshape(-1, 3), s


__all__ = ["icosphere", "fib_sphere", "frames", "tube_topology", "tube_uv", "tubes", "strut_profile", "spheres",
           "sphere_topology", "ferro_base", "Part", "parts", "PARTS",
           "ferro_surface", "ferro_unpack", "filings", "bilinear", "FERRO_S", "FERRO_SAT", "FERRO_GRID", "FERRO_HEAD",
           "FERRO_LEN"]


# ---------------------------------------------------------------------- per organism: the parts to draw
class Part:
    """One drawable piece: vertices (world), a fixed topology, per-vertex attributes, a UV map (optional)."""

    __slots__ = ("verts", "faces", "attrs", "uv")

    def __init__(self, verts: np.ndarray, faces: np.ndarray, attrs: dict | None = None, uv: np.ndarray | None = None):
        self.verts, self.faces, self.attrs, self.uv = verts, faces, attrs or {}, uv


def _collapse(valid: np.ndarray, p: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    return np.where(valid[:, None], p, anchor[None, :])


def tensor_parts(pos: np.ndarray, radius: np.ndarray, members: np.ndarray, extra) -> dict:
    M = members
    st = (M[:, 0] >= 0) & (M[:, 2] == 0)
    cb = (M[:, 0] >= 0) & (M[:, 2] > 0)
    i, j = M[st, 0].astype(int), M[st, 1].astype(int)
    r = M[st, 3] * 0.6
    sv = tubes(pos[i], pos[j], r, r, 8, 9, strut_profile)
    ns, nc = int(st.sum()), int(cb.sum())
    per_s = 8 * 9 + 2
    ci, cj = M[cb, 0].astype(int), M[cb, 1].astype(int)
    rc = M[cb, 3]
    cv = tubes(pos[ci], pos[cj], rc, rc, 4, 2, caps=False)
    hv, hf = spheres(pos, radius * 0.7, 1)
    return {"struts": Part(sv, tube_topology(ns, 8, 9, True), {"stress": np.repeat(M[st, 4], per_s)},
                           tube_uv(ns, 8, 9, True)),
            "cables": Part(cv, tube_topology(nc, 4, 2, False), {"stress": np.repeat(M[cb, 4], 8),
                                                               "act": np.repeat(M[cb, 5], 8)}),
            "hubs": Part(hv, hf)}


@lru_cache(maxsize=4)
def _fold_panels(R: int, C: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ij = lambda k, m: k * (C + 1) + m                                          # noqa: E731
    P = np.array([(ij(k, m), ij(k, m + 1), ij(k + 1, m + 1), ij(k + 1, m)) for k in range(R) for m in range(C)])
    faces = np.arange(4 * len(P)).reshape(-1, 4)
    uv = np.tile(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), (len(P), 1))
    rows = np.repeat([k / max(R - 1, 1) for k in range(R) for m in range(C)], 4)
    return P, faces, uv, rows


def fold_parts(pos: np.ndarray, radius: np.ndarray, members: np.ndarray, extra) -> dict:
    R, C = int(round(extra[0])), int(round(extra[1]))
    P, faces, uv, rows = _fold_panels(R, C)
    return {"panels": Part(pos[P].reshape(-1, 3), faces, {"row": rows, "clap": np.full(len(rows), float(extra[3]))},
                           uv)}


def arbor_parts(pos: np.ndarray, radius: np.ndarray, members: np.ndarray, extra) -> dict:
    M = members
    n = len(M)
    valid = M[:, 0] >= 0
    par = np.where(valid, M[:, 0], 0).astype(int)
    kid = np.arange(n)
    rc = np.where(valid, M[:, 3], 0.0)
    rp = np.minimum(radius[par], 1.25 * rc)
    p0 = _collapse(valid, pos[par], pos[0])
    p1 = _collapse(valid, pos[kid], pos[0])
    sides, rings = 7, 3
    v = tubes(p0, p1, rp, rc, sides, rings, caps=False)
    g_c, g_p = M[:, 4].clip(0.0), np.where(valid, M[par, 4], 0.0).clip(0.0)
    d_c, d_p = M[:, 5].clip(0.0), np.where(valid & (M[par, 0] >= 0), M[par, 5], 0.0).clip(0.0)
    uu = np.linspace(0.0, 1.0, rings)
    glow = (g_p[:, None] + (g_c - g_p)[:, None] * uu[None, :])
    dist = (d_p[:, None] + (d_c - d_p)[:, None] * uu[None, :])
    shed = np.repeat(np.where(valid, M[:, 2], 0.0), sides * rings)
    hub_r = np.where(radius > 0, radius * 1.02, 0.0)
    node_g = np.where(valid, M[:, 4], 0.5).clip(0.0)                             # the heart: half lit
    node_d = np.where(valid, M[:, 5], 0.0).clip(0.0)
    hv, hf = spheres(pos, hub_r, 1)
    per_h = len(icosphere(1)[0])
    return {"vessels": Part(v, tube_topology(n, sides, rings, False),
                            {"glow": np.repeat(glow, sides, axis=1).ravel(), "dist": np.repeat(dist, sides, axis=1).ravel(),
                             "shed": shed}),
            "nodes": Part(hv, hf, {"glow": np.repeat(node_g, per_h), "dist": np.repeat(node_d, per_h),
                                   "shed": np.zeros(len(hv))})}


def ferro_parts(pos: np.ndarray, radius: np.ndarray, members: np.ndarray, extra, subdiv: int = 5,
                seeds: np.ndarray | None = None, t: float = 0.0) -> dict:
    v, mask = ferro_surface(extra, pos[0], subdiv)
    F = ferro_unpack(extra)
    sat = F["sats"]
    dv, df = spheres(sat[:, :3], sat[:, 3], 2)
    out = {"ferro": Part(v, icosphere(subdiv)[1], {"spike": mask}), "droplets": Part(dv, df)}
    if seeds is not None:
        R = max(float(F["R"]), 1e-3)
        ends, strength = filings(extra, pos[0], seeds, t, 0.16 * R / 0.5)
        a, b = ends[0::2], ends[1::2]
        rr = np.full(len(seeds), 0.006 * R / 0.5)
        fv = tubes(a, b, rr, rr * 0.35, 3, 2, caps=False)
        out["filings"] = Part(fv, tube_topology(len(seeds), 3, 2, False), {"field": np.repeat(strength, 6)})
    return out


def truss_parts(pos: np.ndarray, radius: np.ndarray, members: np.ndarray, extra) -> dict:
    M = members
    m = len(M)
    valid = M[:, 0] >= 0
    i = np.where(valid, M[:, 0], 0).astype(int)
    j = np.where(valid, M[:, 1], 0).astype(int)
    r = np.where(valid, M[:, 3], 0.0)
    p0, p1 = _collapse(valid, pos[i], pos[0]), _collapse(valid, pos[j], pos[0])
    sides, rings = 8, 7
    v = tubes(p0, p1, r, r, sides, rings, strut_profile, bow=np.where(valid, M[:, 5], 0.0))
    per = sides * rings + 2
    n = len(pos)
    tdir = np.asarray(extra[0:3], float)
    thrust = np.asarray(extra[7:7 + n], float)
    hit = np.asarray(extra[7 + n:7 + 2 * n], float)
    hv, hf = spheres(pos, radius, 1)
    per_h = len(icosphere(1)[0])
    s = float(np.median(radius)) / 0.012 if len(radius) else 1.0
    on = thrust > 0.05
    L = 0.32 * np.sqrt(np.clip(thrust, 0.0, 3.0)) * min(max(s, 0.5), 2.0) * 0.5
    cr = np.where(on, 0.05 * np.sqrt(np.clip(thrust, 0.0, 3.0)) * min(max(s, 0.5), 2.0) * 0.5, 0.0)
    tip = pos - tdir[None, :] * np.where(on, L, 0.0)[:, None]
    cv = tubes(pos, tip, cr, cr * 0.05, 8, 3, caps=False)
    return {"struts": Part(v, tube_topology(m, sides, rings, True),
                           {"stress": np.repeat(np.where(valid, M[:, 4], 0.0), per),
                            "dying": np.repeat(np.where(valid, M[:, 2], 1.0), per)}, tube_uv(m, sides, rings, True)),
            "hubs": Part(hv, hf, {"thrust": np.repeat(thrust, per_h), "hit": np.repeat(hit, per_h)}),
            "plumes": Part(cv, tube_topology(n, 8, 3, False), {"thrust": np.repeat(thrust, 24)},
                           tube_uv(n, 8, 3, False))}


PARTS = {0: tensor_parts, 1: fold_parts, 2: arbor_parts, 3: ferro_parts, 4: truss_parts}


def parts(kind: int, pos: np.ndarray, radius: np.ndarray, members: np.ndarray, extra, **kw) -> dict:
    """Everything to draw for a Bionic organism of this kind (0 tensor .. 4 truss)."""
    fn = PARTS[int(kind)]
    pos, radius = np.asarray(pos, float), np.asarray(radius, float)
    members = np.asarray(members, float) if members is not None else np.zeros((0, 6))
    extra = np.asarray(extra, float) if extra is not None else np.zeros(0)
    return fn(pos, radius, members, extra, **kw) if kind == 3 else fn(pos, radius, members, extra)
