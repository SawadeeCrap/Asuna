"""Morphology analysis: find the body plan of an arbitrary generated mesh.

Method (topological, works for humans, spiders, insects, robots, tails ...):

1. geodesic *centre* of the surface: minimum of the approximate average
   geodesic distance (AGD) computed from farthest-point samples;
2. f(v) = geodesic distance from that centre;
3. superlevel-set **merge tree** of f with persistence simplification
   (union-find sweep from the extremities inwards).  Leaves are extremity
   tips (feet, hands, head top, tail tip, antenna tips), internal nodes are
   junctions where limbs join (ankle, crotch, shoulder, coxa ...);
4. every surviving arc is sliced into level sets of f; slice centroids form
   the arc's centreline and slice spreads give its radius;
5. arcs are grouped into *limbs* (tip -> attachment), classified by role
   (ground-contact legs, head, arms, tail, appendages).

The result is a :class:`Morphology` that the rig fitters turn into bones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import meshgraph as mg


@dataclass
class Arc:
    id: int
    start: int                    # vertex where the arc is born (tip or junction)
    start_f: float
    end: int | None = None        # vertex where it merges into its parent
    end_f: float | None = None
    children: list[int] = field(default_factory=list)
    parent: int | None = None
    verts: np.ndarray | None = None
    centerline: np.ndarray | None = None   # (K, 3) ordered from start (tip side) to end
    radii: np.ndarray | None = None        # (K,)
    slice_f: np.ndarray | None = None      # (K,)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def length(self) -> float:
        if self.centerline is None or len(self.centerline) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).sum())

    @property
    def radius(self) -> float:
        if self.radii is None or len(self.radii) == 0:
            return 0.0
        return float(np.median(self.radii))


@dataclass
class Limb:
    """A chain of arcs from an extremity tip to where it attaches to the body."""
    name: str
    role: str                       # leg | arm | head | tail | antenna | appendage
    arcs: list[int]                 # tip-first
    tip: np.ndarray
    tips: list[np.ndarray]          # all extremity tips merged into this limb (toe, heel ...)
    centerline: np.ndarray          # tip -> attachment
    radii: np.ndarray
    attach_vertex: int
    side: str = "C"                 # L / R / C
    ground_contact: bool = False

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.centerline, axis=0), axis=1).sum())


@dataclass
class Morphology:
    verts: np.ndarray
    faces: np.ndarray
    graph: mg.MeshGraph
    height: float
    center_vertex: int
    f: np.ndarray
    arcs: dict[int, Arc]
    root_arc: int
    vert_arc: np.ndarray
    limbs: list[Limb] = field(default_factory=list)
    body_plan: str = "unknown"      # biped | quadruped | hexapod | octopod | multileg | legless
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    notes: list[str] = field(default_factory=list)

    def limbs_by_role(self, role: str) -> list[Limb]:
        return [l for l in self.limbs if l.role == role]

    def summary(self) -> dict:
        return {
            "body_plan": self.body_plan,
            "height": round(self.height, 4),
            "vertices": int(self.verts.shape[0]),
            "arcs": len(self.arcs),
            "limbs": [
                {"name": l.name, "role": l.role, "side": l.side, "length": round(l.length, 4),
                 "tip": [round(float(x), 4) for x in l.tip], "ground": l.ground_contact,
                 "tips": len(l.tips)}
                for l in self.limbs
            ],
            "notes": self.notes,
        }


# --------------------------------------------------------------------------- merge tree

def _merge_tree(g: mg.MeshGraph, f: np.ndarray, tau: float) -> tuple[dict[int, Arc], np.ndarray, int]:
    nbrs, _ = g.adjacency_lists()
    order = np.argsort(-f, kind="stable")
    uf = mg.UnionFind(g.n)
    processed = [False] * g.n
    comp: dict[int, dict] = {}
    arcs: dict[int, Arc] = {}
    vert_arc = np.full(g.n, -1, dtype=np.int64)
    absorbed: dict[int, int] = {}
    fl = f.tolist()

    def new_arc(v: int, children: list[int] | None = None) -> int:
        aid = len(arcs)
        arcs[aid] = Arc(aid, v, fl[v], children=list(children or []))
        return aid

    for v in order.tolist():
        roots = {uf.find(u) for u in nbrs[v] if processed[u]}
        processed[v] = True
        if not roots:
            aid = new_arc(v)
            comp[v] = {"arc": aid, "birth": fl[v]}
            vert_arc[v] = aid
            continue
        if len(roots) == 1:
            r = roots.pop()
            info = comp.pop(r)
            nr = uf.union(r, v)
            comp[nr] = info
            vert_arc[v] = info["arc"]
            continue
        infos = [comp.pop(r) for r in roots]
        infos.sort(key=lambda i: i["birth"], reverse=True)
        sig = [i for i in infos if i["birth"] - fl[v] >= tau]
        nr = v
        for r in roots:
            nr = uf.union(nr, r)
        birth = max(i["birth"] for i in infos)
        if len(sig) <= 1:
            keep = sig[0] if sig else infos[0]
            for i in infos:
                if i is not keep:
                    absorbed[i["arc"]] = keep["arc"]
            comp[nr] = {"arc": keep["arc"], "birth": birth}
            vert_arc[v] = keep["arc"]
        else:
            for i in infos:
                if i not in sig:
                    absorbed[i["arc"]] = sig[0]["arc"]
            child_ids = [i["arc"] for i in sig]
            for c in child_ids:
                arcs[c].end = v
                arcs[c].end_f = fl[v]
            aid = new_arc(v, child_ids)
            for c in child_ids:
                arcs[c].parent = aid
            comp[nr] = {"arc": aid, "birth": birth}
            vert_arc[v] = aid

    def resolve(a: int) -> int:
        while a in absorbed:
            a = absorbed[a]
        return a

    vert_arc = np.array([resolve(int(a)) for a in vert_arc.tolist()], dtype=np.int64)
    alive = set(vert_arc.tolist())
    arcs = {k: a for k, a in arcs.items() if k in alive}
    for a in arcs.values():
        a.children = [c for c in a.children if c in arcs]
    roots = [a.id for a in arcs.values() if a.parent is None]
    root = max(roots, key=lambda k: int((vert_arc == k).sum()))
    return arcs, vert_arc, root


def _slice_arc(arc: Arc, verts: np.ndarray, f: np.ndarray, step: float) -> None:
    vs = arc.verts
    if vs is None or len(vs) == 0:
        arc.centerline = np.zeros((0, 3))
        arc.radii = np.zeros(0)
        arc.slice_f = np.zeros(0)
        return
    fv = f[vs]
    hi = arc.start_f
    lo = arc.end_f if arc.end_f is not None else float(fv.min())
    n = max(1, int(math.ceil((hi - lo) / step)))
    edges = np.linspace(hi, lo, n + 1)
    pts, rad, fs = [], [], []
    for k in range(n):
        a, b = edges[k + 1], edges[k]
        m = (fv >= a) & (fv <= b) if k == 0 else (fv >= a) & (fv < b)
        if m.sum() < 3:
            continue
        P = verts[vs[m]]
        c = P.mean(axis=0)
        pts.append(c)
        rad.append(float(np.linalg.norm(P - c, axis=1).mean()))
        fs.append(0.5 * (a + b))
    if not pts:
        c = verts[vs].mean(axis=0)
        pts, rad, fs = [c], [float(np.linalg.norm(verts[vs] - c, axis=1).mean())], [hi]
    arc.centerline = np.array(pts)
    arc.radii = np.array(rad)
    arc.slice_f = np.array(fs)


# --------------------------------------------------------------------------- public

def analyze(verts: np.ndarray, faces: np.ndarray, *, persistence: float = 0.045,
            fps_samples: int = 12, min_slenderness: float = 2.2,
            ground_fraction: float = 0.04) -> Morphology:
    """Analyse a (proxy) mesh in normalised space (Z up, ground at z = 0)."""
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    g = mg.build_graph(verts, faces)
    comp = mg.connected_components(g)
    if comp.max() > 0:
        # Keep the largest shell only (loose debris is common in generated meshes).
        counts = np.bincount(comp)
        keep = comp == int(np.argmax(counts))
        remap = -np.ones(g.n, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        fmask = keep[faces].all(axis=1)
        verts = verts[keep]
        faces = remap[faces[fmask]]
        g = mg.build_graph(verts, faces)
    height = float(verts[:, 2].max() - verts[:, 2].min())
    _, fields = mg.farthest_point_sampling(g, fps_samples)
    agd = fields.mean(axis=0)
    center = int(np.argmin(agd))
    f = mg.dijkstra(g, [center])
    tau = persistence * height
    arcs, vert_arc, root = _merge_tree(g, f, tau)
    step = max(0.012 * height, 2.0 * g.mean_edge)
    for a in arcs.values():
        a.verts = np.nonzero(vert_arc == a.id)[0]
        _slice_arc(a, verts, f, step)

    morph = Morphology(verts, faces, g, height, center, f, arcs, root, vert_arc)
    _absorb_bumps(morph, min_slenderness)
    _build_limbs(morph, ground_fraction)
    _classify(morph)
    return morph


def _absorb_bumps(m: Morphology, min_slenderness: float) -> None:
    """Remove stubby leaf arcs (buttocks, breasts, knuckles) that are not limbs."""
    changed = True
    while changed:
        changed = False
        for a in list(m.arcs.values()):
            if not a.is_leaf or a.parent is None:
                continue
            tip_z = m.verts[a.start, 2]
            if tip_z < 0.04 * m.height:
                continue  # ground contacts (toes, heels, claws) are always kept
            r = max(a.radius, 1e-6)
            if a.length / r >= min_slenderness and a.length >= 0.05 * m.height:
                continue
            parent = m.arcs[a.parent]
            parent.children.remove(a.id)
            m.vert_arc[a.verts] = parent.id
            parent.verts = np.nonzero(m.vert_arc == parent.id)[0]
            del m.arcs[a.id]
            m.notes.append(f"absorbed bump arc {a.id} (len {a.length:.3f}, r {r:.3f})")
            # A junction with a single remaining child is no longer a junction: fuse.
            if len(parent.children) == 1:
                child = m.arcs[parent.children[0]]
                child.end, child.end_f = parent.end, parent.end_f
                child.parent = parent.parent
                if parent.parent is not None:
                    gp = m.arcs[parent.parent]
                    gp.children = [child.id if c == parent.id else c for c in gp.children]
                else:
                    m.root_arc = child.id
                m.vert_arc[parent.verts] = child.id
                child.verts = np.nonzero(m.vert_arc == child.id)[0]
                del m.arcs[parent.id]
            changed = True
            break
    step = max(0.012 * m.height, 2.0 * m.graph.mean_edge)
    for a in m.arcs.values():
        _slice_arc(a, m.verts, m.f, step)


def _leaf_arcs_under(m: Morphology, aid: int) -> list[int]:
    a = m.arcs[aid]
    if a.is_leaf:
        return [aid]
    out: list[int] = []
    for c in a.children:
        out.extend(_leaf_arcs_under(m, c))
    return out


def _build_limbs(m: Morphology, ground_fraction: float) -> None:
    H = m.height
    ground_z = ground_fraction * H
    leaves = [a for a in m.arcs.values() if a.is_leaf]
    used: set[int] = set()
    limbs: list[Limb] = []

    # 1. Ground contacts: group sibling tips that merge quickly (toe + heel -> one foot).
    ground = [a for a in leaves if m.verts[a.start, 2] <= ground_z]
    foot_merge = 0.16 * H
    for a in ground:
        if a.id in used:
            continue
        chain = [a.id]
        tips = [m.verts[a.start]]
        cur = a
        # Climb while the parent only gathers ground tips of the same foot.
        while cur.parent is not None:
            par = m.arcs[cur.parent]
            under = _leaf_arcs_under(m, par.id)
            if all(m.verts[m.arcs[u].start, 2] <= ground_z for u in under) and \
               (a.start_f - par.start_f) <= foot_merge and len(under) <= 3:
                for u in under:
                    if u not in chain:
                        used.add(u)
                        tips.append(m.verts[m.arcs[u].start])
                chain.append(par.id)
                cur = par
                continue
            break
        used.update(chain)
        limbs.append(_make_limb(m, "leg", chain, tips, ground=True))

    # 2. Remaining leaves are free extremities (head, arms, tail, antennae).
    for a in leaves:
        if a.id in used:
            continue
        used.add(a.id)
        limbs.append(_make_limb(m, "free", [a.id], [m.verts[a.start]], ground=False))
    m.limbs = limbs


def _make_limb(m: Morphology, role: str, chain: list[int], tips: list[np.ndarray],
               ground: bool) -> Limb:
    # The limb's own centreline is the top arc of the chain (above merged sub-tips),
    # prefixed by the tip arc for the extremity itself.
    top = m.arcs[chain[-1]]
    first = m.arcs[chain[0]]
    if len(chain) > 1:
        cl = np.concatenate([first.centerline, top.centerline], axis=0)
        rr = np.concatenate([first.radii, top.radii])
    else:
        cl, rr = top.centerline, top.radii
    tip = np.array(m.verts[first.start])
    attach = top.end if top.end is not None else top.start
    return Limb(name="", role=role, arcs=list(chain), tip=tip, tips=[np.array(t) for t in tips],
                centerline=cl, radii=rr, attach_vertex=int(attach), ground_contact=ground)


def _classify(m: Morphology) -> None:
    H = m.height
    legs = [l for l in m.limbs if l.ground_contact]
    free = [l for l in m.limbs if not l.ground_contact]
    n_legs = len(legs)
    if n_legs == 2:
        m.body_plan = "biped"
    elif n_legs == 4:
        m.body_plan = "quadruped"
    elif n_legs == 6:
        m.body_plan = "hexapod"
    elif n_legs == 8:
        m.body_plan = "octopod"
    elif n_legs == 0:
        m.body_plan = "legless"
    else:
        m.body_plan = "multileg"

    fwd = m.forward
    left = np.cross(np.array([0.0, 0.0, 1.0]), fwd)

    def side_of(p: np.ndarray) -> str:
        s = float(np.dot(p, left))
        if abs(s) < 0.03 * H:
            return "C"
        return "L" if s > 0 else "R"

    for l in legs:
        l.side = side_of(l.tip)

    if m.body_plan == "biped":
        # Head: the free limb whose tip is highest; arms: lateral free limbs below it.
        free.sort(key=lambda l: -l.tip[2])
        if free:
            free[0].role = "head"
            free[0].side = "C"
        for l in free[1:]:
            l.side = side_of(l.tip)
            lateral = abs(float(np.dot(l.tip, left)))
            if lateral > 0.08 * H and l.length > 0.15 * H:
                l.role = "arm"
            elif float(np.dot(l.tip, fwd)) < -0.05 * H:
                l.role = "tail"
            else:
                l.role = "appendage"
    else:
        for l in free:
            l.side = side_of(l.tip)
            along = float(np.dot(l.tip, fwd))
            if along > 0.0:
                l.role = "antenna" if l.radius_ok(H) else "head"
            else:
                l.role = "tail"

    # Names: role + side + index ordered front-to-back.
    by_key: dict[tuple[str, str], list[Limb]] = {}
    for l in m.limbs:
        by_key.setdefault((l.role, l.side), []).append(l)
    for (role, side), ls in by_key.items():
        ls.sort(key=lambda l: -float(np.dot(l.tip, fwd)))
        for i, l in enumerate(ls):
            if len(ls) == 1:
                l.name = f"{role}_{side.lower()}" if side != "C" else role
            else:
                l.name = f"{role}_{side.lower()}{i + 1}" if side != "C" else f"{role}{i + 1}"


def _limb_radius_ok(self: Limb, H: float) -> bool:
    """Thin long free limbs in front are antennae; thick ones are heads."""
    r = float(np.median(self.radii)) if len(self.radii) else 0.0
    return r < 0.03 * H and self.length > 0.08 * H


Limb.radius_ok = _limb_radius_ok  # type: ignore[attr-defined]
