"""Surface graph utilities for rig analysis (pure numpy + heapq, no scipy).

Hunyuan3D meshes are dense marching-cubes surfaces (often > 1M triangles).
All topology analysis runs on a decimated *proxy* (~10-40k vertices), which
these helpers handle in a few hundred milliseconds per geodesic sweep.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MeshGraph:
    verts: np.ndarray          # (V, 3)
    faces: np.ndarray          # (F, 3) int
    edges: np.ndarray          # (E, 2) int, unique, i < j
    edge_len: np.ndarray       # (E,)
    indptr: np.ndarray         # CSR row pointers (V + 1)
    indices: np.ndarray        # CSR neighbour indices
    weights: np.ndarray        # CSR edge lengths
    _adj_cache: tuple | None = None

    @property
    def n(self) -> int:
        return int(self.verts.shape[0])

    def neighbors(self, v: int) -> np.ndarray:
        return self.indices[self.indptr[v]:self.indptr[v + 1]]

    def adjacency_lists(self) -> tuple[list[list[int]], list[list[float]]]:
        """Python-list adjacency (much faster than numpy indexing inside heap loops)."""
        if self._adj_cache is None:
            ip = self.indptr.tolist()
            ind = self.indices.tolist()
            w = self.weights.tolist()
            nbrs = [ind[ip[i]:ip[i + 1]] for i in range(self.n)]
            wts = [w[ip[i]:ip[i + 1]] for i in range(self.n)]
            self._adj_cache = (nbrs, wts)
        return self._adj_cache

    @property
    def mean_edge(self) -> float:
        return float(self.edge_len.mean()) if self.edge_len.size else 0.0


def build_graph(verts: np.ndarray, faces: np.ndarray) -> MeshGraph:
    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e.sort(axis=1)
    e = np.unique(e, axis=0)
    e = e[e[:, 0] != e[:, 1]]
    L = np.linalg.norm(verts[e[:, 0]] - verts[e[:, 1]], axis=1)
    n = verts.shape[0]
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    w = np.concatenate([L, L])
    order = np.argsort(src, kind="stable")
    src, dst, w = src[order], dst[order], w[order]
    counts = np.bincount(src, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return MeshGraph(verts, faces, e, L, indptr, dst, w)


def dijkstra(g: MeshGraph, sources, max_dist: float = math.inf,
             source_dist=None) -> np.ndarray:
    """Multi-source geodesic (edge-path) distances."""
    nbrs, wts = g.adjacency_lists()
    dist = [math.inf] * g.n
    heap: list[tuple[float, int]] = []
    srcs = list(np.atleast_1d(sources).tolist())
    for k, s in enumerate(srcs):
        d0 = 0.0 if source_dist is None else float(source_dist[k])
        if d0 < dist[s]:
            dist[s] = d0
            heap.append((d0, s))
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        d, u = pop(heap)
        if d > dist[u] or d > max_dist:
            continue
        nu = nbrs[u]
        wu = wts[u]
        for k in range(len(nu)):
            v = nu[k]
            nd = d + wu[k]
            if nd < dist[v]:
                dist[v] = nd
                push(heap, (nd, v))
    return np.array(dist)


def dijkstra_labels(g: MeshGraph, seed_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic Voronoi partition: nearest seed set (label) and distance for every vertex."""
    nbrs, wts = g.adjacency_lists()
    dist = [math.inf] * g.n
    label = [-1] * g.n
    heap: list[tuple[float, int, int]] = []
    for lab, seeds in enumerate(seed_lists):
        for s in seeds:
            if dist[s] > 0.0:
                dist[s] = 0.0
                label[s] = lab
                heap.append((0.0, s, lab))
    heapq.heapify(heap)
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        d, u, lab = pop(heap)
        if d > dist[u] or label[u] != lab:
            continue
        nu = nbrs[u]
        wu = wts[u]
        for k in range(len(nu)):
            v = nu[k]
            nd = d + wu[k]
            if nd < dist[v]:
                dist[v] = nd
                label[v] = lab
                push(heap, (nd, v, lab))
    return np.array(label), np.array(dist)


def farthest_point_sampling(g: MeshGraph, k: int, start: int | None = None) -> tuple[list[int], np.ndarray]:
    """Geodesic farthest point sampling. Returns samples and per-sample distance fields."""
    if start is None:
        start = int(np.argmax(g.verts[:, 2]))
    samples = [start]
    fields = [dijkstra(g, [start])]
    mind = fields[0].copy()
    for _ in range(k - 1):
        nxt = int(np.argmax(np.where(np.isfinite(mind), mind, -1.0)))
        samples.append(nxt)
        d = dijkstra(g, [nxt])
        fields.append(d)
        mind = np.minimum(mind, d)
    return samples, np.stack(fields, axis=0)


def connected_components(g: MeshGraph, mask: np.ndarray | None = None) -> np.ndarray:
    """Component id per vertex (-1 for masked-out vertices)."""
    nbrs, _ = g.adjacency_lists()
    comp = np.full(g.n, -1, dtype=np.int64)
    allowed = np.ones(g.n, bool) if mask is None else mask
    cid = 0
    for s in range(g.n):
        if comp[s] >= 0 or not allowed[s]:
            continue
        stack = [s]
        comp[s] = cid
        while stack:
            u = stack.pop()
            for v in nbrs[u]:
                if comp[v] < 0 and allowed[v]:
                    comp[v] = cid
                    stack.append(v)
        cid += 1
    return comp


def laplacian_smooth_values(g: MeshGraph, values: np.ndarray, iterations: int,
                            alpha: float = 0.5, locked: np.ndarray | None = None) -> np.ndarray:
    """Iteratively average per-vertex values (V,) or (V, K) with their 1-ring."""
    vals = np.array(values, dtype=float, copy=True)
    e = g.edges
    deg = np.bincount(np.concatenate([e[:, 0], e[:, 1]]), minlength=g.n).astype(float)
    deg[deg == 0] = 1.0
    for _ in range(int(iterations)):
        acc = np.zeros_like(vals)
        np.add.at(acc, e[:, 0], vals[e[:, 1]])
        np.add.at(acc, e[:, 1], vals[e[:, 0]])
        mean = acc / (deg[:, None] if vals.ndim == 2 else deg)
        new = (1.0 - alpha) * vals + alpha * mean
        if locked is not None:
            new[locked] = vals[locked]
        vals = new
    return vals


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


def vertex_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric vertex area (one third of adjacent triangle areas)."""
    a = verts[faces[:, 1]] - verts[faces[:, 0]]
    b = verts[faces[:, 2]] - verts[faces[:, 0]]
    fa = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    va = np.zeros(verts.shape[0])
    for k in range(3):
        np.add.at(va, faces[:, k], fa / 3.0)
    return va
