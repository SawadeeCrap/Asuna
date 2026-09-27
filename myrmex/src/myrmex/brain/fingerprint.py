"""A morphology as a few dozen numbers - never a mesh.

* ``w``: the blend of forms (the engine's softmax over its latent ``z``);
* ``geo``: eight shape descriptors of the actual point cloud (size, elongation, flatness, asymmetry,
  clumping, reach, lumpiness, twist - half-turns round the long axis) - what the body looks like, whatever
  made it so;
* ``mat``: the material state (cohesion, stiffness, damping, repulsion, persistence, dispersion) where the
  organism has one;
* ``bodies``: how many parts it flies as (1 = one body; a flock splits into up to 4).

``embed`` turns a fingerprint into one vector whose Euclidean distance is the morphological distance:
Hellinger distance between blends (0.7), standardised geometry (0.25), material and body count (0.05 each).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GEO = ("size", "elong", "flat", "asym", "clump", "reach", "lumpy", "twist")
W_FORM, W_GEO, W_MAT, W_BODY = 0.7, 0.25, 0.05, 0.05


def geometry(X: np.ndarray) -> np.ndarray:
    """Eight descriptors of a point cloud (n x 3), invariant to position and orientation."""
    X = np.asarray(X, float)
    X = X - X.mean(0)
    n = len(X)
    if n < 4:
        return np.zeros(len(GEO))
    C = X.T @ X / n
    ev, V = np.linalg.eigh(C)                              # ascending
    l3, l2, l1 = np.maximum(ev, 1e-9)
    r = np.linalg.norm(X, axis=1)
    rms = math.sqrt(float((r * r).mean())) + 1e-9
    P = X @ V                                              # principal frame (last column = longest axis)
    sd = P.std(0) + 1e-9
    asym = float(np.mean(np.abs((P ** 3).mean(0)) / sd ** 3))
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    clump = float(np.min(D, 1).mean() / rms)
    reach = float(r.max() / rms)
    lumpy = float(r.std() / (r.mean() + 1e-9))
    return np.array([math.log(rms), 0.5 * math.log(l1 / l2), 0.5 * math.log(l2 / l3), asym, clump, reach, lumpy,
                     _twist(P)])


def _twist(P: np.ndarray, k: int = 10) -> float:
    """How far the body turns round its long axis, in half-turns: in slices along the axis, the direction of
    each cross-section (its second moment about the axis - a ribbon's width, a coil's offset) and how that
    direction rotates from slice to slice, weighted by how clearly each slice has a direction.  A twisted
    ribbon or a coil read high; a straight rod, a flat ribbon, a plain bend or a ball read ~0."""
    if len(P) < 2 * k:
        return 0.0
    order = np.argsort(P[:, 2])
    ang, clear = [], []
    for c in np.array_split(order, k):
        q = P[c, :2]
        m = q.T @ q / len(q)
        ang.append(0.5 * math.atan2(2.0 * m[0, 1], m[0, 0] - m[1, 1]))
        clear.append(math.hypot(m[0, 0] - m[1, 1], 2.0 * m[0, 1]) / (m[0, 0] + m[1, 1] + 1e-12))
    turn = 0.0
    for i in range(k - 1):
        d = (ang[i + 1] - ang[i] + 0.5 * math.pi) % math.pi - 0.5 * math.pi
        turn += d * min(clear[i], clear[i + 1]) ** 2
    return abs(turn) / math.pi


@dataclass
class Fingerprint:
    w: np.ndarray                                          # blend over the vocabulary's forms
    geo: np.ndarray                                        # GEO descriptors
    mat: np.ndarray = field(default_factory=lambda: np.zeros(0))
    bodies: int = 1

    def dominant(self, forms: tuple) -> str:
        return forms[int(np.argmax(self.w))] if len(self.w) else ""

    def to_list(self) -> list:
        return [float(x) for x in np.concatenate([self.w, self.geo, self.mat, [self.bodies]])]


@dataclass
class Scale:
    """Standardisation of the geometry (from the organism's own forms: what counts as a big change here)."""
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def from_geometry(cls, clouds: np.ndarray | None) -> "Scale":
        if clouds is None or len(clouds) < 2:
            return cls(np.zeros(len(GEO)), np.ones(len(GEO)))
        G = np.array([geometry(c) for c in clouds])
        return cls(G.mean(0), np.maximum(G.std(0), 0.05))


MAT_SPAN = np.array([1.0, 1.6, 0.6, 0.7, 2.8, 0.9])        # ranges of the six material parameters


def embed(fp: Fingerprint, scale: Scale) -> np.ndarray:
    parts = [np.sqrt(np.clip(fp.w, 0.0, None)) * W_FORM,
             np.clip((fp.geo - scale.mu) / scale.sd, -4.0, 4.0) * (W_GEO / math.sqrt(len(GEO)))]
    if len(fp.mat):
        parts.append(np.asarray(fp.mat, float)[:6] / MAT_SPAN[:len(fp.mat)] * (W_MAT / math.sqrt(6)))
    parts.append([W_BODY * min(max(fp.bodies, 1) - 1, 3) / 3.0])
    return np.concatenate(parts)


def distance(a: Fingerprint, b: Fingerprint, scale: Scale) -> float:
    return float(np.linalg.norm(embed(a, scale) - embed(b, scale)))


def blend_cloud(w: np.ndarray, clouds: np.ndarray) -> np.ndarray:
    """The target cloud of a blend (the engine's own rule: a weighted sum of centred forms)."""
    keep = w > 0.01
    return np.tensordot(w[keep], clouds[keep], axes=1)


__all__ = ["GEO", "Fingerprint", "Scale", "geometry", "embed", "distance", "blend_cloud"]
