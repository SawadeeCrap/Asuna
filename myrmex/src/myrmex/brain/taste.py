"""The performer's taste, learned from Good / Bad: P(good) of a change of form.

A logistic regression over named features of a proposed change, refit (a few Newton steps, L2) on every
rating - small data on purpose: tens of ratings already move it, and it never needs a GPU.  The features
say what the change *looks like*, the same way for the trainer's proposals and the live brain's candidates:

* ``geo:*``  - the predicted shape's eight descriptors (size, elongation, flatness, asymmetry, clumping,
  reach, lumpiness, twist), standardised by the organism's own forms;
* ``dfm:*``  - the deformation (stretch, width, height, twist, bend, ripple, size), each over its range;
* ``mat:*``  - the material state asked for;
* ``form:*`` - the share of each form in the blend; ``mix`` - how much of a blend it is (0: one pure form);
* ``change`` - how far it is from what the body is now (the brain's morphological distance).

Stored per organism next to the decision log (``taste-<organism>.json``): the rows it learned from, so a
later version can refit them with other features.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .fingerprint import GEO, Scale

DFM_SPAN = {"stretch": 0.8, "width": 0.7, "height": 0.7, "twist": 1.5, "bend": 1.0, "ripple": 0.35, "size": 0.35}


def features(fp, emb, cur_emb, blend: dict, material: str | None, deform: dict | None, scale: Scale) -> dict:
    """The named features of one change (``fp``/``emb``: its predicted fingerprint and embedding)."""
    x = {}
    geo = np.asarray(fp.geo, float)
    for i, g in enumerate(GEO):
        x["geo:" + g] = float(np.clip((geo[i] - scale.mu[i]) / scale.sd[i], -4.0, 4.0)) / 2.0
    for k, span in DFM_SPAN.items():
        v = float((deform or {}).get(k, 0.0))
        if v:
            x["dfm:" + k] = v / span
    if material:
        x["mat:" + material] = 1.0
    s = sum(max(0.0, float(v)) for v in blend.values()) or 1.0
    for f, v in blend.items():
        if v > 0:
            x["form:" + f] = float(v) / s
    x["mix"] = 1.0 - max((max(0.0, float(v)) / s for v in blend.values()), default=1.0)   # 0 a pure form .. 0.5
    x["change"] = float(np.linalg.norm(np.asarray(emb, float) - np.asarray(cur_emb, float)))
    return x


def candidate_features(c, cur_emb, scale: Scale) -> dict:
    """A live candidate (candidates.Candidate, evaluated) as taste features."""
    return features(c.fp, c.emb, cur_emb, c.blend, c.material, c.deform, scale)


class Taste:
    """P(good) of a change; ``rows`` are (features, 1/0)."""

    LAM = 1.0                      # L2 on the weights (not the bias): with few ratings the model stays near 0

    def __init__(self, path: str = ""):
        self.path = path
        self.rows: list[tuple[dict, float]] = []
        self.w: dict[str, float] = {}
        self.b = 0.0
        self.hits: list[float] = []        # was the prediction right, rating by rating (before learning from it)

    # ------------------------------------------------------------------ using it
    def logit(self, x: dict) -> float:
        return self.b + sum(self.w.get(k, 0.0) * float(v) for k, v in x.items())

    def p(self, x: dict) -> float:
        z = max(-30.0, min(30.0, self.logit(x)))
        return 1.0 / (1.0 + math.exp(-z))

    @property
    def n(self) -> int:
        return len(self.rows)

    def accuracy(self, last: int = 20) -> float | None:
        h = self.hits[-last:]
        return float(np.mean(h)) if len(h) >= 5 else None

    # ------------------------------------------------------------------ learning
    def add(self, x: dict, good: bool) -> None:
        if self.rows:
            self.hits.append(1.0 if (self.p(x) > 0.5) == bool(good) else 0.0)
        self.rows.append((dict(x), 1.0 if good else 0.0))
        self.fit()

    def fit(self, iters: int = 15) -> None:
        if not self.rows:
            self.w, self.b = {}, 0.0
            return
        names = sorted({k for x, _ in self.rows for k in x})
        X = np.array([[float(x.get(k, 0.0)) for k in names] + [1.0] for x, _ in self.rows])
        y = np.array([t for _, t in self.rows])
        w = np.zeros(X.shape[1])
        reg = np.full(X.shape[1], self.LAM)
        reg[-1] = 1e-3
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
            g = X.T @ (p - y) + reg * w
            H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        self.w = {k: float(v) for k, v in zip(names, w[:-1]) if abs(v) > 1e-9}
        self.b = float(w[-1])

    def top(self, k: int = 6) -> list[tuple[str, float]]:
        """What it likes (+) and dislikes (-) most."""
        return sorted(self.w.items(), key=lambda kv: -abs(kv[1]))[:k]

    # ------------------------------------------------------------------ keeping it
    def to_dict(self) -> dict:
        return {"version": 1, "rows": [[x, y] for x, y in self.rows], "hits": self.hits[-200:]}

    def save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, self.path)
        except OSError:
            pass

    @classmethod
    def load(cls, path: str) -> "Taste":
        t = cls(path)
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            t.rows = [(dict(x), float(y)) for x, y in d.get("rows", []) if isinstance(x, dict)]
            t.hits = [float(h) for h in d.get("hits", [])]
            t.fit()
        except (OSError, ValueError, TypeError):
            pass
        return t


def taste_path(folder: str, organism: str) -> str:
    return os.path.join(folder, f"taste-{organism or 'organism'}.json") if folder else ""


__all__ = ["Taste", "features", "candidate_features", "taste_path", "DFM_SPAN"]
