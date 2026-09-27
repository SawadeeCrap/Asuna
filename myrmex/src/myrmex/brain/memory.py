"""Morphological memory: which forms the organism has been, for how long, and when - compactly.

Three tiers, all over fingerprint embeddings (never meshes):

* **prototypes** - an episodic archive: every distinct form it settled into (leader clustering, radius
  ``radius``), with the goal that produced it (blend + material), first / last seen, arrivals, total dwell,
  its parent form, the operation that led there and the context (music energy) - at most ``capacity``;
* **trace** - the last ``trace_s`` seconds of observations (repetition, oscillation, residence);
* **returns** - how much each old form "wants" to come back: rises with the time since it was last seen
  (past a refractory period), with how long it was held, and with a similar musical context; falls with
  how often it came back lately.  The organism can return - just not in a loop.

Feedback (``mark``) raises or lowers a form's salience: the performer's "good" / "bad" pads - and the
labels a later Kev fine-tune would learn from.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .fingerprint import Fingerprint


@dataclass
class Prototype:
    id: int
    emb: np.ndarray
    fp: Fingerprint
    goal: dict                                  # {"blend": {form: share}, "material": name | None}
    first: float
    last: float
    visits: int = 1                             # separate arrivals
    dwell: float = 0.0                          # seconds held
    parent: int | None = None
    op: str = ""
    energy: float = 0.0                         # music energy when first reached
    salience: float = 0.0                       # performer feedback (+ good, - bad)

    def summary(self, forms: tuple) -> str:
        top = np.argsort(self.fp.w)[::-1][:2]
        parts = [f"{forms[i]} {self.fp.w[i]:.0%}" for i in top if self.fp.w[i] > 0.12]
        return " + ".join(parts) or forms[int(top[0])]


class MorphMemory:
    def __init__(self, radius: float = 0.3, capacity: int = 96, trace_s: float = 180.0):
        self.radius, self.capacity, self.trace_s = radius, capacity, trace_s
        self.protos: dict[int, Prototype] = {}
        self.trace: deque[tuple[float, int]] = deque()
        self.arrivals: deque[tuple[float, int, str]] = deque(maxlen=256)   # (t, proto, op)
        self.current: int | None = None
        self._next = 0
        self._t = None

    # ------------------------------------------------------------------ observing
    def nearest(self, emb: np.ndarray) -> tuple[int | None, float]:
        if not self.protos:
            return None, math.inf
        ids = list(self.protos)
        E = np.stack([self.protos[i].emb for i in ids])
        d = np.linalg.norm(E - emb, axis=1)
        j = int(np.argmin(d))
        return ids[j], float(d[j])

    def observe(self, t: float, fp: Fingerprint, emb: np.ndarray, goal: dict | None = None, op: str = "",
                energy: float = 0.0) -> int:
        """The body as it is now -> the prototype it is in (a new one when it is somewhere new)."""
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        pid, d = self.nearest(emb)
        if pid is None or d > self.radius:
            pid = self._next
            self._next += 1
            self.protos[pid] = Prototype(pid, emb.copy(), fp, dict(goal or {}), t, t, 1, 0.0, self.current, op, energy)
            self._evict(t, {pid, self.current})
            self.arrivals.append((t, pid, op))
        else:
            p = self.protos[pid]
            p.emb += 0.08 * (emb - p.emb)                   # drifts a little with what it becomes
            if pid != self.current:
                p.visits += 1
                self.arrivals.append((t, pid, op))
                if goal:
                    p.goal = dict(goal)
            p.last = t
        if self.current is not None and self.current in self.protos:
            self.protos[self.current].dwell += dt
        self.current = pid
        self.trace.append((t, pid))
        while self.trace and t - self.trace[0][0] > self.trace_s:
            self.trace.popleft()
        return pid

    def _evict(self, t: float, protect: set) -> None:
        """Keep at most ``capacity`` forms: the least valuable goes - an old one before a recent one, never the
        form it is in or the one it just found."""
        def keep(p):                                        # long-held, liked and recent forms stay
            return math.log1p(p.dwell) + 2.0 * p.salience + math.log1p(p.visits) - (t - p.last) / 600.0
        while len(self.protos) > self.capacity:
            recent = {pid for _, pid in self.trace}
            pool = [p for p in self.protos.values() if p.id not in protect and p.id not in recent] or \
                   [p for p in self.protos.values() if p.id not in protect]
            if not pool:
                return
            del self.protos[min(pool, key=keep).id]

    # ------------------------------------------------------------------ reading
    def residence(self, t: float) -> float:
        """How long it has been in the current form."""
        start = t
        for ti, pid in reversed(self.trace):
            if pid != self.current:
                break
            start = ti
        return t - start

    def recent_arrivals(self, t: float, window: float) -> list:
        return [(ti, pid, op) for ti, pid, op in self.arrivals if t - ti <= window]

    def repetition(self, pid: int | None, t: float, window: float = 120.0) -> float:
        """Arrivals at ``pid`` in the last ``window`` s, recency weighted (1 = just now)."""
        if pid is None:
            return 0.0
        return sum(0.5 ** ((t - ti) / (window / 2)) for ti, p, _ in self.arrivals if p == pid and t - ti <= window)

    def oscillating(self, pid: int | None, t: float, window: float = 40.0) -> bool:
        """Going back to the form before last, quickly: A -> B -> A."""
        arr = [a for a in self.arrivals if t - a[0] <= window]
        return pid is not None and len(arr) >= 2 and arr[-2][1] == pid and arr[-1][1] != pid

    def return_scores(self, t: float, energy: float, refractory: float = 45.0, tau: float = 60.0) -> list:
        """[(prototype, 0..1 pull)] for every form it could come back to, strongest first."""
        out = []
        for p in self.protos.values():
            if p.id == self.current or p.salience < -0.5:
                continue
            since = t - p.last
            ripe = 1.0 / (1.0 + math.exp(-(since - refractory) / (0.25 * tau)))
            held = 1.0 - math.exp(-p.dwell / 8.0)
            context = math.exp(-abs(energy - p.energy) / 0.35)
            tired = 1.0 / (1.0 + self.repetition(p.id, t, 300.0))
            pull = ripe * held * (0.5 + 0.5 * context) * tired * (1.0 + 0.5 * max(0.0, p.salience))
            out.append((p, pull))
        out.sort(key=lambda x: -x[1])
        return out

    def mark(self, delta: float, pid: int | None = None) -> None:
        pid = self.current if pid is None else pid
        if pid in self.protos:
            self.protos[pid].salience = max(-2.0, min(3.0, self.protos[pid].salience + delta))

    def summary(self, t: float, forms: tuple, n: int = 4) -> dict:
        """What a decision needs to know about the past, in a few words (quantised: stable between decisions)."""
        arr = list(self.arrivals)[-n:]
        recent = []
        for k, (ti, pid, _) in enumerate(arr):
            nxt = arr[k + 1][0] if k + 1 < len(arr) else t
            if pid in self.protos:
                recent.append((self.protos[pid].summary(forms), int(round((nxt - ti) / 5.0) * 5)))
        cur = self.protos.get(self.current)
        return {"forms_known": len(self.protos), "recent": recent,
                "current": cur.summary(forms) if cur else "", "held_s": int(self.residence(t) // 5 * 5),
                "repeating": bool(cur and self.repetition(cur.id, t) > 1.5)}


__all__ = ["MorphMemory", "Prototype"]
