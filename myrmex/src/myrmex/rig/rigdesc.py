"""Engine-independent rig description (bones + landmarks + chains).

A :class:`RigDescription` is what the fitters produce from a
:class:`~myrmex.rig.morphology.Morphology` and what both sides consume:

* the Blender side builds an armature from it and skins the mesh,
* the motion engine builds a :class:`~myrmex.body.bodyplan.BodyPlan` from it.

It is plain JSON so the user can inspect / hand-edit it, and so a rig that
was built by other means (Rigify, Auto-Rig Pro, Mixamo, UniRig) can be
described with a small mapping file instead.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class BoneDesc:
    name: str
    head: list[float]
    tail: list[float]
    parent: str | None = None
    roll_ref: list[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
    role: str = "generic"
    side: str = "C"
    limb: str | None = None
    connect: bool = False
    deform: bool = True

    @property
    def head_v(self) -> np.ndarray:
        return np.asarray(self.head, dtype=float)

    @property
    def tail_v(self) -> np.ndarray:
        return np.asarray(self.tail, dtype=float)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.tail_v - self.head_v))


@dataclass
class ChainDesc:
    """An ordered bone chain with a semantic role for the motion engine."""
    name: str
    role: str                      # spine | leg | arm | head | tail | antenna | hair | skirt
    bones: list[str]
    side: str = "C"
    extra: dict = field(default_factory=dict)


@dataclass
class RigDescription:
    body_plan: str
    height: float
    forward: list[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
    up: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    bones: list[BoneDesc] = field(default_factory=list)
    chains: list[ChainDesc] = field(default_factory=list)
    landmarks: dict[str, list[float]] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ access
    def bone(self, name: str) -> BoneDesc:
        for b in self.bones:
            if b.name == name:
                return b
        raise KeyError(name)

    def has_bone(self, name: str) -> bool:
        return any(b.name == name for b in self.bones)

    def chain(self, name: str) -> ChainDesc:
        for c in self.chains:
            if c.name == name:
                return c
        raise KeyError(name)

    def chains_by_role(self, role: str) -> list[ChainDesc]:
        return [c for c in self.chains if c.role == role]

    def landmark(self, name: str) -> np.ndarray:
        return np.asarray(self.landmarks[name], dtype=float)

    def add_bone(self, name, head, tail, parent=None, roll_ref=None, role="generic",
                 side="C", limb=None, connect=False, deform=True) -> BoneDesc:
        b = BoneDesc(name, [float(x) for x in head], [float(x) for x in tail], parent,
                     [float(x) for x in (roll_ref if roll_ref is not None else self.forward)],
                     role, side, limb, connect, deform)
        self.bones.append(b)
        return b

    def children_of(self, name: str) -> list[BoneDesc]:
        return [b for b in self.bones if b.parent == name]

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "format": "myrmex.rig/1",
            "body_plan": self.body_plan,
            "height": self.height,
            "forward": self.forward,
            "up": self.up,
            "bones": [asdict(b) for b in self.bones],
            "chains": [asdict(c) for c in self.chains],
            "landmarks": self.landmarks,
            "params": self.params,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RigDescription":
        rd = cls(d["body_plan"], float(d["height"]), list(d.get("forward", [0, -1, 0])),
                 list(d.get("up", [0, 0, 1])))
        rd.bones = [BoneDesc(**b) for b in d.get("bones", [])]
        rd.chains = [ChainDesc(**c) for c in d.get("chains", [])]
        rd.landmarks = {k: list(v) for k, v in d.get("landmarks", {}).items()}
        rd.params = dict(d.get("params", {}))
        rd.notes = list(d.get("notes", []))
        return rd

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "RigDescription":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def validate(self) -> list[str]:
        problems = []
        names = [b.name for b in self.bones]
        if len(names) != len(set(names)):
            problems.append("duplicate bone names")
        for b in self.bones:
            if b.parent is not None and b.parent not in names:
                problems.append(f"{b.name}: unknown parent {b.parent}")
            if b.length < 1e-4:
                problems.append(f"{b.name}: zero length")
        for c in self.chains:
            for bn in c.bones:
                if bn not in names:
                    problems.append(f"chain {c.name}: unknown bone {bn}")
        return problems
