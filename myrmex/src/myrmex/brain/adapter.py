"""The brain's two hands on an engine: read the body (a Snapshot), set a direction (a Decision).

Only through what the engines already have - nothing is added to their physics:

* colony family (v3-v13) and the Bionic line (v14-v18): the lead body's latent goal ``z_goal`` (a blend is
  the inverse of the engine's own softmax), its material goal, its intent timers (the engine's own state
  machine waits ``hold`` seconds before choosing again), and ``trigger_event``;
* Mimetic Polyalloy (v2, v5): the same on the engine itself;
* Black Nanomaterial (v1): the morphology's target descriptor vector (a blend of its named morphologies).

The engine keeps every reaction of its own (evading, striking, dashing on the kick, the glove's gestures);
while a brain target is current and the organism is free again, ``maintain`` puts the target back after a
reaction window - 4 s after the engine's own choices, 1.2 s after a lunge on the kick (the Spear's dash
into its lance would otherwise hold it in the lance through a whole drop) - so the reaction is seen and the
exploration goes on.  The hand always wins: while the
glove sculpts the form (or holds it frozen) the adapter reports "not free" and changes nothing.
"""
from __future__ import annotations

import math

import numpy as np

from .candidates import material_vector
from .core import Decision, Snapshot
from .fingerprint import Fingerprint, geometry
from .vocab import Vocabulary, vocabulary

FREE_INTENTS = ("CRUISE", "HOVER", "EXPLORE", "DISPLAY", "PATROL", "FORMATION")
QUIET_S = 4.0                         # the engine's own reactions must pause this long before the target comes back
LUNGE_S = 1.2                         # ... a lunge on the kick (Spear dash, Swarm surge, Blade slash: one form, peak
LUNGE_PEAK = 3.3                      #     >= 3.3) is a flash: the direction is back after this long


def softmax3(z: np.ndarray) -> np.ndarray:
    e = np.exp(3.0 * (z - z.max()))
    return e / e.sum()


ABSENT = math.exp(-7.5)               # an absent form sits 2.5 below the peak - as in the engines' own goals


def inverse_blend(w: np.ndarray, strength: float) -> np.ndarray:
    """A z_goal whose softmax (the engine's temperature 3) is the blend ``w``: the peak at ``strength``."""
    w = np.asarray(w, float)
    w = np.maximum(w / w.max(), ABSENT)
    return strength + np.log(w) / 3.0


def blend_of(z: np.ndarray, forms: tuple, floor: float = 0.06) -> dict:
    w = softmax3(np.asarray(z, float))
    out = {forms[i]: float(w[i]) for i in np.argsort(w)[::-1] if w[i] >= floor}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s else {forms[int(np.argmax(w))]: 1.0}


class Adapter:
    family = ""

    def __init__(self, engine, organism: str = ""):
        self.engine = engine
        self.vocab: Vocabulary = vocabulary(engine, organism)
        self.target: np.ndarray | None = None         # the brain's current z_goal
        self.material: np.ndarray | None = None
        self.until = -1e9
        self.reacted_at: float | None = None          # the last time the engine set a goal of its own
        self.lunge = False                            # ... and whether that was a lunge on the kick
        self._seen: np.ndarray | None = None
        self.applied = 0
        self.restored = 0

    # --- per family
    def _body(self):
        return self.engine

    def _intent(self) -> str:
        return getattr(self._body(), "intent", "CRUISE")

    def _points(self, state) -> np.ndarray:
        pos = getattr(state, "pos", None)
        return np.asarray(pos, float) if pos is not None else np.zeros((0, 3))

    def _set_timers(self, hold: float) -> None:
        b = self._body()
        b.intent_t = 0.0
        if hasattr(b, "dwell"):
            b.dwell = hold
        if hasattr(b, "intent_dwell"):
            b.intent_dwell = hold

    def _sculpted(self) -> bool:
        gc = self.engine.glove.ctrl
        return bool(gc.active and ((gc.finger_mode == "morph" and getattr(self.engine, "sculpt", None))
                                   or gc.freeze > 0.5))

    # --- reading
    def free(self) -> bool:
        return self._intent() in FREE_INTENTS and not self._sculpted()

    def due(self, lookahead: float) -> bool:
        """The engine's own state machine chooses the next form within ``lookahead`` s."""
        b = self._body()
        dwell = getattr(b, "dwell", getattr(b, "intent_dwell", 1e9))
        return self.free() and b.intent_t + lookahead >= dwell

    def snapshot(self, t: float, state, user, controls: dict, music: str = "", lookahead: float = 1.0) -> Snapshot:
        b = self._body()
        w = softmax3(np.asarray(b.z, float))
        mat = np.asarray(getattr(b, "mat", np.zeros(0)), float)
        X = self._points(state)
        geo = geometry(X) if len(X) >= 4 else np.zeros(8)
        fp = Fingerprint(w, geo, mat.copy(), self._bodies())
        inp = self.engine.inp
        return Snapshot(t, fp, blend_of(b.z_goal, self.vocab.forms), self.free(), self.due(lookahead),
                        float(inp.energy), bool(inp.playing), music, user, dict(controls or {}))

    def _bodies(self) -> int:
        return 1

    # --- acting
    def apply(self, d: Decision, t: float) -> None:
        b = self._body()
        w = self.vocab.weights(d.blend)
        if w.sum() <= 0:
            return
        self.target = inverse_blend(w, d.strength)
        b.z_goal = self.target.copy()
        mv = material_vector(d.material) if self.vocab.materials else None
        self.material = mv
        if mv is not None and hasattr(b, "mat_goal"):
            b.mat_goal = mv.copy()
        self._set_timers(d.hold)
        if d.event:
            self.engine.trigger_event(d.event)
        self.until = t + d.hold
        self.reacted_at, self._seen = None, self.target.copy()
        self.applied += 1

    def maintain(self, t: float) -> None:
        """After the engine's own reactions (a dash on the kick, an evasion ...) have paused ``QUIET_S``, the
        brain's direction again - while it is current and the organism is free."""
        if self.target is None or t > self.until:
            self.target = None
            return
        b = self._body()
        zg = np.asarray(b.z_goal, float)
        if self._seen is None or np.abs(zg - self._seen).max() > 1e-6:
            self._seen = zg.copy()
            if np.abs(zg - self.target).max() >= 0.3:
                self.reacted_at = t                     # a goal of the engine's own
                self.lunge = bool(zg.max() >= LUNGE_PEAK and (zg > 0.5).sum() == 1)
        if self.reacted_at is None or np.abs(zg - self.target).max() < 0.3:
            return
        if t - self.reacted_at >= (LUNGE_S if self.lunge else QUIET_S) and self.free():
            b.z_goal = self.target.copy()
            self._seen = self.target.copy()
            if self.material is not None and hasattr(b, "mat_goal"):
                b.mat_goal = self.material.copy()
            self.reacted_at = None
            self.restored += 1

    def release(self) -> None:
        self.target = None


class ColonyAdapter(Adapter):
    """v3-v13: the lead body."""
    family = "colony"

    def _body(self):
        return self.engine.bodies[0]

    def _points(self, state) -> np.ndarray:
        X = np.asarray(self.engine.x, float)
        own = getattr(self.engine, "own", None)
        return X[own == 0] if own is not None and (own == 0).sum() >= 4 else X

    def _bodies(self) -> int:
        return len(self.engine._alive())


class PolyalloyAdapter(Adapter):
    family = "polyalloy"

    def _points(self, state) -> np.ndarray:
        return np.asarray(self.engine.x, float)


class BionicAdapter(Adapter):
    family = "bionic"

    def free(self) -> bool:
        return self._intent() in ("CRUISE", "HOVER", "EXPLORE", "DISPLAY") and not self._sculpted()


class CreatureAdapter(Adapter):
    """v1 Black Nanomaterial: a blend of named morphologies as the descriptor target."""
    family = "creature"

    def __init__(self, engine, organism: str = ""):
        super().__init__(engine, organism)
        from ..creature.morphology import MORPHS
        self.M = np.array([MORPHS[f] for f in self.vocab.forms], float)

    def _intent(self) -> str:
        return "CRUISE" if self.engine.beh.state not in ("OVERLOADED", "RECOVERING") else self.engine.beh.state

    def _w(self) -> np.ndarray:
        d = np.linalg.norm(self.M - self.engine.morph.vec[None, :], axis=1)
        return softmax3(-d * 3.0)

    def snapshot(self, t: float, state, user, controls: dict, music: str = "", lookahead: float = 1.0) -> Snapshot:
        X = self._points(state)
        w = self._w()
        fp = Fingerprint(w, geometry(X) if len(X) >= 4 else np.zeros(8), np.zeros(0), 1)
        dst = np.linalg.norm(self.M - self.engine.morph.dst[None, :], axis=1)
        inp = self.engine.inp
        beh = self.engine.beh
        due = self.free() and beh.mem.time_in_state + lookahead >= beh.dwell
        return Snapshot(t, fp, {self.vocab.forms[int(np.argmin(dst))]: 1.0}, self.free(), due, float(inp.energy),
                        bool(inp.playing), music, user, dict(controls or {}))

    def apply(self, d: Decision, t: float) -> None:
        w = self.vocab.weights(d.blend)
        if w.sum() <= 0:
            return
        m = self.engine.morph
        m.src, m.dst, m.t0 = m.vec.copy(), w @ self.M, self.engine.t
        m.name = self.vocab.forms[int(np.argmax(w))]
        self.engine.beh.mem.time_in_state = 0.0
        self.engine.beh.dwell = d.hold
        if d.event:
            self.engine.trigger_event(d.event)
        self.until, self.applied = t + d.hold, self.applied + 1

    def maintain(self, t: float) -> None:
        pass


def make_adapter(engine, organism: str = "") -> Adapter:
    if hasattr(engine, "bodies") and hasattr(type(engine), "SHAPE_SET"):
        return ColonyAdapter(engine, organism)
    if hasattr(type(engine), "REGIMES") and len(type(engine).REGIMES) > 1:
        return BionicAdapter(engine, organism)
    if hasattr(engine, "z") and hasattr(type(engine), "VOCAB"):
        return PolyalloyAdapter(engine, organism)
    return CreatureAdapter(engine, organism)


def user_cue(glove_state, glove_ctrl, now: float, preset: str = "") -> "object":
    """The hand as the brain reads it - from the glove state the session already keeps."""
    from .candidates import UserCue
    if glove_state is None or not getattr(glove_state, "present", False):
        return UserCue()
    ext = np.asarray(glove_state.ext, float)
    spin = min(1.0, float(np.abs(glove_state.omega).max()) / 6.0)
    motion = min(1.0, float(np.linalg.norm(glove_state.pos_v)) / 2.5)
    g = next((name for tg, name in reversed(glove_state.gestures) if now - tg < 3.0), "")
    sculpt = bool(glove_ctrl is not None and glove_ctrl.active and
                  (glove_ctrl.finger_mode == "morph" or glove_ctrl.freeze > 0.5))
    return UserCue(True, float(ext.mean()), spin, motion, g, sculpt)


__all__ = ["Adapter", "ColonyAdapter", "PolyalloyAdapter", "BionicAdapter", "CreatureAdapter", "make_adapter",
           "inverse_blend", "blend_of", "softmax3", "user_cue", "FREE_INTENTS"]
