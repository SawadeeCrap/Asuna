"""The morphology brain: memory + novelty + candidates + an arbiter, at a low decision rate.

One call per decision tick (``step``): the body as it is now (a ``Snapshot``) in, at most one ``Decision``
out - a target blend of forms, a material, perhaps an event, how strongly and for how long.  The engine
does the rest continuously (its latent inertia is the transition).

Modes (the benchmark's A-G; ``deterministic`` is the production default, ``kev_candidates`` adds Kev):

    current         A  the engine alone (the brain only watches)
    random          B  a random form or blend at each decision
    novelty         C  the most novel form, greedily (no identity, continuity or memory)
    kev_direct      D  Kev picks the operation itself, no memory, no candidates
    kev_memory      E  as D, with the memory summary in the state
    kev_candidates  F  candidates + novelty + memory, Kev chooses (confidence-gated, deterministic fallback)
    deterministic   G  candidates + novelty + memory, a Boltzmann arbiter over plain utilities

Autonomy is not a parameter blend: it sets how often the brain may take the next decision away from the
engine's own state machine, how exploratory its choices are, how far from the organism's identity it may
wander, how much it defers to the hand, and the budget of radical changes.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .candidates import Candidate, UserCue, evaluate, generate
from .fingerprint import Fingerprint, Scale, embed
from .kev_client import (INTENSITY, KevClient, KevError, candidates_questions, choice_confidence, liked_questions,
                         operations_questions, state_text)
from .memory import MorphMemory
from .vocab import Vocabulary

MODES = {"A": "current", "B": "random", "C": "novelty", "D": "kev_direct", "E": "kev_memory", "F": "kev_candidates",
         "G": "deterministic"}
KEV_MODES = ("kev_direct", "kev_memory", "kev_candidates")


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * min(1.0, max(0.0, u))


@dataclass
class BrainControls:
    autonomy: float = 0.5           # 0 the performer .. 1 the organism
    novelty: float = 0.6            # exploration pressure
    persistence: float = 0.5        # how long a form is held
    mutation: float = 0.4           # how far a single change may go (blend shares, material)
    returns: float = 0.5            # how readily it comes back to old forms
    memory: float = 0.6             # memory depth (how long forms stay "recent")
    rate_hz: float = 1.0            # decision ticks per second
    min_confidence: float = 0.2     # Kev: below this, the deterministic arbiter decides

    CC = {"brain_autonomy": "autonomy", "brain_novelty": "novelty", "brain_persistence": "persistence",
          "brain_mutation": "mutation", "brain_return": "returns", "brain_memory": "memory"}

    def merged(self, controls: dict | None) -> "BrainControls":
        """MIDI / app overrides (0..1 each; brain_rate 0..1 -> 0.25..2 Hz)."""
        d = asdict(self)
        for cc, key in self.CC.items():
            v = (controls or {}).get(cc)
            if v is not None and math.isfinite(float(v)) and float(v) >= 0.0:
                d[key] = min(1.0, float(v))
        r = (controls or {}).get("brain_rate")
        if r is not None and float(r) >= 0.0:
            d["rate_hz"] = 0.25 * 8.0 ** min(1.0, float(r))
        return BrainControls(**d)

    def interval(self) -> float:
        """Seconds between the brain's own changes beyond the engine's decision points (autonomy > 0.6)."""
        return lerp(14.0, 3.0, self.autonomy) * lerp(0.6, 1.6, self.persistence)


@dataclass
class BrainConfig:
    enabled: bool = False
    mode: str = "deterministic"
    controls: BrainControls = field(default_factory=BrainControls)
    kev_url: str = ""
    kev_timeout: float = 0.8
    kev_candidates: int = 6
    kev_ask: str = "choice"          # Kev's question: "choice" (which of these?) | "liked" (would the performer like
    #                                  each? - the question the Train page teaches it, brain/trainer.py)
    memory_size: int = 96
    seed: int = 0
    log_dir: str = ""                # a JSONL decision log per organism and day there ("" = none)
    reshape: bool = True             # candidates that stretch / twist / bend the form (engines that can: colony family)
    taste: bool = True               # your Good / Bad from the Train page weigh in (taste-<organism>.json in log_dir)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BrainConfig":
        d = dict(d or {})
        ctl = BrainControls(**{k: float(v) for k, v in (d.pop("controls", None) or {}).items()
                               if k in BrainControls.__dataclass_fields__})
        m = str(d.get("mode", "") or "deterministic").strip()
        mode = MODES.get(m.upper(), m) if len(m) == 1 else m
        return cls(enabled=bool(d.get("enabled", False)), mode=mode if mode in MODES.values() else "deterministic",
                   controls=ctl, kev_url=str(d.get("kev_url", "") or ""), kev_timeout=float(d.get("kev_timeout", 0.8)),
                   kev_candidates=int(d.get("kev_candidates", 6)),
                   kev_ask="liked" if str(d.get("kev_ask", "")) == "liked" else "choice",
                   memory_size=int(d.get("memory_size", 96)), seed=int(d.get("seed", 0)),
                   log_dir=str(d.get("log_dir", "") or ""), reshape=bool(d.get("reshape", True)),
                   taste=bool(d.get("taste", True)))

    def to_dict(self) -> dict:
        return {"enabled": self.enabled, "mode": self.mode, "controls": asdict(self.controls), "kev_url": self.kev_url,
                "kev_timeout": self.kev_timeout, "kev_candidates": self.kev_candidates, "kev_ask": self.kev_ask,
                "memory_size": self.memory_size, "seed": self.seed, "log_dir": self.log_dir, "reshape": self.reshape,
                "taste": self.taste}


# Feature flags (read once, when a live session starts; the app's own settings are the everyday way).
ENV = {"MYRMEX_BRAIN_ENABLED": "enabled", "MYRMEX_BRAIN_MODE": "mode", "MYRMEX_BRAIN_KEV_URL": "kev_url",
       "MYRMEX_BRAIN_KEV_TIMEOUT": "kev_timeout", "MYRMEX_BRAIN_MEMORY_SIZE": "memory_size"}
ENV_CONTROLS = {"MYRMEX_BRAIN_HZ": "rate_hz", "MYRMEX_BRAIN_AUTONOMY": "autonomy", "MYRMEX_BRAIN_NOVELTY": "novelty",
                "MYRMEX_BRAIN_PERSISTENCE": "persistence", "MYRMEX_BRAIN_MUTATION": "mutation",
                "MYRMEX_BRAIN_RETURN": "returns", "MYRMEX_BRAIN_MEMORY": "memory",
                "MYRMEX_BRAIN_MIN_CONFIDENCE": "min_confidence"}


def with_env(d: dict | None, environ: dict | None = None) -> dict:
    """The brain's settings (a BrainConfig dict) with any ``MYRMEX_BRAIN_*`` environment flags on top:
    ENABLED (1/0), MODE (A-G or a name), HZ, AUTONOMY, NOVELTY, PERSISTENCE, MUTATION, RETURN, MEMORY (0..1),
    MEMORY_SIZE, MIN_CONFIDENCE, KEV_URL, KEV_TIMEOUT.  Malformed values are ignored."""
    env = os.environ if environ is None else environ
    out = dict(d or {})
    ctl = dict(out.get("controls") or {})
    for var, key in ENV.items():
        v = env.get(var)
        if v is None or v == "":
            continue
        try:
            if key == "enabled":
                out[key] = v.strip().lower() in ("1", "true", "yes", "on")
            elif key in ("kev_timeout",):
                out[key] = float(v)
            elif key == "memory_size":
                out[key] = max(8, int(v))
            else:
                out[key] = v.strip()
        except ValueError:
            continue
    for var, key in ENV_CONTROLS.items():
        v = env.get(var)
        if v is None or v == "":
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if math.isfinite(x):
            ctl[key] = max(0.05, min(10.0, x)) if key == "rate_hz" else max(0.0, min(1.0, x))
    if ctl:
        out["controls"] = ctl
    return out


@dataclass
class Snapshot:
    t: float
    fp: Fingerprint
    blend: dict                      # the engine's current target blend
    free: bool = True                # the engine may change form now (not evading, striking, sculpted ...)
    due: bool = False                # the engine's own state machine is about to choose the next form
    energy: float = 0.0
    playing: bool = False
    music: str = ""
    user: UserCue = field(default_factory=UserCue)
    controls: dict = field(default_factory=dict)


@dataclass
class Decision:
    t: float
    op: str
    blend: dict
    material: str | None = None
    event: str | None = None
    strength: float = 2.6
    hold: float = 8.0
    source: str = "det"
    confidence: float | None = None
    novelty: float = 0.0
    utility: float = 0.0
    label: str = ""
    latency_ms: float = 0.0
    probs: dict | None = None
    deform: dict | None = None       # the form stretched / twisted / bent ... (creature.colony.DEFORM names)

    def to_dict(self) -> dict:
        return asdict(self)


class BrainCore:
    def __init__(self, vocab: Vocabulary, cfg: BrainConfig | None = None, kev: KevClient | None = None):
        self.vocab, self.cfg = vocab, cfg or BrainConfig()
        self.scale = Scale.from_geometry(vocab.geometry)
        self.memory = MorphMemory(capacity=self.cfg.memory_size)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.kev = kev if kev is not None else _client(self.cfg)
        self.last_commit = -1e9
        self.last_radical = -1e9
        self.last_event = 0.0
        self.recent_ops: list[str] = []                      # the kinds of change made lately (variety)
        self.pending_goal: tuple[float, dict] | None = None
        self.stats = {"decisions": 0, "commits": 0, "holds": 0, "yields": 0, "kev_calls": 0, "kev_fail": 0,
                      "kev_lowconf": 0, "kev_ms": [], "step_ms": [], "steps": 0, "step_ms_total": 0.0}
        self.log: list[dict] = []
        self.logged = 0                                     # decision-log lines so far (the log keeps the last 2000)
        self.taste = None
        self.load_taste()

    def load_taste(self) -> None:
        """Your Good / Bad from the Train page (``taste-<organism>.json`` in the log folder), when there is some."""
        from .taste import Taste, taste_path
        self.taste = None
        path = taste_path(self.cfg.log_dir, self.vocab.organism)
        if self.cfg.taste and path and os.path.isfile(path):
            t = Taste.load(path)
            self.taste = t if t.n >= 5 else None

    # ------------------------------------------------------------------ the tick
    def step(self, snap: Snapshot) -> Decision | None:
        t0 = time.perf_counter()
        try:
            return self._step(snap)
        finally:
            ms = (time.perf_counter() - t0) * 1000.0
            self.stats["step_ms"].append(ms)
            del self.stats["step_ms"][:-500]
            self.stats["steps"] += 1
            self.stats["step_ms_total"] += ms

    def _step(self, snap: Snapshot) -> Decision | None:
        ctl = self.cfg.controls.merged(snap.controls)
        t = snap.t
        emb = embed(snap.fp, self.scale)
        goal = None
        if self.pending_goal is not None and t - self.pending_goal[0] < 8.0:
            goal = self.pending_goal[1]
        self.memory.observe(t, snap.fp, emb, goal or {"blend": dict(snap.blend)}, energy=snap.energy)
        mode = self.cfg.mode
        if mode == "current" or ctl.autonomy <= 0.01:
            return None
        if not snap.free or snap.user.sculpt:
            self.stats["yields"] += 1
            return None
        interval = ctl.interval()
        # When: the engine's own decision points are taken over with probability = autonomy (the organism's
        # tempo of change stays its own); above 0.6 it also changes between them, every ``interval``.
        if snap.due:
            if self.rng.random() >= ctl.autonomy:
                return None
        elif not (ctl.autonomy > 0.6 and t - self.last_commit >= interval
                  and self.memory.residence(t) >= 0.45 * interval):
            return None
        if snap.user.present and snap.user.motion > 0.5 and ctl.autonomy < 0.5:
            self.stats["yields"] += 1                         # the hand is busy: let it lead
            return None
        self.stats["decisions"] += 1
        cands = generate(self.vocab, snap.blend, snap.fp, self.memory, t, snap.energy, ctl, self.rng,
                         reshape=self.cfg.reshape and self.vocab.family == "colony")
        self._cands = cands
        if mode == "random":
            pool = [c for c in cands if c.op in ("SHIFT", "HYBRID")]
            c = pool[int(self.rng.integers(len(pool)))] if pool else None
            return self._commit(snap, c, ctl, "random", interval) if c else None
        evaluate(cands, self.vocab, self.scale, snap.fp, self.memory, t, snap.energy, snap.user, ctl)
        if mode == "novelty":
            pool = [c for c in cands if c.op in ("SHIFT", "HYBRID")]
            c = max(pool, key=lambda c: c.feats["novelty"]) if pool else None
            return self._commit(snap, c, ctl, "novelty", interval) if c else None
        U = self._utilities(cands, ctl, snap)
        if mode in KEV_MODES and self.kev is not None:
            got = self._kev_choose(mode, cands, U, ctl, snap)
            if got is not None:
                c, source, conf, probs, level, ms = got
                return self._commit(snap, c, ctl, source, interval, level, kev=(conf, probs, ms))
        i = self._boltzmann(U, ctl)
        return self._commit(snap, cands[i], ctl, "det" if mode not in KEV_MODES else "fallback", interval,
                            utility=float(U[i]))

    # ------------------------------------------------------------------ deterministic arbiter
    def _utilities(self, cands: list[Candidate], ctl: BrainControls, snap: Snapshot) -> np.ndarray:
        t, u = snap.t, snap.user
        w_nov = (0.4 + 1.2 * ctl.novelty) * (1.0 + 0.8 * u.motion if u.present else 1.0)
        w_con = 0.8 * (1.0 - 0.6 * ctl.mutation)
        w_idn = 0.6 * (1.0 - 0.5 * ctl.autonomy)
        w_usr = 1.2 * (1.0 - 0.6 * ctl.autonomy) if u.present else 0.0
        w_ret = 0.9 * ctl.returns
        gap = lerp(60.0, 20.0, ctl.autonomy * ctl.novelty)
        interval = ctl.interval()
        held = min(1.0, self.memory.residence(t) / interval)
        # Large events and pushing the form further are changes novelty hardly sees (the shape stays): an
        # appetite for an event grows since the last one (faster with autonomy x novelty, stronger with the
        # music's energy); a form held a while may be pushed further instead of left.
        appetite = 1.0 - math.exp(-(t - self.last_event) / lerp(150.0, 40.0, ctl.autonomy * ctl.novelty))
        ops = self.recent_ops[-6:]
        U = np.zeros(len(cands))
        cur_emb = embed(snap.fp, self.scale) if self.taste is not None else None
        for i, c in enumerate(cands):
            f = c.feats
            v = (w_nov * f["novelty"] + w_con * f["continuity"] + w_idn * f["identity"] + w_usr * f["user"]
                 + w_ret * f["pull"] - 1.0 * f["repetition"] - 1.5 * f["oscillation"])
            if ops and c.op != "HOLD":                        # not the same kind of change over and over
                v -= 0.8 * ops.count(c.op) / len(ops)
            if f["radical"] and t - self.last_radical < gap:
                v -= 1.5
            if c.op == "HOLD":
                v += 0.6 * ctl.persistence * max(0.0, 1.0 - self.memory.residence(t) / (2.0 * interval))
            if c.op == "EVENT":
                v += 0.9 * appetite * (0.4 + snap.energy)
                if c.event == "COLLAPSE":
                    v -= 0.8 * (1.0 - ctl.autonomy)
            elif c.op in ("INTENSIFY", "DISSOLVE"):
                v += 0.5 * held * (0.5 + ctl.mutation)
            if cur_emb is not None and c.op != "HOLD" and c.fp is not None:   # your Good / Bad (Train page)
                from .taste import candidate_features
                v += 0.35 * float(np.clip(self.taste.logit(candidate_features(c, cur_emb, self.scale)), -3.0, 3.0))
            c.utility = float(v)
            U[i] = v
        return U

    def _boltzmann(self, U: np.ndarray, ctl: BrainControls) -> int:
        tau = 0.06 + 0.22 * ctl.autonomy * (0.5 + ctl.novelty)
        p = np.exp((U - U.max()) / tau)
        p = p / p.sum()
        return int(self.rng.choice(len(U), p=p))

    # ------------------------------------------------------------------ Kev
    def _kev_choose(self, mode: str, cands: list[Candidate], U: np.ndarray, ctl: BrainControls, snap: Snapshot):
        summary = self.memory.summary(snap.t, self.vocab.forms) if mode != "kev_direct" else None
        state = state_text(self.vocab.organism, self.vocab.free, self.vocab.signature(), summary)
        ctx = {"music": snap.music or ("playing" if snap.playing else "silent") + f", energy {snap.energy:.1f}",
               "hand": _hand_words(snap.user)}
        liked = mode == "kev_candidates" and self.cfg.kev_ask == "liked"
        if mode == "kev_candidates":
            order = np.argsort(-U)[:max(2, self.cfg.kev_candidates)]
            pool = [cands[i] for i in order]
            if liked:                                         # "would the performer like it?" for each (Train page)
                questions = liked_questions([c.describe() for c in pool], _blend_words(snap.fp.w, self.vocab.forms))
                keys = list(questions)
            else:
                questions = candidates_questions(pool, ctx)
                keys = [chr(ord("A") + i) for i in range(len(pool))]
        else:
            pool, keys = [], []
            by_form = {}
            for c in cands:
                if c.op == "SHIFT":
                    by_form[f"become {next(iter(c.blend))}"] = c
                elif c.op == "EVENT":
                    by_form[str(c.event).lower()] = c
                elif c.op == "HOLD":
                    by_form["stay as it is"] = c
            keys, pool = list(by_form), list(by_form.values())
            questions = operations_questions(keys, ctx)
        self.stats["kev_calls"] += 1
        self.last_request = (state, questions, keys)          # (the Kev benchmark replays real requests)
        t0 = time.perf_counter()
        try:
            ans = self.kev.ask(state, questions)
        except KevError:
            self.stats["kev_fail"] += 1
            return None
        ms = (time.perf_counter() - t0) * 1000.0
        self.stats["kev_ms"].append(ms)
        del self.stats["kev_ms"][:-500]
        if liked:
            probs = [float(ans[k].get("noul", 0.0)) for k in keys]
        else:
            probs = [float(ans["next"]["probabilities"].get(k, 0.0)) for k in keys]
        s = sum(probs)
        if s <= 0:
            self.stats["kev_fail"] += 1
            return None
        pk = np.array(probs) / s
        conf = choice_confidence(list(pk))
        level = None if liked else float(ans["intensity"].get("score", 1.0))
        if mode != "kev_candidates":                         # D / E: Kev's own pick
            j = int(np.argmax(pk))
            return pool[j], "kev", conf, dict(zip(keys, map(float, pk))), level, ms
        hi = max(0.5, ctl.min_confidence + 0.25)
        Ud = np.array([c.utility for c in pool])
        tau = 0.06 + 0.22 * ctl.autonomy * (0.5 + ctl.novelty)
        pd = np.exp((Ud - Ud.max()) / tau)
        pd = pd / pd.sum()
        if conf >= hi:
            j, source = int(np.argmax(pk)), "kev"
        elif conf >= ctl.min_confidence:
            mix = (1.0 - conf) * pd + conf * pk
            j, source = int(self.rng.choice(len(pool), p=mix / mix.sum())), "kev+det"
        else:
            self.stats["kev_lowconf"] += 1
            j, source = int(self.rng.choice(len(pool), p=pd)), "det-lowconf"
        return pool[j], source, conf, dict(zip(keys, map(float, pk))), level, ms

    # ------------------------------------------------------------------ committing
    def _commit(self, snap: Snapshot, c: Candidate | None, ctl: BrainControls, source: str, interval: float,
                level: float | None = None, utility: float = 0.0, kev: tuple | None = None) -> Decision | None:
        if c is None or c.op == "HOLD":
            self.stats["holds"] += 1
            self.last_commit = snap.t - 0.5 * interval          # look again a little sooner
            self._log(snap, c, source, kev, None)
            return None
        lv = 1.0 + 1.2 * ctl.mutation if level is None else level
        strength = float(np.clip(c.strength * (0.85 + 0.12 * lv), 1.8, 3.8))
        hold = lerp(4.0, 14.0, ctl.persistence) * float(self.rng.uniform(0.8, 1.25)) * (0.85 + 0.1 * lv)
        nov = float(c.feats.get("novelty", 0.0)) if c.feats else 0.0
        if c.feats and c.feats.get("radical"):
            self.last_radical = snap.t
        if c.op == "EVENT":
            self.last_event = snap.t
        self.recent_ops = (self.recent_ops + [c.op])[-12:]
        self.last_commit = snap.t
        self.stats["commits"] += 1
        self.pending_goal = (snap.t, {"blend": dict(c.blend), "material": c.material})
        d = Decision(snap.t, c.op, dict(c.blend), c.material, c.event, strength, hold, source, None, nov,
                     utility or c.utility, c.describe(), deform=dict(c.deform) if c.deform else None)
        if kev is not None:
            d.confidence, d.probs, d.latency_ms = kev
        self._log(snap, c, source, kev, d)
        return d

    def _log(self, snap: Snapshot, c: Candidate | None, source: str, kev: tuple | None, d: Decision | None) -> None:
        """One line per decision (committed or held): what it was, what it could become, what was chosen."""
        cands = sorted(getattr(self, "_cands", None) or [], key=lambda x: -x.utility)
        self.log.append({
            "t": round(snap.t, 2), "current": _blend_words(snap.fp.w, self.vocab.forms),
            "target": _blend_words(self.vocab.weights(snap.blend), self.vocab.forms) if snap.blend else "",
            "candidates": [{"op": x.op, "label": x.describe(), "novelty": round(float(x.feats.get("novelty", 0.0)), 3),
                            "utility": round(float(x.utility), 3)} for x in cands[:8] if x.feats],
            "chosen": c.describe() if c is not None else "HOLD", "op": c.op if c is not None else "HOLD",
            "source": source, "confidence": None if kev is None else round(float(kev[0]), 3),
            "probs": None if kev is None else {k: round(v, 3) for k, v in kev[1].items()},
            "kev_ms": None if kev is None else round(float(kev[2]), 1),
            "strength": None if d is None else round(d.strength, 2), "hold": None if d is None else round(d.hold, 1),
            "novelty": round(float(c.feats.get("novelty", 0.0)), 3) if c is not None and c.feats else 0.0,
            "feats": {k: round(float(v), 3) for k, v in ((c.feats or {}) if c is not None else {}).items()}})
        del self.log[:-2000]
        self.logged += 1

    # ------------------------------------------------------------------ feedback
    def mark(self, good: bool) -> None:
        """The performer liked (or not) what it is now - memory salience (and a label for a later fine-tune)."""
        self.memory.mark(1.0 if good else -1.0)


def _client(cfg: BrainConfig) -> KevClient | None:
    """The Kev client of a config (None without a URL, or with one that is not local - then the deterministic
    arbiter decides)."""
    if not cfg.kev_url:
        return None
    try:
        return KevClient(cfg.kev_url, cfg.kev_timeout)
    except KevError:
        return None


def _blend_words(w, forms: tuple, floor: float = 0.12) -> str:
    w = np.asarray(w, float)
    if not len(w) or w.sum() <= 0:
        return ""
    w = w / w.sum()
    return " + ".join(f"{forms[i]} {w[i]:.0%}" for i in np.argsort(w)[::-1][:3] if w[i] >= floor)


def _hand_words(u: UserCue) -> str:
    if not u.present:
        return "not present"
    shape = "open" if u.open > 0.65 else ("closed into a fist" if u.open < 0.35 else "half open")
    extra = []
    if u.spin > 0.4:
        extra.append("turning")
    if u.motion > 0.4:
        extra.append("moving fast")
    if u.gesture:
        extra.append(f"just made a {u.gesture.lower()}")
    return ", ".join([shape] + extra)


__all__ = ["BrainCore", "BrainConfig", "BrainControls", "Snapshot", "Decision", "MODES", "KEV_MODES", "INTENSITY",
           "with_env", "ENV", "ENV_CONTROLS"]
