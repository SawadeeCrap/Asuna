"""Training the brain's taste: one change of the body at a time, held until the performer says Good or Bad.

The *Train* page of the app (``LiveSession.set_training``): the organism alone - no music, no hand, no knobs,
no effects, its own state machine held, the camera circling it slowly - and the brain shows one change of its
body at a time.  The change stays until you judge it; Good or Bad (or Skip) brings the next one at once.

A proposal is a change the body can really make:

* a form of the organism's latent space - one of its own, or any form it knows (``range`` "own" | "all") -
  pure or a blend of two;
* the form changed beyond the blend where the engine can (the colony family, creature.colony.deform):
  stretched or squashed along its axes, twisted, bent, rippled, scaled - ``spread`` (0..1) sets how far;
* a material state (fluid ... hard, dispersed).

Every step prepares 30, keeps the 6 farthest (in the brain's morphological distance) from what the body is
and from what was just shown - each step is a clear change - and one of the 6 is shown:

* **Kev** picks when a local Kev server is set: for each of the 6 the yes / no question "would the performer
  like it to become ...?" (Kev's ``noul``, kev_client.liked_question); the likeliest yes is shown (``source``
  "kev"), now and then another one (``explore``: its no-go areas get verdicts too);
* otherwise the **taste model** (brain/taste.py) learned from your verdicts so far picks, with exploration
  that shrinks as it learns (``taste`` / ``explore``).

What is kept (``log_dir``, the app's ``~/Myrmex/brain``):

* ``train-<organism>.jsonl`` - every verdict: the change, its features, who picked, what Kev and the taste
  model expected;
* ``kev-<organism>.jsonl`` - Kev's fine-tune records: exactly the question Kev is asked, labelled with your
  verdict (``kev.train --data`` reads them as they are, see docs/MORPHOLOGY_BRAIN.md);
* ``taste-<organism>.json`` - the taste model, refit on every verdict; the live brain adds it to its arbiter.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .candidates import _blend_name, deform_words, predict, random_deform
from .core import Decision, _blend_words
from .fingerprint import Fingerprint, Scale, embed
from .kev_client import KevClient, KevError, liked_question, liked_questions, state_text
from .taste import Taste, features, taste_path
from .vocab import MATERIALS, SITUATIONAL, Vocabulary

N_PREPARED = 30                     # proposals prepared per step
N_OPTIONS = 6                       # ... the farthest of them considered (what Kev is asked about)
MATERIAL_WORDS = {"FLUID": "fluid", "ELASTIC": "elastic", "COHESIVE": "cohesive", "STRUCTURED": "structured",
                  "HIGH_STIFFNESS": "hard", "DISPERSED": "dispersed"}


@dataclass
class Proposal:
    step: int
    blend: dict
    material: str | None
    deform: dict
    label: str
    fp: Fingerprint | None = None
    emb: np.ndarray | None = None
    x: dict = field(default_factory=dict)            # the taste model's features
    source: str = ""                                 # kev | taste | explore
    p_kev: float | None = None                       # Kev: p(the performer likes it)
    p_taste: float | None = None                     # the taste model: p(good)
    options: list = field(default_factory=list)      # the labels of the 6 considered
    before: str = ""                                 # what the body was (the question's "it is now ...")
    kev_ms: float | None = None

    def decision(self, t: float = 0.0) -> Decision:
        """For the adapter: held until the verdict (``hold`` ~ forever)."""
        return Decision(t, "TRAIN", dict(self.blend), self.material, None, 2.8, 1e9, self.source, self.p_kev,
                        label=self.label, deform=dict(self.deform) or None)

    def summary(self) -> dict:
        return {"step": self.step, "label": self.label, "source": self.source,
                "p_kev": None if self.p_kev is None else round(self.p_kev, 3),
                "p_taste": None if self.p_taste is None else round(self.p_taste, 3),
                "kev_ms": None if self.kev_ms is None else round(self.kev_ms, 1)}


def describe(blend: dict, deform: dict | None, material: str | None) -> str:
    parts = [_blend_name(blend)]
    w = deform_words(deform)
    if w:
        parts.append(w)
    if material:
        parts.append(MATERIAL_WORDS.get(material, material.lower()))
    return ", ".join(parts)


def _count_lines(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


class Trainer:
    """Proposals, the pick among them, the verdicts (see the module docstring).  No engine here."""

    def __init__(self, vocab: Vocabulary, log_dir: str = "", range_: str = "all", spread: float = 0.8,
                 kev: KevClient | None = None, deform: bool = True, seed: int | None = None):
        self.vocab = vocab
        self.scale = Scale.from_geometry(vocab.geometry)
        self.range = "own" if range_ == "own" else "all"
        self.spread = min(1.0, max(0.0, float(spread)))
        self.kev = kev
        self.can_deform = bool(deform)
        self.rng = np.random.default_rng(seed)
        self.log_dir = log_dir
        org = vocab.organism or "organism"
        self.paths = {"log": os.path.join(log_dir, f"train-{org}.jsonl") if log_dir else "",
                      "kev": os.path.join(log_dir, f"kev-{org}.jsonl") if log_dir else "",
                      "taste": taste_path(log_dir, org)}
        self.taste = Taste.load(self.paths["taste"]) if self.paths["taste"] else Taste()
        self.kev_records = _count_lines(self.paths["kev"]) if self.paths["kev"] else 0
        self.state = state_text(vocab.organism, vocab.free, vocab.signature(), None)
        self.shown: list[Proposal] = []
        self.current: Proposal | None = None
        self.step = 0
        self.counts = {"good": 0, "bad": 0, "skip": 0}
        self.kev_hits: list[float] = []
        self.error = ""

    # ------------------------------------------------------------------ settings
    def configure(self, range_: str | None = None, spread: float | None = None) -> None:
        if range_ is not None:
            self.range = "own" if range_ == "own" else "all"
        if spread is not None:
            self.spread = min(1.0, max(0.0, float(spread)))

    def forms(self) -> list[str]:
        own = [f for f in self.vocab.free if f in self.vocab.forms]
        if self.range == "own" and own:
            return own
        return [f for f in self.vocab.forms if f not in SITUATIONAL]

    # ------------------------------------------------------------------ proposals
    def _candidate(self) -> Proposal:
        rng, forms = self.rng, self.forms()
        f = forms[int(rng.integers(len(forms)))]
        blend = {f: 1.0}
        if len(forms) > 1 and rng.random() < 0.3:                     # now and then a blend of two
            g = forms[int(rng.integers(len(forms) - 1))]
            g = g if g != f else forms[-1]
            m = float(rng.uniform(0.55, 0.8))
            blend = {f: m, g: 1.0 - m}
        mat = str(MATERIALS[int(rng.integers(len(MATERIALS)))]) if self.vocab.materials and rng.random() < 0.6 else None
        d = random_deform(rng, self.spread) if self.can_deform else {}
        return Proposal(0, blend, mat, d, describe(blend, d, mat))

    def propose(self, cur: Fingerprint) -> Proposal:
        """The next change to show, far from what the body is and from what was just shown."""
        cur_emb = embed(cur, self.scale)
        pool = [self._candidate() for _ in range(N_PREPARED)]
        for p in pool:
            p.fp = predict(self.vocab, p.blend, p.material, None, cur, p.deform or None)
            p.emb = embed(p.fp, self.scale)
            p.x = features(p.fp, p.emb, cur_emb, p.blend, p.material, p.deform, self.scale)
        refs = [cur_emb] + [q.emb for q in self.shown[-4:] if q.emb is not None]
        options: list[Proposal] = []
        while len(options) < min(N_OPTIONS, len(pool)):             # farthest-point: each a clear change
            R = refs + [o.emb for o in options]
            rest = [p for p in pool if all(p is not o for o in options)]
            options.append(max(rest, key=lambda p: min(float(np.linalg.norm(p.emb - r)) for r in R)))
        before = self.shown[-1].label if self.shown else _blend_words(cur.w, self.vocab.forms)
        j, source = self._pick(options, before)
        pick = options[j]
        pick.source, pick.before = source, before
        pick.options = [o.label for o in options]
        pick.p_taste = self.taste.p(pick.x) if self.taste.n else None
        self.step += 1
        pick.step = self.step
        self.current = pick
        self.shown = (self.shown + [pick])[-12:]
        return pick

    def _pick(self, options: list[Proposal], before: str) -> tuple[int, str]:
        if self.kev is not None:
            t0 = time.perf_counter()
            try:
                ans = self.kev.ask(self.state, liked_questions([o.label for o in options], before))
                pk = [float(ans[f"c{i}"].get("noul", 0.0)) for i in range(len(options))]
                ms = (time.perf_counter() - t0) * 1000.0
                for o, p in zip(options, pk):
                    o.p_kev, o.kev_ms = p, ms
                self.error = ""
                if self.rng.random() < 0.15:                          # the ones Kev would not show get verdicts too
                    return int(self.rng.integers(len(options))), "explore"
                return int(np.argmax(pk)), "kev"
            except (KevError, KeyError, TypeError, ValueError) as e:
                self.error = f"Kev: {e}"
        n = self.taste.n
        sigma = 2.0 / math.sqrt(1.0 + n / 8.0)                       # explores while it knows little
        scores = [self.taste.logit(o.x) + sigma * float(self.rng.standard_normal()) for o in options]
        return int(np.argmax(scores)), ("taste" if n >= 10 else "explore")

    # ------------------------------------------------------------------ verdicts
    def rate(self, good: bool | None) -> bool:
        """Good (True), Bad (False) or Skip (None) for the change shown.  -> whether there was one."""
        p = self.current
        if p is None:
            return False
        self.current = None
        if good is None:
            self.counts["skip"] += 1
            self._write(self.paths["log"], {"kind": "skip", **self._row(p)})
            return True
        good = bool(good)
        self.counts["good" if good else "bad"] += 1
        if p.p_kev is not None:
            self.kev_hits.append(1.0 if (p.p_kev > 0.5) == good else 0.0)
        self._write(self.paths["log"], {"kind": "verdict", "good": good, **self._row(p)})
        rec = {"state": self.state, "questions": {"liked": {**liked_question(p.label, p.before), "label": good}}}
        if self._write(self.paths["kev"], rec):
            self.kev_records += 1
        self.taste.add(p.x, good)
        self.taste.save()
        return True

    def _row(self, p: Proposal) -> dict:
        return {"organism": self.vocab.organism, "step": p.step, "label": p.label, "before": p.before,
                "blend": p.blend, "material": p.material, "deform": p.deform, "source": p.source,
                "p_kev": p.p_kev, "p_taste": p.p_taste, "options": p.options, "features": p.x,
                "range": self.range, "spread": self.spread}

    def _write(self, path: str, rec: dict) -> bool:
        if not path:
            return False
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"wall": round(time.time(), 2), **rec} if "questions" not in rec else rec,
                                   default=float) + "\n")
            return True
        except (OSError, TypeError, ValueError) as e:
            self.error = f"{os.path.basename(path)}: {e}"
            return False

    # ------------------------------------------------------------------ reading
    def status(self) -> dict:
        t = self.taste
        likes = [(k, round(v, 2)) for k, v in t.top(12) if v > 0][:4]
        dislikes = [(k, round(v, 2)) for k, v in t.top(12) if v < 0][:4]
        kh = self.kev_hits[-20:]
        return {"step": self.step, "counts": dict(self.counts), "range": self.range, "spread": round(self.spread, 2),
                "current": self.current.summary() if self.current is not None else None,
                "taste": {"n": t.n, "accuracy": t.accuracy(), "likes": likes, "dislikes": dislikes},
                "kev": {"on": self.kev is not None, "accuracy": float(np.mean(kh)) if len(kh) >= 5 else None,
                        "records": self.kev_records},
                "files": dict(self.paths), "error": self.error}


class TrainingRun:
    """The training mode on a live organism (``LiveSession.set_training``): the engine held, one proposal on
    the body at a time.  ``begin`` / ``end`` / ``snapshot`` / the proposal's application run under the
    session's lock; preparing a proposal (Kev: ~0.15 s) runs on a helper thread - the engine never waits."""
    MORPH_TAU = 0.5                  # a new form is there in ~1.5 s
    PLASTIC = 0.35                   # ... and the network takes it on sooner

    def __init__(self, backend, organism: str, cfg: dict | None = None, log_dir: str = "", kev_url: str = ""):
        from .adapter import make_adapter
        cfg = dict(cfg or {})
        self.backend, self.engine = backend, backend.engine
        self.adapter = make_adapter(self.engine, organism)
        kev, self.error = None, ""
        if cfg.get("kev") and kev_url:
            try:
                kev = KevClient(kev_url, timeout=float(cfg.get("kev_timeout", 3.0)))
            except KevError as e:
                self.error = f"Kev: {e}"
        self.trainer = Trainer(self.adapter.vocab, log_dir, cfg.get("range", "all"), float(cfg.get("spread", 0.8)),
                               kev, deform=self.adapter.supports_deform)
        self.busy = False
        self.saved: dict = {}

    # --- under the session lock
    def begin(self) -> None:
        from ..creature.config import PARAMS
        e = self.engine
        self.saved = {"morph_tau": getattr(e, "morph_tau", None), "plastic_scale": getattr(e, "plastic_scale", 1.0)}
        if hasattr(e, "morph_tau"):
            e.morph_tau, e.plastic_scale = self.MORPH_TAU, self.PLASTIC
        for p in PARAMS:                                          # the knobs back to the organism's own
            e.set_parameter(p, None)
        plan = getattr(e, "PLAN", {})
        for b in self.adapter._all():
            if hasattr(b, "intent") and "HOVER" in plan and b.intent in ("CRUISE", "EXPLORE", "DISPLAY", "PATROL",
                                                                         "FORMATION", "HOVER"):
                b.intent = "HOVER"                                # calm, in place: the shape is what is seen
        self.adapter._set_timers(1e9, everyone=True)

    def end(self) -> None:
        e = self.engine
        if hasattr(e, "morph_tau"):
            e.morph_tau = self.saved.get("morph_tau")
            e.plastic_scale = self.saved.get("plastic_scale", 1.0)
        for b in self.adapter._all():
            if hasattr(b, "dfm_goal"):
                b.dfm_goal = np.zeros(len(b.dfm_goal))            # the form as it is again
        self.adapter._set_timers(2.0, everyone=True)            # its own state machine goes on in 2 s
        self.adapter.release()

    def snapshot(self) -> Fingerprint:
        from .candidates import UserCue
        return self.adapter.snapshot(0.0, self.backend.state, UserCue(), {}).fp

    # --- from the app or a MIDI pad (any thread - the engine's own included: the work is on a helper thread)
    def next(self, lock) -> bool:
        """Prepare and show the next change.  False: one is already being prepared."""
        return self._job(lock, None, False)

    def rate(self, good: bool | None, lock) -> bool:
        """The verdict on the change shown (learning from it, writing it down), then the next one at once."""
        if self.trainer.current is None:
            return False
        return self._job(lock, good, True)

    def _job(self, lock, good, rated: bool) -> bool:
        if self.busy:
            return False
        self.busy = True

        def work():
            try:
                if rated and not self.trainer.rate(good):
                    return
                with lock:
                    fp = self.snapshot()
                p = self.trainer.propose(fp)
                with lock:
                    self.adapter.apply(p.decision(), 0.0, everyone=True)
            except Exception as e:                               # never the engine's problem
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self.busy = False
        threading.Thread(target=work, name="myrmex-train", daemon=True).start()
        return True

    def status(self) -> dict:
        st = self.trainer.status()
        st.update(on=True, busy=self.busy, organism=self.adapter.vocab.organism, family=self.adapter.vocab.family,
                  deform=self.adapter.supports_deform, forms=len(self.trainer.forms()))
        if self.error and not st.get("error"):
            st["error"] = self.error
        return st


__all__ = ["Trainer", "TrainingRun", "Proposal", "describe", "N_PREPARED", "N_OPTIONS"]
