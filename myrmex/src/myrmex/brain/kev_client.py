"""A local Kev server as the brain's arbiter - the documented System One API, standard library only.

Kev (github.com/jaredpalmer/kev) answers typed questions about a *state* in one prefill pass: ``choice``
(a distribution over named options), ``score`` (an expected level over ordered levels) and ``noul``
(p(yes)); probabilities are temperature-calibrated in its training domain.  Run it next to Myrmex:

    uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009     # MLX on Apple Silicon

What the brain asks (never free text back):

* ``next``      choice   - which of the (pre-computed, pre-validated) candidates to become;
* ``intensity`` score    - subtle .. radical (sets strength and how long it holds).

The state is the part that changes slowly (the organism, its forms, its memory summary - quantised), the
per-decision context goes into the question's instructions: Kev caches a repeated state, so a decision
that only changes the question pays for the question rows alone (Kev README: 28 ms vs 149 ms, Kev-0.8B,
Apple M5).  Kev never computes distances, never invents forms and never runs per frame.
"""
from __future__ import annotations

import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request

LETTERS = "ABCDEFGHIJKL"
INTENSITY = ["subtle", "moderate", "strong", "radical"]


class KevError(RuntimeError):
    pass


def is_local(url: str) -> bool:
    """Local-first: this machine or the local network (an IP in a private / loopback / link-local range,
    ``localhost``, ``*.local``) - never a hosted API."""
    host = (urllib.parse.urlsplit(url).hostname or "").strip("[]").lower()
    if host in ("localhost",) or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


class KevClient:
    def __init__(self, url: str = "http://127.0.0.1:8009", timeout: float = 0.8, model: str = "kev-latest",
                 api_key: str | None = None):
        if not is_local(url):
            raise KevError(f"{url}: the morphology brain only talks to a Kev server on this machine or the local "
                           "network (no hosted APIs)")
        self.url, self.timeout, self.model, self.api_key = url.rstrip("/"), float(timeout), model, api_key
        self.calls = self.failures = 0
        self.last_latency_ms = self.last_model_ms = None
        self._open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open   # (local: no proxies)

    def _post(self, path: str, body: dict) -> dict:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.url + path, data=data, method="POST",
                                     headers={"content-type": "application/json",
                                              **({"authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        t0 = time.perf_counter()
        self.calls += 1
        try:
            with self._open(req, timeout=self.timeout) as r:
                out = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            self.failures += 1
            raise KevError(f"{type(e).__name__}: {e}") from None
        self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_model_ms = out.get("latency_ms")
        return out

    def models(self) -> dict:
        """GET /v1/models as the server returns it (Kev: ``{"models": [{"name", "run", "backend", ...}]}``)."""
        try:
            with self._open(self.url + "/v1/models", timeout=max(self.timeout, 2.0)) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise KevError(f"{type(e).__name__}: {e}") from None

    def ask(self, state, questions: dict) -> dict:
        """-> answers (validated): {id: {...}} as Kev returns them."""
        out = self._post("/v1/systemone", {"state": state, "model": self.model, "questions": questions})
        ans = out.get("answers")
        if not isinstance(ans, dict) or set(ans) != set(questions):
            self.failures += 1
            raise KevError("malformed answer")
        for qid, q in questions.items():
            a = ans[qid]
            if a.get("type") != q["type"]:
                self.failures += 1
                raise KevError(f"answer {qid}: wrong type")
            if q["type"] in ("choice", "score"):
                probs = a.get("probabilities")
                if not isinstance(probs, dict) or not probs or abs(sum(probs.values()) - 1.0) > 0.05:
                    self.failures += 1
                    raise KevError(f"answer {qid}: bad probabilities")
        return ans


# ---------------------------------------------------------------------------- what is asked
def _words(f: dict) -> str:
    nov = "new" if f["novelty"] > 0.55 else ("fresh" if f["novelty"] > 0.3 else "familiar")
    cont = "smooth" if f["continuity"] > 0.6 else ("a clear change" if f["continuity"] > 0.3 else "abrupt")
    idn = "typical of it" if f["identity"] > 0.6 else ("unusual for it" if f["identity"] < 0.3 else "")
    hand = "follows the hand" if f["user"] > 0.3 else ("against the hand" if f["user"] < -0.3 else "")
    rep = "just done again" if f["repetition"] > 1.0 else ""
    return ", ".join(x for x in (nov, cont, idn, hand, rep) if x)


def state_text(organism: str, forms: tuple, signature: str, summary: dict | None) -> dict:
    """The slowly changing part (Kev caches it)."""
    s = {"organism": organism, "forms it can take": ", ".join(forms[:14]), "its signature form": signature}
    if summary is not None:
        s["memory"] = {"current form": summary.get("current", ""), "held for": f"{summary.get('held_s', 0)} s",
                       "recent forms": "; ".join(f"{n} ({d} s)" for n, d in summary.get("recent", [])),
                       "forms remembered": summary.get("forms_known", 0),
                       "repeating itself": "yes" if summary.get("repeating") else "no"}
    return s


def context_text(ctx: dict) -> str:
    return f"Music: {ctx.get('music', 'unknown')}. Hand: {ctx.get('hand', 'none')}."


def candidates_questions(cands: list, ctx: dict) -> dict:
    criteria = {LETTERS[i]: f"{c.describe()} - {_words(c.feats)}".rstrip(" -") for i, c in enumerate(cands)}
    return {"next": {"type": "choice", "criteria": criteria,
                     "instructions": "An artificial organism is exploring its own morphology. " + context_text(ctx) +
                     " Choose its next transformation: new but growing out of the current form, still recognisably "
                     "itself, not repeating what it just did, following the hand when the hand is active."},
            "intensity": {"type": "score", "criteria": INTENSITY,
                          "instructions": "How strong should that transformation be, given the music and the hand?"}}


def operations_questions(ops: list[str], ctx: dict) -> dict:
    """Direct mode: Kev picks the operation itself (no candidates, no novelty features)."""
    return {"next": {"type": "choice", "criteria": {op: None for op in ops},
                     "instructions": "An artificial organism is exploring its own morphology. " + context_text(ctx) +
                     " Which transformation should it make next?"},
            "intensity": {"type": "score", "criteria": INTENSITY,
                          "instructions": "How strong should that transformation be?"}}


def choice_confidence(p: list[float]) -> float:
    """Kev / TypeSafe: (p_max - 1/K) / (1 - 1/K)."""
    K = len(p)
    return 1.0 if K <= 1 else (max(p) - 1.0 / K) / (1.0 - 1.0 / K)


def describe(info) -> str:
    """One line about the served model, from GET /v1/models: Kev's own ``{"models": [...]}`` ("kev-latest
    (jaredpalmer/kev-0.8b, mlx, bfloat16)"), or an OpenAI-style ``{"data": [{"id"}]}`` list."""
    ms = info.get("models") if isinstance(info, dict) else None
    if isinstance(ms, list) and ms and isinstance(ms[0], dict):
        m = ms[0]
        extra = ", ".join(str(m[k]) for k in ("run", "backend", "dtype") if m.get(k))
        return str(m.get("name", "?")) + (f" ({extra})" if extra else "")
    data = info.get("data") if isinstance(info, dict) else None
    if isinstance(data, list) and data:
        return ", ".join(str(d.get("id", "?")) for d in data[:3] if isinstance(d, dict)) or "a model"
    return "a model"


__all__ = ["KevClient", "KevError", "is_local", "LETTERS", "INTENSITY", "state_text", "candidates_questions",
           "operations_questions", "choice_confidence", "describe"]
