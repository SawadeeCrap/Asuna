"""How well an organism explores its morphology - measured on what it actually became, the same way for
every mode (the brain's own novelty numbers are never used to grade it).

The realised timeline (a fingerprint every ``dt`` seconds) is clustered offline (leader clustering, the
same radius as the memory); an *arrival* is a change of cluster that lasts at least ``min_stay`` (3) seconds - a kick's dash or a quick
evasion is a reaction, not a new form.

    unique_forms            clusters held >= 2 s
    top_share               time in the most occupied cluster (settling: high)
    occupancy_entropy       how evenly time spreads over the clusters it reached (0 .. 1)
    arrivals_per_min        how often it becomes something else
    repetition_rate         arrivals at a cluster already visited in the previous 60 s
    oscillation_rate        arrivals that make A -> B -> A within 30 s
    return_per_10min        arrivals at a cluster last seen >= 120 s before (returning, not looping)
    novelty_mean / _max     each arrival's distance to the k nearest earlier states (time-decayed)
    transition_diversity    normalised entropy of cluster -> cluster transitions
    residence_s             mean time spent per arrival
    jitter                  twitching: the blend's mean second difference while *not* changing form (a smooth
                            transition is not jitter; wobble and back-and-forth are)
    geo_jitter              the same on the shape itself (the geometry descriptors, in the forms' spread): on the
                            real engine this is what is seen - its physics filters the latent's noise
    coverage                share of the free vocabulary that was dominant at least once
    hybrid_share            share of time with no form above 70 % (blends)
    identity                time-averaged identity (how much it stayed itself)
    late_discovery          new clusters found in the last third / first third (keeps exploring or saturates)
    radical_per_10min       arrivals that jump farther than a radical change (candidates.RADICAL: the whole body
                            becomes another form, nothing of the old one kept)
    radical_share           ... as a share of all arrivals (full-body resets: occasional, not constant)
    brain_share             arrivals within 6 s after a brain commit (the brain's influence on what happens)
    hand_follow             correlation of the hand's openness with the body's size (while the hand is there)

``diagnose`` turns them into the questions asked of a run: does it repeat too often, get stuck, oscillate,
become chaotic or too conservative, explore enough, return to old forms?  Thresholds are stated there.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


def cluster(E: np.ndarray, radius: float = 0.3) -> np.ndarray:
    """Leader clustering of embeddings in time order -> label per row."""
    cents: list[np.ndarray] = []
    counts: list[int] = []
    labels = np.zeros(len(E), int)
    for i, e in enumerate(E):
        if cents:
            d = np.linalg.norm(np.stack(cents) - e, axis=1)
            j = int(np.argmin(d))
            if d[j] <= radius:
                labels[i] = j
                counts[j] += 1
                cents[j] = cents[j] + (e - cents[j]) / min(counts[j], 20)
                continue
        cents.append(e.copy())
        counts.append(1)
        labels[i] = len(cents) - 1
    return labels


def arrivals(labels: np.ndarray, dt: float, min_stay: float = 1.0) -> list[tuple[int, int]]:
    """[(index, label)] of changes of cluster that last >= min_stay (flicker ignored)."""
    need = max(1, int(round(min_stay / dt)))
    out, i, n = [], 0, len(labels)
    cur = labels[0] if n else -1
    if n:
        out.append((0, int(cur)))
    while i < n:
        if labels[i] != cur and i + need <= n and np.all(labels[i:i + need] == labels[i]):
            cur = labels[i]
            out.append((i, int(cur)))
            i += need
            continue
        i += 1
    return out


def evaluate(samples: dict, free_idx: list, identity: np.ndarray, dt: float, brain_commits: list | None = None,
             radius: float = 0.3, min_stay: float = 3.0) -> dict:
    """``samples``: {"t", "emb", "w", "size", "hand_open" (nan if absent)} arrays over time."""
    E, W, T = np.asarray(samples["emb"]), np.asarray(samples["w"]), np.asarray(samples["t"])
    n = len(E)
    if n < 4:
        return {}
    labels = cluster(E, radius)
    arr = arrivals(labels, dt, min_stay)
    dur = T[-1] - T[0] + dt
    counts = Counter(labels.tolist())
    unique = sum(1 for c in counts.values() if c * dt >= 2.0)
    last_seen: dict[int, float] = {}
    rep = osc = ret = 0
    nov = []
    seen_idx: list[int] = []
    for k, (i, lab) in enumerate(arr):
        t = T[i]
        if k > 0:
            if lab in last_seen and t - last_seen[lab] <= 60.0:
                rep += 1
            if lab in last_seen and t - last_seen[lab] >= 120.0:
                ret += 1
            if k >= 2 and arr[k - 2][1] == lab and t - T[arr[k - 2][0]] <= 30.0:
                osc += 1
            if seen_idx:
                P = E[seen_idx]
                age = t - T[seen_idx]
                d = np.linalg.norm(P - E[i], axis=1) + (1.0 - 0.5 ** (age / 120.0)) * 0.6
                nov.append(float(np.sort(d)[:4].mean()))
        # every sample of the cluster just left counts as "seen" (for novelty)
        j_end = arr[k + 1][0] if k + 1 < len(arr) else n
        seen_idx.extend(range(i, j_end, max(1, int(2.0 / dt))))
        last_seen[lab] = T[j_end - 1]
    m = max(1, len(arr) - 1)
    trans = Counter((arr[k][1], arr[k + 1][1]) for k in range(len(arr) - 1))
    if len(trans) > 1:
        p = np.array(list(trans.values()), float)
        p /= p.sum()
        tdiv = float(-(p * np.log(p)).sum() / math.log(len(p)))
    else:
        tdiv = 0.0
    cent = {lab: E[labels == lab].mean(0) for lab in set(labels.tolist())}
    radical = sum(1 for k in range(1, len(arr)) if np.linalg.norm(cent[arr[k][1]] - cent[arr[k - 1][1]]) > 0.75)
    stay = np.ones(n, bool)
    for i, _ in arr:
        stay[max(0, i - 2):i + 3] = False
    d2 = np.abs(np.diff(W, n=2, axis=0)).sum(1)             # twitching: the blend's second difference
    ok = stay[2:]
    jitter = float(d2[ok].mean()) if ok.any() else 0.0
    G = np.asarray(samples.get("geo", []), float)
    geo_jitter = float(np.abs(np.diff(G, n=2, axis=0)).mean(1)[ok].mean()) if len(G) == n and ok.any() else None
    dom = W.argmax(1)
    domv = W.max(1)
    covered = {int(d) for d, v in zip(dom, domv) if v >= 0.5}
    coverage = len(covered & set(free_idx)) / max(1, len(free_idx))
    thirds = np.array_split(np.arange(n), 3)
    first_new = len({labels[i] for i in thirds[0]})
    late_new = len({labels[i] for i in thirds[2]} - {labels[i] for i in np.concatenate(thirds[:2])})
    occ = np.array(list(counts.values()), float)
    occ /= occ.sum()
    occ_ent = float(-(occ * np.log(occ)).sum() / math.log(len(occ))) if len(occ) > 1 else 0.0
    out = {"minutes": round(float(dur / 60.0), 1), "unique_forms": unique, "top_share": round(float(occ.max()), 3),
           "occupancy_entropy": round(occ_ent, 3), "arrivals_per_min": round(float(len(arr) / dur * 60), 2),
           "repetition_rate": round(rep / m, 3), "oscillation_rate": round(osc / m, 3),
           "return_per_10min": round(float(ret / dur * 600), 2),
           "novelty_mean": round(float(np.mean(nov)), 3) if nov else 0.0,
           "novelty_max": round(float(np.max(nov)), 3) if nov else 0.0,
           "transition_diversity": round(tdiv, 3), "residence_s": round(float(dur / max(1, len(arr))), 1),
           "jitter": round(jitter, 3), "geo_jitter": None if geo_jitter is None else round(geo_jitter, 3),
           "coverage": round(coverage, 3),
           "hybrid_share": round(float((domv < 0.7).mean()), 3),
           "identity": round(float((W @ identity).mean()), 3),
           "late_discovery": round(late_new / max(1, first_new), 3),
           "radical_per_10min": round(float(radical / dur * 600), 2), "radical_share": round(radical / m, 3)}
    if brain_commits:
        bc = np.asarray(brain_commits, float)
        hits = sum(1 for i, _ in arr[1:] if np.any((T[i] - bc >= 0) & (T[i] - bc <= 6.0)))
        out["brain_share"] = round(hits / m, 3)
    else:
        out["brain_share"] = 0.0
    ho = np.asarray(samples.get("hand_open", []), float)
    sz = np.asarray(samples.get("size", []), float)
    ok = np.isfinite(ho) if len(ho) == n else np.zeros(n, bool)
    if ok.sum() > 20 and np.std(ho[ok]) > 1e-3 and np.std(sz[ok]) > 1e-6:
        out["hand_follow"] = round(float(np.corrcoef(ho[ok], sz[ok])[0, 1]), 3)
    else:
        out["hand_follow"] = None
    return out


# (metric, test, what it means) - a run "has" the problem when the test holds
PROBLEMS = (
    ("stuck", lambda m: m["top_share"] > 0.45 or m["residence_s"] > 60.0, "one form holds > 45 % of the time"),
    ("repetitive", lambda m: m["repetition_rate"] > 0.35, "> 35 % of arrivals revisit a form seen within 60 s"),
    ("oscillating", lambda m: m["oscillation_rate"] > 0.12, "> 12 % of arrivals are A -> B -> A within 30 s"),
    ("chaotic", lambda m: m["residence_s"] < 6.0, "forms last < 6 s on average"),
    ("resets", lambda m: m["radical_share"] > 0.75, "> 75 % of its changes are full-body jumps, not transformations"),
    ("conservative", lambda m: m["arrivals_per_min"] < 1.0 or m["coverage"] < 0.5,
     "< 1 new form a minute, or half of its forms never reached"),
)
QUALITIES = (
    ("explores", lambda m: m["coverage"] >= 0.8 and m["occupancy_entropy"] >= 0.75,
     ">= 80 % of its forms reached, time spread evenly (entropy >= 0.75)"),
    ("returns", lambda m: m["return_per_10min"] >= 1.0, "comes back to a form not seen for >= 2 min, >= 1x / 10 min"),
    ("keeps discovering", lambda m: m["late_discovery"] > 0.0, "new forms still appear in the last third"),
)


def diagnose(m: dict) -> dict:
    """-> {"problems": [...], "qualities": [...]} of one run's metrics (see PROBLEMS / QUALITIES)."""
    if not m:
        return {"problems": [], "qualities": []}
    return {"problems": [name for name, test, _ in PROBLEMS if test(m)],
            "qualities": [name for name, test, _ in QUALITIES if test(m)]}


__all__ = ["evaluate", "cluster", "arrivals", "diagnose", "PROBLEMS", "QUALITIES"]
