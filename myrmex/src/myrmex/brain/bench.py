"""``python morphology_brain_test.py`` - the morphology brain without Blender (docs/MORPHOLOGY_BRAIN.md).

    python morphology_brain_test.py                          1 minute of the Spear, the decision log
    python morphology_brain_test.py --organism colony --minutes 5 --log out/brain/colony.jsonl
    python morphology_brain_test.py --mode F --kev http://127.0.0.1:8009      Kev chooses among the candidates
    python morphology_brain_test.py --vocab spear            what the brain can ask of an organism
    python morphology_brain_test.py --benchmark              modes A B C G: 3 organisms x 3 seeds, 1/5/30/60 min
    python morphology_brain_test.py --benchmark --kev URL    + D E F (Kev, one request at a time)
    python morphology_brain_test.py --sweep rate             decision rate 0.25 .. 4 Hz (cost vs behaviour)
    python morphology_brain_test.py --sweep autonomy         autonomy 0 .. 1 (its influence, measured)
    python morphology_brain_test.py --engine real            the real engine at 120 Hz instead of the abstract one
    python morphology_brain_test.py --ipc-bench              the worker beside a live 120 Hz session: tick cost,
                                                             IPC round trip, serialisation, CPU, RAM, start-up
    python morphology_brain_test.py --kev-bench --kev URL    Kev on this Mac: new / cached state, question count,
                                                             option-order flips, confidence (real requests)

Modes: A current (the engine alone) - B random - C novelty (greedy) - D Kev without memory - E Kev + memory -
F candidates + novelty + memory, Kev chooses - G the same, a deterministic arbiter chooses (the default).
Nothing here needs Blender, MIDI or the network (Kev modes: a Kev server on this machine).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .core import KEV_MODES, MODES
from .metrics import PROBLEMS, QUALITIES, diagnose

LETTER = {v: k for k, v in MODES.items()}
OUT = os.path.join("out", "brain")
ORGANISMS = ("spear", "colony", "swarm")
SWEEPS = {"rate": ("rate_hz", (0.25, 0.5, 1.0, 2.0, 4.0)), "autonomy": ("autonomy", (0.0, 0.25, 0.5, 0.75, 1.0)),
          "novelty": ("novelty", (0.0, 0.3, 0.6, 1.0)), "persistence": ("persistence", (0.0, 0.5, 1.0)),
          "mutation": ("mutation", (0.0, 0.4, 0.8)), "returns": ("returns", (0.0, 0.5, 1.0)),
          "memory": ("memory", (0.0, 0.6, 1.0))}
# columns of the reports: (metric, header, format)
EXPLORE = (("unique_forms", "unique forms", "{:.0f}"), ("top_share", "top share", "{:.2f}"),
           ("occupancy_entropy", "entropy", "{:.2f}"), ("novelty_mean", "novelty mean", "{:.2f}"),
           ("novelty_max", "novelty max", "{:.2f}"), ("transition_diversity", "transitions", "{:.2f}"),
           ("coverage", "coverage", "{:.2f}"), ("late_discovery", "late new", "{:.2f}"),
           ("hybrid_share", "hybrid time", "{:.2f}"))
COHERE = (("repetition_rate", "repetition", "{:.2f}"), ("oscillation_rate", "oscillation", "{:.3f}"),
          ("return_per_10min", "returns /10 min", "{:.1f}"), ("residence_s", "residence s", "{:.1f}"),
          ("arrivals_per_min", "changes /min", "{:.1f}"), ("jitter", "jitter", "{:.3f}"),
          ("geo_jitter", "shape jitter", "{:.3f}"), ("radical_share", "radical share", "{:.2f}"),
          ("identity", "identity", "{:.2f}"), ("brain_share", "brain share", "{:.2f}"))


def mode_name(x: str) -> str:
    x = x.strip()
    name = MODES.get(x.upper(), x) if len(x) == 1 else x
    if name not in MODES.values():
        raise SystemExit(f"unknown mode {x!r}: {' '.join(f'{k}={v}' for k, v in MODES.items())}")
    return name


def controls_of(args) -> dict:
    out = {}
    for k in ("autonomy", "novelty", "persistence", "mutation", "returns", "memory", "min_confidence"):
        v = getattr(args, k, None)
        if v is not None:
            out[k] = float(v)
    if getattr(args, "rate", None) is not None:
        out["rate_hz"] = float(args.rate)
    return out


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def _save(name: str, md: str, data) -> tuple[str, str]:
    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = os.path.join(OUT, f"{name}-{stamp}")
    with open(base + ".md", "w") as f:
        f.write(md)
    with open(base + ".json", "w") as f:
        json.dump(data, f, indent=1, default=_json)
    return base + ".md", base + ".json"


# ---------------------------------------------------------------------------- the decision log
def format_entry(e: dict, wide: bool = False) -> str:
    music = str(e.get("music", "")).split(",")[0]
    out = [f"[{e['t']:7.1f} s] music {music:<5}  hand {e.get('hand', '-')}"]
    out.append(f"    now      {e.get('current', '')}" + (f"    (heading for {e['target']})" if e.get("target") else ""))
    cands = e.get("candidates") or []
    if cands:
        shown = cands[:8 if wide else 5]
        out.append("    options  " + "\n             ".join(
            f"{c['op']:<9} {c['label']:<52.52} novelty {c['novelty']:.2f}  utility {c['utility']:+.2f}" for c in shown))
    if e.get("probs"):
        out.append("    Kev      " + "  ".join(f"{k} {v:.2f}" for k, v in e["probs"].items())
                   + f"    confidence {e['confidence']:.2f}  {e.get('kev_ms')} ms")
    if e.get("op") == "HOLD":
        out.append(f"    chosen   HOLD      stays as it is  [{e['source']}]")
    else:
        out.append(f"    chosen   {e['op']:<9} {e['chosen']}  [{e['source']}, novelty {e['novelty']:.2f}, "
                   f"strength {e['strength']}, holds ~{e['hold']} s]")
    mem = e.get("memory") or {}
    if mem:
        recent = " -> ".join(f"{n} ({d} s)" if d else f"{n} (< 5 s)" for n, d in mem.get("recent", []))
        out.append(f"    history  {mem.get('forms_known', 0)} forms remembered; {recent}"
                   + ("   [repeating itself]" if mem.get("repeating") else ""))
    return "\n".join(out)


def _metrics_table(rows: list[tuple[str, dict]], cols=EXPLORE + COHERE) -> str:
    head = "| " + " | ".join(["metric"] + [r[0] for r in rows]) + " |"
    sep = "|" + "---|" * (len(rows) + 1)
    lines = [head, sep]
    for key, name, fmt in cols:
        vals = []
        for _, m in rows:
            v = m.get(key)
            vals.append("-" if v is None else fmt.format(v))
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def demo(args) -> None:
    from .sandbox import run
    mode = mode_name(args.mode)
    if mode in KEV_MODES and not args.kev:
        raise SystemExit(f"mode {LETTER[mode]} ({mode}) needs a Kev server: --kev http://127.0.0.1:8009")
    ctl = controls_of(args)
    log = None
    if args.log:
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        log = open(args.log, "w")
        log.write(json.dumps({"kind": "run", "organism": args.organism, "mode": mode, "minutes": args.minutes,
                              "seed": args.seed, "engine": args.engine, "controls": ctl}) + "\n")
    n = [0]

    def on(e):
        n[0] += 1
        if not args.quiet:
            print(format_entry(e, args.wide), flush=True)
        if log is not None:
            log.write(json.dumps({"kind": "decision", **e}, default=_json) + "\n")

    print(f"Morphology brain - {args.organism}, mode {LETTER[mode]} ({mode}), {args.minutes:g} min, seed {args.seed}, "
          f"{args.engine} engine" + (f", Kev {args.kev}" if mode in KEV_MODES else "") + "\n")
    r = run(args.organism, mode, args.minutes, args.seed, ctl, kev_url=args.kev or "", hand=not args.no_hand,
            engine=args.engine, on_decision=on, kev_timeout=args.kev_timeout)
    rows = [(f"{LETTER[mode]} {mode}", r["metrics"])]
    if mode != "current" and not args.no_compare:
        base = run(args.organism, "current", args.minutes, args.seed, ctl, hand=not args.no_hand, engine=args.engine)
        rows.append(("A current (the engine alone)", base["metrics"]))
    print("\n" + _metrics_table(rows))
    for label, m in rows:
        dg = diagnose(m)
        print(f"\n{label}: problems {', '.join(dg['problems']) or 'none'}; qualities {', '.join(dg['qualities']) or '-'}")
    st = r["stats"]
    print(f"\n{n[0]} decision points, {st['commits']} changes made, {st['holds']} holds, {st['yields']} yields to the "
          f"hand / the engine; brain CPU {st['cpu_ms_per_min']} ms per minute (step p95 {st['step_ms_p95']} ms)"
          + (f"; Kev {st['kev_calls']} calls, {st['kev_fail']} failed, {st['kev_lowconf']} below confidence, "
             f"median {st['kev_ms_median']} ms" if mode in KEV_MODES else ""))
    if log is not None:
        log.write(json.dumps({"kind": "result", "metrics": r["metrics"], "stats": st,
                              "baseline": rows[1][1] if len(rows) > 1 else None}, default=_json) + "\n")
        log.close()
        print(f"log: {args.log}")


# ---------------------------------------------------------------------------- the benchmark
def _task(kw: dict) -> dict:
    from .sandbox import run
    r = run(**kw)
    return {**{k: r[k] for k in ("organism", "mode", "seed", "engine", "minutes", "metrics", "by_minutes", "stats")},
            "controls": dict(kw.get("controls") or {})}


def _mean(rows: list[dict], key: str):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(v)) if v else None


def _aggregate(results: list[dict], minutes: float, mode: str, organism: str | None = None) -> tuple[dict, dict]:
    runs = [r for r in results if r["mode"] == mode and (organism is None or r["organism"] == organism)]
    ms = [r["by_minutes"].get(minutes) or (r["metrics"] if r["minutes"] == minutes else None) for r in runs]
    ms = [m for m in ms if m]
    agg = {k: _mean(ms, k) for k, _, _ in EXPLORE + COHERE}
    probs = {name: sum(1 for m in ms if name in diagnose(m)["problems"]) / max(1, len(ms)) for name, _, _ in PROBLEMS}
    quals = {name: sum(1 for m in ms if name in diagnose(m)["qualities"]) / max(1, len(ms)) for name, _, _ in QUALITIES}
    return agg, {"problems": probs, "qualities": quals, "runs": len(ms)}


def _report(results: list[dict], durations: list[float], modes: list[str], organisms: list[str], seeds: int,
            title: str, controls: dict) -> str:
    md = [f"# {title}", "",
          f"{len(organisms)} organisms ({', '.join(organisms)}) x {seeds} seeds, modes "
          f"{' '.join(LETTER[m] for m in modes)}; controls {json.dumps(controls) if controls else 'defaults'}. "
          "Every metric is measured on what the body became (its fingerprint every 0.5 s), the same way for every "
          "mode - see brain/metrics.py.  Means over organisms and seeds.", ""]
    for d in durations:
        md.append(f"## {d:g} min")
        rows = []
        diag = []
        for m in modes:
            agg, dg = _aggregate(results, d, m)
            rows.append((f"{LETTER[m]} {m}", agg))
            diag.append((m, dg))
        md.append("")
        md.append(_metrics_table(rows))
        md.append("")
        md.append("| mode | " + " | ".join(name for name, _, _ in PROBLEMS) + " | "
                  + " | ".join(name for name, _, _ in QUALITIES) + " |")
        md.append("|" + "---|" * (1 + len(PROBLEMS) + len(QUALITIES)))
        for m, dg in diag:
            md.append(f"| {LETTER[m]} {m} | " + " | ".join(f"{dg['problems'][n]:.0%}" for n, _, _ in PROBLEMS) + " | "
                      + " | ".join(f"{dg['qualities'][n]:.0%}" for n, _, _ in QUALITIES) + " |")
        md.append("")
        md.append("(share of runs showing it: " + "; ".join(f"{n} = {why}" for n, _, why in PROBLEMS + QUALITIES) + ")")
        md.append("")
    md.append("## Per organism (longest run)")
    md.append("")
    d = max(durations)
    keys = ("unique_forms", "top_share", "occupancy_entropy", "repetition_rate", "oscillation_rate",
            "return_per_10min", "novelty_mean", "residence_s", "jitter", "radical_share", "brain_share")
    md.append("| organism | mode | " + " | ".join(keys) + " |")
    md.append("|" + "---|" * (2 + len(keys)))
    for o in organisms:
        for m in modes:
            agg, _ = _aggregate(results, d, m, o)
            md.append(f"| {o} | {LETTER[m]} | " + " | ".join("-" if agg.get(k) is None else f"{agg[k]:.3g}"
                                                             for k in keys) + " |")
    md.append("")
    st = {}
    for m in modes:
        runs = [r for r in results if r["mode"] == m]
        rs = [r["stats"] for r in runs]
        st[m] = {"commits_per_min": np.mean([r["stats"]["commits"] / r["minutes"] for r in runs]),
                 "cpu_ms_per_min": np.mean([s["cpu_ms_per_min"] for s in rs]),
                 "step_ms_p95": np.mean([s["step_ms_p95"] or 0 for s in rs]),
                 "kev_ms_median": _mean([{"k": s.get("kev_ms_median")} for s in rs], "k"),
                 "kev_fail": sum(s.get("kev_fail", 0) for s in rs), "kev_calls": sum(s.get("kev_calls", 0) for s in rs),
                 "kev_lowconf": sum(s.get("kev_lowconf", 0) for s in rs)}
    md.append("## Cost")
    md.append("")
    md.append("| mode | changes made /min | brain CPU ms /min | step p95 ms | Kev calls | Kev failed | "
              "Kev below confidence | Kev median ms |")
    md.append("|---|---|---|---|---|---|---|---|")
    for m in modes:
        s = st[m]
        md.append(f"| {LETTER[m]} {m} | {s['commits_per_min']:.2f} | {s['cpu_ms_per_min']:.0f} | {s['step_ms_p95']:.1f} | "
                  f"{s['kev_calls']} | {s['kev_fail']} | {s['kev_lowconf']} | "
                  f"{'-' if s['kev_ms_median'] is None else round(s['kev_ms_median'], 1)} |")
    md.append("")
    return "\n".join(md)


def _run_tasks(tasks: list[dict], jobs: int) -> list[dict]:
    det = [t for t in tasks if t["mode"] not in KEV_MODES]
    kev = [t for t in tasks if t["mode"] in KEV_MODES]
    out: list[dict] = []
    t0 = time.perf_counter()
    total = len(tasks)

    def note(r):
        out.append(r)
        print(f"  {len(out)}/{total}  {r['organism']:<8} {LETTER[r['mode']]} seed {r['seed']}  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    if jobs > 1 and len(det) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for r in ex.map(_task, det):
                note(r)
    else:
        for t in det:
            note(_task(t))
    for t in kev:                                   # one local Kev server: one request at a time
        note(_task(t))
    return out


def benchmark(args) -> None:
    real = args.engine == "real"
    modes = [mode_name(m) for m in (args.modes or (["A", "B", "C", "G"] + (["D", "E", "F"] if args.kev else [])))]
    if real and not args.modes:
        modes = ["current", "deterministic"]
    if any(m in KEV_MODES for m in modes):
        _check_kev(args.kev)
    durations = sorted(float(d) for d in (args.durations or ([2.0] if real else [1, 5, 30, 60])))
    organisms = args.organisms or (["spear", "colony"] if real else list(ORGANISMS))
    seeds = args.seeds or (2 if real else 3)
    ctl = controls_of(args)
    T = max(durations)
    tasks = [dict(organism=o, mode=m, minutes=T, seed=s, controls=ctl, kev_url=args.kev if m in KEV_MODES else "",
                  checkpoints=durations, kev_timeout=max(args.kev_timeout, 5.0), hand=not args.no_hand,
                  engine=args.engine) for o in organisms for m in modes for s in range(seeds)]
    print(f"{len(tasks)} runs of {T:g} simulated minutes ({args.engine} engine), {args.jobs} at a time ...")
    wall = time.perf_counter()
    results = _run_tasks(tasks, args.jobs)
    title = f"Morphology brain benchmark ({args.engine} engine)"
    md = _report(results, durations, modes, organisms, seeds, title, ctl)
    md += f"\n(wall time {time.perf_counter() - wall:.0f} s)\n"
    print("\n" + md)
    paths = _save("benchmark", md, {"modes": modes, "organisms": organisms, "seeds": seeds, "durations": durations,
                                    "controls": ctl, "engine": args.engine, "results": results})
    print(f"saved {paths[0]} and {paths[1]}")


def sweep(args) -> None:
    key, values = SWEEPS[args.sweep]
    values = [float(v) for v in (args.values or values)]
    mode = mode_name(args.mode if args.mode != "G" or not args.modes else args.modes[0])
    organisms = args.organisms or ["spear", "colony"]
    seeds = args.seeds or 3
    minutes = args.minutes if args.minutes and args.minutes != 1.0 else 10.0
    base = controls_of(args)
    tasks = []
    for v in values:
        for o in organisms:
            for s in range(seeds):
                tasks.append(dict(organism=o, mode=mode, minutes=minutes, seed=s, controls={**base, key: v},
                                  kev_url=args.kev if mode in KEV_MODES else "", hand=not args.no_hand,
                                  engine=args.engine, kev_timeout=max(args.kev_timeout, 5.0)))
    print(f"sweep {args.sweep}: {values}, mode {LETTER[mode]}, {len(tasks)} runs of {minutes:g} min ...")
    results = _run_tasks(tasks, args.jobs)
    rows = []
    for v in values:
        rs = [r for r in results if abs(float(r["controls"][key]) - v) < 1e-9]
        ms = [r["metrics"] for r in rs]
        agg = {k: _mean(ms, k) for k, _, _ in EXPLORE + COHERE}
        agg["cpu_ms_per_min"] = float(np.mean([r["stats"]["cpu_ms_per_min"] for r in rs]))
        agg["changes_made_per_min"] = float(np.mean([r["stats"]["commits"] / minutes for r in rs]))
        agg["ticks_per_min"] = float(np.mean([r["stats"]["steps"] / minutes for r in rs]))
        rows.append((f"{key} {v:g}", agg))
    cols = (("ticks_per_min", "brain ticks /min", "{:.0f}"), ("changes_made_per_min", "changes made /min", "{:.2f}"),
            ("cpu_ms_per_min", "brain CPU ms /min", "{:.0f}")) + EXPLORE + COHERE
    md = (f"# Sweep: {args.sweep}\n\nmode {LETTER[mode]} ({mode}), {', '.join(organisms)} x {seeds} seeds, "
          f"{minutes:g} min each; other controls {json.dumps(base) if base else 'defaults'}.\n\n"
          + _metrics_table(rows, cols) + "\n")
    print("\n" + md)
    paths = _save(f"sweep-{args.sweep}", md, {"key": key, "values": values, "mode": mode, "organisms": organisms,
                                             "seeds": seeds, "minutes": minutes, "results": results})
    print(f"saved {paths[0]} and {paths[1]}")


# ---------------------------------------------------------------------------- the vocabulary
def show_vocab(args) -> None:
    from .fingerprint import GEO
    from .sandbox import organism_class
    from .vocab import vocabulary
    for org in ([args.vocab] if args.vocab != "all" else list(ORGANISMS) + ["hive", "cyber_hive", "blade", "cloud",
                                                                           "crawler", "osseous_colony"]):
        v = vocabulary(organism_class(org), org)
        print(f"\n{org} ({v.family}): {len(v.forms)} forms, {len(v.free)} it may be asked to take")
        print("  forms      " + ", ".join(f"{f}{'*' if f in v.free else ''}" for f in v.forms) + "   (* free)")
        print("  identity   " + ", ".join(f"{v.forms[i]} {v.identity[i]:.2f}" for i in np.argsort(-v.identity)[:6]))
        print("  events     " + (", ".join(v.events) or "-"))
        print("  materials  " + (", ".join(v.materials) or "-"))
        if v.geometry is not None:
            from .fingerprint import Scale, blend_cloud, geometry
            G = np.array([geometry(blend_cloud(np.eye(len(v.forms))[i], v.geometry)) for i in range(len(v.forms))])
            sc = Scale.from_geometry(v.geometry)
            Z = (G - sc.mu) / sc.sd
            print("  the most ... of its forms (shape descriptors, in the forms' own spread):")
            for j, g in enumerate(GEO):
                order = np.argsort(-Z[:, j])
                print(f"    {g:<6} most: {', '.join(v.forms[i] for i in order[:3]):<32} least: "
                      f"{', '.join(v.forms[i] for i in order[-3:])}")


# ---------------------------------------------------------------------------- the worker beside a live session
def _proc_usage(pid: int) -> tuple[float, float] | None:
    """(RSS MB, CPU seconds) of a process - /proc on Linux, ps on macOS."""
    try:
        with open(f"/proc/{pid}/status") as f:
            rss = next(int(line.split()[1]) for line in f if line.startswith("VmRSS")) / 1024.0
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return rss, (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, StopIteration, IndexError, ValueError):
        pass
    try:
        out = subprocess.run(["ps", "-o", "rss=,time=", "-p", str(pid)], capture_output=True, text=True,
                             timeout=5).stdout.split()
        secs = 0.0
        for part in out[1].replace("-", ":").split(":"):
            secs = secs * 60.0 + float(part)
        return int(out[0]) / 1024.0, secs
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _rss_mb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024.0 * 1024.0) if sys.platform == "darwin" else r / 1024.0


def _ipc_session(organism: str, brain: dict, secs: float) -> tuple[dict, dict | None]:
    """One live session (no outputs, internal clock) for ``secs`` s: -> (measurements, a snapshot payload)."""
    from ..realtime.session import LiveConfig, LiveSession
    t_make = time.perf_counter()
    ses = LiveSession(LiveConfig(backend=organism, clock="internal", out=[], brain=brain), start_inputs=False)
    steps: list[float] = []
    orig = ses.step

    def timed(now, orig=orig, steps=steps):
        t0 = time.perf_counter()
        r = orig(now)
        steps.append(time.perf_counter() - t0)
        return r
    ses.step = timed
    btick: list[float] = []
    if ses.brain is not None:                          # the brain's share of every engine tick
        orig_tick = ses.brain.tick

        def timed_tick(*a, orig_tick=orig_tick, btick=btick, **k):
            t0 = time.perf_counter()
            r = orig_tick(*a, **k)
            btick.append(time.perf_counter() - t0)
            return r
        ses.brain.tick = timed_tick
    ses.start()
    ready_s = None
    if ses.brain is not None:
        while time.perf_counter() - t_make < 30.0 and not ses.brain.ready:
            time.sleep(0.01)
        ready_s = time.perf_counter() - t_make if ses.brain.ready else None
    steps.clear()                                      # (the boot is start-up cost, measured apart)
    btick.clear()
    cpu0, wall0 = time.process_time(), time.perf_counter()
    w0 = _proc_usage(ses.brain.proc.pid) if ses.brain is not None and ses.brain.proc is not None else None
    time.sleep(secs)
    w1 = _proc_usage(ses.brain.proc.pid) if ses.brain is not None and ses.brain.proc is not None else None
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    st = ses.brain.stats if ses.brain is not None else None
    probe = None
    if ses.brain is not None:
        probe = ses.brain._payload(0.0, None, None, None, {}, "", 1.0)
        n_nodes = len(ses.brain.adapter._points(None))
    ses.stop()
    a = np.array(steps) * 1000.0
    out = {"ticks": len(a), "tick_ms_mean": float(a.mean()), "tick_ms_p95": float(np.percentile(a, 95)),
           "tick_ms_p99": float(np.percentile(a, 99)), "tick_ms_max": float(a.max()),
           "over_budget": int((a > 1000.0 / 120.0).sum()), "main_cpu_pct": 100.0 * cpu / wall,
           "main_rss_peak_mb": _rss_mb()}
    if st is not None:
        tu = np.array(btick) * 1e6 if btick else np.zeros(1)
        rtt = np.array(st["rtt_ms"]) if st["rtt_ms"] else np.zeros(1)
        wm = np.array(st["worker_ms"]) if st["worker_ms"] else np.zeros(1)
        out.update({"worker_start_s": ready_s, "snapshots": st["sent"], "answered": st["answered"],
                    "applied": st["applied"], "timeouts": st["timeouts"], "stale": st["stale"],
                    "restarts": st["restarts"], "brain_tick_us_median": float(np.median(tu)),
                    "brain_tick_us_p99": float(np.percentile(tu, 99)), "brain_tick_us_max": float(tu.max()),
                    "rtt_ms_median": float(np.median(rtt)), "rtt_ms_p95": float(np.percentile(rtt, 95)),
                    "rtt_ms_max": float(rtt.max()), "worker_ms_median": float(np.median(wm)),
                    "worker_ms_p95": float(np.percentile(wm, 95)), "worker_ms_max": float(wm.max()),
                    "nodes": n_nodes})
        if w0 and w1:
            out["worker_rss_mb"] = w1[0]
            out["worker_cpu_pct"] = 100.0 * (w1[1] - w0[1]) / secs
    return out, probe


def ipc_bench(args) -> None:
    secs = args.seconds
    rate = args.rate or 1.0
    reps = max(1, args.repeat)
    runs: dict[str, list[dict]] = {"brain off": [], "brain on": []}
    payload_probe = None
    brain_on = {"enabled": True, "mode": "deterministic", "controls": {"rate_hz": rate, "autonomy": 0.75}}
    for k in range(reps):                               # alternating: the machine's drift hits both alike
        for label, brain in (("brain off", {}), ("brain on", brain_on)):
            print(f"  {label}, run {k + 1}/{reps} ({secs:g} s) ...", flush=True)
            m, probe = _ipc_session(args.organism, brain, secs)
            runs[label].append(m)
            payload_probe = probe or payload_probe

    def agg(ms: list[dict]) -> dict:
        keys = {k for m in ms for k, v in m.items() if isinstance(v, (int, float)) and v is not None}
        out = {k: float(np.mean([m[k] for m in ms if m.get(k) is not None])) for k in keys}
        out["tick_ms_max"] = float(max(m["tick_ms_max"] for m in ms))
        out["over_budget"] = int(round(np.mean([m["over_budget"] for m in ms])))
        for k in ("snapshots", "applied", "timeouts", "stale", "answered", "restarts"):
            if k in out:
                out[k] = int(sum(m.get(k, 0) for m in ms))
        for k in ("rtt_ms_max", "worker_ms_max", "brain_tick_us_max"):
            if k in out:
                out[k] = float(max(m[k] for m in ms))
        return out
    res = {label: agg(ms) for label, ms in runs.items()}
    ser = None
    if payload_probe is not None:
        msg = ("snap", payload_probe)
        n = 2000
        t0 = time.perf_counter()
        for _ in range(n):
            b = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        dump_us = (time.perf_counter() - t0) / n * 1e6
        t0 = time.perf_counter()
        for _ in range(n):
            pickle.loads(b)
        load_us = (time.perf_counter() - t0) / n * 1e6
        ser = {"bytes": len(b), "dumps_us": dump_us, "loads_us": load_us}
    off, on = res["brain off"], res["brain on"]
    md = [f"# The brain worker beside a live session ({args.organism}, 120 Hz, {reps} x {secs:g} s each way, "
          f"alternating; brain at {rate:g} Hz, autonomy 0.75)", "",
          "Means over the runs (max: the worst of them; ticks over budget: per run).", "",
          "| | brain off | brain on |", "|---|---|---|"]
    for k, name, fmt in (("tick_ms_mean", "engine tick mean ms", "{:.2f}"), ("tick_ms_p95", "engine tick p95 ms", "{:.2f}"),
                         ("tick_ms_p99", "engine tick p99 ms", "{:.2f}"), ("tick_ms_max", "engine tick max ms", "{:.2f}"),
                         ("over_budget", "ticks over 8.33 ms", "{:d}"), ("main_cpu_pct", "app process CPU %", "{:.1f}"),
                         ("main_rss_peak_mb", "app process peak RSS MB", "{:.0f}")):
        f = (lambda v: fmt.format(int(round(v)))) if "d}" in fmt else fmt.format
        md.append(f"| {name} | {f(off[k])} | {f(on[k])} |")
    for label in ("brain off", "brain on"):
        md.append(f"| {label}: tick mean / p99 per run, ms | "
                  + ", ".join(f"{m['tick_ms_mean']:.2f} / {m['tick_ms_p99']:.2f}" for m in runs[label]) + " | |")
    md.append("")
    md.append("| brain | value |")
    md.append("|---|---|")
    for k, name, fmt in (("worker_start_s", "worker start-up (spawn + imports + vocabulary) s", "{:.2f}"),
                         ("brain_tick_us_median", "brain cost in the engine tick, median us", "{:.1f}"),
                         ("brain_tick_us_p99", "... p99 us", "{:.0f}"),
                         ("brain_tick_us_max", "... max us", "{:.0f}"),
                         ("rtt_ms_median", "snapshot -> decision round trip, median ms", "{:.1f}"),
                         ("rtt_ms_p95", "... p95 ms", "{:.1f}"), ("rtt_ms_max", "... worst ms", "{:.1f}"),
                         ("worker_ms_median", "worker compute per snapshot, median ms", "{:.2f}"),
                         ("worker_ms_p95", "... p95 ms (a decision: candidates + geometry)", "{:.1f}"),
                         ("worker_ms_max", "... max ms", "{:.1f}"),
                         ("worker_cpu_pct", "worker CPU % (one core)", "{:.2f}"),
                         ("worker_rss_mb", "worker RSS MB", "{:.0f}"),
                         ("snapshots", "snapshots sent", "{:d}"), ("applied", "decisions applied", "{:d}"),
                         ("timeouts", "timeouts", "{:d}"), ("stale", "dropped as stale / busy", "{:d}"),
                         ("nodes", "nodes read per snapshot (<= 160 sent)", "{:d}")):
        v = on.get(k)
        md.append(f"| {name} | {'-' if v is None else fmt.format(int(round(v)) if 'd}' in fmt else v)} |")
    if ser:
        md.append(f"| snapshot message size bytes | {ser['bytes']} |")
        md.append(f"| serialise / deserialise us | {ser['dumps_us']:.0f} / {ser['loads_us']:.0f} |")
    md.append("")
    md.append("The engine tick is measured round the whole session step (inputs, engine, camera, stream, brain). "
              "The round trip includes waiting for the next 120 Hz tick to read the answer (<= 8.3 ms).")
    text = "\n".join(md) + "\n"
    print(text)
    paths = _save("ipc", text, {"organism": args.organism, "seconds": secs, "rate": rate, "repeat": reps,
                               "results": res, "runs": runs, "serialisation": ser})
    print(f"saved {paths[0]} and {paths[1]}")


# ---------------------------------------------------------------------------- Kev on this machine
def _check_kev(url: str | None) -> dict:
    from .kev_client import KevClient, KevError
    if not url:
        raise SystemExit("Kev modes need a local Kev server: --kev http://127.0.0.1:8009\n"
                         "  uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009")
    try:
        return KevClient(url, timeout=5.0).models()
    except KevError as e:
        raise SystemExit(f"no Kev server at {url}: {e}\n"
                         "  uv run --extra serve python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009")


class _Recorder:
    """A Kev client that records what would be asked and answers nothing (the brain falls back): real requests
    for the Kev benchmark, captured from a run of the sandbox."""

    def __init__(self):
        self.requests: list[tuple[dict, dict]] = []

    def ask(self, state, questions):
        from .kev_client import KevError
        self.requests.append((json.loads(json.dumps(state)), json.loads(json.dumps(questions))))
        raise KevError("recording")


def kev_bench(args) -> None:
    from .kev_client import KevClient, KevError, choice_confidence
    from .sandbox import run
    info = _check_kev(args.kev)
    cli = KevClient(args.kev, timeout=30.0)
    rec = _Recorder()
    minutes = args.minutes if args.minutes and args.minutes != 1.0 else 5.0
    run(args.organism, "kev_candidates", minutes, args.seed, controls_of(args), kev=rec, kev_candidates=12)
    reqs = rec.requests[: args.requests]
    if not reqs:
        raise SystemExit("the sandbox made no decisions to ask about")
    print(f"Kev at {args.kev}: {json.dumps(info)[:200]}")
    print(f"{len(reqs)} real requests from {minutes:g} simulated minutes of {args.organism}\n")

    def ask(state, questions):
        t0 = time.perf_counter()
        ans = cli.ask(state, questions)
        return ans, (time.perf_counter() - t0) * 1000.0, cli.last_model_ms

    def cut(questions, k):
        q = json.loads(json.dumps(questions))
        crit = q["next"]["criteria"]
        q["next"]["criteria"] = {key: crit[key] for key in list(crit)[:k]}
        return q
    # new state (first time) vs the same state again (Kev's prefix cache) - 6 options, as the brain asks
    first, again, model_first, model_again, conf, fails = [], [], [], [], [], 0
    for state, questions in reqs:
        q = cut(questions, 6)
        try:
            a, ms, mm = ask(state, q)
            first.append(ms)
            model_first.append(mm or 0.0)
            a2, ms2, mm2 = ask(state, q)
            again.append(ms2)
            model_again.append(mm2 or 0.0)
            p = list(a["next"]["probabilities"].values())
            conf.append(choice_confidence([x / sum(p) for x in p]))
        except KevError:
            fails += 1
    # question size: 2 .. 12 options
    sizes = {}
    for k in (2, 4, 6, 8, 12):
        ms_k = []
        for state, questions in reqs[:10]:
            if len(questions["next"]["criteria"]) < k:
                continue
            try:
                ms_k.append(ask(state, cut(questions, k))[1])
            except KevError:
                fails += 1
        if ms_k:
            sizes[k] = float(np.median(ms_k))
    # option order: the same candidates, re-lettered in 3 other orders -> the same one chosen?
    rng = np.random.default_rng(args.seed)
    flips = tried = 0
    for state, questions in reqs[:20]:
        q = cut(questions, 6)
        crit = q["next"]["criteria"]
        texts = list(crit.values())
        try:
            base = ask(state, q)[0]["next"]["choice"]
        except KevError:
            fails += 1
            continue
        chosen = crit[base]
        for _ in range(3):
            perm = rng.permutation(len(texts))
            q2 = json.loads(json.dumps(q))
            q2["next"]["criteria"] = {chr(ord("A") + i): texts[j] for i, j in enumerate(perm)}
            try:
                c2 = ask(state, q2)[0]["next"]["choice"]
            except KevError:
                fails += 1
                continue
            tried += 1
            flips += q2["next"]["criteria"][c2] != chosen
    conf_a = np.array(conf) if conf else np.zeros(1)
    lo, hi = 0.2, 0.5
    md = [f"# Kev on this machine ({args.kev})", "",
          f"{len(reqs)} requests the brain really makes ({args.organism}, {minutes:g} simulated minutes), "
          f"6 candidates + an intensity score per request unless stated.", "",
          "| | median | p95 | max |", "|---|---|---|---|"]
    for name, xs in (("round trip, new state ms", first), ("round trip, same state again (cached) ms", again),
                     ("model time, new state ms", model_first), ("model time, cached ms", model_again)):
        if xs:
            md.append(f"| {name} | {np.median(xs):.0f} | {np.percentile(xs, 95):.0f} | {np.max(xs):.0f} |")
    md.append("")
    md.append("| options | median round trip ms |")
    md.append("|---|---|")
    for k, v in sizes.items():
        md.append(f"| {k} | {v:.0f} |")
    md.append("")
    md.append(f"Option-order flips: {flips} of {tried} re-orderings chose a different candidate "
              f"({(flips / tried if tried else 0):.0%}).")
    md.append(f"Confidence (p_max - 1/K) / (1 - 1/K): median {np.median(conf_a):.2f}, "
              f"p10 {np.percentile(conf_a, 10):.2f}, p90 {np.percentile(conf_a, 90):.2f}. Under the policy "
              f"(>= {hi} Kev decides, {lo} .. {hi} blended with the deterministic arbiter, < {lo} deterministic): "
              f"{(conf_a >= hi).mean():.0%} / {((conf_a >= lo) & (conf_a < hi)).mean():.0%} / {(conf_a < lo).mean():.0%}.")
    md.append(f"Failed requests: {fails}.")
    text = "\n".join(md) + "\n"
    print(text)
    paths = _save("kev", text, {"url": args.kev, "info": info, "first_ms": first, "again_ms": again,
                               "model_first_ms": model_first, "model_again_ms": model_again, "sizes": sizes,
                               "flips": flips, "tried": tried, "confidence": conf, "fails": fails})
    print(f"saved {paths[0]} and {paths[1]}")


# ---------------------------------------------------------------------------- the command line
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="morphology_brain_test.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--organism", default="spear", help="spear, colony, swarm, hive, blade ... (default spear)")
    ap.add_argument("--mode", default="G", help="A B C D E F G or a name (default G, the deterministic brain)")
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=("abstract", "real"), default="abstract",
                    help="abstract: the engine's morphology logic, fast (default); real: the organism engine at 120 Hz")
    ap.add_argument("--kev", default="", help="a local Kev server, e.g. http://127.0.0.1:8009")
    ap.add_argument("--kev-timeout", type=float, default=0.8)
    ap.add_argument("--log", default="", help="write the decision log as JSON lines")
    ap.add_argument("--quiet", action="store_true", help="no decision log on screen")
    ap.add_argument("--wide", action="store_true", help="more candidates per decision on screen")
    ap.add_argument("--no-hand", action="store_true", help="no scripted hand")
    ap.add_argument("--no-compare", action="store_true", help="skip the engine-alone comparison run")
    for k in ("autonomy", "novelty", "persistence", "mutation", "returns", "memory", "min_confidence"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
    ap.add_argument("--rate", type=float, default=None, help="decision ticks per second (default 1)")
    g = ap.add_argument_group("benchmarks")
    g.add_argument("--benchmark", action="store_true")
    g.add_argument("--modes", nargs="*", default=None)
    g.add_argument("--durations", nargs="*", type=float, default=None, help="minutes (default 1 5 30 60)")
    g.add_argument("--organisms", nargs="*", default=None)
    g.add_argument("--seeds", type=int, default=None)
    g.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    g.add_argument("--sweep", choices=tuple(SWEEPS), default=None)
    g.add_argument("--values", nargs="*", type=float, default=None)
    g.add_argument("--vocab", nargs="?", const="spear", default=None, help="an organism's action vocabulary (or all)")
    g.add_argument("--ipc-bench", action="store_true")
    g.add_argument("--seconds", type=float, default=30.0, help="--ipc-bench: seconds per session (default 30)")
    g.add_argument("--repeat", type=int, default=3, help="--ipc-bench: sessions each way, alternating (default 3)")
    g.add_argument("--kev-bench", action="store_true")
    g.add_argument("--requests", type=int, default=40, help="--kev-bench: real requests to replay (default 40)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.vocab:
        show_vocab(args)
    elif args.benchmark:
        benchmark(args)
    elif args.sweep:
        sweep(args)
    elif args.ipc_bench:
        ipc_bench(args)
    elif args.kev_bench:
        kev_bench(args)
    else:
        demo(args)
    return 0


__all__ = ["main", "parser", "format_entry", "demo", "benchmark", "sweep", "ipc_bench", "kev_bench"]


if __name__ == "__main__":                      # python -m myrmex.brain.bench (e.g. with the app's own Python)
    raise SystemExit(main())
