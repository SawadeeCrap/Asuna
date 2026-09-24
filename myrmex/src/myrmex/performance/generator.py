"""Offline performance generation: MusicTimeline + body plan -> Performance.

This is the primary high-quality workflow::

    music project -> analyse whole timeline -> simulate behaviour + body at a
    fixed internal rate -> sample poses at the render frame rate -> bake.

The simulation is causal (the creature only "hears" the past, plus a small
configurable lookahead that aligns physical response peaks with the audio),
so the realtime engine produces the same kind of motion.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field

import numpy as np

from ..behavior.engine import BehaviorEngine, EngineConfig
from ..motion.biped import BipedMotor
from ..motion.bodyplan import BipedPlan
from ..music.features import FEATURE_GROUPS, FeatureExtractor, analyze_features, energy_reference
from ..music.phrases import EventDetector, MusicEvent, analyze_structure
from ..music.predictor import GroovePredictor
from ..music.timeline import MusicTimeline
from .performance import Performance, Recorder

DRIVE_CHANNELS = ("arousal", "tension", "agitation", "groove", "curiosity", "confidence", "fatigue",
                  "boredom", "attention", "startle", "suspense")
FEATURE_CHANNELS = ("energy", "density", "pitch", "impulse", "rhythm", "silence", "novelty", "trend")


@dataclass
class GenerateOptions:
    seed: int = 0
    fps: float = 30.0
    sim_rate: float = 120.0
    start: float = 0.0
    duration: float | None = None
    tail: float = 1.5
    engine: dict = field(default_factory=dict)
    style: dict = field(default_factory=dict)
    progress: bool = False


def generate_biped(tl: MusicTimeline, plan: BipedPlan, opts: GenerateOptions | None = None) -> Performance:
    opts = opts or GenerateOptions()
    t_start = _time.time()
    cfg = EngineConfig.from_dict(opts.engine)
    F = analyze_features(tl, 30.0)
    structure = analyze_structure(tl, F, 30.0)
    fx = FeatureExtractor(tl, energy_ref=energy_reference(tl))
    pred = GroovePredictor(groups=FEATURE_GROUPS)
    det = EventDetector()
    motor = BipedMotor(plan, seed=opts.seed, style=opts.style or None)
    engine = BehaviorEngine(plan.height, motor.psi0, opts.seed, cfg, structure)
    rec = Recorder(plan.sk.names, opts.fps)
    dur = opts.duration if opts.duration is not None else tl.duration + opts.tail
    sub = max(1, int(round(opts.sim_rate / opts.fps)))
    dt = 1.0 / (opts.fps * sub)
    n_ticks = int(round(dur * opts.fps)) * sub
    notes = tl.notes
    k = 0
    lookahead = float(cfg.lookahead)
    section_starts = [(s.start, s) for s in structure.sections[1:]]
    si = 0
    for i in range(n_ticks):
        t = opts.start + i * dt
        th = t + lookahead
        batch = []
        while k < len(notes) and notes[k].time <= th:
            batch.append(notes[k])
            k += 1
        fr = fx.update(th, batch)
        events: list[MusicEvent] = det.update(fr, dt)
        for g, s in pred.advance(fr.bar, fr.bar_phase):
            events.append(MusicEvent(t, "omission", float(s), g))
        while si < len(section_starts) and section_starts[si][0] <= th:
            sec = section_starts[si][1]
            events.append(MusicEvent(t, "phrase", 1.0, data={"label": sec.label, "bar": sec.start_bar}))
            si += 1
        onsets = []
        for n in batch:
            g = fx.group_of(n)
            bar, in_bar = tl.tempo.bar_position(n.beat if n.beat is not None else tl.tempo.beats(n.time))
            bpb = tl.tempo.beats_per_bar(n.beat or 0.0)
            s, entry = pred.observe(g, bar, in_bar / bpb, n.velocity)
            onsets.append((g, float(n.velocity), s))
            if s > 0.5:
                events.append(MusicEvent(t, "surprise", s, g))
            if entry:
                events.append(MusicEvent(t, "entry", float(n.velocity), g))
        spb = tl.tempo.beats_per_bar(fr.beat) * 60.0 / max(fr.tempo, 1.0)
        antic = pred.anticipation(fr.bar_phase, spb)
        cmd = engine.update(t, dt, fr, onsets, events, motor.pos, motor.psi, motor.speed_ref, antic)
        pose = motor.update(dt, cmd)
        for e in events:
            if e.type != "onset":
                rec.event(t, e.type, strength=e.strength, group=e.group, **{k2: v for k2, v in e.data.items()})
        if i % sub == sub - 1:
            ch = {f"drive_{name}": getattr(engine.drives, name) for name in DRIVE_CHANNELS}
            for name in FEATURE_CHANNELS:
                ch[f"music_{name}"] = getattr(fr, name)
            ch["speed"] = motor.speed_ref
            ch["groove_phase"] = engine.groove.phase % 1.0
            rec.add(pose.delta, ch, {"behavior": engine.behavior_name, "strategy": engine.strategy,
                                     "section": engine.section})
    for (t_ev, kind, data) in motor.events:
        rec.event(t_ev, kind, **data)
    meta = {
        "generator": "myrmex.biped",
        "seed": opts.seed,
        "music_source": tl.source,
        "music_duration": tl.duration,
        "audio_path": tl.audio_path,
        "sections": [s.as_dict() for s in structure.sections],
        "engine": {k2: v for k2, v in cfg.__dict__.items() if k2 != "mappings"},
        "behavior_log": engine.log,
        "sim_seconds": round(_time.time() - t_start, 2),
        "body_plan": plan.rig.body_plan,
        "height": plan.height,
    }
    perf = rec.build(meta)
    return perf
