"""One simulation step shared by the offline generator and the realtime session.

    onsets ─► features ─► predictor/events ─► behaviour ─► motor ─► pose

Keeping this in one place guarantees that a live performance and a baked one
behave identically for identical input.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..behavior.engine import BehaviorEngine, EngineConfig
from ..motion.biped import BipedMotor
from ..motion.bodyplan import BipedPlan
from ..music.features import FEATURE_GROUPS, ControlFrame, FeatureExtractor
from ..music.phrases import EventDetector, MusicEvent, Structure
from ..music.predictor import GroovePredictor
from ..music.timeline import MusicTimeline, NoteEvent


@dataclass
class TickResult:
    pose: object
    frame: ControlFrame
    events: list[MusicEvent]
    command: object


class PerformanceCore:
    def __init__(self, plan: BipedPlan, timeline: MusicTimeline, seed: int = 0,
                 config: EngineConfig | None = None, structure: Structure | None = None,
                 energy_ref: float | None = None, style: dict | None = None):
        self.plan = plan
        self.tl = timeline
        self.cfg = config or EngineConfig()
        self.fx = FeatureExtractor(timeline, energy_ref=energy_ref)
        self.pred = GroovePredictor(groups=FEATURE_GROUPS)
        self.det = EventDetector()
        motor_style = dict(style or {})
        if self.cfg.mode == "runway" and "preset" not in motor_style and self.cfg.style:
            motor_style["preset"] = self.cfg.style
        self.motor = BipedMotor(plan, seed=seed, style=motor_style or None)
        self.engine = BehaviorEngine(plan.height, self.motor.psi0, seed, self.cfg, structure)
        self.structure = structure
        self._section_idx = 0

    def tick(self, t: float, dt: float, notes: list[NoteEvent], *, hear_time: float | None = None,
             beat: float | None = None, tempo: float | None = None, beats_per_bar: float | None = None,
             curves: dict | None = None) -> TickResult:
        th = t if hear_time is None else hear_time
        fr = self.fx.update(th, notes, curves, beat=beat, tempo=tempo, beats_per_bar=beats_per_bar)
        events: list[MusicEvent] = self.det.update(fr, dt)
        for g, s in self.pred.advance(fr.bar, fr.bar_phase):
            events.append(MusicEvent(t, "omission", float(s), g))
        if self.structure is not None:
            secs = self.structure.sections
            while self._section_idx + 1 < len(secs) and secs[self._section_idx + 1].start <= th:
                self._section_idx += 1
                sec = secs[self._section_idx]
                events.append(MusicEvent(t, "phrase", 1.0, data={"label": sec.label, "bar": sec.start_bar}))
        onsets = []
        bpb = beats_per_bar or 4.0
        for n in notes:
            g = self.fx.group_of(n)
            if n.beat is not None and beat is None:
                bar, in_bar = self.tl.tempo.bar_position(n.beat)
                bpb_n = self.tl.tempo.beats_per_bar(n.beat)
                phase = in_bar / bpb_n
            else:
                bar, phase = fr.bar, fr.bar_phase
            s, entry = self.pred.observe(g, bar, phase, n.velocity)
            onsets.append((g, float(n.velocity), s))
            if s > 0.5:
                events.append(MusicEvent(t, "surprise", s, g))
            if entry:
                events.append(MusicEvent(t, "entry", float(n.velocity), g))
        spb = (beats_per_bar or self.tl.tempo.beats_per_bar(fr.beat)) * 60.0 / max(fr.tempo, 1.0)
        antic = self.pred.anticipation(fr.bar_phase, spb)
        m = self.motor
        cmd = self.engine.update(t, dt, fr, onsets, events, m.pos, m.psi, m.speed_ref, antic)
        pose = m.update(dt, cmd)
        return TickResult(pose, fr, events, cmd)
