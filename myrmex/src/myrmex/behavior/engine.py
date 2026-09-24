"""The behaviour engine: music + memory + drives -> a stream of MotionCommands.

Layers (fast -> slow)::

    onsets/events ──► reactions (habituation, variation)          ~0.1–0.5 s
    features ──────► drives (arousal, tension, groove, boredom…)  ~0.5–20 s
    drives ────────► behaviour selection (weighted, novelty)      ~1–8 s
    sections/phrases ► strategy (still/sway/stroll/display/…)     ~bars–phrases

Everything random comes from named seeded streams, so the same project +
seed reproduces the same performance, while a new seed gives a different
but equally coherent one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..motion.command import MotionCommand
from ..music.features import FEATURE_GROUPS, ControlFrame
from ..music.phrases import MusicEvent, Structure
from ..util.mathutil import clamp, rot_z, wrap_angle
from ..util.rng import RngStreams
from .drives import Drives
from .gaze import Gaze
from .groove import GrooveOscillator
from .mapping import DEFAULT_MAPPINGS, MappingSet
from .novelty import RecencyTracker
from .reactions import Reactions
from .runway import RunwayDirector
from .vocab_biped import SECTION_STRATEGIES, STRATEGIES, VOCABULARY, Ctx

SMOOTH_FIELDS = {"crouch": 0.12, "rise": 0.15, "lean": 0.18, "side_lean": 0.18, "twist": 0.15,
                 "weight_bias": 0.2, "shoulder_raise": 0.12, "head_nod": 0.08, "hip_sway": 0.5,
                 "arm_swing": 0.5, "energy": 0.4, "tension": 0.25, "stride_scale": 0.6,
                 "cadence_scale": 0.6, "step_height": 0.5, "sharpness": 0.4, "lateral": 0.15}


@dataclass
class EngineConfig:
    mode: str = "runway"            # runway (continuous beat-locked walking) | free (wander / dance in place)
    style: str = "catwalk"          # motor style preset used by the generator in runway mode
    stage_radius: float = 3.0
    variation: float = 0.5          # 0 = predictable, 1 = adventurous choices
    novelty: float = 0.6            # strength of the anti-repetition penalty
    sensitivity: float = 1.0        # overall responsiveness to musical events
    lookahead: float = 0.05         # s: hear events slightly early to align visual peaks (offline only)
    mappings: dict = field(default_factory=lambda: dict(DEFAULT_MAPPINGS))
    strategy_weights: dict = field(default_factory=dict)     # optional overrides per strategy
    behavior_weights: dict = field(default_factory=dict)     # optional overrides per behaviour

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class BehaviorEngine:
    def __init__(self, height: float, psi0: float, seed: int = 0, config: EngineConfig | None = None,
                 structure: Structure | None = None, stage_center: np.ndarray | None = None):
        self.cfg = config or EngineConfig()
        self.rs = RngStreams(seed)
        self.height = height
        self.psi0 = psi0
        self.structure = structure
        self.stage_center = np.zeros(3) if stage_center is None else np.asarray(stage_center, float)
        self.drives = Drives()
        self.mapping = MappingSet.from_dict(self.cfg.mappings)
        self.groove = GrooveOscillator(self.rs.stream("groove"))
        self.gaze = Gaze(self.rs.stream("gaze"), 0.93 * height, FEATURE_GROUPS)
        self.reactions = Reactions(self.rs.stream("reactions"), mode=self.cfg.mode)
        self.runway = RunwayDirector(self.rs.stream("runway"), height, psi0, seed) if self.cfg.mode == "runway" else None
        self.beh_hist = RecencyTracker(half_life=25.0, strength=1.0)
        self.strat_hist = RecencyTracker(half_life=60.0, strength=0.8)
        self.gesture_hist = RecencyTracker(half_life=30.0, strength=1.2)
        self.strategy = "still"
        self.strategy_t0 = 0.0
        self.behavior = None
        self.section = "intro"
        self._smooth: dict[str, float] = {}
        self.last_cmd = MotionCommand()
        self.log: list[tuple[float, str, str]] = []
        self.pending_interrupt: tuple[str, dict] | None = None
        self._last_interrupt = -100.0

    # ------------------------------------------------------------------ helpers
    def _section_label(self, t: float, fr: ControlFrame) -> str:
        if self.structure is not None:
            s = self.structure.section_at(t)
            if s is not None:
                return s.label
        d = self.drives
        if fr.silence > 0.7:
            return "silence"
        if fr.trend > 0.25:
            return "build"
        if d.arousal > 0.7:
            return "peak"
        if d.arousal < 0.3:
            return "break" if t > 20.0 else "intro"
        return "groove"

    def _choose_strategy(self, t: float) -> str:
        d = self.drives
        base = dict(SECTION_STRATEGIES.get(self.section, SECTION_STRATEGIES["groove"]))
        mod = {"still": 0.3 + 1.5 * (1.0 - d.arousal), "sway": 0.3 + 2.0 * d.groove,
               "stroll": 0.5 + d.curiosity, "travel": 0.3 + d.arousal * (1.0 - d.fatigue),
               "display": 0.3 + d.confidence + d.arousal, "agitated": 0.2 + 2.0 * d.agitation,
               "suspense": 0.2 + 2.0 * max(d.tension, d.suspense)}
        rng = self.rs.stream("strategy")
        names, weights = [], []
        for s in STRATEGIES:
            w = base.get(s, 0.08) * mod[s] * self.cfg.strategy_weights.get(s, 1.0)
            w *= self.strat_hist.penalty(t, s, self.cfg.novelty)
            if s == self.strategy:
                w *= 0.7 + 0.8 * (1.0 - d.boredom)
            names.append(s)
            weights.append(w ** (1.0 / (0.5 + self.cfg.variation)))
        return names[rng.weighted_index(weights)]

    def _choose_behavior(self, ctx: Ctx, forced: str | None = None, params: dict | None = None):
        rng = self.rs.stream("behavior")
        if forced is None:
            table = STRATEGIES[self.strategy]
            names, weights = [], []
            prev = self.behavior.name if self.behavior else None
            for name, cls in VOCABULARY.items():
                base = table.get(name, 0.04)
                if name in ("recoil",):
                    base = 0.0
                w = base * max(0.0, cls.affinity(ctx)) * self.cfg.behavior_weights.get(name, 1.0)
                w *= self.beh_hist.penalty(ctx.t, name, self.cfg.novelty)
                if name == prev:
                    w *= 0.35
                if cls.locomotion and ctx.drives.fatigue > 0.8:
                    w *= 0.4
                names.append(name)
                weights.append(max(w, 0.0) ** (1.0 / (0.5 + self.cfg.variation)))
            forced = names[rng.weighted_index(weights)]
        # Gesture novelty: expose per-kind penalties to the Gesture behaviour.
        for k in VOCABULARY["gesture"].KINDS:
            ctx.mods["_gesture_penalty_" + k] = self.gesture_hist.penalty(ctx.t, k, 1.0)
        beh = VOCABULARY[forced](ctx, **(params or {}))
        if forced == "gesture":
            self.gesture_hist.add(ctx.t, beh.kind)
        self.beh_hist.add(ctx.t, forced)
        self.behavior = beh
        self.log.append((ctx.t, self.strategy, forced))
        return beh

    def _smooth_cmd(self, cmd: MotionCommand, dt: float) -> None:
        for f, tau in SMOOTH_FIELDS.items():
            v = float(getattr(cmd, f))
            s = self._smooth.get(f, v)
            s += (v - s) * (1.0 - math.exp(-dt / tau))
            self._smooth[f] = s
            setattr(cmd, f, s)

    # ------------------------------------------------------------------ main
    def update(self, t: float, dt: float, fr: ControlFrame, onsets: list[tuple[str, float, float]],
               events: list[MusicEvent], motor_pos: np.ndarray, motor_psi: float, motor_speed: float,
               anticipation: dict | None = None) -> MotionCommand:
        """``onsets`` is a list of (group, velocity, surprise)."""
        mods = self.mapping.evaluate(fr.as_dict(), dt)
        d = self.drives
        surprise = max((s for _, _, s in onsets), default=0.0)
        antic = max((v for v in (anticipation or {}).values()), default=0.0)
        effort = clamp(motor_speed / 1.2 + 0.3 * d.arousal, 0.0, 1.0)
        section = self._section_label(t, fr)
        d.update(fr, dt, surprise=surprise, anticipation=antic, effort=effort, events=events,
                 strategy_age=t - self.strategy_t0, section_label=section,
                 sensitivity=self.cfg.sensitivity)
        beat_dur = 60.0 / max(fr.tempo, 30.0)
        self.groove.update(dt, fr.beat, fr.tempo, d.groove * float(mods.get("responsiveness", 1.0)), d.arousal)
        ctx = Ctx(t, dt, fr, d, mods, self.rs.stream("behavior"), np.asarray(motor_pos, float), motor_psi,
                  motor_speed, self.stage_center, self.cfg.stage_radius, beat_dur, self.groove, self.gaze,
                  section, self.strategy, anticipation or {}, self.height, events)
        # ---------------- strategy (slow layer)
        new_section = section != self.section
        self.section = section
        if self.runway is not None:
            return self._update_runway(ctx, dt, mods, onsets, events, motor_pos, motor_psi, motor_speed,
                                       anticipation)
        phrase = any(e.type in ("phrase", "drop", "break") for e in events)
        if new_section or (phrase and d.boredom > 0.35) or self.behavior is None or \
                (t - self.strategy_t0 > 45.0 and d.boredom > 0.6):
            s = self._choose_strategy(t)
            if s != self.strategy or self.behavior is None:
                self.strategy = s
                self.strategy_t0 = t
                self.strat_hist.add(t, s)
                ctx.strategy = s
                if self.behavior is not None and self.behavior.interruptible and \
                        self.behavior.progress(ctx) > 0.3:
                    self.behavior = None
        # ---------------- interrupts (event-driven), in priority order
        forced, fparams = None, None
        rb = self.rs.stream("behavior")
        kinds = {e.type: e for e in events}
        if "drop" in kinds and kinds["drop"].strength > 0.3:
            opts = ["pose", "rise", "groove", "walk", "spin"]
            w = [0.5 + d.confidence, 0.4 + d.startle, 0.5 + d.groove, 0.4 + d.arousal, 0.15 + 0.5 * d.confidence]
            w = [wi * self.beh_hist.penalty(t, o, self.cfg.novelty) for wi, o in zip(w, opts)]
            forced = opts[rb.weighted_index(w)]
            self._last_interrupt = t
        elif "silence_start" in kinds and kinds["silence_start"].strength > 0.3:
            forced = "hesitate"
            self.reactions.freeze(t, kinds["silence_start"].strength)
            self._last_interrupt = t
        elif surprise > 0.6 * (2.0 - self.cfg.sensitivity) and d.startle > 0.5 and \
                t - self._last_interrupt > 5.0 and self.section not in ("drop", "peak"):
            self._last_interrupt = t
            if surprise > 0.85 and rb.chance(0.6):
                forced = "recoil"
            else:
                forced = rb.choice(["turn", "look_around", "hesitate"])
                if forced == "turn":
                    ang = rb.uniform(1.0, 2.4) * rb.choice([-1.0, 1.0])
                    fparams = {"heading": motor_psi + ang}
        for e in events:
            if e.type == "omission":
                self.reactions.freeze(t, 0.5 * e.strength)
            elif e.type == "entry" and e.group:
                self.gaze.orient(t, e.group, 0.6 * e.strength)
                d.attention = min(1.0, d.attention + 0.3 * e.strength)
        if self.behavior is None or self.behavior.finished(ctx) or (forced and self.behavior.interruptible):
            self._choose_behavior(ctx, forced, fparams)
        # ---------------- base command from drives + mappings
        cmd = MotionCommand()
        cmd.energy = clamp(0.25 + 0.7 * d.arousal + mods.get("energy", 0.0), 0.0, 1.0)
        cmd.tension = clamp(0.12 + 0.6 * d.tension + 0.3 * d.startle + mods.get("tension", 0.0), 0.0, 1.0)
        cmd.sharpness = clamp(0.3 + 0.5 * d.arousal + mods.get("sharpness", 0.0), 0.0, 1.0)
        cmd.hip_sway = clamp((0.45 + 0.5 * d.confidence) * mods.get("hip_sway", 1.0), 0.0, 1.5)
        cmd.cadence_scale = mods.get("cadence_scale", 1.0) * (0.92 + 0.2 * d.arousal)
        cmd.stride_scale = mods.get("stride_scale", 1.0) * (0.85 + 0.3 * d.confidence)
        cmd.step_height = mods.get("step_height", 1.0) * (0.8 + 0.4 * d.arousal)
        cmd.arm_swing = mods.get("arm_swing", 1.0) * (0.6 + 0.6 * d.arousal)
        cmd.crouch = mods.get("crouch", 0.0) + 0.08 * d.fatigue
        cmd.lean = mods.get("lean", 0.0)
        cmd.heading = None
        cmd.turn_rate = 0.0
        self.behavior.apply(ctx, cmd)
        # ---------------- gaze
        walking = motor_speed > 0.25
        self.gaze.update(t, d.curiosity, d.agitation, walking, fr.pitch, float(mods.get("gaze_wander", 1.0)))
        R = rot_z(wrap_angle(motor_psi))
        fwd = R @ np.array([1.0, 0.0, 0.0])
        left = R @ np.array([0.0, 1.0, 0.0])
        gt = self.gaze.world_target(np.array([motor_pos[0], motor_pos[1], 0.0]), fwd, left)
        gt[2] += float(mods.get("gaze_height", 0.0)) * 0.6
        cmd.gaze_target = gt
        cmd.gaze_weight = clamp(0.55 + 0.45 * d.attention, 0.0, 1.0)
        # ---------------- smoothing, then fast reactions on top
        self._smooth_cmd(cmd, dt)
        for group, vel, sur in onsets:
            self.reactions.trigger(t, group, vel, sur, d, mods, self.gaze,
                                   anticipated=(anticipation or {}).get(group, 0.0))
        self.reactions.update(t, dt)
        self.reactions.apply(t, cmd, anticipation, d.groove)
        cmd.crouch = clamp(cmd.crouch, 0.0, 0.9)
        cmd.rise = clamp(cmd.rise, 0.0, 1.0)
        cmd.shoulder_raise = clamp(cmd.shoulder_raise, 0.0, 1.0)
        cmd.weight_bias = clamp(cmd.weight_bias, -1.0, 1.0)
        self.last_cmd = cmd
        return cmd

    def _update_runway(self, ctx: Ctx, dt: float, mods: dict, onsets, events, motor_pos, motor_psi,
                       motor_speed, anticipation) -> MotionCommand:
        d, t, fr = self.drives, ctx.t, ctx.fr
        self.strategy = "runway"
        for e in events:
            if e.type == "entry" and e.group:
                self.gaze.orient(t, e.group, 0.4 * e.strength)
                d.attention = min(1.0, d.attention + 0.3 * e.strength)
        cmd = MotionCommand()
        cmd.energy = clamp(0.3 + 0.65 * d.arousal + mods.get("energy", 0.0), 0.0, 1.0)
        cmd.tension = clamp(0.1 + 0.35 * d.tension + 0.2 * d.startle + mods.get("tension", 0.0), 0.0, 0.8)
        cmd.sharpness = clamp(0.4 + 0.5 * d.arousal + mods.get("sharpness", 0.0), 0.0, 1.0)
        cmd.crouch = 0.35 * mods.get("crouch", 0.0)
        self.runway.apply(ctx, cmd, events, self.structure)
        cmd.hip_sway *= mods.get("hip_sway", 1.0)
        cmd.arm_swing *= mods.get("arm_swing", 1.0)
        R = rot_z(wrap_angle(motor_psi))
        fwd = R @ np.array([1.0, 0.0, 0.0])
        left = R @ np.array([0.0, 1.0, 0.0])
        gt = self.gaze.world_target(np.array([motor_pos[0], motor_pos[1], 0.0]), fwd, left)
        gt[2] += float(mods.get("gaze_height", 0.0)) * 0.3
        cmd.gaze_target = gt
        cmd.gaze_weight = 0.9
        self._smooth_cmd(cmd, dt)
        for group, vel, sur in onsets:
            self.reactions.trigger(t, group, vel, sur, d, mods, self.gaze,
                                   anticipated=(anticipation or {}).get(group, 0.0))
        self.reactions.update(t, dt)
        self.reactions.apply(t, cmd, None, 0.0)
        cmd.crouch = clamp(cmd.crouch, 0.0, 0.5)
        cmd.shoulder_raise = clamp(cmd.shoulder_raise, 0.0, 1.0)
        cmd.weight_bias = clamp(cmd.weight_bias, -1.0, 1.0)
        label = "pose" if not self.runway.walking else ("strut:" + (self.runway.flourish.kind
                                                                   if self.runway.flourish else "walk"))
        if not self.log or self.log[-1][2] != label:
            self.log.append((t, "runway", label))
        self._runway_label = label
        self.last_cmd = cmd
        return cmd

    @property
    def behavior_name(self) -> str:
        if self.runway is not None:
            return getattr(self, "_runway_label", "strut")
        return self.behavior.name if self.behavior else ""
