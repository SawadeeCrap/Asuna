"""Internal drives: the creature's slowly changing "mood" (animation control variables).

These are not emotions and not AI – they are smoothed control signals with
different time constants, fed by musical features and events, which the
behaviour layer reads to decide *how* to move:

=============  =================================================  ===========
drive          grows with                                           timescale
=============  =================================================  ===========
arousal        energy, density, drops, startles                     fast (0.6/2.5 s)
tension        rising trend (builds), suspense before big events,   medium (3/2 s)
               silence right after activity
agitation      dense *irregular* rhythms, surprises, novelty        fast (0.4/1.8 s)
groove         steady, regular, repeating rhythm                    slow (4/1.5 s)
curiosity      sparse / novel / quiet passages                      medium (2/4 s)
confidence     low-end weight (kick + bass) and loudness            slow (2/4 s)
fatigue        sustained physical effort                            very slow
boredom        repetition over time in one strategy                 very slow
attention      salient events (spikes, decays)                      fast
startle        surprising events (spikes, decays)                   very fast
=============  =================================================  ===========
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _follow(x: float, target: float, dt: float, attack: float, release: float) -> float:
    tau = attack if target > x else release
    return x + (target - x) * (1.0 - math.exp(-dt / max(tau, 1e-4)))


@dataclass
class Drives:
    arousal: float = 0.2
    tension: float = 0.2
    agitation: float = 0.0
    groove: float = 0.0
    curiosity: float = 0.5
    confidence: float = 0.3
    fatigue: float = 0.0
    boredom: float = 0.0
    attention: float = 0.3
    startle: float = 0.0
    suspense: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)

    def update(self, fr, dt: float, *, surprise: float = 0.0, anticipation: float = 0.0,
               effort: float = 0.0, events: list | None = None, strategy_age: float = 0.0,
               section_label: str | None = None, sensitivity: float = 1.0) -> None:
        e = fr.energy
        # ---- event spikes
        for ev in events or []:
            if ev.type == "drop":
                self.arousal = min(1.0, self.arousal + 0.35 * ev.strength)
                self.tension *= 0.35
                self.startle = max(self.startle, 0.5 * ev.strength)
                self.boredom *= 0.3
            elif ev.type == "break":
                self.tension = min(1.0, self.tension + 0.1)
                self.curiosity = min(1.0, self.curiosity + 0.2)
            elif ev.type == "silence_start":
                self.suspense = min(1.0, self.suspense + 0.6 * ev.strength)
                self.attention = min(1.0, self.attention + 0.5 * ev.strength)
            elif ev.type == "silence_end":
                self.startle = max(self.startle, 0.3 * ev.strength)
                self.suspense *= 0.3
            elif ev.type == "phrase":
                self.boredom *= 0.75
            elif ev.type == "omission":
                self.attention = min(1.0, self.attention + 0.3 * ev.strength)
        if surprise > 0.05:
            self.startle = max(self.startle, min(1.0, surprise * sensitivity))
            self.attention = min(1.0, self.attention + 0.6 * surprise)
            self.boredom *= (1.0 - 0.5 * surprise)
        # ---- continuous targets
        s = sensitivity
        arousal_t = min(1.0, (0.72 * e + 0.2 * fr.density + 0.25 * self.startle) * (0.7 + 0.3 * s))
        self.arousal = _follow(self.arousal, arousal_t, dt, 0.6, 2.5)
        build = max(0.0, fr.trend) * (0.6 + 0.4 * e)
        tension_t = min(1.0, 0.15 + 0.9 * build + 0.5 * anticipation + 0.6 * self.suspense
                        + (0.25 if section_label == "build" else 0.0))
        self.tension = _follow(self.tension, tension_t, dt, 3.0, 2.0)
        irregular = fr.density * (1.0 - fr.rhythm)
        agit_t = min(1.0, 1.1 * irregular + 0.6 * self.startle + 0.25 * fr.novelty)
        self.agitation = _follow(self.agitation, agit_t, dt, 0.4, 1.8)
        groove_t = fr.rhythm * min(1.0, e * 1.3) * (1.0 - 0.5 * fr.novelty) * (0.6 + 0.4 * fr.repetition)
        self.groove = _follow(self.groove, groove_t, dt, 4.0, 1.5)
        cur_t = max(0.0, min(1.0, 0.55 * (1.0 - e) + 0.6 * fr.novelty + 0.3 * fr.silence - 0.4 * self.fatigue))
        self.curiosity = _follow(self.curiosity, cur_t, dt, 2.0, 4.0)
        conf_t = min(1.0, 0.8 * fr.low_activity * (0.5 + 0.5 * e) + 0.2 * fr.velocity)
        self.confidence = _follow(self.confidence, conf_t, dt, 2.0, 4.0)
        # ---- slow integrators
        self.fatigue = max(0.0, min(1.0, self.fatigue + dt * (0.012 * effort - 0.01 * (1.0 - effort))))
        bored_t = min(1.0, fr.repetition * (1.0 - fr.novelty) * min(1.0, strategy_age / 24.0))
        self.boredom = _follow(self.boredom, bored_t, dt, 12.0, 3.0)
        # ---- fast decays
        self.attention = max(0.0, self.attention - dt * 0.6 * self.attention) + 0.0
        self.attention = max(self.attention, 0.25 + 0.3 * self.curiosity)
        self.startle *= math.exp(-dt / 0.6)
        self.suspense *= math.exp(-dt / (6.0 if fr.silence > 0.5 else 1.0))
