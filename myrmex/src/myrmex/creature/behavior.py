"""Autonomous behaviour: drives, temporal memory, a state machine and locomotion intent.

States describe how the organism allocates material and movement - not emotions.
Transitions come from music, time, internal oscillators and accumulated instability; they
emit reconfiguration *events* rather than scripted animation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STATES = ("REST", "CURIOUS", "ALERT", "AGITATED", "AGGRESSIVE", "EXPANSIVE", "CONTRACTED", "HUNTING",
          "DEFENSIVE", "OVERLOADED", "REORGANIZING", "RECOVERING")
# morphology, locomotion speed, appendage activity, favoured appendage, surface activity, stiffness, noise
PROFILE = {
    "REST":         ("COMPACT", 0.06, 0.05, "SENSOR", 0.15, 1.0, 0.15),
    "CURIOUS":      ("ELONGATED", 0.35, 0.35, "SENSOR", 0.30, 0.9, 0.25),
    "ALERT":        ("DEFENSIVE", 0.12, 0.30, "SPIKE", 0.35, 1.3, 0.20),
    "AGITATED":     ("REORGANIZING", 0.45, 0.50, "WHIP", 0.70, 0.8, 0.60),
    "AGGRESSIVE":   ("AGGRESSIVE", 0.70, 0.60, "BLADE", 0.60, 1.4, 0.35),
    "EXPANSIVE":    ("MULTI", 0.30, 0.85, "TENDRIL", 0.50, 0.7, 0.30),
    "CONTRACTED":   ("COMPACT", 0.04, 0.00, "SPIKE", 0.25, 1.5, 0.10),
    "HUNTING":      ("QUADRUPED", 0.85, 0.30, "LIMB", 0.45, 1.1, 0.25),
    "DEFENSIVE":    ("DEFENSIVE", 0.10, 0.50, "SPIKE", 0.40, 1.6, 0.20),
    "OVERLOADED":   ("EXPLOSIVE", 0.40, 0.90, "WHIP", 1.00, 0.6, 0.90),
    "REORGANIZING": ("REORGANIZING", 0.15, 0.20, "CABLE", 0.60, 0.7, 0.80),
    "RECOVERING":   ("COMPACT", 0.10, 0.10, "SENSOR", 0.20, 0.9, 0.20),
}


@dataclass
class Memory:
    recent_energy: float = 0.0
    recent_impulse: float = 0.0
    arousal: float = 0.1
    arousal_v: float = 0.0
    agitation: float = 0.0
    fatigue: float = 0.0
    instability: float = 0.0
    time_in_state: float = 0.0
    time_since_reconfig: float = 0.0
    overload_time: float = 0.0
    last_burst: float = -1e9


class BehaviorSystem:
    def __init__(self, rng, stage_radius: float):
        self.rng = rng
        self.state = "REST"
        self.mem = Memory()
        self.stage = stage_radius
        self.goal = np.zeros(3)
        self.heading = rng.uniform(-math.pi, math.pi)
        self.dwell = 5.0
        self.collapsed_at = None
        self.log: list[tuple[float, str]] = []

    def profile(self, key: str):
        i = ("morph", "speed", "appendage", "archetype", "surface", "stiffness", "noise").index(key)
        return PROFILE[self.state][i]

    def _choose(self, params: dict) -> str:
        m = self.mem
        a, ag, en = m.arousal, m.agitation, m.recent_energy
        w = {
            "REST": 1.2 * (1 - a) ** 2, "CURIOUS": 0.6 + 0.8 * (1 - ag) * a, "ALERT": 0.3 + 0.9 * m.recent_impulse,
            "AGITATED": 1.4 * ag, "AGGRESSIVE": 1.6 * a * params["aggression"], "EXPANSIVE": 1.2 * en * params["expansion"],
            "CONTRACTED": 0.8 * params["contraction"] * (1 - en), "HUNTING": 1.1 * a * params["speed"],
            "DEFENSIVE": 0.5 * params["rigidity"] + 0.4 * m.recent_impulse, "OVERLOADED": 2.5 * max(0.0, a - 0.8) * en,
            "REORGANIZING": 0.8 * m.instability, "RECOVERING": 1.2 * m.fatigue,
        }
        w[self.state] *= 0.35                            # prefer change once dwell time is over
        names = list(w)
        return names[self.rng.weighted_index([max(1e-4, w[n]) for n in names])]

    def update(self, dt: float, t: float, inp, params: dict) -> list[tuple]:
        m, ev = self.mem, []
        k_e = 1.0 - math.exp(-dt / 4.0)
        m.recent_energy += (inp.energy - m.recent_energy) * k_e
        m.recent_impulse = max(m.recent_impulse * math.exp(-dt / 1.0), inp.transient)
        target = min(1.0, 0.75 * inp.energy + 0.35 * m.recent_impulse + 0.3 * inp.spectral_flux) * params["reactivity"] + \
            0.15 * (1 - params["reactivity"])
        w = 2 * math.pi * 0.35                           # arousal has inertia (underdamped: overshoots a little)
        m.arousal_v += dt * (w * w * (target - m.arousal) - 2 * 0.7 * w * m.arousal_v)
        m.arousal = min(1.0, max(0.0, m.arousal + dt * m.arousal_v))
        m.agitation += (min(1.0, inp.spectral_flux + 0.6 * inp.high) - m.agitation) * (1 - math.exp(-dt / 1.5))
        m.fatigue = min(1.0, max(0.0, m.fatigue + dt * (0.03 * (m.arousal - 0.6))))
        m.instability = min(1.5, m.instability + dt * (0.25 * inp.spectral_flux + 0.1 * params["instability"]))
        m.time_in_state += dt
        m.time_since_reconfig += dt
        # --- state machine
        coh = params["coherence"]
        if m.time_in_state > self.dwell:
            new = self._choose(params)
            m.time_in_state = 0.0
            self.dwell = self.rng.uniform(3.0, 9.0) * (0.6 + 0.8 * coh)
            if new != self.state:
                self.state = new
                self.log.append((t, new))
                ev.append(("MORPHOLOGY_SHIFT", self.profile("morph")))
        # --- events
        if inp.transient > 0.55 and m.arousal > 0.45 and t - m.last_burst > 1.2 + 2.0 * coh:
            m.last_burst = t
            ev.append(("APPENDAGE_BURST", self.profile("archetype")))
        if m.instability > 1.0:
            m.instability = 0.0
            m.time_since_reconfig = 0.0
            ev.append(("MORPHOLOGY_SHIFT", "REORGANIZING"))
        if self.state == "OVERLOADED":
            m.overload_time += dt
            if m.overload_time > 3.0 and self.collapsed_at is None:
                self.collapsed_at = t
                ev.append(("COLLAPSE", None))
        else:
            m.overload_time = 0.0
        if self.collapsed_at is not None and t - self.collapsed_at > 2.5:
            self.collapsed_at = None
            self.state, m.time_in_state = "RECOVERING", 0.0
            ev.append(("RECONSTRUCTION", None))
        if m.time_since_reconfig > 25.0 + 20.0 * coh:
            m.time_since_reconfig = 0.0
            ev.append(("MASS_REBALANCE", None))
        return ev

    def locomotion_goal(self, dt: float, core: np.ndarray, params: dict) -> tuple[np.ndarray, float]:
        """Wandering attractor with directional persistence, steered back inside the stage."""
        speed = self.profile("speed") * (0.4 + 1.2 * params["speed"]) * (0.5 + self.mem.arousal)
        self.heading += dt * self.rng.normal(0.0, 1.0) * (0.6 + 1.5 * params["noise"]) * (1.2 - params["coherence"])
        to_c = -core[:2]
        dist = float(np.linalg.norm(to_c))
        if dist > 0.6 * self.stage:                      # turn home gently
            want = math.atan2(to_c[1], to_c[0])
            dh = (want - self.heading + math.pi) % (2 * math.pi) - math.pi
            self.heading += dh * min(1.0, dt * 0.8 * dist / self.stage)
        d = np.array([math.cos(self.heading), math.sin(self.heading), 0.0])
        self.goal = core * np.array([1, 1, 0]) + d * speed * 0.8
        return self.goal, speed
