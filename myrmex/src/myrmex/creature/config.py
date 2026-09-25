"""Configuration of the creature engine (physics, morphology, audio response, parameters)."""
from __future__ import annotations

from dataclasses import dataclass, field

# Macro parameters (0..1).  Each can be automatic (behaviour decides) or overridden live.
PARAMS = ("aggression", "arousal", "expansion", "contraction", "fluidity", "rigidity", "asymmetry",
          "tendril_activity", "instability", "surface_activity", "mass_shift", "reactivity", "noise",
          "coherence", "mutation", "speed", "density", "kick_mode", "obstacle_rate", "altitude",
          "swarm", "armor", "mechanism", "hunt",
          "architecture", "pattern", "nanoswarm", "memory")

DEFAULT_PARAMS = {"aggression": 0.3, "arousal": 0.3, "expansion": 0.5, "contraction": 0.3, "fluidity": 0.6,
                  "rigidity": 0.4, "asymmetry": 0.4, "tendril_activity": 0.4, "instability": 0.25,
                  "surface_activity": 0.4, "mass_shift": 0.3, "reactivity": 0.6, "noise": 0.3, "coherence": 0.6,
                  "mutation": 0.3, "speed": 0.4, "density": 0.5,
                  # Mimetic Polyalloy (v2) only: kick -> A impulse / B obstacle / C pressure / E turbulence / mix
                  "kick_mode": 0.9, "obstacle_rate": 0.3, "altitude": 0.5,
                  # Polyalloy Colony (v3) only: flock splitting, armour plates / hardening waves,
                  # articulated mechanisms (wings, rotors, tendrils, legs), prey hunting
                  "swarm": 0.5, "armor": 0.5, "mechanism": 0.6, "hunt": 0.4,
                  # Polyalloy Hive (v4) only: builds structures from itself, reaction-diffusion patterns,
                  # free-flying nanomachine streams, musical phrase memory
                  "architecture": 0.5, "pattern": 0.5, "nanoswarm": 0.6, "memory": 0.6}


@dataclass
class PhysicsConfig:
    # natural frequency (Hz) and damping ratio per structural level: heavy core, elastic whips
    core: tuple[float, float] = (0.9, 0.85)
    primary: tuple[float, float] = (1.6, 0.55)
    secondary: tuple[float, float] = (2.8, 0.42)
    gravity: float = 5.0            # on appendages (cables sag, limbs fall)
    max_speed: float = 14.0
    repulsion: float = 40.0         # keeps primary masses from interpenetrating
    ground: bool = True


@dataclass
class MorphologyConfig:
    n_primary: int = 6
    n_secondary: int = 12
    max_appendages: int = 6
    segments: int = 7
    size: float = 1.5               # overall scale (m)
    total_material: float = 1.0     # conserved volume (relative units)
    min_core_fraction: float = 0.16
    transition_time: float = 2.8    # s for a morphology transition


@dataclass
class AudioResponse:
    bass_expansion: float = 1.0
    transient_impulse: float = 1.0
    high_vibration: float = 1.0
    flux_instability: float = 1.0
    energy_arousal: float = 1.0
    amplitude_surface: float = 1.0
    tempo_oscillation: float = 1.0


@dataclass
class CreatureConfig:
    seed: int = 0
    variation: float = 0.5
    sim_rate: float = 120.0
    stage_radius: float = 3.5
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    audio: AudioResponse = field(default_factory=AudioResponse)
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
