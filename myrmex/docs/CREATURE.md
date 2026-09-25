# Black Nanomaterial Creature (backend `creature`)

The creature is a finite quantity of self-organising material, not a character: its anatomy
(limbs, tendrils, spikes, fins, legs) is a temporary configuration that emerges from behaviour.

## Architecture

```
Ableton / VCV / MIDI / audio ──► existing input layer (InputHub, ClockHub, FeatureExtractor)   [unchanged]
                                   │  CreatureControlInput (bass, mid, high, energy, transient,
                                   │  spectral_flux, amplitude, tempo, beat, beat_phase)
                                   ▼
LiveSession ── backend "humanoid" ─► PerformanceCore + BipedMotor ─► bone poses (MYRP)          [unchanged]
            └─ backend "creature" ─► CreatureBackend ─► CreatureEngine ─► CreatureState (MYRC) ─► Blender
```

`src/myrmex/creature/`: `config` (all parameters), `control` (input normalisation, ParameterSet),
`behavior` (drives, memory, 12 states, events, wandering attractor), `nodes` (point masses on
per-node springs: inertia, overshoot, damping, repulsion, ground), `material` (MassField:
volumes always sum to TOTAL_MATERIAL, appendages are paid by the core), `morphology` (10 body
configurations blended continuously, 9 appendage archetypes as chains of nodes), `engine`
(the loop), `protocol` (UDP state), `backend` (session glue, recording).

Simulation and presentation are separate: the engine streams node positions / radii /
surface activity (~2.7 KB per frame); Blender (`blender/myrmex_blender/creature.py`) turns
every node into an element of one metaball field (a single continuous substance), drives the
nanomaterial shader (near-black metallic, coat highlights, flowing microstructure, faint
amber energy traces on transients) and the dark studio (large soft key, two rims, black floor).

## Loop (120 Hz)

behaviour (drives, state, events) → parameters (auto or manual) → morphology blend →
audio pressure (bass inflates, transients compress then rebound) → appendage growth →
mass allocation (radii, node masses = inertia) → stiffness per level (+ coherent left/right
asymmetry) → locomotion goal → targets (primary / secondary / appendage chains, follow-the-leader)
→ spring integration → surface / glow.  Feedback: arousal → forces → morphology → mass → inertia → motion.

## API

`CreatureEngine(cfg).set_input(CreatureControlInput)`, `.set_parameter(name, value | None)`,
`.trigger_event(name)`, `.update(dt) -> CreatureState`, `.state()`.
Events: MORPHOLOGY_SHIFT, MASS_REBALANCE, APPENDAGE_BURST, COLLAPSE, RECONSTRUCTION.

## Control

* Parameters (0..1, Auto = behaviour decides): aggression, arousal, expansion, contraction, fluidity,
  rigidity, asymmetry, tendril_activity, instability, surface_activity, mass_shift, reactivity, noise,
  coherence, mutation, speed, density.  App: *Creature* tab.  OSC: `/myrmex/control fluidity 0.8`
  (negative = back to automatic).  Ableton: a track named **Myrmex** with a Rack - name the macros like
  the parameters.  MIDI: *MIDI* tab bindings (any CC / note → any parameter, trigger or camera control;
  curve, invert, range, smoothing; notes as trigger / gate / toggle / velocity / music group; Learn).
* Notes on the control channel (16) or the Myrmex track: C3 morphology shift, D3 appendage burst,
  E3 camera cut, F3 collapse, G3 reconstruction, A3 mass rebalance.
* Camera: *Camera & Output* tab - Automatic director or Manual (the chosen shot is held until you
  pick another; keys 1-9 pick shots, 0 = automatic); distance, height, orbit, lens, smoothness also
  as controls `cam_mode`, `cam_distance`, `cam_height`, `cam_orbit`, `cam_lens`, `cam_smooth`.

## Debug, recording, determinism

*Creature* tab → Debug shows the control network (nodes + links) in Blender.  The same seed gives
the same performance for the same input.  With recording on, stopping the engine writes
`nanomaterial_take_*.npz` (see *Takes → video* below).  Viewport speed: metaball
*Resolution Viewport* (default 0.03) on the CreatureBody data; render resolution 0.02 or finer.


# Mimetic Polyalloy (backend `polyalloy`, creature v2)

An airborne, finite, self-reconfiguring material - not a robot that transforms, a material that can
temporarily become robot-like structures.  Procedural (no learning); only the visual logic of the
"carbon-based mimetic polyalloy" brief is used.

* **Finite material**: 96 structural nodes with fixed mass; nothing is ever created.  Dispersion,
  splitting and reassembly move the same material around.
* **Adaptive elastic network**: k-nearest-neighbour springs; over-stretched links break (separation),
  cohesive material rebuilds its network (recombination).  Fragments = connected components.
* **Material states** FLUID / ELASTIC / COHESIVE / STRUCTURED / HIGH_STIFFNESS / DISPERSED are blends of
  cohesion, stiffness, damping, repulsion, persistence, dispersion - physics, not visual modes.
  Hardened material thins into beads along its internal strut lattice: the skeleton shows.
* **Morphological attractors** CORE, SPINDLE, RING, SHIELD, BLADES, LATTICE, WINGS, CLOUD: every node owns
  a fixed material coordinate, each attractor maps it to a place, so the body flows between
  configurations.  A latent vector with inertia blends them (formation → stability → use → reconfiguration).
* **Flight**: distributed thrust against gravity, cruise / hover / altitude band; the turn rate comes from
  the body's actual moment of inertia (spread-out shapes turn slower).
* **Kick = physical event** (knob *kick mode*): impulse · obstacle (a sphere flies at it) · pressure wave ·
  turbulence · mix.  Obstacles are answered by a varied strategy (never the same three times):
  local split and flow-around, shield + stiffen, full dispersion, or a dodge - then reassembly.
  *obstacle rate* also spawns obstacles by itself; *altitude* sets the flight band.
* **Aerial camera** (9 modes, also the app's shot buttons 1-9): observe · follow · approach · retreat ·
  orbit · lock · track · impact (shake + reframe on big events) · recovery.
* **Blender**: one metaball body with the polyalloy shader (near-black, microscopic cell segmentation),
  an internal strut lattice (Geometry Nodes tubes that appear as the material hardens), obstacle spheres,
  and a render-time micro-machine layer (tiny hexagonal plates instanced over the surface; panel toggle
  *Micro-machines in viewport*).

Control notes (Myrmex track / control channel): 60 morphology shift, 62 blades, 65 collapse (disperse),
67 reconstruction, 69 rebalance, 71 obstacle, 72 impulse, 74 pressure, 76 turbulence; 64 = camera cut.

# Takes → video (both organisms, and the humanoid)

1. *Camera & Output* → tick **Record**, play your set, stop the engine (or *Save take now*).
   A take stores the body every frame, the live camera and the song position.
2. Export the track from Ableton **from bar 1** and pick it as *Song for renders*.
3. **Open last take in Blender** - the scene is rebuilt as ordinary animation (metaball keyframes,
   strut lattice as a Point Cache, obstacles, lights, one camera per shot + markers, the song lined up
   by the recorded song position).  Change the look if you like, then *Myrmex → Render Video*
   (or F12 / Render Animation).  Or **Render last take → .mp4** renders in the background (H.264 + AAC,
   next to the take; sizes 1920×1080, 1080×1920 for reels, square, 4K).
4. In Blender: *Myrmex* panel → *Takes → video* → **Import Take** works for any take
   (`nanomaterial_take_*`, `polyalloy_take_*`, humanoid `take_*` with the character's .blend open).
