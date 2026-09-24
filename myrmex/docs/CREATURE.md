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
`creature_take_*.npz` (per-frame node positions, radii, surface, glow).  Viewport speed: metaball
*Resolution Viewport* (default 0.03) on the CreatureBody data; render resolution 0.02 or finer.
