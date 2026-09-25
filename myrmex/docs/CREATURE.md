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

# Polyalloy Colony (backend `colony`, creature v3)

The same finite polyalloy, 128 nodes, now able to act as several bodies and to build moving machinery.

* **Flock**: on a drop (energy high for a while, knob *swarm*) the material splits into 2-4 autonomous
  bodies - each a complete smaller shape sized by its share of the material - that fly in formation,
  scout, flank, attack prey from several sides, or part around an obstacle and close behind it.  On a
  breakdown they fly back and fuse (their elastic networks re-form into one).
* **Per-node material field**: every node has its own cohesion / stiffness / damping / dispersion; it relaxes
  to its body's state, diffuses along the network and hardens where it is hit.  Back-beat hits (snares) send
  **hardening waves** through the body: hexagonal **armour plates** rise in a travelling band and sink back
  (knob *armor*); struts show only where the material is locally hard.  The network is plastic: it slowly
  accepts the shape it is held in (morphological inertia).
* **Mechanisms** (knob *mechanism*): wings flap with the beat, blade rotors and rings spin with the energy,
  six tendrils carry travelling waves, the spindle pumps, a crown of spikes pulses on kicks, and when it
  perches it walks on six legs in a tripod gait locked to the beat.
* **Prey** (knob *hunt*): a small dark lure with a faint warm core flies around; the colony hunts it, closes
  around it in a shell, carries it for two bars and lets it burst out.
* The aerial camera follows the story: retreats when the flock splits, pushes in on the catch, tracks the
  walk on the ground, observes the merge.

Control notes (in addition to v2): 77 split · 79 merge · 81 hardening wave · 83 hunt · 84 perch.

# Polyalloy Hive (backend `hive`, creature v4)

A two-scale material: everything the Colony does, plus a real micro layer and a memory.

* **Nanomachine swarm** (knob *nanoswarm*): 1536 micro-machines ride on the structural nodes and circulate over
  the surface.  Where the material loosens, is hit or scattered they come off and fly as smoke-like streams in
  a swirling flow, then match the body's speed, home back and re-attach.  Calm music: ~15 % in the air;
  a drop: over half.  While the colony is split, couriers stream between the bodies - living bridges.
* **Emergent morphogenesis** (knob *pattern*): a Gray-Scott reaction-diffusion system runs on the elastic
  network; activator peaks push the surface out into spines and fins that migrate, split and fade.  Hits seed
  new peaks; energy and highs tune the chemistry.
* **Living architecture** (knob *architecture*): like army ants bridging with their own bodies, it leaves part
  of its material behind as a twisted pillar, an arch or a ring gate on its path (the tail detaches and flies
  there), circles it or flies through the gate, and calls the material back on a drop or after a while.
  Manual *Build* waits (up to 8 s) until the organism is free to build.
* **Phrase memory** (knob *memory*): every 4 bars it fingerprints the passage; when a passage returns it returns
  to the form it had then (with variation), otherwise it learns the new one.

Control notes (in addition to v3): 86 build · 88 recall.  Frames of v3/v4 exceed macOS's 9216-byte UDP
limit, so every frame above 8 KB travels in fragments and Blender reassembles it.

# The Osseous line (backends `osseous`, `osseous_colony`, `osseous_hive` - creatures v5, v6, v7)

Bony, aggressive versions of v2, v3 and v4 - the liquid originals stay as they are.  Same physics,
behaviour and features as the originals, plus:

* **Bone-link skeleton** instead of strut tubes.  The tubes followed random neighbour links and
  retracted to their midpoints - they floated like shards.  The skeleton is a minimum spanning tree
  through each body (no crossings, no shortcuts through empty space; a spine with ribs and limbs).
  Every link is an articulated bone segment - knuckled ends, a thin waist, a sharp dorsal crest facing
  out, a twist, a hooked tip - with gaps between links; each link grows where the material ossifies,
  dissolves where it liquefies and keeps mutating its proportions with its own phase.
* **Ossification**: hardness and *aggression* turn the material to bone; ossified nodes thin into beads
  strung along the bones; hardening waves ossify in travelling bands.
* **Bony, aggressive forms** (the style of bionic vertebrae, not their anatomy): SPINE (knuckled ridge with
  swept-back thorns that jump on kicks), CLAW (two hooked claws), MANDIBLE (claws snapping shut on every
  beat, v6/v7), SCYTHE (curved blades swinging with the bar), THORN (quills), CARAPACE (faceted shell with a
  keel).  *aggression* favours them.
* **Strike**: instead of evading it can harden into blades and lunge - obstacles are knocked away (HIT),
  prey is struck before it is enveloped.  Events STRIKE and OSSIFY.
* v6/v7: **bony scutes** (raised, swept back into a spike) instead of flat hexagonal plates.
  v7: **quill volley** on a drop (bristling + nanomachines fired outwards, event QUILLS), ring gates with
  **fangs**, reaction-diffusion spines turn to bone.

Control notes (in addition to v2-v4): 89 strike · 91 ossify · 93 quill volley.  Takes of every polyalloy
organism rebuild their struts / bone links and plates / scutes each frame from the recorded skeleton
(no big caches); open them through the app so the Myrmex add-on is active.

# Cyber Hive (backend `cyber_hive`, creature v8)

The Osseous Hive (v7) rebuilt as a hi-tech machine organism: same physics and behaviour (flock,
structures, nanomachine swarm, pattern, phrase memory, strikes, quill volleys), a different material.

* **White nanomaterial** body (ceramic gloss over microscopic cells) with faint *circuit seams* and a
  *scan band* in soft light acid green (low contrast: the lines glow, they don't glare).  The pattern
  rides with the body (its centre and heading are written to the shader every frame).
* **Rails** instead of bones: the skeleton through each body is drawn as white hexagonal modules with
  two light lines along them and a glowing collar that slides like a piston (`CyberRails`).
* **Hex panels** instead of scutes: white tiles aligned with the direction of travel, each with a
  light ring (`CyberPanels`).
* **Light per node** (sent in the stream, recorded in takes): a scan front sweeps the body tail → head
  every second bar, hardening waves and hits light up, the reaction-diffusion pattern glows, pulses of
  light run along the rails and panels in the direction of travel.
* **Machine forms**: HALO (a core in two gyroscope rings turning apart), ARRAY (two panel arrays that
  fold notch by notch), PRISM (a hexagonal crystal whose rings turn against each other like a lock) -
  their mechanisms move in robotic steps on the beat.
* Events **SCAN** (note 95) and **GLITCH** (note 96: parts of the body jump in quantized steps and
  flicker; also by itself on hard transients), plus STRIKE / OSSIFY (= lock) / QUILLS from v7.
* Swarm: white chips, one in five a small green light.

Colours live in the materials `MyrmexCyberWhite` (body), `MyrmexCyberHull`, `MyrmexCyberLine` (the
light: *Emission Color*, strength = light x pulses) and `MyrmexCyberMote` - change them in Blender and
**Save Look** keeps them.

# Your look in Blender (saved between sessions)

Tune materials, lights, world, colour management and render settings, then *Myrmex panel → Save Look*.
Humanoid: saved in the character's .blend (next to its rig.json).  Organisms: `~/Myrmex/looks/<type>.blend`
- the app opens it for live sessions and for take renders instead of building the default studio
(*Character → Your look in Blender* shows which looks exist, *Forget saved look* deletes one).
**Keep my Blender settings** (on by default): Myrmex no longer changes EEVEE / colour / shadows / samples,
neither when going live nor in *Render Video* (which then only sets size, frame rate and the output file).
Ctrl+Z, redo or opening another file while live no longer stops the stream: the link finds its objects
again by itself; *Start Live* also works in organism scenes (no armature needed).

# Takes → video (all organisms, and the humanoid)

1. *Camera & Output* → tick **Record**, play your set, stop the engine (or *Save take now*).
   A take stores the body every frame, the live camera and the song position.
2. Export the track from Ableton **from bar 1** and pick it as *Song for renders*.
3. **Open last take in Blender** - the scene is rebuilt as ordinary animation (metaball keyframes,
   strut lattice as a Point Cache, obstacles, lights, one camera per shot + markers, the song lined up
   by the recorded song position).  Change the look if you like, then *Myrmex → Render Video*
   (or F12 / Render Animation).  Or **Render last take → .mp4** renders in the background (H.264 + AAC,
   next to the take; sizes 1920×1080, 1080×1920 for reels, square, 4K).
4. In Blender: *Myrmex* panel → *Takes → video* → **Import Take** works for any take
   (`nanomaterial_take_*`, `polyalloy_take_*`, `colony_take_*`, `hive_take_*`, humanoid `take_*` with the
   character's .blend open).  Hive takes also write `*_swarm.pc2` (the nanomachines) next to the take.
