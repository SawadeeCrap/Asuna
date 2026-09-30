# Module manifest — Fusion Clone (`FusionClone` / `FusionClone`)

Generated from `src/FusionCloneModule.hpp` (the single source of truth for ids, ranges and defaults). Panel: 16 HP (240 × 380 px), layout constants in
`src/PanelLayout.hpp`.

## Plugin

| Field | Value |
|---|---|
| slug / module slug | `FusionClone` / `FusionClone` |
| version | 2.0.1 (patch-data version 2) |
| tags | Effect, Oscillator |
| license | GPL-3.0-or-later |
| Rack API | Rack 2 (`ARCH` any; C++11) |

## Parameters

| Id | Name | Range | Default | Snap | Notes |
|---|---|---|---|---|---|
| `VOICES_PARAM` | Voices (1 original + N−1 clones) | 1 … 16 | 8 | yes | 1 = original only |
| `SPREAD_PARAM` | Spread (detune) | 0 … 1 | 0.35 | | shaped by `SPREAD_CURVE_PARAM` |
| `DRIFT_PARAM` | Drift | 0 … 1 | 0.25 | | σ = 1.2 cents × value, correlation time from `DRIFT_RATE_PARAM` |
| `CHARACTER_PARAM` | Character | 0 … 1 | 0.30 | | level/colour tolerance, saturation asymmetry (HIGH/ULTRA), pitch jitter |
| `PHASE_PARAM` | Phase divergence | 0 … 1 | 0.60 | | fresh random draw per lock event |
| `HARMONIC_PARAM` | Harmonic divergence | 0 … 1 | 0.30 | | smooth, correlated across harmonics |
| `WIDTH_PARAM` | Stereo width | 0 … 1 | 0 | | linear pan, L+R = mono sum for any value |
| `MIX_PARAM` | Mix | 0 … 1 | 1 | | 0 = original only |
| `OUTPUT_PARAM` | Output | −24 … +12 dB | 0 | | |
| `QUALITY_PARAM` | Quality | ECO, BALANCED, HIGH, ULTRA | BALANCED | switch | see ARCHITECTURE §7 |
| `ALGO_PARAM` | Algorithm | CLASSIC, FUSION | CLASSIC | switch | |
| `ORIG_PARAM` | Original only | off, on | off | switch | |
| `RANDOM_PARAM` | Randomize voice seeds | button | | | edge triggered, new 32-bit seed |
| `DETUNE_RANGE_PARAM` | Detune range at SPREAD = 100 % | 5 … 60 cents | 20 | | expert (context menu) |
| `SPREAD_CURVE_PARAM` | Spread curve (exponent) | 0.8 … 3 | 1.6 | | expert |
| `DRIFT_RATE_PARAM` | Drift rate | 0 … 1 | 0.5 | | expert; τ = 30 s × 0.07^value |
| `DRIFT_CORR_PARAM` | Drift correlation between voices | 0 … 1 | 0.25 | | expert |
| `FUSION_SHIFT_PARAM` | Fusion detune-cluster amount | 0 … 1 | 0.30 | | expert; only in FUSION mode |
| `SHIFT_MODE_PARAM` | Fusion shift mode | RATIO, HZ | RATIO | switch | expert; RATIO = Doppler/BBD delay reading, HZ = SSB shift |
| `SUMMING_PARAM` | Analog-style summing saturation | 0 … 1 | 0.35 | | expert |
| `LEVEL_LAW_PARAM` | Level law | 0 … 0.5 | 0.075 | | expert; total gain N^−(0.5 − value): 0 = equal power, 0.5 = unity sum |

## Ports

| Id | Direction | Name | Scaling |
|---|---|---|---|
| `AUDIO_INPUT` | in | Audio (Fusion VCO2 out) | ±5 V nominal; polyphonic cable → channel 1 only (POLY light) |
| `VOICES_CV_INPUT` | in | Voices CV | 0 … 10 V = 1 … 16 (1.5 voices per volt, rounded, clamped) |
| `SPREAD_CV_INPUT` | in | Spread CV | 0.1 per volt, added to the knob, clamped 0 … 1 |
| `DRIFT_CV_INPUT` | in | Drift CV | as above |
| `CHARACTER_CV_INPUT` | in | Character CV | as above |
| `MIX_CV_INPUT` | in | Mix CV | as above |
| `L_OUTPUT` | out | Left | ±5 V nominal |
| `R_OUTPUT` | out | Right | identical to L at WIDTH = 0 |

Bypass routes: `AUDIO_INPUT` → `L_OUTPUT` and `R_OUTPUT`.

## Lights

| Id | Meaning |
|---|---|
| `LOCK_LIGHT` | Clones active (brightness = lock weight) |
| `ACQ_LIGHT` | Connected, audible input, not locked (acquiring / pass-through) |
| `POLY_LIGHT` | Polyphonic input connected; only channel 1 is used |

## Patch data (`dataToJson`)

```json
{ "version": 2, "seed": 1592627764 }
```

* `seed` — 32-bit seed of the voice population (`RAND`, *Randomize*). Restoring it reproduces the same clone population **bit for bit**
  (verified by `tests/test_module.cpp`). A patch without `seed` loads with the default seed.
* All other state is in the standard Rack parameter list. Rack "Randomize" only draws a new seed (it keeps VOICES / QUALITY / MODE).

## Threading and real-time rules

* `process()` allocates nothing, takes no locks, and never calls into the OS. All buffers, FFT plans and tables are allocated in
  `onSampleRateChange()` (called by Rack on the engine thread when the module is added or the rate changes).
* The analysis hop is a resumable job (one bounded piece per sample) and table construction is time-sliced (one voice at a time, up to five
  short stages, one stage per sample) and double-buffered with a cross-fade; parameter changes are smoothed per sample; a QUALITY change
  reconfigures without reallocating. The worst single `process()` call measured is 100 – 170 µs at 16 voices (`docs/BENCHMARKS.md`).
* The GUI reads a seqlock-protected status snapshot written every 1024 samples; it never touches the DSP state.

## Diagnostics in Rack's log

The module writes one line to Rack's `log.txt` when it enters the engine and one when it leaves it (never from `process()`), so that a bug report or an
automated test can see that the plugin was instantiated and what the engine was doing:

```
[info src/FusionCloneModule.hpp:119 onAdd] Fusion Clone: module added
[info src/FusionCloneModule.hpp:130 onRemove] Fusion Clone: module removed; last state LOCKED, repeating unit 261.63 Hz, 16 voice(s), input -5.7 dB, safety-net hits 0
```

`last state` is `LOCKED`, `ACQUIRING` or `PASS-THRU` (the display's states); `safety-net hits` is the engine's counter of non-finite / absurd output samples
that had to be caught (always 0 in every test). `tests/rack_smoke.sh` reads these lines from a real Rack.

Two warnings come from the panel widget (GUI thread, checked every 2 s, never from `process()`), each rate limited:

```
[warn src/FusionClone.cpp:… step] Fusion Clone: the lock was dropped 24 times in the last 2 s (last reason 2: 1 novelty, 2 tracker mismatch, 3 coherence, 4 doubling); repeating unit 166.00 Hz, input -12.3 dB. The clones flicker while this lasts; please report the pitch and waveform.
[warn src/FusionClone.cpp:… step] Fusion Clone: the engine's safety net intervened (1 time(s) so far); this should never happen, please report it.
```

The first is the symptom of the bug fixed in 2.0.1 (see `ARCHITECTURE.md` §9.7): a lock that is dropped and re-acquired many times a second makes the
clone layer stutter. Rack itself logs `Loaded plugin FusionClone v2.0.1`, which tells which build is installed.
