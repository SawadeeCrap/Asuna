# Benchmarks

All numbers below were measured **on the build machine, not on the target**: an Intel Xeon @ 2.8 GHz (4 cores, 33 MB L3) in a Linux container,
`g++ 13 -O3 -march=native` (tools) / `-O2 -march=native` (tests), 48 kHz, pffft as FFT backend. Nothing was measured on Apple silicon (the
NEON code path is compiled and inspected but never executed here), and nothing was measured inside Rack. Treat the figures as *relative*
(quality mode against quality mode, voices against voices, pitch against pitch) and as evidence that the worst case is bounded; expect an M4 Pro
core to be at least as fast per sample, but that is an expectation, not a measurement.

Method: the engine is **deterministic**, so every cell is measured as the *fastest of three identical runs* (or, for per-sample and per-block
figures, the per-sample/per-block minimum over three runs), which removes scheduler noise and interrupts and leaves the compute cost. Input: a
band-limited sawtooth (harmonics to 0.46 fs) at 1.0 = 5 V ×4 level, 6 s per cell with a 1 s warm-up (lock and first tables) that is not timed.
Parameters: SPREAD 0.5, DRIFT 0.3, CHARACTER 0.3, HARMONIC 0.3, PHASE 0.7, WIDTH 0.3, CLASSIC mode. Reproduce with

```sh
make -C tools PFFFT=/path/to/pffft bench && tools/bench 6 md     # CPU, worst block, FUSION layer, bloom latency
make -C tests PFFFT=/path/to/pffft realtime                      # allocation counter, finiteness, per-sample cost
```

## 1. CPU load — % of one core, steady state (100 % = the real-time limit of one core of *this* machine)

| quality | pitch | 1 voice | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| ECO | 20 Hz | 3.07 | 3.53 | 3.94 | 5.06 | 7.63 |
| ECO | 110 Hz | 2.34 | 2.59 | 2.91 | 3.67 | 4.88 |
| ECO | 440 Hz | 3.35 | 3.83 | 4.09 | 5.02 | 6.17 |
| ECO | 3520 Hz | 2.71 | 3.07 | 3.37 | 4.06 | 5.43 |
| BALANCED | 20 Hz | 2.93 | 3.43 | 3.95 | 4.84 | 7.06 |
| BALANCED | 110 Hz | 2.47 | 2.86 | 3.45 | 4.08 | 5.82 |
| BALANCED | 440 Hz | 3.37 | 3.88 | 4.16 | 4.86 | 6.66 |
| BALANCED | 3520 Hz | 2.68 | 3.06 | 3.45 | 4.28 | 6.00 |
| HIGH | 20 Hz | 6.21 | 6.95 | 8.59 | 11.36 | 18.02 |
| HIGH | 110 Hz | 2.94 | 3.29 | 3.78 | 4.85 | 7.19 |
| HIGH | 440 Hz | 3.84 | 4.30 | 4.91 | 6.16 | 8.33 |
| HIGH | 3520 Hz | 3.66 | 4.22 | 5.07 | 5.71 | 8.18 |
| ULTRA | 20 Hz | 6.10 | 7.00 | 8.46 | 11.57 | 18.10 |
| ULTRA | 110 Hz | 3.45 | 4.00 | 4.63 | 6.01 | 9.01 |
| ULTRA | 440 Hz | 4.10 | 4.54 | 5.10 | 6.59 | 9.79 |
| ULTRA | 3520 Hz | 3.73 | 4.02 | 4.75 | 5.81 | 8.58 |

Reading the table:

* **The "1 voice" column is not free.** VOICES = 1 leaves the original untouched, but tracker, level detector and analyser keep running so that
  raising VOICES (or a VOICES CV) finds the lock already established: 2.3 – 6.2 % of a core. This is a deliberate design choice (immediate
  response over idle CPU); a lower-power idle mode would delay the bloom-in by its full analysis time (§4).
* **Marginal cost of one clone** ((16 voices − 1 voice)/15): ECO 0.17 – 0.30 %, BALANCED 0.22 – 0.28 %, HIGH 0.28 – 0.30 % (0.79 % at 20 Hz),
  ULTRA 0.32 – 0.38 % (0.80 % at 20 Hz) of a core. Oscillator reads scale with the voice count; table rebuilds (one IFFT of up to 8192
  points per voice per publication interval) dominate at very low pitch, where the tables are largest and HIGH/ULTRA publish every 4–5 ms.
* Pitch has little influence except at 20 Hz in HIGH/ULTRA (large tables, 4-period windows, 32-tap resampling).
* FUSION algorithm layer (two extra table readers per clone, 16 voices, BALANCED, 110 Hz): RATIO mode 10.9 %, HZ mode (Hilbert pair per clone)
  9.9 % of a core, against 5.8 % in CLASSIC mode.

## 2. Worst case — the most expensive audio block

Rack calls `process()` once per sample inside the audio driver's block; what causes a dropout is the *worst block*, not the mean.
Most expensive 256-sample block at 16 voices, as a percentage of the block's real-time budget (5.33 ms), steady state:

| quality | 20 Hz | 110 Hz | 440 Hz | 3520 Hz |
|---|---|---|---|---|
| ECO | 16.4 | 7.9 | 7.7 | 8.1 |
| BALANCED | 13.4 | 8.3 | 10.6 | 7.5 |
| HIGH | 27.7 | 13.0 | 10.9 | 9.1 |
| ULTRA | 31.6 | 15.2 | 13.3 | 9.4 |

Cost of a **single call** of `process()` (`tests/test_realtime.cpp`, R2; 16 voices, saw + sub oscillator, CHARACTER 0.5, FUSION layer on; *every*
sample of the 3 s run counts, including the acquisition of the note; budget per sample at 48 kHz is 20.8 µs):

| quality | pitch | mean µs | 99.9 % µs | 99.99 % µs | max µs | samples > 100 µs |
|---|---|---|---|---|---|---|
| ECO | 20 Hz | 1.35 | 56.8 | 57.8 | 145.8 | 1 |
| ECO | 110 Hz | 1.03 | 44.6 | 52.8 | 119.6 | 1 |
| ECO | 880 Hz | 1.35 | 14.5 | 17.8 | 102.5 | 1 |
| BALANCED | 20 Hz | 1.30 | 56.6 | 58.9 | 147.4 | 1 |
| BALANCED | 110 Hz | 1.14 | 45.2 | 58.9 | 109.8 | 1 |
| BALANCED | 880 Hz | 1.41 | 15.0 | 17.8 | 102.5 | 1 |
| HIGH | 20 Hz | 3.27 | 76.9 | 86.8 | 162.0 | 5 |
| HIGH | 110 Hz | 1.55 | 30.9 | 39.4 | 119.9 | 1 |
| HIGH | 880 Hz | 1.69 | 19.3 | 26.5 | 113.4 | 1 |
| ULTRA | 20 Hz | 3.25 | 76.8 | 82.2 | 145.8 | 2 |
| ULTRA | 110 Hz | 1.99 | 31.0 | 39.4 | 109.1 | 1 |
| ULTRA | 880 Hz | 1.87 | 19.1 | 26.1 | 109.0 | 1 |

The **maximum single-sample cost is 100 – 165 µs** (one event per note: the acquisition, which runs the period refinement and starts the
first table builds); in steady state it is 21 – 131 µs. Before the real-time hardening the worst single sample was 0.3 – 3 ms. The remaining
peaks are the FFT stage of an analysis hop (≤ 100 µs at 32 768 points), the finishing stage (harmonic alignment and publication), the first
stage of a table build, and the acquisition.

Allocation check (R1): **zero heap allocations or frees inside `setParams()` and `process()`** over 7 s of input that includes locks, drops, pitch
steps, VOICES/SPREAD/DRIFT/WIDTH/MIX/CHARACTER/ORIG changes every 100 ms, a quality switch every second, DC, noise, ±1e9 and NaN/Inf samples;
every output sample finite and bounded (R3).

## 3. What each optimisation bought (same machine, same test)

| change | worst single sample | remark |
|---|---|---|
| before (direct grid fill, whole-hop FFT and extraction in one call, whole-table builds, direct-scan period refinement) | 0.3 – 3.1 ms (2.5 ms of it in the period refinement at acquisition) | analysis grid fill alone 0.44 ms at 20 Hz + sub after the SIMD change |
| explicit SSE2/NEON sinc dot product | 34 → 13 – 15 ns per grid point (−58 %, `-O2`) | scalar float reductions cannot be auto-vectorised without `-ffast-math` |
| time-sliced analysis hop (512-point chunks) + precomputed Hann windows | 0.75 ms → 0.3 ms at 20 Hz | the window was recomputed (400 µs) whenever the window length changed |
| per-voice table build in ≤ 5 stages, one per sample | 0.3 ms → 0.1 ms | CHARACTER shaping at HIGH/ULTRA cost ~150 µs per voice in one piece |
| period refinement by FFT cross-correlation | 2.5 ms → 0.1 ms at acquisition | verified equal to the direct scan to 0.0003 sample |
| lazily built statics moved into `prepare()` | (eliminates two heap operations on the first read) | sinc kernels and log₂ table |

## 4. Latency

* **Original path: 0 samples** (direct connection, no delay line; measured exactly 0 in `tests/test_engine.cpp`).
* **Clone bloom-in** — the time from a note onset (2 ms attack) until the clone weight exceeds 90 %; it is a few periods of the tracked
  repeating unit plus a fixed ≈ 8 ms tracker/lock overhead, essentially independent of QUALITY:

| pitch | ECO | BALANCED | HIGH | ULTRA |
|---|---|---|---|---|
| 20.0 Hz | 267.3 ms | 267.3 ms | 267.8 ms | 267.8 ms |
| 41.2 Hz | 186.8 ms | 182.7 ms | 183.1 ms | 181.0 ms |
| 110.0 Hz | 77.9 ms | 68.8 ms | 69.0 ms | 66.0 ms |
| 440.0 Hz | 20.7 ms | 20.7 ms | 20.9 ms | 20.9 ms |
| 1760.0 Hz | 10.2 ms | 10.2 ms | 10.4 ms | 10.4 ms |

* Pitch steps: the clones re-acquire in 3 ms (fifth up) to 6 ms (semitone) at 110 Hz (`research/results/transient.txt`, T2); the original is
  heard throughout.
* The module shows the bloom time for the current pitch (`(M + 3)` periods) in its context menu; the Rack module reports **0 samples** of
  latency for the audio path, because the original is never delayed.

## 5. Not measured

CPU inside Rack with other modules running, on macOS/Apple silicon (NEON path, Apple's `libm`, Rack's own pffft build), with Rack's default
`-march` flags (Rack builds plugins for `nehalem` on x86-64 and generic ARMv8 on arm64), at 44.1/96/192 kHz (the engine is tested for correctness
there, `tests/test_engine.cpp` robustness section, not benchmarked), or under memory pressure. `tools/bench` is the tool to run on the target.
