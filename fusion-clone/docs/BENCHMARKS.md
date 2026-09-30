# Benchmarks

The numbers in §1 – §4 were measured **on the build machine, not on the target**: an Intel Xeon @ 2.8 GHz (4 cores, 33 MB L3) in a Linux
container, `g++ 13 -O3 -march=native` (tools) / `-O2 -march=native` (tests), 48 kHz, pffft as FFT backend. §6 repeats the benchmark on an
**Apple M1 virtual machine** (GitHub Actions `macos-14`, 3 cores, shared): there the NEON code path ran and every DSP test passed. Nothing was
measured on a real Mac or inside a Rack session with other modules (§5). Treat the figures as *relative* (quality mode against quality mode,
voices against voices, pitch against pitch) and as evidence that the worst case is bounded; a current Apple-silicon desktop/laptop core is
expected to be at least as fast as the M1 VM core, but that is an expectation, not a measurement.

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
| ECO | 20 Hz | 3.10 | 3.79 | 4.46 | 6.71 | 9.13 |
| ECO | 110 Hz | 2.38 | 2.65 | 3.01 | 3.86 | 5.30 |
| ECO | 440 Hz | 3.53 | 3.74 | 4.21 | 4.83 | 6.17 |
| ECO | 3520 Hz | 2.85 | 3.46 | 3.60 | 4.53 | 6.18 |
| BALANCED | 20 Hz | 3.04 | 3.53 | 4.22 | 6.09 | 8.62 |
| BALANCED | 110 Hz | 2.74 | 3.13 | 3.50 | 4.30 | 6.32 |
| BALANCED | 440 Hz | 3.64 | 3.83 | 4.19 | 4.84 | 6.50 |
| BALANCED | 3520 Hz | 2.83 | 3.17 | 3.54 | 4.38 | 6.62 |
| HIGH | 20 Hz | 6.02 | 7.11 | 8.84 | 12.21 | 18.82 |
| HIGH | 110 Hz | 3.04 | 3.49 | 3.99 | 5.26 | 7.61 |
| HIGH | 440 Hz | 3.98 | 4.27 | 4.97 | 6.65 | 8.55 |
| HIGH | 3520 Hz | 3.78 | 4.08 | 4.75 | 5.92 | 8.30 |
| ULTRA | 20 Hz | 5.93 | 7.23 | 8.86 | 12.56 | 20.73 |
| ULTRA | 110 Hz | 3.57 | 4.00 | 4.60 | 6.02 | 9.00 |
| ULTRA | 440 Hz | 4.01 | 4.46 | 5.05 | 6.61 | 9.33 |
| ULTRA | 3520 Hz | 3.71 | 4.03 | 4.70 | 6.03 | 8.76 |

Reading the table:

* **The "1 voice" column is not free.** VOICES = 1 leaves the original untouched, but tracker, level detector and analyser keep running so that
  raising VOICES (or a VOICES CV) finds the lock already established: 2.4 – 6.0 % of a core. This is a deliberate design choice (immediate
  response over idle CPU); a lower-power idle mode would delay the bloom-in by its full analysis time (§4).
* **Marginal cost of one clone** ((16 voices − 1 voice)/15): ECO 0.18 – 0.22 % (0.40 % at 20 Hz), BALANCED 0.19 – 0.25 % (0.37 % at 20 Hz),
  HIGH 0.30 % (0.85 % at 20 Hz), ULTRA 0.34 – 0.36 % (0.99 % at 20 Hz) of a core. Oscillator reads scale with the voice count; table rebuilds (one IFFT of up to 8192
  points per voice per publication interval) dominate at very low pitch, where the tables are largest and HIGH/ULTRA publish every 4–5 ms.
* Pitch has little influence except at 20 Hz in HIGH/ULTRA (large tables, 4-period windows, 32-tap resampling).
* FUSION algorithm layer (two extra table readers per clone, 16 voices, BALANCED, 110 Hz): RATIO mode 10.74 %, HZ mode (Hilbert pair per clone)
  8.43 % of a core, against 6.3 % in CLASSIC mode.

## 2. Worst case — the most expensive audio block

Rack calls `process()` once per sample inside the audio driver's block; what causes a dropout is the *worst block*, not the mean.
Most expensive 256-sample block at 16 voices, as a percentage of the block's real-time budget (5.33 ms), steady state:

| quality | 20 Hz | 110 Hz | 440 Hz | 3520 Hz |
|---|---|---|---|---|
| ECO | 17.5 | 8.1 | 8.6 | 8.1 |
| BALANCED | 18.6 | 8.4 | 7.6 | 9.9 |
| HIGH | 28.3 | 11.2 | 9.4 | 9.3 |
| ULTRA | 29.3 | 13.7 | 14.3 | 10.1 |

Cost of a **single call** of `process()` (`tests/test_realtime.cpp`, R2; 16 voices, saw + sub oscillator, CHARACTER 0.5, FUSION layer on; *every*
sample of the 3 s run counts, including the acquisition of the note; budget per sample at 48 kHz is 20.8 µs):

| quality | pitch | mean µs | 99.9 % µs | 99.99 % µs | max µs | samples > 100 µs |
|---|---|---|---|---|---|---|
| ECO | 20 Hz | 1.49 | 56.0 | 57.1 | 144.8 | 1 |
| ECO | 110 Hz | 1.08 | 44.5 | 47.1 | 114.0 | 1 |
| ECO | 880 Hz | 1.37 | 14.4 | 17.6 | 121.2 | 1 |
| BALANCED | 20 Hz | 1.46 | 56.5 | 61.6 | 152.0 | 1 |
| BALANCED | 110 Hz | 1.19 | 44.5 | 55.2 | 128.9 | 1 |
| BALANCED | 880 Hz | 1.41 | 14.3 | 17.0 | 103.1 | 1 |
| HIGH | 20 Hz | 3.64 | 84.6 | 113.4 | 163.5 | 49 |
| HIGH | 110 Hz | 1.62 | 30.8 | 39.0 | 135.5 | 1 |
| HIGH | 880 Hz | 1.72 | 19.2 | 26.0 | 102.7 | 1 |
| ULTRA | 20 Hz | 3.59 | 87.2 | 104.5 | 167.1 | 31 |
| ULTRA | 110 Hz | 2.09 | 30.9 | 38.2 | 125.3 | 1 |
| ULTRA | 880 Hz | 1.91 | 19.3 | 26.2 | 113.4 | 1 |

The **maximum single-sample cost is 100 – 170 µs** (one event per note: the acquisition, which runs the period refinement and starts the
first table builds); in steady state it is 21 – 135 µs. Before the real-time hardening the worst single sample was 0.3 – 3 ms. The remaining
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
  repeating unit plus a fixed ≈ 8 ms tracker/lock overhead, essentially independent of QUALITY (the weight itself is a two-stage smoother):

| pitch | ECO | BALANCED | HIGH | ULTRA |
|---|---|---|---|---|
| 20.0 Hz | 256.6 ms | 256.6 ms | 257.1 ms | 257.1 ms |
| 41.2 Hz | 176.0 ms | 172.0 ms | 172.3 ms | 170.3 ms |
| 110.0 Hz | 73.0 ms | 63.9 ms | 64.1 ms | 61.1 ms |
| 440.0 Hz | 19.5 ms | 19.5 ms | 19.7 ms | 19.7 ms |
| 1760.0 Hz | 9.1 ms | 9.1 ms | 9.3 ms | 9.3 ms |

* Pitch steps: the clones re-acquire in 3.5 ms (fifth up) to 6.5 ms (semitone) at 110 Hz (`research/results/transient.txt`, T2); the original is
  heard throughout.
* The module shows the bloom time for the current pitch (`(M + 3)` periods) in its context menu; the Rack module reports **0 samples** of
  latency for the audio path, because the original is never delayed.

## 5. Not measured

CPU inside Rack with other modules running, on real Mac hardware (§6 is a shared virtual machine), with Rack's default `-march` flags on the
build machine (Rack builds plugins for `nehalem` on x86-64 and generic ARMv8 on arm64; §6 uses the arm64 flags, §1 – §4 use `-march=native`), at
44.1/96/192 kHz (the engine is tested for correctness there, `tests/test_engine.cpp` robustness section, not benchmarked), or under memory
pressure. `tools/bench` is the tool to run on the target.

## 6. Apple silicon — GitHub Actions `macos-14` (Apple M1, virtual, 3 cores, shared)

Produced by the `DSP tests (macos-14)` job of `.github/workflows/fusion-clone.yml` (run 36640266691, commit `8806b4b`; run 36639407374 gave the same
picture within the noise below). The benchmark is built with **the flags Rack uses for the plugin**: `-std=c++11 -O3 -funsafe-math-optimizations
-fno-omit-frame-pointer -march=armv8-a+fp+simd`, pffft (NEON) as FFT backend; same method and input as §1. The DSP tests (`make -C tests all`,
including the lock matrix, dynamics, artifact, real-time and stress tests) passed on this machine both with plain `-O2` and with those flags.

CPU load, % of one core of this VM (100 % = real-time limit of one core), fastest of three identical runs per cell:

| quality | pitch | 1 voice | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| ECO | 20 Hz | 1.68 | 2.05 | 2.28 | 3.33 | 4.94 |
| ECO | 110 Hz | 0.80 | 0.95 | 1.16 | 1.63 | 2.24 |
| ECO | 440 Hz | 1.37 | 1.29 | 1.45 | 1.94 | 2.76 |
| ECO | 3520 Hz | 1.63 | 1.43 | 1.65 | 2.13 | 2.90 |
| BALANCED | 20 Hz | 1.85 | 1.96 | 2.46 | 3.28 | 4.98 |
| BALANCED | 110 Hz | 0.99 | 1.09 | 1.36 | 1.81 | 3.60 |
| BALANCED | 440 Hz | 1.33 | 1.55 | 1.41 | 1.98 | 2.89 |
| BALANCED | 3520 Hz | 1.47 | 1.57 | 1.65 | 2.11 | 3.86 |
| HIGH | 20 Hz | 3.72 | 4.44 | 6.31 | 9.88 | 18.75 |
| HIGH | 110 Hz | 1.24 | 1.47 | 1.91 | 2.56 | 4.22 |
| HIGH | 440 Hz | 1.39 | 1.53 | 1.88 | 2.52 | 4.50 |
| HIGH | 3520 Hz | 1.83 | 2.10 | 2.49 | 3.35 | 4.97 |
| ULTRA | 20 Hz | 3.63 | 4.54 | 6.14 | 9.60 | 16.02 |
| ULTRA | 110 Hz | 1.65 | 1.83 | 2.25 | 3.23 | 5.05 |
| ULTRA | 440 Hz | 1.33 | 1.65 | 2.39 | 3.36 | 4.49 |
| ULTRA | 3520 Hz | 1.84 | 1.90 | 2.41 | 3.32 | 5.03 |

Most expensive 256-sample block at 16 voices, % of the block's budget (5.33 ms):

| quality | 20 Hz | 110 Hz | 440 Hz | 3520 Hz |
|---|---|---|---|---|
| ECO | 8.6 | 3.8 | 3.5 | 4.8 |
| BALANCED | 21.9 | 4.3 | 4.1 | 4.0 |
| HIGH | 21.2 | 6.2 | 4.8 | 5.7 |
| ULTRA | 22.5 | 6.6 | 5.5 | 6.3 |

FUSION algorithm layer (16 voices, BALANCED, 110 Hz): RATIO mode 5.15 %, HZ mode 3.89 %. Clone bloom-in times are the same as in §4 (they are a property
of the algorithm, not of the machine). Cost of a single call of `process()` (R2, first run of the workflow, `-O2`): **141 – 171 µs at most in every
cell; 201 samples above 100 µs at 20 Hz HIGH/ULTRA (all others: one), none above 250 µs**, against 20.8 µs of budget per sample and 5.3 ms per block.

What this does and does not say:

* At 16 voices the steady load is **2 – 5 % of one core in ECO/BALANCED and up to 19 % at 20 Hz in HIGH/ULTRA**, and the worst 256-sample block never
  needs more than 23 % of its budget — on a shared virtual M1 core. Against the Xeon of §1 the same cell costs between about **40 % and 100 %**
  of the Xeon's load (ECO 110 Hz / 16 voices 2.24 % against 5.30 %; the 20 Hz HIGH cell is the same on both, 18.8 % against 18.8 %).
* The virtual machine is shared and noisy: cells differ by up to ±30 % between two runs of the same commit (for example the BALANCED 110 Hz / 16-voice
  cell and the BALANCED 20 Hz worst block moved between runs). Read the table as an order of magnitude, not as a specification.
* It is **not** a measurement of a real Mac, and not of a Rack session: Rack's own engine, the GUI thread, other modules and the audio driver are
  absent (§5).
