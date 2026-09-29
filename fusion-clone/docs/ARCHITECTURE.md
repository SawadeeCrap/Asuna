# Architecture — period-synchronous harmonic analysis + free-running oscillator bank

This document explains **what the DSP does, why it was chosen over the alternatives that were tried, and where it stops being valid**.
Everything measured here was measured on *synthetic* signals from the project's hypothesis model of the source (`research/common/
fusion_source.hpp`, see `docs/RESEARCH.md` for what is and is not known about the real Fusion VCO2). Nothing here was listened to, and no
real Fusion VCO2 was available.

Contents: 1 requirements · 2 overview · 3 signal path · 4 candidates tested and why this one · 5 voice divergence model · 6 tracking,
transients, latency · 7 FFT-size and low-frequency study, quality modes · 8 implementation notes (aliasing, real time, determinism, failure
modes) · 9 limitations · 10 acceptance tests.

---

## 1. Requirements and what they force

| Requirement | Consequence for the design |
|---|---|
| One physical Fusion VCO2 in, 1 original + up to 15 *independent* virtual VCO2s out | Clones must have **free-running phase and independent pitch**. Anything derived from the input's own phase (delay taps, pitch-shifter read heads, phase-vocoder synthesis) stays partly *slaved* to the source and to its siblings. |
| The original is never replaced; VOICES = 1 is the original | Original path is a **direct connection** (zero latency, bit transparent apart from level/DC bookkeeping); clones are *added* in parallel. |
| The clones must carry the source's fingerprint (waveform, sub, tube colour, harmonic distribution) | Clones are built from a **measured harmonic model of the source**, not from a generic waveform. |
| 20 Hz … C8 (16 Hz … 4.2 kHz), including a sub oscillator that halves the repeating unit | The analysis must adapt its window to the *period* (a 20 Hz note needs ~100 ms of signal, a 2 kHz note ~1 ms) and treat "period" as the *repeating unit*. |
| Not chorus, not supersaw, not a generic pitch shifter, not an audible phase vocoder | Clone frequencies are *ratios of the tracked source frequency*, produced by an oscillator, not by re-timing the source. |
| Real time in Rack, no allocation/locks on the audio thread, deterministic, patch-saveable | Time-sliced analysis, pre-allocated buffers, seeds instead of stored random data (§8). |

## 2. Overview

```
                                   ┌──────────────── original path: direct, 0 samples, DC exact ───────────────────────────┐
                                   │                                                                                       ▼
 AUDIO IN ──┬─► level / AC-RMS / DC tracker ──────────────────────────────► clone level follows source envelope     ┌───────────────┐
            │                                                                                                       │ power-        │
            ├─► PitchTracker: 5 overlapping YIN lanes, sub-aware (repeating unit), significance gate                │ complementary │──► WIDTH ─► analog
            │        │ coarse period P0 (≈1 %)                                                                       │ mix: original │    summing ─► MIX
            │        ▼                                                                                               │ + N−1 clones  │    ─► OUTPUT
            │   period refinement (FFT cross-correlation, ≈0.01 sample)                                              └───────▲───────┘
            │        │ start value                                                                                           │
            ├─► MirrorRing (131072 samples ≈ 2.7 s)                                                                                   │
            │        ▼                                                                                                       │
            └─► CycleAnalyzer  ── resample the last M periods onto an ANGLE grid (Nc points per period, warped by the tracked │
                 (time-sliced)     frequency and slope) → Hann → FFT → harmonic j lives exactly on bin M·j                     │
                     │             → harmonic set c_j (amplitude + phase, j ≤ J ≤ 4095)                                        │
                     │             → phase-slope between successive hops → frequency measurement → 2-state Kalman (ν, dν/dt)  │
                     ▼                                                                                                         │
              lock state machine (IDLE → ACQ → LOCKED) + novelty detector (onsets, pitch steps, waveform switches)             │
                     │ smoothed harmonic set                                                                                   │
                     ▼                                                                                                         │
       per voice (one voice, one short stage per sample):  divergence curve × anti-alias taper → IFFT → period table ─────────┤
                     │  (double buffered, cross-faded)                                                                         │
                     ▼                                                                                                         │
       oscillator bank: phase_v += ν(now)·ratio_v ; ratio_v = static detune × OU drift ; 16/32-tap sinc table read ───────────┘
```

The two ideas that make it work:

1. **Computed order tracking (angle-domain analysis).** The analyser does not window the input in *time*. It resamples the last M periods
   onto a grid that is uniform in *cycles* (Nc points per period), following the tracked frequency and its slope. A note with vibrato or a
   glide becomes exactly periodic in that domain. With an integer number of periods in the window, harmonic j falls on FFT bin **M·j**, and the
   Hann window has exact zeros at ±2 bins, so neighbouring harmonics do not leak into each other — no peak picking, no bin quantisation, no
   window-length/resolution trade-off at low pitch (the window *is* M periods long, whatever the period).
2. **Free-running oscillators from a measured table.** Each clone reads a wavetable (one period of the source's band-limited waveform, rebuilt
   from the harmonic set with the clone's own smooth spectral divergence) with its **own phase accumulator**. The only thing the clone takes
   from the source in real time is the *frequency* (times its private ratio) and the *level envelope*. Phases are never re-synchronised,
   so the clones are independent oscillators, not filtered copies.

## 3. Signal path in detail

### 3.1 Input conditioning
* NaN/Inf/±1e6 are replaced by 0 before anything else. Levels are normalised so 1.0 = nominal 5 V.
* **AC-RMS envelope** over about one table period (the mean over the same window is subtracted): a DC offset (asymmetric tube stage, unipolar
  pulse) is not part of the tone and must not inflate the clone level. **DC passes through at unity** on the output path
  (`orig = g·in + (1−g)·dcSlow`), so changing VOICES never thumps.
* The clone gain follows `envInst / tableRms` with a 1 ms slew, so amplitude modulation of the source (VCA, tremolo) reaches the clones at once.

### 3.2 PitchTracker (coarse, ~1 %)
Five overlapping lanes (7–62, 50–220, 180–800, 640–2800, 2200–6500 Hz *repeating-unit* frequency) each decimate the input to ≈16× their top
frequency, so a YIN difference function needs only 16 … 71 lags. First dip below an absolute threshold (de Cheveigné & Kawahara 2002) with
parabolic interpolation, then a **multiplicity test**: if 2×, 3× or 4× the found period is *significantly* better, the composite really
repeats at that longer period — a sub oscillator — and the longer period is reported. A **significance gate** rejects lanes whose short window
sees only a flat stretch of a sparse waveform (narrow pulse at low pitch: normalised YIN is scale free and would "find" a period in ripple):
the window must carry ≥ 5 % of the input's long-term AC power. Cost: a few 10⁴ MACs per lane run.

The tracker only seeds the analyser. Its known weakness is a fractional-lag artefact at very high pitch (period 11.5 samples at C8: 2× the
period is closer to an integer than 1×, so the tracker reports an octave down); the engine corrects that with an **odd-harmonic test**
(a real sub puts energy on the odd harmonics of the doubled unit; if they carry < −37 dB the true period is half).

### 3.3 Period refinement
`Σ(a−b)² = Σa² + Σb² − 2Σab`: the correlation term for all integer lags in ±3.5 % of the coarse period comes from one FFT cross-correlation
(16 384 points, ≈0.1 ms; the direct scan it replaced cost 2.5 ms at 20 Hz) plus a parabolic peak. Verified equal to the direct scan to
0.0003 sample on nine signal/pitch combinations.

### 3.4 CycleAnalyzer
Per analysis hop (Q = clamp(⌊period / max-hop⌋, 1, 8) hops per period: about 3–6 ms apart at low and mid pitch, one period apart at high pitch):

1. **Geometry.** Window length L = M·Nc with Nc = nextPow2(1.15·period) clamped to [64, Nc_max] (grid step ≤ 0.87 samples: no decimation of
   the input). The window centre and the grid positions follow the tracker's frequency at the centre and a *smoothed* copy of its slope
   (grid warp `s(b) = −2b / (ν + √(ν² − 2·chirp·b))`), applied only when it moves a grid point by more than 0.02 sample.
2. **Resample** L points with a windowed-sinc reader (16 or 32 taps, Kaiser β = 8.6, cut-off 0.92 Nyquist; explicit SSE2/NEON dot product).
3. **Hann + FFT.** Harmonic coefficients c_j are read *directly* from bins M·j (j ≤ J = min(Nc/2−1, ⌊0.49·period⌋)). An **inter-harmonic
   residual** (Hann leakage of the harmonic bins removed) gives a *periodicity index*: 1 for a perfectly periodic input, ≈0.5 for noise. It is
   the basis of "trust this measurement less" and of the sub-appearing detector.
4. **Frequency measurement.** A symmetric window reports the source phase at its *centre*. The phase of c_j / c_j,prev is 2π·j·δ for a pure
   time shift, so the magnitude-weighted **phase slope** over harmonics gives the shift δ to a small fraction of a sample. The instantaneous
   frequency measurement is then an *exact inversion*, with no loop and therefore no bandwidth/dead-time trade-off:
   `ν̄ = (δ + Δbias + Δθ − ΔA) / Δt_c`
   (Δθ: advance of the reference phase, ΔA: change of window-centre offset, Δbias: the known chirp-bias of the applied grid warp).
5. **Robustness of the measurement.** (a) harmonics beyond `j_max = 0.15 / (M·ε)` (4 … 96; ε = recently observed prediction error) are ignored, so a
   coarse start value bootstraps cleanly; (b) the mean of the last Q measurements — *exactly one waveform period* because hops sit on
   multiples of 1/Q cycle — cancels the periodic leakage bias; (c) measurement variance comes from the least-squares residual and is inflated
   as the periodicity index falls; (d) hop-to-hop shifts beyond 0.25 cycle are declared incoherent.
6. **Two-state Kalman filter** (frequency, slope) with white-acceleration process noise (`accelRel` = 25 /s², enough for 8 Hz ±30 cent vibrato),
   5σ innovation cap and numerical-failure fallback. The same filter predicts the frequency at the next window centre (keeps the grid sharp)
   and *now* (drives the clone oscillators, so glides and vibrato are followed without the M/2-period analysis delay).

The analysis hop is a **resumable job** (§8.2): geometry fixed when the hop is due, then grid fill in 512-point chunks, FFT, extraction,
alignment/Kalman/publication, one bounded piece per sample.

### 3.5 Harmonic set → clone tables
* The set of each hop is complex-averaged over hops (valid because successive sets are aligned in the content-anchored frame).
* Publication is rate limited (4 … 12 ms). Each voice's table is built in ≤ 5 short stages, **one stage per sample**: (1) divergence curve ×
  taper written into a spectrum, (2) IFFT (Nc ≤ 8192), (3–4) for HIGH/ULTRA with CHARACTER > 0: static asymmetric waveshaping of the table and
  re-band-limiting in the frequency domain, (5) copy into the spare table and start a cross-fade (≈0.85 × the publication interval).
* **Anti-alias limit** per voice: harmonics above `0.49 / (ratio·(1 + shiftFrac))` are removed, the top 10 % of the surviving harmonics are
  tapered with cos² (the FUSION layer's RATIO copies run up to `1 + shiftFrac` faster).

### 3.6 Oscillator bank
`phase_v += ν_now · ratio_v`, table read with the same sinc kernel (16 or 32 taps). ν_now is the Kalman prediction for the current sample
plus the slew's own lag (1.5 ms), slewed once for all voices. `ratio_v` (static detune × OU drift) advances linearly between 64-sample
control ticks.

### 3.7 Mixing
* **Power-complementary original gain**: `go = g·√(Ne − (Ne−1)·w²)` with lock weight w ∈ [0,1] — the total power stays at its steady-state
  value while the clones fade in, so there is no level bump at bloom-in.
* **Level law**: per-voice gain `N^−(0.5 − law)`, default law 0.075 (independent oscillators add in power; a small positive law leaves the sum
  slightly louder as N grows, 1.8 dB at N = 16; law 0 = equal power, 0.5 = unity sum).
* **WIDTH**: per-voice pan `L = 1 − a`, `R = 1 + a` with `a = 0.25·WIDTH·pan_v`: L + R equals the mono sum for *any* WIDTH (mono compatible by
  construction; checked to 1.5·10⁻⁷).
* **Analog-style summing**: symmetric `k·tanh(x/k)` (k = 20 V equivalent) blended by SUMMING, only while clones are present.
* MIX crossfades dry input against the processed sum; OUTPUT is a smoothed trim.

## 4. Candidates tested, and why this one

The specification listed eleven approaches (A–K). The harness (`research/`) implements each as *an offline clone renderer* fed the same
synthetic source, and scores the sum "original + N−1 clones" against a **bank of N genuinely independent oscillators** built from the same
source model with the same tuning offsets (`renderBank`). Letters as in the specification:

| | Approach | Implementation in the harness | Status |
|---|---|---|---|
| A | conventional phase vocoder | Bernsee-style instantaneous-frequency PV, N = 4096 and 16384 (`A`, `A'`) | prototyped |
| B | identity phase locking | Laroche–Dolson peak locking, N = 4096 / 16384 (`B`, `B'`) | prototyped |
| C | sinusoidal modelling / partial tracking | STFT peaks + McAulay–Quatieri tracking + additive resynthesis | prototyped |
| D | additive partial resynthesis | harmonic heterodyne with an **oracle f0** (an upper bound for STFT partial methods) | prototyped |
| E | frequency-domain shifting | additive-Hz single-sideband shift (Bode style, Hilbert FIR), fundamental-matched | prototyped |
| F | time-domain pitch shifting | F1 rotating two-tap delay (chorus / classic shifter); F2 period-synchronous resampler with jump-by-period | prototyped |
| G | PSOLA-like | pitch-synchronous overlap-add on the same period source | prototyped |
| H | hybrid time-frequency | 2-band multi-resolution identity-locked PV (16384 below 300 Hz, 2048 above) | prototyped |
| I | instantaneous-frequency analysis | *is the analysis inside A, B, H*; in J it is the phase-slope/Kalman measurement — not a synthesiser on its own | folded in |
| J | hybrid spectral + oscillator bank | **this work** | production |
| K | "more appropriate modern approach found during research" | computed order tracking (Fyfe & Munck 1997) as the analysis front end of J | folded into J |

A note on honesty: the search for K found *techniques* (order tracking, tracking filters), not a ready-made cloner; there is no neural or
learned component anywhere, and none was tried.

### 4.1 Proxies, and what they can and cannot say

| Metric | What it probes |
|---|---|
| `lvl|dB|` | mean absolute deviation (dB) of the eight lowest harmonic *cluster* energies from a common offset: does the candidate keep the source's harmonic amplitude distribution? (reference = 0) |
| `spread ×` | width of each harmonic's line cluster relative to the reference bank (independent oscillators spread harmonic k by k × the detune; a frequency shifter spreads them equally in Hz) |
| `IH dB` | energy between harmonic clusters relative to the harmonic bands: phase-vocoder smear, windowing leakage, aliasing (lower is cleaner; the reference bank has its own floor) |
| `ripple` | std-dev over harmonics of the output/input amplitude ratio: comb filtering (chorus, phase cancellation between voices) |
| `CV`, `recur` | coefficient of variation and recurrence (max normalised autocovariance at 0.15–3 s) of a high band's envelope: a bank of independent oscillators beats like a random process (Rayleigh, CV → 0.52, recurrence low); a chorus / supersaw beats *periodically* |

They are **probes for named failure modes**, not a perceptual model, and their steady-state values are close for many candidates at mid/high
pitch. The discriminating conditions are low pitch, many voices and transients.

### 4.2 Results (final engine; complete tables in `research/results/compare_*.txt`)

*Saw 20 Hz, N = 4* (reference: `IH` −80.3 dB, `ripple` 0.37, `recur` 0.991):

| candidate | lvl\|dB\| | spread × | IH dB | ripple |
|---|---|---|---|---|
| **J (this work)** | **0.39** | **1.00** | **−79.7** | **0.32** |
| A phase vocoder 4096 | 1.07 | 0.99 | −44.1 | 0.94 |
| A′ phase vocoder 16384 | 0.62 | 1.00 | −54.3 | 0.38 |
| B identity locking 4096 | 0.86 | 0.50 | −44.7 | 1.66 |
| B′ identity locking 16384 | 0.63 | 1.00 | −50.3 | 0.37 |
| C sinusoidal tracking | 0.39 | 0.49 | −32.7 | 0.51 (envelope statistics collapse: CV 0.005) |
| D heterodyne, oracle f0 | 0.67 | 1.03 | −45.0 | 0.38 |
| E SSB frequency shift | 0.39 | 0.51 | −80.0 | 0.02 |
| F1 rotating delay | 0.85 | 0.99 | −80.0 | 0.70 |
| F2 period resampler | 0.66 | 1.01 | −79.8 | 0.37 |
| G PSOLA | 0.44 | 1.01 | −79.7 | 0.47 |
| H multi-resolution PV | 0.63 | 1.00 | −50.3 | 0.70 |

*Saw 110 Hz, N = 16* (reference: `IH` −90.4 dB, `ripple` 0.10, CV 0.518, `recur` 0.781):

| candidate | lvl\|dB\| | spread × | IH dB | ripple |
|---|---|---|---|---|
| **J** | **0.18** | **1.00** | **−96.8** | 0.17 |
| A 4096 / A′ 16384 | 0.58 / 0.49 | 1.00 / 0.98 | −57.3 / −86.9 | 0.50 / 0.50 |
| B 4096 / B′ 16384 | 0.60 / 0.53 | 1.00 / 0.99 | −58.4 / −88.5 | 0.50 / 0.50 |
| C sinusoidal | 0.58 | 1.01 | −101.0 | 0.50 |
| D oracle heterodyne | 0.64 | 1.01 | −92.0 | 0.54 |
| E SSB shift | 0.11 | **0.24** | −90.4 | 0.00 |
| F1 rotating delay | **1.40** | 1.00 | −91.0 | **1.27** |
| F2 period resampler | 0.60 | 1.01 | −84.2 | 0.54 |
| G PSOLA | 0.12 | 1.01 | −87.7 | 0.15 |
| H multi-resolution PV | 0.67 | 1.00 | −78.8 | 0.56 |

*Saw + Doppler-style DETUNE stage (source hypothesis), 110 Hz, N = 8* (reference: `IH` −48.0 dB, `ripple` 0.17, `recur` 0.48): J 0.22 / 1.00 /
−52.6 / **0.30**; B 0.24 / 1.01 / −48.0 / 0.17; G 0.22 / 1.03 / −47.4 / 0.12; H 0.19 / 1.01 / −48.2 / 0.13; E 0.22 / 0.83 / −48.4 / 0.00;
F2 1.19 / 1.10 / **−28.0** / 1.00. J's `recur` is 0.68 against 0.48 for the reference: see limitation 9.2 (sideband modulation is shared by
all clones).

Texture statistics over many random realisations (`research/results/texture.txt`; band envelope CV and recurrence, mean ± sd over 10 seeds):

| scene | independent bank (reference) | **J** | F1 rotating delay | evenly spaced detunes ("supersaw") |
|---|---|---|---|---|
| saw 110 Hz, N = 8 | 0.504 ± 0.026 / 0.468 ± 0.197 | **0.499 ± 0.016 / 0.476 ± 0.178** | 0.508 / 0.359 | 0.468 / **0.997** |
| saw 110 Hz, N = 16 | 0.502 / 0.226 | **0.513 / 0.246** | 0.504 / 0.194 | 0.490 / **0.994** |
| saw 220 Hz, N = 8 | 0.503 / 0.470 | **0.494 / 0.471** | 0.497 / 0.455 | 0.457 / **0.999** |
| saw 55 Hz, N = 8 | 0.500 / 0.468 | **0.499 / 0.465** | 0.503 / 0.481 | 0.512 / **1.000** |

Transients (`research/results/transient.txt`, N = 8, note-on with a 2 ms attack at 110 Hz; the offline candidates get their best case, no
extra lag): pre-attack echo relative to the steady level — reference −0.0 dB, **J −3.6**, A +0.9, B +6.0, C +1.5, F2 +0.5, G +2.4, H −7.0 (but
96 ms rise); overshoot — reference 2.4 dB, **J 0.1**, B 9.1, C 6.1, G 2.4. J's clones *bloom in* over a few periods while the untouched original
provides the attack; the price is the bloom time (§6).

### 4.3 Why J, in one paragraph per rejected family

* **Phase vocoders (A, B, H).** With N = 4096 the inter-harmonic floor is −44 … −58 dB at 20–110 Hz (a 4096-point window cannot separate 20 Hz
  partials): the "phasey/washy" signature. N = 16384 fixes the floor at 110 Hz but blurs attacks and pitch steps (B′ overshoot 7.5 dB, H rise 96 ms)
  and still shows ripple 0.5 at N = 16 voices, because all clones are made from the *same* frames and cancel/add coherently. Identity locking
  helps peak coherence but not the low-pitch resolution problem.
* **Sinusoidal tracking (C) and heterodyne partial resynthesis (D).** Excellent when it works (IH −101 dB at 110 Hz) but at 20–41 Hz the peak
  picker/tracker breaks (IH −33 … −28 dB, envelope statistics collapse at 20 Hz), it needs f0 (D was *given* the oracle f0 and still failed at low
  pitch) and in the harness (unoptimised, offline) it needs tens of seconds per 14 s of audio (J: 0.9 s). Nothing exploits that the source is *exactly periodic in angle*.
* **Frequency shifting (E).** Reproduces neither the harmonic-proportional cluster spread (spread × 0.20 … 0.51 in every scene but the Doppler
  one) nor harmonic ratios: it is a different sound (inharmonic beating), useful only as the literal-SSB reading of the DETUNE text (the HZ
  mode of the optional FUSION layer).
* **Delay-line pitch shifting (F1).** Level deviation 0.4 … 1.4 dB (0.8 typical) and ripple up to 1.27: it is a comb filter with a moving notch, the chorus
  signature the specification forbids. **Period-synchronous resampling (F2) and PSOLA (G)** are the strongest competitors at mid pitch, but each
  clone is a *re-timed copy of the source*: their pitch/epoch tracker must be right on every cycle (F2 collapses on the DETUNE-sideband source,
  IH −28 dB), they cannot follow a sub-oscillator that changes the repeating unit, and every clone inherits the source's phase noise.
* **Supersaw / evenly spaced detunes.** Envelope recurrence 0.99 … 1.00 against 0.23 … 0.47 for independent oscillators: a *periodic* beating
  pattern, audible as a chorus (`texture.txt`).

J is the only approach that at the same time reproduces the source's harmonic profile to ≤ 0.4 dB, keeps the inter-harmonic floor at the
reference level from 20 Hz upward, produces independent-population beat statistics, leaves the attack to the untouched original, and costs
< 1 % of a core per clone (`docs/BENCHMARKS.md`).

## 5. Voice divergence model

All per-voice quantities are **derived deterministically from (seed, voice index)** (§8.3) as *unit* draws, and scaled by the user controls.

| Aspect | Model | Controls |
|---|---|---|
| **Static detune** | Gaussian quantile of a golden-ratio low-discrepancy sequence (prefix-stable: raising VOICES never moves existing clones), central 20 % of the probability mass skipped so no clone sits on the original; `cents = z_v · range/2.4 · SPREAD^curve` | SPREAD, expert: range (5–60 cents, default 20), curve (default 1.6) |
| **Drift** | Ornstein–Uhlenbeck (exact discretisation, bounded ±3σ): σ = 1.2 cents × DRIFT, time constant `30·0.07^rate` s (30 s … 2.1 s); a shared "temperature" component (correlation ρ, default 0.25); level drift σ 0.12 dB × DRIFT, tilt drift σ 0.25 dB × DRIFT | DRIFT, expert: rate, correlation |
| **Phase decoupling** | Fresh random start phase per lock event (`frac(θ + PHASE·u)`), plus a static smooth per-voice *phase dispersion* (two log-frequency sinusoids, ±0.12 rad × PHASE) | PHASE |
| **Correlated harmonic divergence** | A smooth curve over log₂(harmonic): tilt (≈ ±0.45 × σ_tilt dB across 4 octaves) + three bells (0.35 dB scale) + absolute-frequency HF corner + odd/even balance (±0.12 dB scale), interpolated on a 96-point grid — neighbouring harmonics move *together* (no per-partial noise) | HARMONIC |
| **Subtle nonlinear divergence** | Static asymmetric `tanh` shaping of the table (≤ ~1 % THD; only at HIGH/ULTRA, re-band-limited so it cannot alias), per-voice level tolerance (σ 0.3 dB × CHARACTER), ±0.01 cent white pitch jitter (no additive noise is generated: the noise of the source stays with the original) | CHARACTER |
| **Voice correlation** | Independent OU drifts + shared component ρ; independent phases; correlation between clones and original is what remains of the *shared* harmonic model (limitation 9.1) | expert: drift correlation |
| **Level normalisation** | `N^−(0.5 − law)`, power-complementary original gain (§3.7) | expert: level law |
| **Stereo** | Linear pan from a separate low-discrepancy sequence; mono-compatible for any width | WIDTH |
| **Summing** | Symmetric tanh soft clip, applied only while clones are present | expert: analog summing |
| **FUSION layer (optional)** | Two extra table readers per clone modelling the documented BBD "frequency shifter" cluster: RATIO mode = ratio pushed apart by a soft-square LFO (Doppler / time-varying-delay reading, ±(3 + 22·amount) cents), HZ mode = additive single-sideband offsets (Hilbert pair + carriers, (0.35 + 5.5·amount) Hz); each voice has its own LFO phase and rate tolerance; LFO rate and depth rise together with the knob, as the product text says of the hardware. **Hypothesis**, see `docs/RESEARCH.md` §2.3 | MODE, expert: Fusion layer, shift model |

## 6. Tracking, transients, latency

**States.** IDLE → (two consistent confident tracker estimates, ≥ M+1 periods of clean history) ACQ → LOCKED when coherence > 0.92, at least
two good hops and periodicity > 0.30. A 0.6 s watchdog abandons an acquisition that produces no usable set.

**Drops** (`dropCount` reasons): 1 novelty — the one-period residual `x(n) − x(n−P)` rises 6× above its slow estimate and above 10 % of the level
within a few ms (onsets, pitch steps, waveform switches; slow natural modulation never opens that gap); 2 tracker mismatch — three confident
tracker estimates incompatible with P, P/2, 2P; 3 coherence < 0.6 or periodicity < 0.12 twice in a row; 4 re-lock at the *doubled* period
(the SUB knob turned up during a held note: the tracker now reports 2P and the single-period model explains the signal poorly). On a drop the
lock weight falls in ≈1.2 ms and the original carries on alone.

**Glides and vibrato** do not drop the lock: the Kalman slope carries them (D1: ±15 cent 5 Hz vibrato at 55 / 110 / 220 / 880 Hz, zero drops,
clone-versus-source pitch error 5.2 / 1.8 / 0.7 / 0.08 cent rms in BALANCED; 11.4 / 3.7 / 1.4 / 0.15 in HIGH — M = 4 doubles the measurement
latency, physics rather than tuning: §7). A 110 → 220 Hz glide over 1 s: 0 unlocked samples, clone pitch error 0.19 cent.

**Latency.** The original path has **0 samples** of latency (a direct connection; measured exactly 0 in `tests/test_engine.cpp`). The clones
*bloom in*: the time until the clone weight exceeds 90 % after an onset — 267 ms at 20 Hz, 183 ms at 41 Hz, 66–78 ms at 110 Hz, 21 ms at
440 Hz, 10 ms at 1.76 kHz, essentially independent of QUALITY (`docs/BENCHMARKS.md`); it is a few periods of the tracked unit, not a fixed
delay. A pitch step re-acquires in 3–6 ms at 110 Hz (T2, `research/results/transient.txt`).

## 7. FFT-size / analysis-window study, low-frequency operation, quality modes

Program: `research/exp_fft_study.cpp` → `research/results/fft_study.txt`.

**S1 — exactness of the harmonic model.** An additive sawtooth with amplitude 1/j up to 0.46·fs (analytic truth) is analysed for 1.5 s;
error of the harmonic amplitudes in dB, as max over harmonics below 0.30·fs / max over 0.30–0.46·fs, for window M = 2 and M = 4 periods
(N_c ≤ 8192 gives the same numbers as N_c ≤ 4096 in every row):

| f0 (Hz) | N_c ≤ 1024: M=2 / M=4 | N_c ≤ 2048: M=2 / M=4 | N_c ≤ 4096: M=2 / M=4 |
|---|---|---|---|
| 20 | **2.85 / 0.0 · 22.4 / 0.0** (J 511) | 0.0006 / 4.3 · 0.0005 / 4.0 (J 1023) | 0.0003 / 6.0 · 0.0003 / 6.0 (J 1175) |
| 30 | **5.5 / 6.0 · 4.0 / 4.5** (J 511) | 0.0004 / 6.0 · 0.0012 / 6.0 (J 784) | 0.0004 / 6.0 · 0.0012 / 6.0 |
| 55 | 0.0004 / 5.9 · 0.0004 / 5.9 | same | same |
| 110 | 0.0003 / 5.6 · 0.0004 / 5.6 | same | same |
| 440 | 0.0004 / 5.6 · 0.0009 / 5.6 | same | same |
| 1760 – 4186 | 0.0001 – 0.0002 / 2.0 – 2.5 | same | same |

* Where the grid has at least as many points per period as the input has samples (N_c ≥ period), the harmonic amplitudes below 14.4 kHz are
  exact to **≤ 0.0012 dB**, independent of M and of the pitch: the harmonic sits on its bin and the Hann window has a zero at ±2 bins.
* Where it has fewer, the resampler decimates without an anti-alias filter and content above the grid's Nyquist folds into the harmonics:
  2.8 – 22 dB errors at 20–30 Hz with N_c ≤ 1024. With N_c ≤ 2048 a *plain* 20 Hz saw is still exact below 14.4 kHz (step 1.17 samples), but the
  4800-sample repeating unit of a 20 Hz note **with a sub oscillator** (step 2.3 samples) was not: an earlier ECO with N_c ≤ 2048 failed the
  lock-robustness matrix there (frequency excursions of 5–8 cents on `saw+sub` and `pulse+sub+tube`). **That is why ECO and BALANCED use
  N_c ≤ 4096 and HIGH/ULTRA ≤ 8192** (step 1.17 samples resp. 0.59 at the 20 Hz + sub extreme).
* The second number of each pair is the price of a compact resampling kernel: between 0.30·fs (14.4 kHz) and the kernel cut-off (0.46·fs,
  22 kHz) the reader droops (table S1b). It affects the *clones only*; the original is untouched.

**S1b — interpolation kernel** (M = 2, N_c ≤ 4096; max error in dB below 0.30·fs / 0.30–0.40·fs / 0.40–0.46·fs):

| f0 (Hz) | 16 taps (ECO, BALANCED) | 32 taps (HIGH, ULTRA) |
|---|---|---|
| 20 | 0.0058 / 1.43 / 6.02 | 0.0008 / 0.127 / 6.02 |
| 55 | 0.0042 / 1.43 / 5.96 | 0.0007 / 0.125 / 5.90 |
| 110 | 0.0037 / 1.38 / 5.83 | 0.0003 / 0.109 / 5.64 |
| 440 | 0.0018 / 1.20 / 5.83 | 0.0007 / 0.062 / 5.64 |

**S2 — vibrato tracking (analyser alone, ±15 cents at 5 Hz):** rms error of the frequency prediction for "now":

| f0 (Hz) | 41.2 | 55 | 110 | 220 | 440 | 880 |
|---|---|---|---|---|---|---|
| M = 2 | 8.64 | 4.89 | 1.70 | 0.60 | 0.16 | 0.05 |
| M = 4 | 17.79 | 10.77 | 3.37 | 1.17 | 0.31 | 0.08 |

**S3 — window length:** M × period = 100 / 200 ms at 20 Hz (M = 2 / 4), 48 / 97 ms at 41.2 Hz, 18 / 36 ms at 110 Hz, 4.5 / 9 ms at 440 Hz,
1.1 / 2.3 ms at 1.76 kHz.

**Decision.** M is a trade of *measurement latency and vibrato tracking* (M = 2 twice as good) against *noise averaging and resolution of
non-harmonic energy between harmonics* (M = 4: two extra bins between harmonics, √2 more averaging). Hence the modes:

| QUALITY | window M | resampling kernel | N_c max | analysis hop (max) | table publication | CHARACTER shaping | intent |
|---|---|---|---|---|---|---|---|
| ECO | 2 periods | 16 taps | 4096 | 6 ms | ≥ 12 ms | — | lowest CPU; tracks vibrato well |
| BALANCED | 2 periods | 16 taps | 4096 | 4 ms | ≥ 7 ms | — | default |
| HIGH | 4 periods | 32 taps | 8192 | 4 ms | ≥ 5 ms | yes | cleanest low notes, flat to 19 kHz |
| ULTRA | 4 periods | 32 taps | 8192 | 3 ms | ≥ 4 ms | yes | densest updates, most CPU |

**20 Hz.** A 20 Hz saw with a sub oscillator repeats every 100 ms (4800 samples): at 48 kHz BALANCED analyses 2 units (200 ms) on a 4096-point
grid (step 1.17 samples), HIGH/ULTRA 4 units (400 ms) on 8192 points. Lock matrix: every waveform family locks at 20 Hz in every mode; C0
(16.35 Hz) locks in 0.20 s (`research/results/engine_checks.txt`); the analyser's steady-state frequency jitter at 20 Hz is 0.001 cent rms.
The price is the bloom time (267 ms), and vibrato tracking at ≤ 55 Hz (S2).

## 8. Implementation notes

### 8.1 Aliasing
* Analysis reads the input at a grid step ≤ 0.87 sample (no decimation) except at the limits discussed in §7.
* Clone tables are band-limited *before* the oscillator: harmonics above `0.49/(ratio·(1+shiftFrac))` removed, top 10 % tapered (cos²), so a
  clone that is detuned up cannot fold. The table has ≥ 1.15 points per output sample, so the images of a sinc-interpolated table read start
  above 0.66·fs, far above the 0.46·fs kernel cut-off.
* CHARACTER's nonlinearity acts on the *table*, then removes everything above J in the frequency domain — **no oversampling is needed at run
  time**. Everything the shaping generates that is higher than the source's band is discarded, not folded.
* Measured: the inter-harmonic floor of J is at or below the reference bank's own at every pitch tested (e.g. −85.8 dB against −78.5 dB at
  1.76 kHz, N = 8).

### 8.2 Real time
* `process()` and `setParams()` are **allocation-free and lock-free** — counted by replacing global `operator new/delete` around every call
  (`tests/test_realtime.cpp`, R1: zero heap operations over 7 s of busy input including quality switches, VOICES changes, hostile input).
  Function-local statics (interpolation kernels, log₂ table) are built in `prepare()`.
* **Time slicing.** Cost peaks are removed rather than averaged: an analysis hop is a resumable job (fill 512 grid points per sample, then FFT,
  extraction, alignment/publish); a voice table is built in ≤ 5 stages, one per sample, one voice at a time; the period refinement is one FFT
  correlation. Worst *single-sample* cost at 16 voices went from 0.3 … 3 ms to **100 … 165 µs including the acquisition of a note** (21 … 131 µs in
  steady state; R2, min of three identical runs to remove scheduler noise), mean 1.0 … 3.3 µs per sample; see `docs/BENCHMARKS.md`.
* **SIMD.** The 16/32-tap sinc dot product (the inner loop of every read) is explicit SSE2 / NEON / scalar (`kernelDot`), verified against a
  double-precision reference (`tests/test_sinc.cpp`) and syntax/codegen-checked for aarch64 with clang; the FFT is pffft (SSE/NEON).
* FFT plans, windows, tables and rings are allocated in `prepare()` for the largest configuration; quality switches only re-configure.

### 8.3 Determinism and patch state
Per-voice seeds: a *personality* `(seed, voice)` (detune quantile, pan, level, tilt, bells, dispersion, drive, bias, LFO phase/rate), plus
one RNG stream per voice for drift/jitter and one per lock event for start phases. The seed is stored in the patch (`dataToJson`), so a saved
patch reproduces the same population; RANDOMIZE draws a new seed; the same seed and the same input give **bit-identical output**
(`tests/test_module.cpp`: reloaded patch differs by 0).

### 8.4 Failure modes (all tested in `tests/test_engine.cpp`, `research/exp_engine_checks.cpp`)
Silence, DC, white noise, ±1e9 garbage, NaN/Inf, hard-clipped saw, a chord, sample rates 44.1 … 192 kHz, disconnected input, polyphonic input
(channel 1 only): output stays finite and bounded, no lock on garbage (noise/DC never lock), clean fall-back to the original.

## 9. Limitations (scientific and practical)

**9.1 Perceptual, not electrical, equivalence.** The clones are *models of independent oscillators built from the source's measured harmonic
set*. They contain no VCO2 circuit model. Whatever the source's signal does not reveal cannot be cloned; whatever it contains is inherited
**coherently** by every clone: the harmonic amplitudes/phases (deliberately, with smooth per-voice divergence on top), and any non-periodic
component the analysis cannot represent (below). Real independent oscillators would each have their own component tolerances in ways that
are not smooth functions of harmonic number; the divergence model is a *plausible statistical stand-in*, not a measurement.

**9.2 What a periodic model cannot carry.** Noise, and the DETUNE section's BBD sidebands (which beat against the main line at the internal
LFO rate), are *not* exactly periodic in the source's period. The original keeps them; the clones receive only what the harmonic model
explains (the periodicity index falls and measurement noise is inflated accordingly). Consequences: clones are somewhat "cleaner" than the
source in that respect, and where the source's detune modulation is strong every clone shows the *same* sideband modulation unless the optional
FUSION layer (independent per-voice LFO) is used — which is itself a **hypothesis** about the hardware (`docs/RESEARCH.md` §2.3: Doppler vs
SSB is undecided without a measurement; `docs/REFERENCE_PROTOCOL.md`).

**9.3 Monophonic, quasi-periodic sources only.** A chord, a noise source, or heavy FM has no single period: the engine does not lock and the
original passes unchanged. Polyphonic input uses channel 1 only.

**9.4 Bloom-in.** Clones appear a few periods after a note onset or pitch step (267 ms at 20 Hz … 10 ms at 1.76 kHz). Fast pitch modulation
(vibrato > ~8 Hz at ≤ 55 Hz, or a pitch *step*) is followed by a re-acquisition, not seamlessly.

**9.5 Band edge.** The analysis kernel droops above 0.30·fs (16 taps: 1.4 dB at 19 kHz; 32 taps: 0.13 dB) and reaches −6 dB at 0.46·fs; the
clones' top octave is slightly darker than the original's. Inaudible for a sawtooth (1/j) but stated.

**9.6 Not verified.** Linking against the real Rack SDK, the Rack GUI, the macOS/Apple-silicon build, real-time CPU on the target machine, and
*listening*. Real-Fusion behaviour (sub sync, detune, tube colour, waveform switching transients) is a hypothesis model; the reference protocol
describes the measurements that would settle it.

## 10. Acceptance tests (specification tests 1 – 15) and where they run

| # | Test | Where | Result |
|---|---|---|---|
| 1 | VOICES = 1 → the input, zero latency | `test_engine` TEST 1, `test_module` | exact (latency 0 samples) |
| 2–5 | VOICES = 2 / 4 / 8 / 16: rising density, independent-oscillator statistics, no chorus/supersaw signature | `test_engine` TESTS 2–5, `research/exp_texture` | CV and recurrence within the reference's spread; supersaw recurrence 0.99 |
| 6 | SPREAD = 0 → clones on the original pitch, no modulation | `test_engine` TEST 6 | pass |
| 7 | DRIFT = 0 → no random movement | `test_engine` TEST 7 | pass |
| 8 | DRIFT = 100 % → slow, bounded, independent instability | `test_engine` TEST 8 (drift velocity at 1 s scale < 3 cent/s) | pass |
| 9, 10 | PHASE = 0 → aligned; PHASE = 100 % → maximum useful divergence | `test_engine` TESTS 9/10 | pass |
| 11, 12 | HARMONIC = 0 → source identity; 100 % → subtle individuality | `test_engine` TESTS 11/12 | pass |
| 13 | WIDTH = 0 mono-compatible; WIDTH > 0 keeps the mono fold-down | `test_engine` TEST 13 | L = R exactly at 0; fold-down deviation 1.5·10⁻⁷ |
| 14, 15 | MIX = 0 → original only; MIX = 100 % → processed sum | `test_engine` TESTS 14/15, `test_module` | exact / differs by 0.79 rel. |

Further suites: `test_lock` (waveform × pitch matrix, 168 cells per quality mode), `test_dynamics` (vibrato, SUB sweep, WAVE switch, note gap),
`test_analyzer`, `test_pitch`, `test_sinc`, `test_fft`, `test_hilbert`, `test_realtime`, `test_module` (real `Module` class, headless).
