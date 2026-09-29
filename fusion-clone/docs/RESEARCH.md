# Research report — Erica Synths Fusion VCO2 and the state of the art in spectral cloning

This report is the output of the research phase that preceded the implementation of **FusionClone**. It has three jobs:

1. say precisely **what is known about the Fusion VCO2** and how well it is known (every technical statement is classified),
2. record **what the design is allowed to assume** about the source as a consequence,
3. list the **DSP literature** the architecture study drew on.

> **Read this first — limits of the evidence.** The sandbox this project was built in blocks outbound access to `ericasynths.lv` (the
> manufacturer's site, product page and manual PDF); fetching them failed with `EGRESS_BLOCKED`. The block was **not** circumvented.
> Everything below about the hardware therefore comes from *search-engine summaries of the manufacturer's product text, retailer pages, a
> review and forum threads* (secondary sources), and **no hardware, recording, schematic or measurement of a real Fusion VCO2 was available**
> (the module's schematics are not in Erica Synths' open-source DIY line either). Circuitry is never invented in this project: whatever
> cannot be sourced is marked INFERRED, SPECULATIVE or UNKNOWN below.

## 1. Classification legend

| Tag | Meaning |
|---|---|
| **CONFIRMED** | Stated by the manufacturer (or quoted from the manufacturer by retailers/reviewers) in the sources listed in §6. Confirmed *as text*; not verified on hardware. |
| **INFERRED** | Not stated, but follows from confirmed statements plus established engineering practice. The inference chain is given. |
| **MEASURED** | Measured by this project. **Only ever on synthetic signals** produced by the project's own hypothesis model of the source (`research/common/fusion_source.hpp`), never on a real module. |
| **SPECULATIVE** | A plausible value or mechanism chosen so that the algorithms can be stress-tested. Labelled as such wherever it appears in code (`// SPECULATIVE`). |
| **UNKNOWN** | Not available from any source used. The design must not depend on it. |

## 2. The Fusion VCO2 — statement by statement

### 2.1 Oscillator core, waveforms, outputs

| # | Statement | Class | Basis |
|---|---|---|---|
| 1 | The core is a "highly stable **AS3340**-based VCO". | CONFIRMED | Manufacturer's product text as reproduced by retailers/reviews (§6 [1]–[5]). |
| 2 | Three waveforms are available **simultaneously**; outputs are **TRI**, **PLS**, **SUB** and the main **OUT**. | CONFIRMED | [1]–[5]. |
| 3 | The main **OUT** carries the waveform selected with **WAVE**; the exact law of the WAVE control (cross-fade curve, whether it passes through intermediate shapes) is not published in the sources used. | CONFIRMED (existence) / **UNKNOWN** (law) | [1]; the sources list WAVE among the controls but not its transfer law. |
| 4 | Tracking over "an 8 octave range", VCO range "C0–C8 and more". | CONFIRMED | Retailer specification lines [3], [6]. |
| 5 | An AS3340 is a clone of the CEM3340 sawtooth-core VCO (saw from the integrator, triangle and pulse derived from it). | INFERRED | General knowledge of the CEM3340/AS3340 family; not from a Fusion-specific source. |
| 6 | The saw/triangle/pulse are *not* perfectly band-limited and contain analog shape errors (slight curvature, reset spike, finite fall time). | INFERRED | Typical of the chip family; magnitude UNKNOWN. |
| 7 | Power: 14 HP, 228 mA at +12 V, 55 mA at −12 V. | CONFIRMED | Retailer specification [6]. |
| 8 | Signal levels (Vpp) of TRI / PLS / OUT / SUB. | **UNKNOWN** | Not in the sources. The module therefore normalises to the nominal 5 V level (1.0 = 5 V) and never assumes a specific level; see the `INPUT` handling in `docs/ARCHITECTURE.md`. |

### 2.2 Sub oscillator

| # | Statement | Class | Basis |
|---|---|---|---|
| 9 | A **transistor-based sub oscillator one octave below** the main oscillator, with a level control. | CONFIRMED | [1]–[5]. |
| 10 | A **COLOR** switch applies a low-pass filter to the sub. | CONFIRMED | [1]–[5]. Cut-off frequency UNKNOWN (the hypothesis model uses 700 Hz, **SPECULATIVE**). |
| 11 | The sub has its own **sync circuitry** ("unique sub sync"). | CONFIRMED | [1]. Behaviour UNKNOWN. |
| 12 | The sub is a divide-by-two of the main cycle (a square at f/2), so the *composite* waveform repeats every **two** main cycles. | INFERRED | "−1 oct transistor sub" + standard flip-flop divider practice. **Consequence for the design: the repeating unit of the analysis is one or two main cycles**, chosen by measurement (`PitchTracker` multiplicity test, `Engine::unitIsDoubled`). |

### 2.3 DETUNE — the section the whole module is about

| # | Statement | Class | Basis |
|---|---|---|---|
| 13 | DETUNE is "**two BBD delay lines that make a frequency shifter** that is mixed back to the principal oscillator in order to **emulate two detuned VCOs**". | CONFIRMED | Manufacturer text as quoted in [1]–[5]. |
| 14 | Turning the knob clockwise increases **both the detune amount and the frequency of an internal LFO**; at the extreme clockwise setting the result is "**crazy frequency beats**". | CONFIRMED | [1]–[5]. |
| 15 | Reviewers describe the result as "thick", "wide", "chorus-like", "bucket-brigade based short delay driving the detune effect". | CONFIRMED (as opinion) | [7], [8]. |
| 16 | Forum discussion: "the two delay lines are modulated slightly to give a chorusing effect", and "the BBD works together with PWM". | Secondary, **unverified** | [9] (search summary of a forum thread; the thread itself was not read). Treated as a hint only. |
| 17 | The manufacturer's phrase "frequency shifter" describes a **true single-sideband (additive-Hz)** shift. | **Not established** | The literal reading (Bode/Moog style SSB, output = input + Δf) is one hypothesis (H-SSB). |
| 18 | It is instead a **variable-delay (Doppler) pitch shifter**: two BBDs whose clock is swept by the internal LFO and cross-faded so that one line is always "fresh". The output pitch ratio is then *multiplicative* (f × r), not additive (f + Δf). | INFERRED (H-DOPPLER) | An analog delay line whose clock rate is modulated changes pitch by the Doppler effect; a two-BBD pitch shifter with out-of-phase ramps is the textbook analog architecture (see the BBD pitch-shifter description in [10]). A BBD cannot realise a true SSB shifter without additional quadrature circuitry that the sources never mention. The marketing word "frequency shifter" is then loose language. |
| 19 | Delay times, clock range, BBD part numbers, LFO waveform, number of BBD stages, which lines are modulated in opposite directions, mix ratios, maximum detune in cents. | **UNKNOWN** | Not in the sources. The hypothesis model uses 3–25 cents and 0.12–2.1 Hz (**SPECULATIVE**). |

**Consequence for the design.** The module must work for *both* H-DOPPLER and H-SSB, so it never assumes either:

* the clones' own detune is applied as a **multiplicative ratio** by default (this is how independent VCOs behave, and it is what the
  Fusion's DETUNE is supposed to emulate),
* the optional **Fusion layer** (context menu → *Shift model*) can add a small detune-cluster around each clone either as a multiplicative
  ratio (**RATIO**, H-DOPPLER) or as an additive frequency shift built from a Niemitalo polyphase-IIR Hilbert pair (**HZ**, H-SSB),
* the `tools/fusion_detune_probe.py` protocol (docs/REFERENCE_PROTOCOL.md) tells a user with a real module how to decide between the two
  in ten minutes: with H-DOPPLER the sideband spacing of harmonic *n* grows ∝ n, with H-SSB it is the same for every harmonic.

### 2.4 Tube stage

| # | Statement | Class | Basis |
|---|---|---|---|
| 20 | The Fusion series "combines vacuum tubes and semiconductors". | CONFIRMED | [1]–[5]. |
| 21 | **TUBE CRUNCH** is "a distinct tube overdrive added on top of the mix" (oscillator + sub + the external audio input). | CONFIRMED | [1]–[5], [3] ("external audio input"). |
| 22 | Which tube, its operating point (plate/heater/bias voltages), and whether the stage is a triode gain stage, a cathode follower or a clipper. | **UNKNOWN** | Searches for the tube type (6N3P / ECC82 / 12AX7 / 6922 …) returned no confirmation. Not guessed. |
| 23 | The transfer function is asymmetric with a soft knee, producing even and odd harmonics. | INFERRED | Generic triode overdrive behaviour. The hypothesis model uses an asymmetric soft-saturation curve (**SPECULATIVE**). |

**Consequence.** CLONES must inherit the tube colour *without* a tube model: they are built from the harmonic content of the
already-saturated signal, so whatever the tube did is in the spectrum the analyser reads. The module's own optional **CHARACTER** saturation is
a mild table-domain tanh that is re-band-limited (HIGH/ULTRA only); it is not an attempt to reproduce the Fusion's tube.

### 2.5 Things a design might be tempted to assume — and must not

* That the composite is periodic. With DETUNE on, the output contains sidebands that beat against the main oscillator; the composite has **no
  exact period**. The analyser measures how much of the window is periodic (`periodicity`) and the clones inherit the sidebands *coherently*
  (documented limitation, `docs/ARCHITECTURE.md` §9).
* That there is a pure sub at exactly f/2 phase-locked to the main cycle at all levels/temperatures.
* That the waveform is drift-free. The engine tracks slow pitch and shape changes; it never freezes a table.

## 3. What the design may therefore assume

1. **A monophonic, quasi-periodic audio-rate signal**, 20 Hz … ~5 kHz fundamental, any of saw / triangle / pulse / sine / mixtures, with sub
   (period doubling), optional detune sidebands, optional tube saturation, analog noise and drift.
2. Nothing about levels: everything is normalised at the input and the module is level-transparent for VOICES = 1.
3. Nothing about *how* detune is made: the cloning engine is a spectral machine that works on the signal it receives.
4. The **clones are independent oscillators in the perceptual sense only** (independent detune, phase, drift and harmonic divergence), **not
   electrically identical to additional hardware oscillators**. This scientific limitation is stated in the module's info menu and in the
   README.

## 4. Evidence produced by this project ("MEASURED")

All of the following were measured on **synthetic** signals from the hypothesis model, with a bank of genuinely independent oscillators as
the reference (see `research/`). None of it validates the model against a real Fusion.

| What | Where |
|---|---|
| Architecture comparison (candidates A–H prototyped, I folded into A/B/J, K = order tracking inside J) on perceptual proxies (cluster spread, inter-harmonic cleanliness, comb ripple, envelope statistics, recurrence, level law) | `research/results/compare_*.txt`, `docs/ARCHITECTURE.md` §4 |
| Transient handling (attacks, steps, gates) | `research/results/transient.txt` |
| Texture / voice-count scaling | `research/results/texture.txt` |
| FFT-size / window-length study, low-frequency (20 Hz) operation, bloom-in latency | `research/results/fft_study.txt`, `docs/ARCHITECTURE.md` §7, `docs/BENCHMARKS.md` |
| Frequency-tracker accuracy on steady, gliding and vibrato input | `tests/test_analyzer.cpp`, `tests/test_dynamics.cpp` |
| Lock robustness over waveform × pitch × quality | `tests/test_lock.cpp` |

## 5. DSP literature that shaped the architecture study

These are background references; they were **not** re-fetched in the sandbox (network policy), so they are cited from prior knowledge.

* **Phase vocoder and its artefacts.** Flanagan & Golden (1966); Portnoff (1976); Dolson (1986); Puckette, *Phase-locked vocoder* (WASPAA
  1995); Laroche & Dolson, *Improved phase vocoder time-scale modification of audio* (IEEE Trans. Speech Audio Process., 1999) and *New
  phase-vocoder techniques for pitch-shifting, harmonizing and other exotic effects* (WASPAA 1999, peak detection + peak shifting, 50 %/75 %
  overlap); Röbel, *A new approach to transient processing in the phase vocoder* (DAFx-03). → candidates A, B, H.
* **Sinusoidal modelling / partial tracking.** McAulay & Quatieri (IEEE Trans. ASSP, 1986); Serra & Smith, *Spectral modeling synthesis*
  (Computer Music J., 1990). → candidates C, D.
* **Time-domain pitch/period manipulation.** Moulines & Charpentier, *Pitch-synchronous waveform processing techniques* (Speech Commun.,
  1990) — PSOLA; WSOLA (Verhelst & Roelands, 1993). → candidates F, G.
* **Frequency shifting.** Bode / Moog analog shifters; digital SSB with Hilbert pairs — Niemitalo's polyphase IIR half-band Hilbert
  transformer (2003) is used in the optional Fusion layer (`src/dsp/Filters.hpp`, verified to ≤ 0.71° phase error and ~50 dB image rejection
  in `tests/test_hilbert.cpp`). → candidate E and the HZ shift model.
* **Pitch detection.** de Cheveigné & Kawahara, *YIN* (JASA 2002) — the basis of the multi-lane `PitchTracker`.
* **Order tracking.** Fyfe & Munck, *Analysis of computed order tracking* (Mech. Syst. Signal Process., 1997) — resampling to a uniform *angle*
  grid so that a modulated periodic signal becomes exactly periodic. This is the core idea of the chosen architecture (candidate J).
* **Windows and leakage.** Harris, *On the use of windows for harmonic analysis with the DFT* (Proc. IEEE, 1978). The Hann window has exact
  zeros at ±2 bins, which is why harmonics of an M-period window sit cleanly on bins M·j.
* **Tracking filters.** Kalman (1960); Kalata, *The tracking index* (IEEE Trans. AES, 1984) — the two-state frequency/slope filter.
* **Stochastic drift.** Uhlenbeck & Ornstein (1930) — the bounded, slow, independent per-voice drift.
* **Analog delay-line pitch shifting.** The generic BBD two-line pitch-shifter architecture (ramp-modulated clocks, out of phase) and its
  distinction from a frequency shifter, as summarised in [10].

## 6. Sources actually consulted

Secondary sources reached through search-engine summaries (the manufacturer's own pages were blocked, see the box at the top):

1. Erica Synths, *Fusion VCO2* product text (as quoted by [2]–[6], [8]) — ericasynths.lv/shop/eurorack-modules/by-series/fusion-series/fusion-vco2/ and
   the manual `ericasynths.lv/media/Fusion_VCO2_manual_web.pdf` (**not fetched**; only its title/link appeared in results).
2. Clockface Modular — *Erica Synths Fusion VCO V2*: clockfacemodular.com/en/products/erica-synths-fusion-vco2
3. Perfect Circuit — *Fusion VCO V2 Tube Oscillator*: perfectcircuit.com/erica-synths-fusion-vco-v2.html
4. Synth Anatomy — *Erica Synths new Fusion VCO 2*: synthanatomy.com/2019/05/erica-synths-fusion-vco-2.html
5. ModularGrid — *Erica Synths Fusion VCO2*: modulargrid.net/e/erica-synths-fusion-vco2
6. Signal Sounds — *Fusion VCO 2 Eurorack valve oscillator*: signalsounds.com/erica-synths-fusion-vco-2-eurorack-valve-oscillator-module
7. gearnews.com — *Erica Synths Fusion VCO V2 is dark and full of terrors*.
8. Review text describing the detune as "wide and glorious … thick and chorus-like".
9. Mod Wiggler — *Emulating Erica Synths Fusion VCO2's De-tune, possible?* (modwiggler.com/forum/viewtopic.php?t=256876) and *Erica Synths Fusion VCO V2*
   (viewtopic.php?t=225572). Summaries only.
10. Search summary of analog BBD pitch-shifter architecture (two BBDs, ramp waveforms out of phase; Doppler effect from clock-rate modulation;
    frequency shifter vs pitch shifter are different effects).

## 7. Open questions that only a real measurement can close

The reference-analysis tools in `tools/` and the protocol in `docs/REFERENCE_PROTOCOL.md` exist to answer these with a recording of a real
module. Until then they are UNKNOWN and the design is independent of them:

1. Multiplicative (Doppler) or additive (SSB) detune? (sideband spacing vs harmonic number)
2. Detune depth range in cents and LFO rate range as a function of the knob; LFO waveform.
3. Whether the two BBD lines are shifted in opposite directions (symmetric cluster) or only one side is used.
4. Level and colour of the tube stage as a function of TUBE CRUNCH (harmonic distortion spectrum).
5. AS3340 waveform imperfections at 20 Hz … 5 kHz and the tracking drift over temperature.
6. How the SUB level interacts with the detune section (is the sub also passed through the BBDs?).
