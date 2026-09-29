# Reference-recording protocol (for someone who owns a real Fusion VCO2)

The project was developed **without access to hardware or recordings**, so several properties of the Fusion VCO2 are UNKNOWN
(`docs/RESEARCH.md` §7). This protocol turns a handful of recordings into answers, using the scripts in `tools/`. Nothing here is required
to *use* the module; it exists so the model and the module's defaults can be corrected against the real thing.

## 0. Setup

* Record the module's **OUT** (or TRI/PLS/SUB as noted) into a DC-coupled interface at **≥ 48 kHz / 24 bit**, mono, levels well below clipping.
  Write the input gain and the interface's nominal level in the file name; absolute level is not needed.
* Patch a **stable pitch**: 1 V/oct from a precise source (sequencer or MIDI-CV), no modulation, no sync, external audio in unplugged,
  TUBE CRUNCH at minimum, SUB at minimum, DETUNE at minimum unless the step says otherwise.
* Let the module warm up ≥ 20 minutes.
* Record **at least 20 s** per file (the detune LFO can be as slow as 0.1 Hz); **60 s** for step 2.
* All scripts need `numpy`; plots need `matplotlib`.

## 1. Clean oscillator (tests: "waveform imperfections", "level", "noise")

For each of C1 (32.7 Hz), C2 (65.4 Hz), C3, C4, C5, C6, C7 record WAVE = saw, WAVE = triangle, WAVE = pulse (PWM at 50 %), 20 s each.

```
python3 tools/analyze_reference.py rec_c2_saw.wav --f0 65.406 --out c2_saw.json --plot c2_saw.png
```

It reports: measured f0 (and drift in cents over the file), harmonic amplitudes/phases up to Nyquist, even/odd ratio, THD+N, noise floor,
DC, and the presence of **sub-harmonic lines at f0/2** (period doubling). Compare against ideal saw (1/n), triangle (1/n², odd only) and
pulse (sinc) to see the AS3340 shape errors.

## 2. DETUNE: multiplicative or additive? (the key question)

Record **C2 (65.4 Hz) saw**, DETUNE at 25 %, 50 %, 75 %, 100 %, 60 s each, everything else at minimum.

```
python3 tools/fusion_detune_probe.py rec_c2_detune50.wav --f0 65.406
```

The probe takes a long FFT, finds the sideband cluster around each harmonic n = 1 … 40 and fits `offset(n) = a + b·n`:

| Result | Meaning | FusionClone setting |
|---|---|---|
| `b / (a + b) > 0.7` | offsets grow with n: **multiplicative ratio (Doppler / variable delay)** — H-DOPPLER | *Shift model → RATIO* (default) |
| `|b| < 0.1·a` | same offset in Hz for every harmonic: **additive frequency shift (SSB)** — H-SSB | *Shift model → HZ* |
| in between | mixture (e.g. delay + shifter) | pick RATIO or HZ by the larger effect, note it |

It also prints: the cluster spread in cents at each harmonic, the LFO rate (from the beating envelope of harmonic 1 and from the sideband
spacing), whether the cluster is symmetric (±Δ around the main line, "two detuned VCOs") or one-sided, and the sideband levels relative to
the carrier at each DETUNE setting. These four numbers set the module's **Fusion layer** parameters (`src/dsp/Engine.hpp`, `controlTick`).

The script has a built-in self test that runs the classifier on synthetic H-DOPPLER and H-SSB signals: `python3 tools/fusion_detune_probe.py --selftest`.

## 3. SUB and COLOR

C2 saw with SUB at 100 %, COLOR off and on, 20 s each. `analyze_reference.py` reports the level of the f0/2 line and the low-pass corner of
the sub (from the line ratio f0/2 : 3f0/2 : 5f0/2 …). Check whether the composite repeats every two cycles even at low SUB levels (it does if
the f0/2 line stays sharp).

## 4. TUBE CRUNCH

C2 saw, TUBE CRUNCH at 0, 25, 50, 75, 100 %, 20 s each. The script reports THD and the even/odd harmonic ratio versus setting; a plot of the
measured transfer curve (input/output) can be reconstructed with the sine-sweep option (`--sweep`) if a sine is recorded through the external
input at several levels.

## 5. Check the module against the recording

Feed a recording (or the live module) through FusionClone at VOICES = 2 … 16 and compare with a *real* multi-oscillator reference (record
2 … 4 real VCOs mixed, if available):

```
python3 tools/analyze_reference.py clones_16.wav --f0 65.406 --compare reference_4vco.wav
```

reports cluster spread, inter-harmonic floor and envelope statistics for both files side by side (the same proxies used in `research/`).

## 6. What to send back

The JSON files from steps 1–4 (a few kB each) are enough to update `docs/RESEARCH.md` §2 from INFERRED/UNKNOWN to MEASURED and to set better
defaults; the WAV files are not needed.
