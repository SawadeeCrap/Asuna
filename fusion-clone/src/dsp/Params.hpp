// FusionClone DSP core — public parameter set and status structures.
#pragma once
#include "Common.hpp"

namespace fc {

enum Quality { QUALITY_ECO = 0, QUALITY_BALANCED = 1, QUALITY_HIGH = 2, QUALITY_ULTRA = 3 };
enum Algorithm { ALGO_CLASSIC = 0, ALGO_FUSION = 1 };
enum ShiftMode { SHIFT_RATIO = 0, SHIFT_HZ = 1 };

/** All user-facing parameters. Normalised 0..1 unless noted. The engine smooths everything that could zipper. */
struct EngineParams {
	// --- primary ---
	int voices = 8;            // 1..16 total voices: 1 original + (voices-1) clones
	float spread = 0.35f;      // detune amount
	float drift = 0.25f;       // slow independent pitch/timbre drift
	float character = 0.30f;   // analog individuality: saturation asymmetry, noise, level tolerance
	float phase = 0.60f;       // phase divergence between voices
	float harmonic = 0.30f;    // harmonic divergence (smooth spectral tilt/bells, odd/even, HF rolloff)
	float width = 0.0f;        // stereo micro-positioning
	float mix = 1.0f;          // dry/wet
	float outputDb = 0.0f;     // output trim
	// --- expert ---
	float detuneRangeCents = 20.f; // detune at spread = 1 (one-sigma of the tolerance distribution ~ range/2.4)
	float spreadCurve = 1.6f;      // spread^curve
	float driftRate = 0.5f;        // 0 = very slow (30 s), 1 = faster (2 s)
	float driftCorrelation = 0.25f; // shared "temperature" component between voices
	float fusionShift = 0.0f;      // FUSION algorithm: amount of the per-voice detune-cluster layer
	int shiftMode = SHIFT_RATIO;
	int algorithm = ALGO_CLASSIC;
	int quality = QUALITY_BALANCED;
	float summing = 0.35f;         // analog-style summing saturation amount
	float levelLaw = 0.075f;       // per-voice gain = N^-(0.5-levelLaw): 0 = equal power, 0.5 = unity-sum
	bool originalOnly = false;     // debug/A-B: bypass all clones
	uint32_t seed = 0x5EED1234u;
};

struct EngineStatus {
	bool locked = false;
	bool acquiring = false;
	float coherence = 0.f;
	float periodicity = 0.f;
	float lockWeight = 0.f;
	double unitFreqHz = 0.0;   // repeating-unit frequency (half the oscillator frequency when a sub-oscillator is present)
	int activeVoices = 1;
	float inputLevelDb = -120.f;
	float latencySamples = 0.f;
	float bloomMs = 0.f;       // time until the clones are fully present after an onset at the current pitch
	int quality = 1;
	int Nc = 0;
	int J = 0;
	// lightweight spectrum snapshot for the GUI (dB of the first kSpecBins table harmonics, seqlock protected)
	static const int kSpecBins = 48;
	float spectrumDb[kSpecBins];
	float voiceCents[kMaxVoices];
	uint32_t specSeq = 0;
	EngineStatus() {
		for (int i = 0; i < kSpecBins; i++)
			spectrumDb[i] = -120.f;
		for (int i = 0; i < kMaxVoices; i++)
			voiceCents[i] = 0.f;
	}
};

} // namespace fc
