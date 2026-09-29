// FusionClone DSP core — virtual voice: personality (deterministic random tolerance set), period table, drift.
#pragma once
#include "Common.hpp"
#include "FFT.hpp"
#include "Filters.hpp"
#include "Params.hpp"
#include "SincInterp.hpp"

namespace fc {

/** Van der Corput / golden-ratio style low-discrepancy value in [0,1) for voice index i, so that any prefix of voices is
    well spread (changing VOICES never moves the existing voices). */
inline double goldenSeq(uint32_t seed, int i) {
	const double phi = 0.6180339887498949;
	double u0 = (hash32(seed ^ 0xC0FFEEu) >> 8) * (1.0 / 16777216.0);
	return frac(u0 + phi * (double) (i + 1));
}

/** Everything that makes one virtual oscillator different from the source. All values are *unit* quantities; the user
    controls (SPREAD, HARMONIC, ...) scale them. Generated deterministically from (seed, voice index). */
struct VoicePersonality {
	double centsUnit = 0.0;   // detune in units of sigma (|value| >= ~0.25)
	float pan = 0.f;          // -1..1
	float levelDb = 0.f;      // static level tolerance, unit sigma (dB)
	float tiltDb = 0.f;       // spectral tilt across 4 octaves, unit sigma (dB)
	float bellAmpDb[3];       // smooth log-frequency ripple
	float bellPeriodOct[3];
	float bellPhase[3];
	float oddEvenDb = 0.f;
	float hfCornerRel = 1.f;  // relative corner of the HF roll-off
	float dispAmp[2];         // phase dispersion (radians at PHASE = 1)
	float dispPeriodOct[2];
	float dispPhase[2];
	float drive = 0.f;        // 0..1, saturation strength for CHARACTER
	float bias = 0.f;         // -1..1, saturation asymmetry
	float noiseGain = 0.f;    // reserved (unused): still drawn so that the sequence of personality values stays stable
	float startPhase = 0.f;   // 0..1, random initial phase (PHASE = 1)
	float lfoPhase = 0.f;     // Fusion detune-cluster LFO phase
	float lfoRate = 1.f;      // relative LFO rate tolerance
	uint32_t rngSeed = 1;

	VoicePersonality() {
		for (int i = 0; i < 3; i++)
			bellAmpDb[i] = bellPeriodOct[i] = bellPhase[i] = 0.f;
		for (int i = 0; i < 2; i++)
			dispAmp[i] = dispPeriodOct[i] = dispPhase[i] = 0.f;
	}

	void generate(uint32_t seed, int voiceIndex) {
		Rng r(hashCombine(seed, (uint32_t) voiceIndex * 2654435761u + 17u));
		// detune: low-discrepancy quantile of a Gaussian, pushed away from 0 so no clone sits on top of the original
		double p = goldenSeq(seed, voiceIndex);
		p = p < 0.5 ? p * 0.8 : 0.2 + p * 0.8; // skip the central 20% of the probability mass -> |z| >= 0.25
		p = clampT(p + 0.02 * (r.uniform() - 0.5), 0.02, 0.98);
		centsUnit = normalQuantile(p);
		// pan: separate low-discrepancy sequence, alternating sides
		double q = goldenSeq(seed ^ 0xBEEF1234u, voiceIndex);
		pan = (float) (q * 2.0 - 1.0);
		levelDb = r.gauss();
		tiltDb = r.gauss();
		for (int i = 0; i < 3; i++) {
			bellAmpDb[i] = r.gauss() / (1.f + i);
			bellPeriodOct[i] = 2.2f + 2.5f * r.uniform() + 1.6f * i;
			bellPhase[i] = (float) kTwoPi * r.uniform();
		}
		oddEvenDb = r.gauss();
		hfCornerRel = std::exp2(0.25f * r.gauss());
		for (int i = 0; i < 2; i++) {
			dispAmp[i] = r.gauss();
			dispPeriodOct[i] = 1.8f + 2.4f * r.uniform() + 2.f * i;
			dispPhase[i] = (float) kTwoPi * r.uniform();
		}
		drive = r.uniform();
		bias = r.bipolar();
		noiseGain = 0.5f + r.uniform();
		startPhase = r.uniform();
		lfoPhase = r.uniform();
		lfoRate = 1.f + 0.08f * r.gauss();
		rngSeed = r.nextU32();
	}
};

/** Per-voice divergence curves sampled on a coarse log2(harmonic) grid; the table builder interpolates. */
struct DivergenceCurve {
	static const int kGrid = 96;
	static const int kMaxOct = 13; // covers harmonics up to 8192
	float gain[kGrid + 1];        // linear
	float phase[kGrid + 1];       // radians
	float oddEven;                // linear factor applied as (1 +- oddEven) to odd/even harmonics
	DivergenceCurve() : oddEven(0.f) {
		for (int i = 0; i <= kGrid; i++) {
			gain[i] = 1.f;
			phase[i] = 0.f;
		}
	}

	/** amountH: HARMONIC 0..1, amountP: PHASE 0..1, tiltDrift: extra time-varying tilt in dB (from DRIFT). fundHz: absolute
	    frequency of harmonic 1 for the voice (used for the absolute-frequency HF roll-off). */
	void build(const VoicePersonality& vp, float amountH, float amountP, float tiltDrift, double fundHz) {
		const float hs = amountH;
		for (int g = 0; g <= kGrid; g++) {
			float x = (float) g * (float) kMaxOct / kGrid; // log2 harmonic index
			// gain in dB: broadband ripple + tilt (about harmonic ~ 2^4), all scaled by HARMONIC
			float db = vp.tiltDb * 0.45f * (x - 4.f) / 4.f + tiltDrift * (x - 4.f) / 4.f;
			for (int i = 0; i < 3; i++)
				db += 0.35f * vp.bellAmpDb[i] * std::cos((float) kTwoPi * x / vp.bellPeriodOct[i] + vp.bellPhase[i]);
			// absolute-frequency HF roll-off (component tolerance of the output stage): -dB grows as (f/fc)^3 beyond ~10 kHz
			double f = fundHz * std::exp2((double) x);
			double fc = 16000.0 * vp.hfCornerRel;
			double rr = f / fc;
			db += (float) (-1.2 * rr * rr * rr * (vp.hfCornerRel - 0.6));
			gain[g] = std::pow(10.f, hs * db / 20.f);
			// phase dispersion (small, smooth): keeps the waveform shape, individualises the phase response
			float ph = 0.f;
			for (int i = 0; i < 2; i++)
				ph += vp.dispAmp[i] * 0.12f * std::sin((float) kTwoPi * x / vp.dispPeriodOct[i] + vp.dispPhase[i]);
			phase[g] = amountP * ph;
		}
		oddEven = hs * (std::pow(10.f, 0.12f * vp.oddEvenDb / 20.f) - 1.f);
	}
};

/** Precomputed log2 table for harmonic indices (1..8192). */
struct Log2Lut {
	std::vector<float> v;
	Log2Lut() {
		v.assign(8193, 0.f);
		for (int i = 1; i <= 8192; i++)
			v[i] = (float) std::log2((double) i);
	}
};
inline const Log2Lut& log2Lut() {
	static const Log2Lut l;
	return l;
}

/** A period table: one cycle of the (band-limited) waveform, with a guard so sinc reads never wrap. */
struct PeriodTable {
	static const int G = 20;
	int Nc = 0;
	AlignedBuffer<float> d; // maxNc + 2G
	float rms = 0.f;

	void alloc(int maxNc) { d.alloc((size_t) maxNc + 2 * G); }

	void finalize(int n) {
		Nc = n;
		for (int i = 0; i < G; i++) {
			d[G - 1 - i] = d[G + n - 1 - i];
			d[G + n + i] = d[G + i];
		}
	}
	void copyFrom(const PeriodTable& o) {
		Nc = o.Nc;
		rms = o.rms;
		std::memcpy(d.data(), o.d.data(), sizeof(float) * (size_t) (o.Nc + 2 * G));
	}
	template <int TAPS>
	inline float read(double phase) const {
		double p = (phase - std::floor(phase)) * Nc;
		int i0 = (int) p;
		float fr = (float) (p - i0);
		const SincKernel<TAPS>& K = sharedSincKernel<TAPS>();
		return K.read(d.data() + G + i0, fr);
	}
};

/** Ornstein-Uhlenbeck process (exact discretisation), bounded to +-3 sigma. The state is kept in units of sigma, so changing sigma (the DRIFT
    control) rescales the output smoothly instead of clamping the old value. */
struct OuProcess {
	float u = 0.f; // unit-variance state
	float x = 0.f; // last output (= u * sigma), read by the table builder
	Rng rng;
	void seed(uint32_t s) {
		rng.reseed(s);
		u = x = 0.f;
	}
	/** advance by dt seconds with time constant tau and stationary std sigma */
	float step(float dt, float tau, float sigma) {
		float a = std::exp(-dt / std::max(tau, 1e-3f));
		float b = std::sqrt(std::max(0.f, 1.f - a * a));
		u = a * u + b * rng.gauss();
		u = clampT(u, -3.f, 3.f);
		x = u * sigma;
		return x;
	}
};

/** Recursive complex oscillator (cos, sin) — avoids per-sample trigonometry for the SSB carriers. */
struct Rotor {
	float c = 1.f, s = 0.f, dc = 1.f, ds = 0.f;
	int n = 0;
	void setFreq(double cyclesPerSample) {
		dc = (float) std::cos(kTwoPi * cyclesPerSample);
		ds = (float) std::sin(kTwoPi * cyclesPerSample);
	}
	void setPhase(double cycles) {
		c = (float) std::cos(kTwoPi * cycles);
		s = (float) std::sin(kTwoPi * cycles);
	}
	inline void step() {
		float nc = c * dc - s * ds;
		s = c * ds + s * dc;
		c = nc;
		if (++n >= 2048) {
			n = 0;
			float r = 1.f / std::sqrt(c * c + s * s);
			c *= r;
			s *= r;
		}
	}
};

/** Runtime state of one clone voice. */
struct VoiceState {
	VoicePersonality vp;
	PeriodTable tab[2];      // double buffer: tab[cur] is playing, tab[cur^1] is the crossfade target
	int cur = 0;
	float alpha = 1.f;       // crossfade tab[cur] -> tab[cur^1]
	float alphaInc = 0.f;
	bool haveTable = false;
	double phase = 0.0;      // table cycles
	double ratio = 1.0;      // current frequency ratio (static detune * drift)
	double ratioTarget = 1.0;
	double ratioInc = 0.0;
	float gain = 0.f;        // smoothed per-voice on/off gain (voice count changes); second stage of a two-stage smoother
	float gainMid = 0.f;     // first stage
	float levelLin = 1.f;    // static level tolerance x drift, smoothed per sample towards levelT (set by the control tick)
	float levelT = 1.f;
	float panL = 1.f, panR = 1.f; // smoothed per sample towards panLT / panRT
	float panLT = 1.f, panRT = 1.f;
	OuProcess drift, levelDrift, tiltDrift;
	float driftCents = 0.f;
	float jitterCents = 0.f;
	Rng noiseRng;
	DivergenceCurve curve;
	// FUSION detune-cluster layer (per-voice, independent LFO)
	double lfoPhase = 0.0;    // cycles
	double lfoInc = 0.0;      // cycles per sample
	double xPhase[2];         // table phases of the two extra oscillators (RATIO mode)
	Rotor carrier[2];         // carriers of the two SSB lines (HZ mode)
	float shiftFrac = 0.f;    // ratio deviation of the extra oscillators
	HilbertPair hil;
	VoiceState() {
		xPhase[0] = xPhase[1] = 0.0;
	}
};

} // namespace fc
