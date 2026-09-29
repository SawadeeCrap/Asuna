// Synthetic "Fusion VCO2-like" source model and independent-oscillator reference bank.
//
// IMPORTANT: this is a *hypothesis model* built from the public product description (see docs/RESEARCH.md):
//   * AS3340-style core: saw / triangle / pulse (harmonic-exact, band-limited additive tables)
//   * transistor sub oscillator one octave down (square from a divide-by-two), COLOR switch = low-pass on the sub
//   * DETUNE = two BBD delay lines feeding a "frequency shifter" mixed back to the main oscillator,
//     modelled here under two competing hypotheses:
//        DetuneModel::Doppler  (time-varying delay, multiplicative pitch ratio)   <- INFERRED for a BBD design
//        DetuneModel::Ssb      (true single-sideband frequency shift, additive Hz)  <- literal reading of the marketing text
//   * TUBE CRUNCH = asymmetric soft saturation after the mix
// None of the numeric parameter ranges below are measured from hardware. They exist so the cloning algorithms can be
// stress-tested against *plausible* Fusion-like spectra (period doubling, sidebands, asymmetry) and against a bank of
// genuinely independent oscillators.
#pragma once
#include "../../src/dsp/FcCommon.hpp"
#include "../../src/dsp/FcFFT.hpp"
#include "../../src/dsp/Filters.hpp"
#include "../../src/dsp/SincInterp.hpp"
#include <functional>

namespace research {

enum DetuneModel { DETUNE_NONE = 0, DETUNE_DOPPLER = 1, DETUNE_SSB = 2 };

struct SourceSpec {
	double fs = 48000.0;
	double seconds = 6.0;
	double f0 = 110.0;
	// main waveform mix
	double wSaw = 1.0, wTri = 0.0, wPulse = 0.0, wSine = 0.0;
	double pulseWidth = 0.5;
	// sub oscillator (-1 oct)
	double sub = 0.0;
	bool colorLpf = false;
	double colorFc = 700.0; // SPECULATIVE
	// DETUNE section
	int detuneModel = DETUNE_NONE;
	double detune = 0.0; // knob 0..1
	double lfoPhase = 0.0;
	double lfoRateJitter = 0.0; // relative, per-instance tolerance
	// TUBE
	double tube = 0.0;
	// imperfections
	double noiseDb = -90.0;
	double driftCents = 0.0; // stationary std-dev of the OU pitch drift
	double driftTau = 6.0;
	double tuneCents = 0.0;
	double startPhase = -1.0; // table phase, <0 = random
	double levelTolDb = 0.0;  // static gain tolerance
	std::vector<double> f0Track; // Hz per sample, overrides f0
	std::vector<float> ampEnv;   // optional VCA envelope
	uint32_t seed = 1;
};

// A periodic table with 32-tap sinc reading (guard-extended).
struct PeriodicTable {
	int N = 0;
	std::vector<float> d; // N + 2*guard
	static const int G = 20;
	void set(const float* tab, int n) {
		N = n;
		d.assign(n + 2 * G, 0.f);
		for (int i = 0; i < n; i++)
			d[G + i] = tab[i];
		for (int i = 0; i < G; i++) {
			d[G - 1 - i] = tab[(n - 1 - i) % n];
			d[G + n + i] = tab[i % n];
		}
	}
	float read(double phase) const {
		double p = (phase - std::floor(phase)) * N;
		int i0 = (int) p;
		float fr = (float) (p - i0);
		const fc::SincKernel<32>& K = fc::sharedSincKernel<32>();
		return K.read(&d[G + i0], fr);
	}
};

inline double tri(double ph) {
	double f = ph - std::floor(ph);
	return 1.0 - 4.0 * std::fabs(f - 0.5);
}

/** Build the (period = 2 main cycles) table for a spec. Returns table and number of harmonics used. */
inline PeriodicTable buildSourceTable(const SourceSpec& s, double maxF0) {
	int M = (int) std::floor(0.92 * s.fs / maxF0); // table harmonics (table fundamental = f0/2)
	M = std::max(M, 8);
	int N = fc::nextPow2((int) std::ceil(2.4 * M));
	N = std::max(64, std::min(N, 65536));
	std::vector<double> a(M + 1, 0.0), b(M + 1, 0.0);
	// main components live at even table harmonics m = 2k
	for (int k = 1; 2 * k <= M; k++) {
		int m = 2 * k;
		if (s.wSaw != 0)
			b[m] += s.wSaw * 2.0 / (fc::kPi * k);
		if (s.wTri != 0 && (k & 1))
			b[m] += s.wTri * 8.0 / (fc::kPi * fc::kPi * k * k) * (((k - 1) / 2) % 2 ? -1.0 : 1.0);
		if (s.wPulse != 0) {
			double w = s.pulseWidth;
			a[m] += s.wPulse * (2.0 / (fc::kPi * k)) * std::sin(2 * fc::kPi * k * w);
			b[m] += s.wPulse * (4.0 / (fc::kPi * k)) * std::pow(std::sin(fc::kPi * k * w), 2.0);
		}
	}
	if (s.wSine != 0 && M >= 2)
		b[2] += s.wSine;
	// sub: square at half rate -> odd table harmonics, phase-aligned so its edges coincide with main reset events
	if (s.sub > 0) {
		for (int m = 1; m <= M; m += 2) {
			double g = s.sub * 4.0 / (fc::kPi * m);
			double fm = m * s.f0 * 0.5;
			double ph = 0.0;
			if (s.colorLpf) {
				double r = fm / s.colorFc;
				g /= std::sqrt(1.0 + r * r);
				ph = -std::atan(r);
			}
			// g*sin(theta + ph) = g cos(ph) sin + g sin(ph) cos
			b[m] += g * std::cos(ph);
			a[m] += g * std::sin(ph);
		}
	}
	fc::RealFFT fft(N);
	std::vector<float> spec(N, 0.f), tab(N, 0.f);
	// x[i] = sum a cos + b sin  <=>  X[m] = (a - j b)/2 with unnormalised inverse
	for (int m = 1; m < N / 2 && m <= M; m++) {
		spec[2 * m] = (float) (a[m] * 0.5);
		spec[2 * m + 1] = (float) (-b[m] * 0.5);
	}
	fft.inverse(spec.data(), tab.data());
	// normalise peak to 0.5 later (after detune/tube we RMS-normalise); here just remove nothing.
	PeriodicTable t;
	t.set(tab.data(), N);
	return t;
}

/** Zero-phase FFT convolution, output aligned with the input (odd-length symmetric kernels). */
inline std::vector<float> fftConvolveSameLocal(const std::vector<float>& x, const std::vector<float>& h) {
	int n = (int) x.size(), m = (int) h.size();
	int L = fc::nextPow2(n + m);
	fc::RealFFT fft(L);
	std::vector<float> a(L, 0.f), b(L, 0.f), A(L), B(L), C(L), c(L);
	std::copy(x.begin(), x.end(), a.begin());
	std::copy(h.begin(), h.end(), b.begin());
	fft.forward(a.data(), A.data());
	fft.forward(b.data(), B.data());
	C[0] = A[0] * B[0];
	C[1] = A[1] * B[1];
	for (int k = 1; k < L / 2; k++) {
		float ar = A[2 * k], ai = A[2 * k + 1], br = B[2 * k], bi = B[2 * k + 1];
		C[2 * k] = ar * br - ai * bi;
		C[2 * k + 1] = ar * bi + ai * br;
	}
	fft.inverse(C.data(), c.data());
	std::vector<float> y(n);
	int off = m / 2;
	for (int i = 0; i < n; i++)
		y[i] = c[i + off] / L;
	return y;
}

inline std::vector<float> renderSource(const SourceSpec& s) {
	const double fs = s.fs;
	const size_t n = (size_t) (s.seconds * fs);
	fc::Rng rng(s.seed);
	double maxF0 = s.f0;
	if (!s.f0Track.empty())
		for (double f : s.f0Track)
			maxF0 = std::max(maxF0, f);
	maxF0 *= fc::centsToRatio(std::fabs(s.tuneCents) + 4 * s.driftCents + 5);
	PeriodicTable tab = buildSourceTable(s, maxF0);

	// --- core oscillator -------------------------------------------------------------------------------------------
	std::vector<float> core(n);
	double ph = s.startPhase >= 0 ? s.startPhase : rng.uniform();
	double drift = 0.0;
	const double dtau = std::max(0.05, s.driftTau);
	const double dsig = s.driftCents * std::sqrt(2.0 / (dtau * fs));
	double tune = fc::centsToRatio(s.tuneCents);
	for (size_t i = 0; i < n; i++) {
		double f = s.f0Track.empty() ? s.f0 : s.f0Track[std::min(i, s.f0Track.size() - 1)];
		if (s.driftCents > 0) {
			drift += -drift / (dtau * fs) + dsig * rng.gauss();
			drift = fc::clampT(drift, -3 * s.driftCents, 3 * s.driftCents);
		}
		double r = tune * fc::centsToRatio(drift);
		ph += f * r / (2.0 * fs);
		ph -= std::floor(ph);
		core[i] = tab.read(ph);
	}
	// remove DC
	{
		double m = 0;
		for (float v : core)
			m += v;
		m /= (double) n;
		for (float& v : core)
			v -= (float) m;
	}
	// normalise the core so its RMS is 0.2
	{
		double e = 0;
		for (float v : core)
			e += (double) v * v;
		double rms = std::sqrt(e / n) + 1e-12;
		float g = (float) (0.2 / rms);
		for (float& v : core)
			v *= g;
	}

	std::vector<float> y = core;

	// --- DETUNE section -----------------------------------------------------------------------------------------------
	if (s.detuneModel != DETUNE_NONE && s.detune > 0) {
		double fl = (0.08 + s.detune * 3.5) * (1.0 + s.lfoRateJitter); // LFO rate rises with the knob (per product description)
		double wet = std::min(1.0, 0.35 + 0.65 * s.detune);
		std::vector<float> A(n, 0.f), B(n, 0.f);
		if (s.detuneModel == DETUNE_DOPPLER) {
			fc::MirrorRing ring;
			ring.alloc(1 << 15);
			const double Dpp = 0.005 * fs; // 5 ms peak-to-peak swing (SPECULATIVE)
			const double D0 = 48.0;
			double lp = s.lfoPhase;
			// prime ring
			for (size_t i = 0; i < n; i++) {
				ring.push(core[i]);
				lp += fl / fs;
				double da = D0 + 0.5 * Dpp * (1.0 + tri(lp));
				double db = D0 + 0.5 * Dpp * (1.0 + tri(lp + 0.5));
				uint64_t newest = ring.count() - 1;
				A[i] = ring.readSinc<32>((double) newest - da);
				B[i] = ring.readSinc<32>((double) newest - db);
			}
		} else {
			// Ideal SSB frequency shifter: analytic signal via a long windowed FIR Hilbert transformer (offline).
			const int H = 4095;
			std::vector<float> hk(H, 0.f);
			for (int k = 0; k < H; k++) {
				int m = k - H / 2;
				if (m & 1) {
					double w = fc::kaiser((double) k, (double) (H - 1), 8.0);
					hk[k] = (float) (2.0 / (fc::kPi * m) * w);
				}
			}
			std::vector<float> hil(n, 0.f);
			// direct convolution is too slow for long signals at 4095 taps; use FFT convolution.
			int L = fc::nextPow2((int) n + H);
			fc::RealFFT f2(L);
			std::vector<float> xa(L, 0.f), ha(L, 0.f), Xa(L), Ha(L), Ya(L), ya(L);
			std::copy(core.begin(), core.end(), xa.begin());
			std::copy(hk.begin(), hk.end(), ha.begin());
			f2.forward(xa.data(), Xa.data());
			f2.forward(ha.data(), Ha.data());
			Ya[0] = Xa[0] * Ha[0];
			Ya[1] = Xa[1] * Ha[1];
			for (int k = 1; k < L / 2; k++) {
				float xr = Xa[2 * k], xi = Xa[2 * k + 1], hr = Ha[2 * k], hi = Ha[2 * k + 1];
				Ya[2 * k] = xr * hr - xi * hi;
				Ya[2 * k + 1] = xr * hi + xi * hr;
			}
			f2.inverse(Ya.data(), ya.data());
			for (size_t i = 0; i < n; i++) {
				size_t j = i + H / 2;
				hil[i] = ya[j] / L;
			}
			// delay `core` by H/2 to align with the Hilbert FIR (linear phase)
			double delta = s.detune * 5.0 * (1.0 + s.lfoRateJitter); // Hz, SPECULATIVE
			double lp = s.lfoPhase;
			for (size_t i = 0; i < n; i++) {
				double t = (double) i / fs;
				double c = std::cos(fc::kTwoPi * (delta * t + lp)), sn = std::sin(fc::kTwoPi * (delta * t + lp));
				float xr = core[i];
				float xh = hil[i];
				// upper sideband: Re{(x + j xh)(c + j sn)} ; lower: Re{(x + j xh)(c - j sn)}
				A[i] = (float) (xr * c - xh * sn);
				B[i] = (float) (xr * c + xh * sn);
			}
		}
		// BBD reconstruction low-pass (SPECULATIVE 7 kHz, 4th order) on the wet lines
		fc::Biquad l1, l2, l3, l4;
		l1.setLowpass(7000, fs, 0.54);
		l2.setLowpass(7000, fs, 1.31);
		l3.setLowpass(7000, fs, 0.54);
		l4.setLowpass(7000, fs, 1.31);
		for (size_t i = 0; i < n; i++) {
			float a = l2.process(l1.process(A[i]));
			float b = l4.process(l3.process(B[i]));
			y[i] = core[i] + (float) wet * 0.6f * (a + b);
		}
	}

	// --- TUBE crunch -------------------------------------------------------------------------------------------------------
	// The analogue stage is continuous-time, and the audio interface band-limits before Rack sees the signal, so the static waveshaper must
	// not alias: run it at 8x and low-pass/decimate (offline polyphase, 96-tap Kaiser).
	if (s.tube > 0) {
		const int OS = 8, TAPS = 96;
		std::vector<float> h = fc::designLowpassFir(TAPS * OS / 2 * 2 + 1, 0.46 / OS, 9.0);
		// upsample by zero-stuffing + filtering (gain OS)
		std::vector<float> up(n * OS, 0.f);
		for (size_t i = 0; i < n; i++)
			up[i * OS] = y[i] * (float) OS;
		up = fftConvolveSameLocal(up, h);
		double drive = 1.0 + 6.0 * s.tube;
		double bias = 0.25 * s.tube;
		double e0 = 0, e1 = 0;
		for (float v : y)
			e0 += (double) v * v;
		double t0 = std::tanh(drive * bias);
		for (size_t i = 0; i < up.size(); i++)
			up[i] = (float) (std::tanh(drive * (up[i] * 5.0 + bias)) - t0);
		up = fftConvolveSameLocal(up, h);
		for (size_t i = 0; i < n; i++) {
			y[i] = up[i * OS];
			e1 += (double) y[i] * y[i];
		}
		float g = (float) std::sqrt(e0 / (e1 + 1e-12));
		for (float& v : y)
			v *= g;
	}

	// --- noise / tolerance / envelope ---------------------------------------------------------------------------------
	if (s.noiseDb > -200) {
		float na = (float) std::pow(10.0, s.noiseDb / 20.0);
		for (float& v : y)
			v += na * rng.gauss() * 0.7071f;
	}
	if (s.levelTolDb != 0) {
		float g = (float) std::pow(10.0, s.levelTolDb / 20.0);
		for (float& v : y)
			v *= g;
	}
	if (!s.ampEnv.empty())
		for (size_t i = 0; i < n; i++)
			y[i] *= s.ampEnv[std::min(i, s.ampEnv.size() - 1)];
	return y;
}

/** Build an independent-oscillator bank: instance 0 is the "real" source; instances 1..N-1 get their own tuning
    offsets, phases, drift processes, detune-LFO phase/rate and tiny level tolerance. Returns the normalised sum. */
inline std::vector<float> renderBank(const SourceSpec& base, int N, const std::vector<double>& centsOffsets, double driftCents,
                                     uint32_t seed) {
	std::vector<float> sum;
	for (int i = 0; i < N; i++) {
		SourceSpec s = base;
		s.seed = fc::hashCombine(seed, (uint32_t) i + 1u);
		fc::Rng r(s.seed ^ 0x77u);
		if (i > 0) {
			s.tuneCents = base.tuneCents + (i - 1 < (int) centsOffsets.size() ? centsOffsets[i - 1] : 0.0);
			s.startPhase = -1.0; // random
			s.driftCents = driftCents;
			s.lfoPhase = r.uniform();
			s.lfoRateJitter = 0.08 * r.gauss();
			s.levelTolDb = 0.3 * r.gauss();
		}
		std::vector<float> y = renderSource(s);
		if (sum.empty())
			sum.assign(y.size(), 0.f);
		for (size_t k = 0; k < y.size(); k++)
			sum[k] += y[k];
	}
	return sum;
}

} // namespace research
