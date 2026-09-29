// Comparison candidates for the architecture study (offline implementations, full-signal in / full-signal out).
//
//   A  plain phase vocoder (Bernsee/smb-style: instantaneous frequency per bin, bin re-mapping, phase accumulation)
//   B  phase vocoder with identity phase locking around spectral peaks (Laroche & Dolson 1999, "shifted peaks")
//   C  sinusoidal modelling: STFT peak picking + McAulay-Quatieri style partial tracking + additive resynthesis
//   D  harmonic additive/heterodyne resynthesis with an *oracle* f0 (upper bound for STFT-based partial methods)
//   E  additive frequency shift (Bode-style single sideband via Hilbert FIR) — matched at the fundamental
//   F1 rotating two-tap delay pitch shifter (the "chorus / classic pitch shifter" baseline)
//   F2 period-synchronous resampler: time-domain read head + jump by whole periods (period from the YIN tracker)
//   G  PSOLA (pitch-synchronous overlap-add), same period source as F2
//   H  2-band multi-resolution identity-locked phase vocoder (long window for lows, short for highs)
//   J  the production engine (harmonic-locked oscillator bank)
//
// Every candidate exposes renderClone(x, ratio, ctx): one clone of the whole input at pitch ratio `ratio`. The driver sums the
// clones with the (undelayed) original. Candidates with algorithmic latency report it so the driver can time-align.
#pragma once
#include "../../src/dsp/Engine.hpp"
#include "../../src/dsp/PitchTracker.hpp"
#include "../common/fusion_source.hpp"
#include "../common/metrics.hpp"
#include <complex>
#include <functional>

namespace research {

struct CloneCtx {
	double fs = 48000;
	double f0 = 110;         // oracle fundamental (Hz) of the *repeating unit's fundamental* — used only by D/F2/G
	double unitPeriod = 0;   // repeating-unit period in samples (2T with a sub oscillator); 0 = fs/f0
	uint32_t seed = 1;
	bool randomPhase = true; // apply an independent random time offset to the clone (steady-state tests)
};

struct Candidate {
	virtual ~Candidate() {}
	virtual const char* name() const = 0;
	virtual std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) = 0;
	virtual int latency(const CloneCtx&) const { return 0; }
	virtual bool supportsVarPitch() const { return false; }
};

// ------------------------------------------------------------------------------------------------------------------------------
// small helpers
// ------------------------------------------------------------------------------------------------------------------------------
inline double wrapPi(double x) {
	return x - fc::kTwoPi * std::floor(x / fc::kTwoPi + 0.5);
}

/** Fractional delay of a whole buffer by D samples (D may be non-integer, >= 0) using 32-tap sinc. */
inline std::vector<float> fractionalDelay(const std::vector<float>& x, double D) {
	std::vector<float> y(x.size(), 0.f);
	fc::MirrorRing ring;
	ring.alloc(fc::nextPow2((int) x.size() + 128));
	// pre-fill ring completely, then read at n - D with future samples available (offline)
	for (float v : x)
		ring.push(v);
	for (int i = 0; i < 40; i++)
		ring.push(0.f);
	for (size_t n = 0; n < x.size(); n++) {
		double pos = (double) n - D;
		if (pos < 20)
			continue;
		y[n] = ring.readSinc<32>(pos);
	}
	return y;
}

/** Kaiser-windowed zero-phase FFT convolution (offline). */
inline std::vector<float> fftConvolveSame(const std::vector<float>& x, const std::vector<float>& h) {
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

// ------------------------------------------------------------------------------------------------------------------------------
// A / B / H : phase vocoder family
// ------------------------------------------------------------------------------------------------------------------------------
class PhaseVocoder : public Candidate {
public:
	PhaseVocoder(int N, int osamp, bool identityLock, const char* nm) : N_(N), osamp_(osamp), lock_(identityLock), name_(nm) {}
	const char* name() const override { return name_.c_str(); }
	int latency(const CloneCtx&) const override { return N_; }

	std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) override {
		std::vector<float> y = process(x, ratio);
		if (c.randomPhase) {
			fc::Rng r(c.seed);
			y = fractionalDelay(y, 40.0 + r.uniform() * (c.unitPeriod > 0 ? c.unitPeriod : c.fs / c.f0));
			y.erase(y.begin(), y.begin() + 0); // (kept for clarity; latency handled by driver)
		}
		return y;
	}

	std::vector<float> process(const std::vector<float>& x, double ratio) const {
		const int N = N_, H = N_ / osamp_, K = N / 2;
		const double expct = fc::kTwoPi * H / N;
		fc::RealFFT fft(N);
		std::vector<float> win(N);
		for (int i = 0; i < N; i++)
			win[i] = fc::hannPeriodic(i, N);
		size_t n = x.size();
		std::vector<float> in(n + 2 * N, 0.f), out(n + 2 * N, 0.f);
		std::copy(x.begin(), x.end(), in.begin() + N);
		std::vector<float> buf(N), spec(N), re(K + 1), im(K + 1), mag(K + 1), ph(K + 1);
		std::vector<double> lastPh(K + 1, 0.0), sumPh(K + 1, 0.0), anaF(K + 1), synMag(K + 1), synF(K + 1);
		std::vector<float> yre(K + 1), yim(K + 1), outSpec(N), frame(N);
		const float scale = 1.f / (N * 0.375f * osamp_);
		for (size_t pos = 0; pos + N <= in.size(); pos += H) {
			for (int i = 0; i < N; i++)
				buf[i] = in[pos + i] * win[i];
			fft.forward(buf.data(), spec.data());
			re[0] = spec[0]; im[0] = 0; re[K] = spec[1]; im[K] = 0;
			for (int k = 1; k < K; k++) { re[k] = spec[2 * k]; im[k] = spec[2 * k + 1]; }
			for (int k = 0; k <= K; k++) {
				mag[k] = std::sqrt(re[k] * re[k] + im[k] * im[k]);
				ph[k] = std::atan2(im[k], re[k]);
				double d = ph[k] - lastPh[k] - k * expct;
				lastPh[k] = ph[k];
				d = wrapPi(d);
				anaF[k] = k + osamp_ * d / fc::kTwoPi; // true frequency in bins
			}
			std::fill(yre.begin(), yre.end(), 0.f);
			std::fill(yim.begin(), yim.end(), 0.f);
			if (!lock_) {
				std::fill(synMag.begin(), synMag.end(), 0.0);
				std::fill(synF.begin(), synF.end(), 0.0);
				for (int k = 0; k <= K; k++) {
					int idx = (int) std::lround(k * ratio);
					if (idx >= 0 && idx <= K) {
						synMag[idx] += mag[k];
						synF[idx] = anaF[k] * ratio;
					}
				}
				for (int k = 0; k <= K; k++) {
					double dev = synF[k] - k;
					dev = fc::kTwoPi * dev / osamp_ + k * expct;
					sumPh[k] += dev;
					yre[k] = (float) (synMag[k] * std::cos(sumPh[k]));
					yim[k] = (float) (synMag[k] * std::sin(sumPh[k]));
				}
			} else {
				// identity phase locking: rigid rotation of each peak's region of influence
				std::vector<int> peaks;
				float mx = 0;
				for (int k = 0; k <= K; k++) mx = std::max(mx, mag[k]);
				for (int k = 2; k < K - 2; k++)
					if (mag[k] > 1e-5f * mx && mag[k] > mag[k - 1] && mag[k] > mag[k + 1] && mag[k] >= mag[k - 2] && mag[k] >= mag[k + 2])
						peaks.push_back(k);
				std::vector<double> newSum(sumPh);
				if (peaks.empty()) {
					for (int k = 0; k <= K; k++) { yre[k] = re[k]; yim[k] = im[k]; }
				}
				for (size_t pi = 0; pi < peaks.size(); pi++) {
					int p = peaks[pi];
					int lo = pi == 0 ? 0 : (peaks[pi - 1] + p) / 2 + 1;
					int hi = pi + 1 == peaks.size() ? K : (p + peaks[pi + 1]) / 2;
					int q = (int) std::lround(p * ratio);
					int shift = q - p;
					double newF = anaF[p] * ratio;
					double dev = newF - q;
					double phiNew = sumPh[q < 0 ? 0 : (q > K ? K : q)] + fc::kTwoPi * dev / osamp_ + q * expct;
					double rot = phiNew - ph[p];
					float cr = (float) std::cos(rot), ci = (float) std::sin(rot);
					for (int k = lo; k <= hi; k++) {
						int t = k + shift;
						if (t < 0 || t > K) continue;
						yre[t] = re[k] * cr - im[k] * ci;
						yim[t] = re[k] * ci + im[k] * cr;
					}
					if (q >= 0 && q <= K) newSum[q] = phiNew;
				}
				for (int k = 0; k <= K; k++)
					if (yre[k] != 0.f || yim[k] != 0.f) sumPh[k] = std::atan2(yim[k], yre[k]);
				for (size_t pi = 0; pi < peaks.size(); pi++) {
					int q = (int) std::lround(peaks[pi] * ratio);
					if (q >= 0 && q <= K) sumPh[q] = newSum[q];
				}
			}
			outSpec[0] = yre[0];
			outSpec[1] = yre[K];
			for (int k = 1; k < K; k++) { outSpec[2 * k] = yre[k]; outSpec[2 * k + 1] = yim[k]; }
			fft.inverse(outSpec.data(), frame.data());
			for (int i = 0; i < N; i++)
				out[pos + i] += frame[i] * win[i] * scale;
		}
		return std::vector<float>(out.begin() + N, out.begin() + N + n);
	}

private:
	int N_, osamp_;
	bool lock_;
	std::string name_;
};

// ------------------------------------------------------------------------------------------------------------------------------
// H : 2-band multi-resolution identity-locked PV
// ------------------------------------------------------------------------------------------------------------------------------
class MultiResPv : public Candidate {
public:
	MultiResPv() : lo_(16384, 4, true, "lo"), hi_(2048, 4, true, "hi") {}
	const char* name() const override { return "H  multi-res PV+IPL (16384|2048 @300Hz)"; }
	int latency(const CloneCtx&) const override { return 16384; }
	std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) override {
		// zero-phase complementary split
		const int T = 8191;
		std::vector<float> h(T);
		std::vector<float> ker = fc::designLowpassFir(T, 300.0 / c.fs, 8.0);
		std::vector<float> lo = fftConvolveSame(x, ker), hi(x.size());
		for (size_t i = 0; i < x.size(); i++) hi[i] = x[i] - lo[i];
		std::vector<float> a = lo_.process(lo, ratio), b = hi_.process(hi, ratio);
		std::vector<float> y(x.size());
		// the two paths have different latencies only in a causal implementation; offline both are zero-latency
		for (size_t i = 0; i < x.size(); i++) y[i] = a[i] + b[i];
		if (c.randomPhase) {
			fc::Rng r(c.seed);
			y = fractionalDelay(y, 40.0 + r.uniform() * (c.unitPeriod > 0 ? c.unitPeriod : c.fs / c.f0));
		}
		return y;
	}
private:
	PhaseVocoder lo_, hi_;
};

// ------------------------------------------------------------------------------------------------------------------------------
// C : sinusoidal modelling with partial tracking (McAulay-Quatieri style)
// ------------------------------------------------------------------------------------------------------------------------------
class SinusoidalMQ : public Candidate {
public:
	explicit SinusoidalMQ(int N) : N_(N) { name_ = "C  sinusoidal MQ tracking (N=" + std::to_string(N) + ")"; }
	const char* name() const override { return name_.c_str(); }
	int latency(const CloneCtx&) const override { return N_; }

	struct Peak { double f, a, ph; };
	struct Track { double f0, f1, a0, a1, phi; bool born; };

	std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) override {
		const int N = N_, H = N_ / 8, K = N / 2;
		fc::RealFFT fft(N);
		std::vector<float> win(N);
		double wsum = 0;
		for (int i = 0; i < N; i++) {
			// Blackman-Harris 4-term (better sidelobes for peak picking)
			double t = fc::kTwoPi * i / N;
			win[i] = (float) (0.35875 - 0.48829 * std::cos(t) + 0.14128 * std::cos(2 * t) - 0.01168 * std::cos(3 * t));
			wsum += win[i];
		}
		size_t n = x.size();
		std::vector<float> y(n + 2 * N, 0.f);
		std::vector<float> in(n + 2 * N, 0.f);
		std::copy(x.begin(), x.end(), in.begin() + N);
		std::vector<float> buf(N), spec(N);
		std::vector<Peak> prevPeaks, peaks;
		std::vector<int> prevTrackIdx;
		struct Live { double f, a, phSyn; bool alive; };
		std::vector<Live> live; // synthesis state per active track
		std::vector<int> owner; // track id per prev peak
		std::vector<double> prevF, prevA;
		bool have = false;
		std::vector<std::pair<double, int>> dummy;
		// per-frame synthesis
		struct Seg { double a0, a1, f0, f1, ph0; };
		size_t frameCount = 0;
		std::vector<Live> tracks; // persistent list; matched by index into prevPeaks
		std::vector<int> trackOfPrev;
		for (size_t pos = 0; pos + N <= in.size(); pos += H, frameCount++) {
			// zero-phase frame: shift so the window centre is at index 0
			for (int i = 0; i < N; i++) buf[(i + N / 2) % N] = in[pos + i] * win[i];
			fft.forward(buf.data(), spec.data());
			// magnitude/phase
			std::vector<float> mag(K + 1);
			mag[0] = std::fabs(spec[0]); mag[K] = std::fabs(spec[1]);
			for (int k = 1; k < K; k++) mag[k] = std::sqrt(spec[2 * k] * spec[2 * k] + spec[2 * k + 1] * spec[2 * k + 1]);
			float mx = *std::max_element(mag.begin(), mag.end());
			peaks.clear();
			for (int k = 3; k < K - 3; k++) {
				if (mag[k] > 3e-4f * mx && mag[k] > mag[k - 1] && mag[k] >= mag[k + 1] && mag[k] > mag[k - 2] && mag[k] >= mag[k + 2]) {
					double la = std::log(mag[k - 1] + 1e-20), lb = std::log(mag[k] + 1e-20), lc = std::log(mag[k + 1] + 1e-20);
					double den = la - 2 * lb + lc;
					double d = den != 0 ? 0.5 * (la - lc) / den : 0.0;
					d = std::max(-0.5, std::min(0.5, d));
					Peak p;
					p.f = (k + d) * c.fs / N;
					p.a = std::exp(lb - 0.25 * (la - lc) * d) * 2.0 / wsum;
					p.ph = std::atan2(spec[2 * k + 1], spec[2 * k]);
					peaks.push_back(p);
				}
			}
			// link to previous peaks (greedy nearest within 0.75 bin + 0.5%)
			std::vector<int> link(peaks.size(), -1);
			std::vector<char> used(prevPeaks.size(), 0);
			for (size_t i = 0; i < peaks.size(); i++) {
				double best = 1e30; int bj = -1;
				for (size_t j = 0; j < prevPeaks.size(); j++) {
					if (used[j]) continue;
					double d = std::fabs(peaks[i].f - prevPeaks[j].f);
					if (d < 0.75 * c.fs / N + 0.005 * peaks[i].f && d < best) { best = d; bj = (int) j; }
				}
				if (bj >= 0) { link[i] = bj; used[bj] = 1; }
			}
			// synthesis of the segment between previous frame centre and this frame centre
			std::vector<Live> nt(peaks.size());
			double segStart = (double) (pos - H + N / 2); // absolute index (in padded coordinates) of previous frame centre
			for (size_t i = 0; i < peaks.size(); i++) {
				double fS = peaks[i].f * ratio; // synthesis frequency
				if (link[i] >= 0) {
					const Live& p = tracks[trackOfPrev[link[i]]];
					nt[i].phSyn = p.phSyn + fc::kTwoPi * 0.5 * (p.f + fS) / c.fs * H;
					addSegment(y, segStart, H, p.a, peaks[i].a, p.f, fS, p.phSyn, c.fs);
				} else {
					// birth: fade in over the segment with the measured phase
					nt[i].phSyn = peaks[i].ph + fc::kTwoPi * fS / c.fs * 0.0;
					// synthesise backwards from the birth phase
					double phBack = peaks[i].ph - fc::kTwoPi * fS / c.fs * H;
					if (have) addSegment(y, segStart, H, 0.0, peaks[i].a, fS, fS, phBack, c.fs);
					nt[i].phSyn = peaks[i].ph;
				}
				nt[i].f = fS; nt[i].a = peaks[i].a; nt[i].alive = true;
			}
			// deaths: fade out the tracks that were not continued
			for (size_t j = 0; j < prevPeaks.size(); j++) if (!used[j] && have) {
				const Live& p = tracks[trackOfPrev[j]];
				addSegment(y, segStart, H, p.a, 0.0, p.f, p.f, p.phSyn, c.fs);
			}
			tracks = nt;
			trackOfPrev.resize(peaks.size());
			for (size_t i = 0; i < peaks.size(); i++) trackOfPrev[i] = (int) i;
			prevPeaks = peaks;
			have = true;
		}
		std::vector<float> out(y.begin() + N, y.begin() + N + n);
		if (c.randomPhase) {
			fc::Rng r(c.seed);
			out = fractionalDelay(out, 40.0 + r.uniform() * (c.unitPeriod > 0 ? c.unitPeriod : c.fs / c.f0));
		}
		return out;
	}

private:
	static void addSegment(std::vector<float>& y, double start, int H, double a0, double a1, double f0, double f1, double ph0, double fs) {
		size_t s = (size_t) std::max(0.0, std::floor(start));
		if (s + H > y.size()) return;
		double w0 = fc::kTwoPi * f0 / fs, w1 = fc::kTwoPi * f1 / fs;
		for (int n = 0; n < H; n++) {
			double t = (double) n;
			double a = a0 + (a1 - a0) * t / H;
			double ph = ph0 + w0 * t + 0.5 * (w1 - w0) * t * t / H;
			y[s + n] += (float) (a * std::cos(ph));
		}
	}
	int N_;
	std::string name_;
};

// ------------------------------------------------------------------------------------------------------------------------------
// D : harmonic additive/heterodyne resynthesis with oracle f0
// ------------------------------------------------------------------------------------------------------------------------------
class HarmonicHeterodyne : public Candidate {
public:
	explicit HarmonicHeterodyne(int N, int Kmax = 260) : N_(N), Kmax_(Kmax) { name_ = "D  harmonic heterodyne, oracle f0 (N=" + std::to_string(N) + ")"; }
	const char* name() const override { return name_.c_str(); }
	int latency(const CloneCtx&) const override { return N_ / 2; }
	std::vector<float> renderClone(const std::vector<float>& x, double ratio, const CloneCtx& c) override {
		const int N = N_, H = N_ / 8;
		double f0 = c.f0;
		if (c.unitPeriod > 0) f0 = c.fs / c.unitPeriod; // use the repeating-unit frequency so sub-harmonics are captured
		int K = std::min(Kmax_, (int) std::floor(0.45 * c.fs / f0));
		size_t n = x.size();
		std::vector<float> win(N);
		double wsum = 0;
		for (int i = 0; i < N; i++) { win[i] = fc::hannPeriodic(i, N); wsum += win[i]; }
		int frames = (int) ((n + H - 1) / H) + 2;
		std::vector<std::vector<std::complex<float>>> X(frames, std::vector<std::complex<float>>(K + 1));
		std::vector<std::complex<double>> rot(K + 1);
		for (int fr = 0; fr < frames; fr++) {
			long centre = (long) fr * H;
			long start = centre - N / 2;
			std::vector<std::complex<double>> acc(K + 1, 0.0), ph(K + 1, 1.0), step(K + 1);
			// direct DTFT at exact harmonic frequencies using per-harmonic rotators
			for (int k = 1; k <= K; k++) {
				double w = -fc::kTwoPi * k * f0 / c.fs;
				step[k] = std::polar(1.0, w);
				ph[k] = std::polar(1.0, w * (double) start);
			}
			for (int i = 0; i < N; i++) {
				long idx = start + i;
				if (idx < 0 || idx >= (long) n) { for (int k = 1; k <= K; k++) ph[k] *= step[k]; continue; }
				double v = x[idx] * win[i];
				for (int k = 1; k <= K; k++) { acc[k] += v * ph[k]; ph[k] *= step[k]; }
			}
			for (int k = 1; k <= K; k++) X[fr][k] = std::complex<float>((float) (acc[k].real() / wsum), (float) (acc[k].imag() / wsum));
		}
		std::vector<float> y(n, 0.f);
		std::vector<std::complex<double>> osc(K + 1), ost(K + 1);
		for (int k = 1; k <= K; k++) {
			double w = fc::kTwoPi * k * f0 * ratio / c.fs;
			ost[k] = std::polar(1.0, w);
			osc[k] = 1.0;
		}
		for (size_t i = 0; i < n; i++) {
			int fr = (int) (i / H);
			double u = (double) (i % H) / H;
			std::complex<double> s = 0;
			for (int k = 1; k <= K; k++) {
				std::complex<double> a = (1 - u) * std::complex<double>(X[fr][k]) + u * std::complex<double>(X[fr + 1][k]);
				s += a * osc[k]; // carrier e^{j 2 pi k f0 r t}: shifted frequency, absolute-time phase
				osc[k] *= ost[k];
			}
			if ((i & 4095) == 4095)
				for (int k = 1; k <= K; k++) osc[k] /= std::abs(osc[k]);
			y[i] = (float) (2.0 * s.real());
		}
		if (c.randomPhase) {
			fc::Rng r(c.seed);
			y = fractionalDelay(y, 40.0 + r.uniform() * (c.unitPeriod > 0 ? c.unitPeriod : c.fs / c.f0));
		}
		return y;
	}
private:
	int N_, Kmax_;
	std::string name_;
};

} // namespace research
