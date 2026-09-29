// Objective proxy metrics used to compare cloning architectures.
//
// None of these is *the* quality criterion (see docs/ARCHITECTURE.md §Evaluation) — they are targeted probes for specific
// failure modes: phase-vocoder smear, chorus-like periodic modulation, comb filtering, transient blur, aliasing, pitch bias.
#pragma once
#include "../../src/dsp/FcCommon.hpp"
#include "../../src/dsp/FcFFT.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace research {

struct Psd {
	std::vector<double> p; // one-sided power per bin (0..nfft/2)
	double binHz = 1.0;
	int nfft = 0;
	double fs = 48000;
};

/** Welch PSD with a Hann window and 50% overlap over x[start, stop). */
inline Psd welch(const std::vector<float>& x, double fs, int nfft, size_t start = 0, size_t stop = (size_t) -1) {
	stop = std::min(stop, x.size());
	Psd r;
	r.nfft = nfft;
	r.fs = fs;
	r.binHz = fs / nfft;
	r.p.assign(nfft / 2 + 1, 0.0);
	fc::RealFFT fft(nfft);
	std::vector<float> w(nfft), buf(nfft), spec(nfft);
	double wsum2 = 0;
	for (int i = 0; i < nfft; i++) {
		w[i] = fc::hannPeriodic(i, nfft);
		wsum2 += (double) w[i] * w[i];
	}
	int segs = 0;
	for (size_t pos = start; pos + nfft <= stop; pos += nfft / 2) {
		for (int i = 0; i < nfft; i++)
			buf[i] = x[pos + i] * w[i];
		fft.forward(buf.data(), spec.data());
		r.p[0] += (double) spec[0] * spec[0];
		r.p[nfft / 2] += (double) spec[1] * spec[1];
		for (int k = 1; k < nfft / 2; k++)
			r.p[k] += (double) spec[2 * k] * spec[2 * k] + (double) spec[2 * k + 1] * spec[2 * k + 1];
		segs++;
	}
	if (segs > 0)
		for (double& v : r.p)
			v /= (double) segs * wsum2 * nfft;
	return r;
}

inline double bandEnergy(const Psd& s, double f1, double f2) {
	int k1 = std::max(0, (int) std::ceil(f1 / s.binHz)), k2 = std::min((int) s.p.size() - 1, (int) std::floor(f2 / s.binHz));
	double e = 0;
	for (int k = k1; k <= k2; k++)
		e += s.p[k];
	return e;
}

inline double rms(const std::vector<float>& x, size_t a = 0, size_t b = (size_t) -1) {
	b = std::min(b, x.size());
	if (b <= a)
		return 0;
	double e = 0;
	for (size_t i = a; i < b; i++)
		e += (double) x[i] * x[i];
	return std::sqrt(e / (b - a));
}

inline double db(double lin) {
	return 20.0 * std::log10(std::max(lin, 1e-12));
}
inline double dbPow(double p) {
	return 10.0 * std::log10(std::max(p, 1e-24));
}

// --------------------------------------------------------------------------------------------------------------------------------
// Harmonic clusters
// --------------------------------------------------------------------------------------------------------------------------------
struct Clusters {
	std::vector<double> energy;   // linear power in the band around each harmonic (index k-1)
	std::vector<double> centroid; // cents relative to k*f0
	std::vector<double> spread;   // std-dev of the cluster in cents
};

inline Clusters harmonicClusters(const Psd& s, double f0, int K, double bandFrac = 0.03) {
	Clusters c;
	for (int k = 1; k <= K; k++) {
		double w = std::min(bandFrac, 0.42 / k);
		double fk = k * f0;
		int k1 = std::max(1, (int) std::ceil(fk * (1 - w) / s.binHz)), k2 = std::min((int) s.p.size() - 1, (int) std::floor(fk * (1 + w) / s.binHz));
		double e = 0, m1 = 0;
		for (int b = k1; b <= k2; b++) {
			e += s.p[b];
			m1 += s.p[b] * (b * s.binHz);
		}
		double cen = e > 0 ? m1 / e : fk;
		double v = 0;
		for (int b = k1; b <= k2; b++) {
			double d = 1200.0 * std::log2(std::max(b * s.binHz, 1e-3) / cen);
			v += s.p[b] * d * d;
		}
		c.energy.push_back(e);
		c.centroid.push_back(1200.0 * std::log2(std::max(cen, 1e-3) / fk));
		c.spread.push_back(e > 0 ? std::sqrt(v / e) : 0.0);
	}
	return c;
}

/** Energy in the "quiet" zones between harmonics relative to energy in harmonic bands, dB. Lower = cleaner.
    Zones are [k+0.3, k+0.7]*f0 for k = 1..Kmax. Only meaningful while cluster half-widths stay below 0.3*f0. */
inline double interHarmonicDb(const Psd& s, double f0, int Kmax, double bandFrac = 0.03) {
	double eh = 0, ez = 0;
	for (int k = 1; k <= Kmax; k++) {
		double w = std::min(bandFrac, 0.25 / k);
		eh += bandEnergy(s, k * f0 * (1 - w), k * f0 * (1 + w));
		ez += bandEnergy(s, (k + 0.3) * f0, (k + 0.7) * f0);
	}
	return dbPow(ez / std::max(eh, 1e-30));
}

/** Harmonic amplitude profile (dB, index k-1) for k = 1..K using wide-enough bands. */
inline std::vector<double> harmonicProfileDb(const Psd& s, double f0, int K) {
	std::vector<double> o;
	for (int k = 1; k <= K; k++) {
		double w = std::min(0.04, 0.4 / k);
		o.push_back(dbPow(bandEnergy(s, k * f0 * (1 - w), k * f0 * (1 + w))));
	}
	return o;
}

/** Std-dev over k of (profile_out - profile_in) after removing the mean: comb-filter ripple in dB. */
inline double combRippleDb(const std::vector<double>& out, const std::vector<double>& in) {
	int K = (int) std::min(out.size(), in.size());
	std::vector<double> d(K);
	double m = 0;
	for (int k = 0; k < K; k++) {
		d[k] = out[k] - in[k];
		m += d[k];
	}
	m /= K;
	double v = 0;
	for (int k = 0; k < K; k++)
		v += (d[k] - m) * (d[k] - m);
	return std::sqrt(v / K);
}

// --------------------------------------------------------------------------------------------------------------------------------
// Envelope / beat texture
// --------------------------------------------------------------------------------------------------------------------------------
struct EnvStats {
	double cv = 0;       // coefficient of variation of the band envelope
	double peakinessDb = 0; // peak / median of the modulation spectrum in 0.3..40 Hz
	double peakHz = 0;
};

/** Envelope of the band [f1,f2] (Hilbert via FFT), statistics over the steady part [a,b). */
inline EnvStats bandEnvelopeStats(const std::vector<float>& x, double fs, double f1, double f2, size_t a, size_t b) {
	b = std::min(b, x.size());
	int n = (int) (b - a);
	int N = fc::nextPow2(n);
	fc::RealFFT fft(N);
	std::vector<float> in(N, 0.f), spec(N), re(N), imv(N);
	for (int i = 0; i < n; i++)
		in[i] = x[a + i]; // rectangular window is fine: the band is selected in the frequency domain and edges are skipped
	fft.forward(in.data(), spec.data());
	// analytic band: keep bins in [f1,f2] doubled; build two real signals (real part and hilbert part) via two inverse transforms
	std::vector<float> sr(N, 0.f), si(N, 0.f);
	int k1 = std::max(1, (int) std::floor(f1 / fs * N)), k2 = std::min(N / 2 - 1, (int) std::ceil(f2 / fs * N));
	for (int k = k1; k <= k2; k++) {
		float xr = spec[2 * k], xi = spec[2 * k + 1];
		sr[2 * k] = xr;
		sr[2 * k + 1] = xi;
		// Hilbert: multiply by -j  => (xr, xi) -> (xi, -xr)
		si[2 * k] = xi;
		si[2 * k + 1] = -xr;
	}
	fft.inverse(sr.data(), re.data());
	fft.inverse(si.data(), imv.data());
	std::vector<float> env(n);
	double mean = 0;
	for (int i = 0; i < n; i++) {
		double r = re[i] / N, h = imv[i] / N;
		env[i] = (float) std::sqrt(r * r + h * h);
		mean += env[i];
	}
	mean /= n;
	// skip edge effects (5%)
	int e0 = n / 20, e1 = n - n / 20;
	double m2 = 0, v = 0;
	for (int i = e0; i < e1; i++)
		m2 += env[i];
	m2 /= (e1 - e0);
	for (int i = e0; i < e1; i++)
		v += (env[i] - m2) * (env[i] - m2);
	v /= (e1 - e0);
	EnvStats r;
	r.cv = std::sqrt(v) / std::max(m2, 1e-12);
	std::vector<float> ev(env.begin() + e0, env.begin() + e1);
	for (float& t : ev)
		t -= (float) m2;
	int nf = std::min(1 << 16, fc::nextPow2((int) ev.size()) >> 1);
	Psd ps = welch(ev, fs, nf);
	int lo = std::max(1, (int) std::ceil(0.3 / ps.binHz)), hi = std::min((int) ps.p.size() - 1, (int) std::floor(40.0 / ps.binHz));
	std::vector<double> seg(ps.p.begin() + lo, ps.p.begin() + hi + 1);
	// smooth with 5-bin boxcar so single-bin speckle does not count as a "line"
	std::vector<double> sm(seg.size(), 0.0);
	for (size_t i = 0; i < seg.size(); i++) {
		double a2 = 0;
		int c = 0;
		for (int j = -2; j <= 2; j++) {
			int q = (int) i + j;
			if (q >= 0 && q < (int) seg.size()) {
				a2 += seg[q];
				c++;
			}
		}
		sm[i] = a2 / c;
	}
	std::vector<double> sorted = sm;
	std::sort(sorted.begin(), sorted.end());
	double med = sorted[sorted.size() / 2];
	size_t im = std::max_element(sm.begin(), sm.end()) - sm.begin();
	r.peakinessDb = dbPow(sm[im] / std::max(med, 1e-30));
	r.peakHz = (lo + (int) im) * ps.binHz;
	return r;
}

// --------------------------------------------------------------------------------------------------------------------------------
// Onset / transient
// --------------------------------------------------------------------------------------------------------------------------------
inline std::vector<float> rmsEnvelope(const std::vector<float>& x, double fs, double winMs = 1.0, double hopMs = 0.25) {
	int win = std::max(4, (int) (winMs * 1e-3 * fs)), hop = std::max(1, (int) (hopMs * 1e-3 * fs));
	std::vector<float> e;
	for (size_t p = 0; p + win <= x.size(); p += hop) {
		double s = 0;
		for (int i = 0; i < win; i++)
			s += (double) x[p + i] * x[p + i];
		e.push_back((float) std::sqrt(s / win));
	}
	return e;
}

struct OnsetStats {
	double preEchoDb = -200;  // max envelope before the onset relative to steady, dB
	double riseMs = 0;        // 10%-90% rise time
	double overshootDb = 0;   // peak in first 150 ms relative to steady
	double delayMs = 0;       // time from t_on to 50% of steady level
};

inline OnsetStats onsetStats(const std::vector<float>& x, double fs, double tOnset, double steadyFrom, double steadyTo, double hopMs = 0.25) {
	std::vector<float> e = rmsEnvelope(x, fs, 1.0, hopMs);
	double eh = hopMs * 1e-3;
	auto idx = [&](double t) { return (int) std::round(t / eh); };
	double st = 0;
	int a = idx(steadyFrom), b = std::min(idx(steadyTo), (int) e.size() - 1);
	for (int i = a; i <= b; i++)
		st += e[i];
	st /= std::max(1, b - a + 1);
	OnsetStats o;
	double pre = 0;
	for (int i = idx(tOnset - 0.030); i < idx(tOnset - 0.001) && i < (int) e.size(); i++)
		if (i >= 0)
			pre = std::max(pre, (double) e[i]);
	o.preEchoDb = db(pre / st);
	double mx = 0;
	int t10 = -1, t50 = -1, t90 = -1;
	for (int i = idx(tOnset - 0.002); i < idx(tOnset + 0.150) && i < (int) e.size(); i++) {
		if (i < 0)
			continue;
		mx = std::max(mx, (double) e[i]);
		double r = e[i] / st;
		if (t10 < 0 && r >= 0.1)
			t10 = i;
		if (t50 < 0 && r >= 0.5)
			t50 = i;
		if (t90 < 0 && r >= 0.9)
			t90 = i;
	}
	o.overshootDb = db(mx / st);
	o.riseMs = (t10 >= 0 && t90 >= 0) ? (t90 - t10) * hopMs : -1;
	o.delayMs = t50 >= 0 ? (t50 * eh - tOnset) * 1e3 : -1;
	return o;
}

/** RMS of the dB difference between two envelopes over [t0,t1] (floor -60 dB). */
inline double envelopeDiffDb(const std::vector<float>& a, const std::vector<float>& b, double fs, double t0, double t1, double steadyA, double steadyB,
                             double hopMs = 0.25) {
	std::vector<float> ea = rmsEnvelope(a, fs, 1.0, hopMs), eb = rmsEnvelope(b, fs, 1.0, hopMs);
	double eh = hopMs * 1e-3;
	int i0 = std::max(0, (int) (t0 / eh)), i1 = std::min((int) std::min(ea.size(), eb.size()) - 1, (int) (t1 / eh));
	double s = 0;
	int c = 0;
	for (int i = i0; i <= i1; i++) {
		double da = std::max(-60.0, db(ea[i] / steadyA)), dbb = std::max(-60.0, db(eb[i] / steadyB));
		s += (da - dbb) * (da - dbb);
		c++;
	}
	return c ? std::sqrt(s / c) : 0.0;
}


/** Envelope recurrence: maximum normalised autocovariance of the band envelope for lags in [0.15 s, 3 s]. A periodic modulation (chorus LFO,
    evenly spaced "supersaw" detunes) makes the envelope repeat (value -> 1); independent oscillators with incommensurate offsets do not. */
inline double envRecurrence(const std::vector<float>& x, double fs, double f1, double f2, size_t a, size_t b) {
	b = std::min(b, x.size());
	int n = (int) (b - a);
	int N = fc::nextPow2(n);
	fc::RealFFT fft(N);
	std::vector<float> in(N, 0.f), spec(N), sr(N, 0.f), si(N, 0.f), re(N), imv(N);
	for (int i = 0; i < n; i++)
		in[i] = x[a + i];
	fft.forward(in.data(), spec.data());
	int k1 = std::max(1, (int) std::floor(f1 / fs * N)), k2 = std::min(N / 2 - 1, (int) std::ceil(f2 / fs * N));
	for (int k = k1; k <= k2; k++) {
		sr[2 * k] = spec[2 * k];
		sr[2 * k + 1] = spec[2 * k + 1];
		si[2 * k] = spec[2 * k + 1];
		si[2 * k + 1] = -spec[2 * k];
	}
	fft.inverse(sr.data(), re.data());
	fft.inverse(si.data(), imv.data());
	// envelope, decimated to ~500 Hz to keep the autocorrelation cheap
	int dec = std::max(1, (int) (fs / 500.0));
	std::vector<float> env;
	for (int i = n / 20; i < n - n / 20; i += dec) {
		double r = re[i] / N, h = imv[i] / N;
		env.push_back((float) std::sqrt(r * r + h * h));
	}
	double m = 0;
	for (float v : env) m += v;
	m /= env.size();
	for (float& v : env) v -= (float) m;
	int M = (int) env.size();
	int L = fc::nextPow2(2 * M);
	fc::RealFFT fftL(L);
	std::vector<float> e2(L, 0.f), E(L), P(L), ac(L);
	std::copy(env.begin(), env.end(), e2.begin());
	fftL.forward(e2.data(), E.data());
	P[0] = E[0] * E[0];
	P[1] = E[1] * E[1];
	for (int k = 1; k < L / 2; k++) {
		P[2 * k] = E[2 * k] * E[2 * k] + E[2 * k + 1] * E[2 * k + 1];
		P[2 * k + 1] = 0.f;
	}
	fftL.inverse(P.data(), ac.data());
	double r0 = ac[0];
	double best = 0;
	double rate = fs / dec;
	for (int lag = (int) (0.15 * rate); lag < std::min(M / 2, (int) (3.0 * rate)); lag++)
		best = std::max(best, (double) ac[lag] / std::max(r0, 1e-30) * ((double) M / (M - lag)));
	return best;
}

} // namespace research
