// FusionClone DSP core — small filters used all over the engine.
#pragma once
#include "FcCommon.hpp"

namespace fc {

/** One-pole low-pass (also used as parameter smoother). */
struct OnePole {
	float y = 0.f;
	float a = 1.f; // coefficient, y += a*(x-y)

	void setCutoff(float fc, float fs) {
		a = 1.f - std::exp(-(float) kTwoPi * clampT(fc / fs, 0.f, 0.49f));
	}
	void setTimeConstant(float seconds, float fs) {
		a = seconds <= 0.f ? 1.f : 1.f - std::exp(-1.f / (seconds * fs));
	}
	float process(float x) {
		y += a * (x - y);
		return y;
	}
	void reset(float v = 0.f) { y = v; }
};

/** First-order DC blocker y[n] = x[n] - x[n-1] + R*y[n-1]. */
struct DcBlocker {
	float x1 = 0.f, y1 = 0.f, R = 0.999f;
	void setCutoff(float fc, float fs) { R = 1.f - (float) kTwoPi * fc / fs; }
	float process(float x) {
		float y = x - x1 + R * y1;
		x1 = x;
		y1 = flushDenormal(y);
		return y;
	}
	void reset() { x1 = y1 = 0.f; }
};

/** RBJ biquad, transposed direct form II. */
struct Biquad {
	float b0 = 1.f, b1 = 0.f, b2 = 0.f, a1 = 0.f, a2 = 0.f;
	float z1 = 0.f, z2 = 0.f;

	void setIdentity() {
		b0 = 1.f;
		b1 = b2 = a1 = a2 = 0.f;
	}
	void set(double B0, double B1, double B2, double A0, double A1, double A2) {
		double inv = 1.0 / A0;
		b0 = (float) (B0 * inv);
		b1 = (float) (B1 * inv);
		b2 = (float) (B2 * inv);
		a1 = (float) (A1 * inv);
		a2 = (float) (A2 * inv);
	}
	void setPeaking(double f0, double fs, double q, double gainDb) {
		double A = std::pow(10.0, gainDb / 40.0);
		double w = kTwoPi * clampT(f0 / fs, 1e-5, 0.49);
		double al = std::sin(w) / (2 * q), c = std::cos(w);
		set(1 + al * A, -2 * c, 1 - al * A, 1 + al / A, -2 * c, 1 - al / A);
	}
	void setLowShelf(double f0, double fs, double gainDb, double slope = 1.0) {
		double A = std::pow(10.0, gainDb / 40.0);
		double w = kTwoPi * clampT(f0 / fs, 1e-5, 0.49);
		double c = std::cos(w), s = std::sin(w);
		double al = s / 2 * std::sqrt((A + 1 / A) * (1 / slope - 1) + 2);
		double sq = 2 * std::sqrt(A) * al;
		set(A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq),
		    (A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq);
	}
	void setHighShelf(double f0, double fs, double gainDb, double slope = 1.0) {
		double A = std::pow(10.0, gainDb / 40.0);
		double w = kTwoPi * clampT(f0 / fs, 1e-5, 0.49);
		double c = std::cos(w), s = std::sin(w);
		double al = s / 2 * std::sqrt((A + 1 / A) * (1 / slope - 1) + 2);
		double sq = 2 * std::sqrt(A) * al;
		set(A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq),
		    (A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq);
	}
	void setLowpass(double f0, double fs, double q) {
		double w = kTwoPi * clampT(f0 / fs, 1e-5, 0.49);
		double al = std::sin(w) / (2 * q), c = std::cos(w);
		set((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al);
	}
	void setHighpass(double f0, double fs, double q) {
		double w = kTwoPi * clampT(f0 / fs, 1e-5, 0.49);
		double al = std::sin(w) / (2 * q), c = std::cos(w);
		set((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al);
	}
	/** All-pass, 2nd order. */
	void setAllpass(double f0, double fs, double q) {
		double w = kTwoPi * clampT(f0 / fs, 1e-5, 0.49);
		double al = std::sin(w) / (2 * q), c = std::cos(w);
		set(1 - al, -2 * c, 1 + al, 1 + al, -2 * c, 1 - al);
	}

	float process(float x) {
		float y = b0 * x + z1;
		z1 = flushDenormal(b1 * x - a1 * y + z2);
		z2 = flushDenormal(b2 * x - a2 * y);
		return y;
	}
	void reset() { z1 = z2 = 0.f; }

	/** Magnitude response at normalised frequency f/fs. */
	double magnitudeAt(double fNorm) const {
		double w = kTwoPi * fNorm;
		double cw = std::cos(w), sw = std::sin(w), c2 = std::cos(2 * w), s2 = std::sin(2 * w);
		double nr = b0 + b1 * cw + b2 * c2, ni = -(b1 * sw + b2 * s2);
		double dr = 1 + a1 * cw + a2 * c2, di = -(a1 * sw + a2 * s2);
		return std::sqrt((nr * nr + ni * ni) / (dr * dr + di * di));
	}
};

/** Windowed-sinc FIR low-pass designer (Kaiser). Returns a symmetric, unit-DC-gain kernel. */
inline std::vector<float> designLowpassFir(int taps, double cutoffNorm /*0..0.5 of fs*/, double beta) {
	std::vector<float> h(taps);
	double sum = 0;
	double mid = (taps - 1) * 0.5;
	for (int i = 0; i < taps; i++) {
		double n = i - mid;
		double s = (std::fabs(n) < 1e-12) ? 2 * cutoffNorm : std::sin(kTwoPi * cutoffNorm * n) / (kPi * n);
		double w = kaiser((double) i, (double) (taps - 1), beta);
		h[i] = (float) (s * w);
		sum += h[i];
	}
	for (int i = 0; i < taps; i++)
		h[i] = (float) (h[i] / sum);
	return h;
}

/** Streaming FIR decimator: push() consumes one input sample and returns true when an output
    sample (available through out()) has been produced. */
class FirDecimator {
public:
	void init(int factor, int taps, double cutoffNorm, double beta = 8.0) {
		factor_ = std::max(1, factor);
		taps_ = taps;
		h_ = designLowpassFir(taps, cutoffNorm, beta);
		buf_.assign(taps * 2, 0.f);
		pos_ = 0;
		count_ = 0;
		out_ = 0.f;
	}
	bool push(float x) {
		// double-length buffer so the FIR read is contiguous
		buf_[pos_] = x;
		buf_[pos_ + taps_] = x;
		pos_++;
		if (pos_ >= taps_)
			pos_ = 0;
		if (++count_ < factor_)
			return false;
		count_ = 0;
		const float* p = &buf_[pos_]; // oldest sample first
		float acc = 0.f;
		for (int i = 0; i < taps_; i++)
			acc += p[i] * h_[i];
		out_ = acc;
		return true;
	}
	float out() const { return out_; }
	int factor() const { return factor_; }
	int taps() const { return taps_; }
	void reset() {
		std::fill(buf_.begin(), buf_.end(), 0.f);
		pos_ = count_ = 0;
		out_ = 0.f;
	}

private:
	int factor_ = 1, taps_ = 1, pos_ = 0, count_ = 0;
	std::vector<float> h_, buf_;
	float out_ = 0.f;
};


/** Polyphase all-pass IIR Hilbert transformer (two branches of four 2nd-order-in-z^-1 all-pass sections). The branch outputs are
    90 degrees apart to within 0.7 degrees from ~0.002 fs to ~0.498 fs (about 100 Hz .. 24 kHz at 48 kHz; below that the image rejection
    of a frequency shifter built on it degrades gracefully). Section: H(z) = (z^-2 - a^2) / (1 - a^2 z^-2). Branch `i` carries the extra
    one-sample delay. Coefficients: O. Niemitalo, checked numerically in tests/test_hilbert.cpp. */
struct HilbertPair {
	struct Ap {
		float c = 0.f, x1 = 0.f, x2 = 0.f, y1 = 0.f, y2 = 0.f;
		inline float process(float x) {
			float y = -c * x + x2 + c * y2;
			x2 = x1;
			x1 = x;
			y2 = y1;
			y1 = y;
			return y;
		}
	};
	Ap a[4], b[4];
	float dly = 0.f;
	HilbertPair() {
		static const double A[4] = {0.6923878, 0.9360654322959, 0.9882295226860, 0.9987488452737};
		static const double B[4] = {0.4021921162426, 0.8561710882420, 0.9722909545651, 0.9952884791278};
		for (int k = 0; k < 4; k++) {
			a[k].c = (float) (A[k] * A[k]);
			b[k].c = (float) (B[k] * B[k]);
		}
	}
	/** i and q are the in-phase and quadrature outputs (analytic signal = i + j*q up to the sign convention checked by the test). */
	inline void process(float x, float& i, float& q) {
		float ya = x, yb = x;
		for (int k = 0; k < 4; k++) {
			ya = a[k].process(ya);
			yb = b[k].process(yb);
		}
		i = dly;
		dly = ya;
		q = yb;
	}
	void reset() {
		for (int k = 0; k < 4; k++) {
			a[k].x1 = a[k].x2 = a[k].y1 = a[k].y2 = 0.f;
			b[k].x1 = b[k].x2 = b[k].y1 = b[k].y2 = 0.f;
		}
		dly = 0.f;
	}
};

} // namespace fc
