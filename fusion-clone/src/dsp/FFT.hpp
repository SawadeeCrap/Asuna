// FusionClone DSP core — real FFT wrapper.
//
// Three interchangeable backends behind one interface and one *packed, ordered* spectrum layout
// (identical to pffft's "ordered" real layout, also documented by rack::dsp::RealFFT::rfft):
//
//     out[0]        = Re F(0)
//     out[1]        = Re F(N/2)
//     out[2k]       = Re F(k)     k = 1 .. N/2-1
//     out[2k+1]     = Im F(k)
//
// Backends (choose with a preprocessor define, default = built-in portable FFT):
//     FC_FFT_PFFFT  -> pffft.h directly with an owned work buffer (plugin build: Rack bundles pffft and exports it; also the harness)
//     FC_FFT_RACK   -> rack::dsp::RealFFT (same pffft underneath, but it lets pffft alloca() its work buffer on the stack)
//     (none)        -> built-in radix-2 FFT (used by unit tests / portable fallback)
//
// Forward transform is unscaled; inverse(forward(x)) == N * x.
#pragma once
#include "Common.hpp"

#if defined(FC_FFT_RACK)
#include <dsp/fft.hpp>
#elif defined(FC_FFT_PFFFT)
#include "pffft.h"
#endif

namespace fc {

#if defined(FC_FFT_RACK)

class RealFFT {
public:
	explicit RealFFT(int n) : n_(n), impl_((size_t) n) {}
	int size() const { return n_; }
	void forward(const float* in, float* out) { impl_.rfft(in, out); }
	void inverse(const float* in, float* out) { impl_.irfft(in, out); }
	static const char* backendName() { return "pffft (rack::dsp::RealFFT)"; }

private:
	int n_;
	rack::dsp::RealFFT impl_;
};

#elif defined(FC_FFT_PFFFT)

class RealFFT {
public:
	// pffft falls back to alloca() (N floats on the *stack*) when no work buffer is passed; that is unsafe on audio threads with small
	// stacks (macOS secondary threads default to 512 KB) and overflows for very large N, so every plan owns an aligned work buffer.
	explicit RealFFT(int n) : n_(n), setup_(pffft_new_setup(n, PFFFT_REAL)) { work_.alloc((size_t) n); }
	~RealFFT() { pffft_destroy_setup(setup_); }
	RealFFT(const RealFFT&) = delete;
	RealFFT& operator=(const RealFFT&) = delete;
	int size() const { return n_; }
	void forward(const float* in, float* out) { pffft_transform_ordered(setup_, in, out, work_.data(), PFFFT_FORWARD); }
	void inverse(const float* in, float* out) { pffft_transform_ordered(setup_, in, out, work_.data(), PFFFT_BACKWARD); }
	static const char* backendName() { return "pffft"; }

private:
	int n_;
	PFFFT_Setup* setup_;
	AlignedBuffer<float> work_;
};

#else

/** Portable radix-2 real FFT (N/2-point complex FFT + split step). Not the fastest, but exact and
    dependency free. N must be a power of two >= 8. */
class RealFFT {
public:
	explicit RealFFT(int n) : n_(n), h_(n / 2) {
		twRe_.resize(h_ / 2 > 0 ? h_ / 2 : 1);
		twIm_.resize(h_ / 2 > 0 ? h_ / 2 : 1);
		for (int k = 0; k < h_ / 2; k++) {
			twRe_[k] = (float) std::cos(-kTwoPi * k / h_);
			twIm_[k] = (float) std::sin(-kTwoPi * k / h_);
		}
		splitRe_.resize(h_ + 1);
		splitIm_.resize(h_ + 1);
		for (int k = 0; k <= h_; k++) {
			splitRe_[k] = (float) std::cos(-kTwoPi * k / n_);
			splitIm_[k] = (float) std::sin(-kTwoPi * k / n_);
		}
		rev_.resize(h_);
		int bits = ilog2(h_);
		for (int i = 0; i < h_; i++) {
			int r = 0;
			for (int b = 0; b < bits; b++)
				if (i & (1 << b))
					r |= 1 << (bits - 1 - b);
			rev_[i] = r;
		}
		zr_.assign(h_, 0.f);
		zi_.assign(h_, 0.f);
		tmpRe_.assign(h_, 0.f); // scratch for inverse(); allocated here so inverse() never allocates
		tmpIm_.assign(h_, 0.f);
	}
	int size() const { return n_; }
	static const char* backendName() { return "builtin radix-2"; }

	void forward(const float* in, float* out) {
		for (int m = 0; m < h_; m++) {
			zr_[rev_[m]] = in[2 * m];
			zi_[rev_[m]] = in[2 * m + 1];
		}
		fftInPlace(false);
		// Split: X[k] = (Z[k] + conj(Z[h-k]))/2 + W^k (Z[k] - conj(Z[h-k]))/(2i)
		float* zr = zr_.data();
		float* zi = zi_.data();
		out[0] = zr[0] + zi[0];
		out[1] = zr[0] - zi[0];
		for (int k = 1; k < h_; k++) {
			float ar = zr[k], ai = zi[k];
			float br = zr[h_ - k], bi = -zi[h_ - k]; // conj(Z[h-k])
			float er = 0.5f * (ar + br), ei = 0.5f * (ai + bi);
			float dr = 0.5f * (ar - br), di = 0.5f * (ai - bi);
			// (d)/(i) = -i*d = (di, -dr)
			float or_ = di, oi = -dr;
			float wr = splitRe_[k], wi = splitIm_[k];
			float pr = wr * or_ - wi * oi;
			float pi = wr * oi + wi * or_;
			out[2 * k] = er + pr;
			out[2 * k + 1] = ei + pi;
		}
	}

	void inverse(const float* in, float* out) {
		// Reverse of the split step: build Z[k] from X[k].
		float* zr = zr_.data();
		float* zi = zi_.data();
		// Z[0]
		float x0 = in[0], xh = in[1];
		float* tr = tmpRe_.data();
		float* ti = tmpIm_.data();
		tr[0] = 0.5f * (x0 + xh);
		ti[0] = 0.5f * (x0 - xh);
		for (int k = 1; k < h_; k++) {
			float xr = in[2 * k], xi = in[2 * k + 1];
			float yr = in[2 * (h_ - k)], yi = -in[2 * (h_ - k) + 1]; // conj(X[h-k])
			float er = 0.5f * (xr + yr), ei = 0.5f * (xi + yi);
			float dr = 0.5f * (xr - yr), di = 0.5f * (xi - yi);
			// O = conj(W^k) * d ; W^k = (wr, wi) -> conj = (wr, -wi)
			float wr = splitRe_[k], wi = -splitIm_[k];
			float or_ = wr * dr - wi * di;
			float oi = wr * di + wi * dr;
			// Z = E + i*O
			tr[k] = er - oi;
			ti[k] = ei + or_;
		}
		for (int m = 0; m < h_; m++) {
			zr[rev_[m]] = tr[m];
			zi[rev_[m]] = ti[m];
		}
		fftInPlace(true);
		for (int m = 0; m < h_; m++) {
			out[2 * m] = zr[m] * 2.f;
			out[2 * m + 1] = zi[m] * 2.f;
		}
	}

private:
	void fftInPlace(bool inverse) {
		float* zr = zr_.data();
		float* zi = zi_.data();
		for (int len = 2; len <= h_; len <<= 1) {
			int half = len >> 1;
			int step = h_ / len;
			for (int i = 0; i < h_; i += len) {
				for (int j = 0; j < half; j++) {
					float wr = twRe_[j * step];
					float wi = inverse ? -twIm_[j * step] : twIm_[j * step];
					int a = i + j, b = a + half;
					float tr = zr[b] * wr - zi[b] * wi;
					float ti = zr[b] * wi + zi[b] * wr;
					zr[b] = zr[a] - tr;
					zi[b] = zi[a] - ti;
					zr[a] += tr;
					zi[a] += ti;
				}
			}
		}
	}
	int n_, h_;
	std::vector<float> twRe_, twIm_, splitRe_, splitIm_, zr_, zi_, tmpRe_, tmpIm_;
	std::vector<int> rev_;
};

#endif

} // namespace fc
