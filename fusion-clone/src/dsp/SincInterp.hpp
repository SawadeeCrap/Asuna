// FusionClone DSP core — windowed-sinc fractional-position reader and a mirrored ring buffer.
//
// Used for (1) angular resampling of the input when extracting cycles, (2) reading clone
// waveforms from period tables, and (3) the time-domain fall-back read heads.
#pragma once
#include "Common.hpp"

// -DFC_NO_SIMD forces the portable scalar dot product (troubleshooting aid for a platform whose intrinsics misbehave)
#if defined(FC_NO_SIMD)
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#define FC_SIMD_SSE 1
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define FC_SIMD_NEON 1
#endif

namespace fc {

/** sum_t s[t] * (c0[t] + w * (c1[t] - c0[t])) for t < TAPS (TAPS a multiple of 4): the inner loop of every windowed-sinc read. Written with explicit
    4-wide SIMD (SSE2 / NEON) so that its speed does not depend on the compiler's auto-vectoriser or on -ffast-math (a scalar float reduction cannot
    be vectorised without reassociation): about 2.4x faster than the plain loop at -O2 and never slower with -O3. */
template <int TAPS>
inline float kernelDot(const float* s, const float* c0, const float* c1, float w) {
#if defined(FC_SIMD_SSE)
	const __m128 vw = _mm_set1_ps(w);
	__m128 acc0 = _mm_setzero_ps(), acc1 = _mm_setzero_ps();
	for (int t = 0; t < TAPS; t += 8) {
		const __m128 a0 = _mm_loadu_ps(c0 + t), b0 = _mm_loadu_ps(c1 + t);
		const __m128 a1 = _mm_loadu_ps(c0 + t + 4), b1 = _mm_loadu_ps(c1 + t + 4);
		acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(s + t), _mm_add_ps(a0, _mm_mul_ps(vw, _mm_sub_ps(b0, a0)))));
		acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(s + t + 4), _mm_add_ps(a1, _mm_mul_ps(vw, _mm_sub_ps(b1, a1)))));
	}
	__m128 acc = _mm_add_ps(acc0, acc1);
	acc = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
	acc = _mm_add_ss(acc, _mm_shuffle_ps(acc, acc, 1));
	return _mm_cvtss_f32(acc);
#elif defined(FC_SIMD_NEON)
	float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
	for (int t = 0; t < TAPS; t += 8) {
		const float32x4_t a0 = vld1q_f32(c0 + t), b0 = vld1q_f32(c1 + t);
		const float32x4_t a1 = vld1q_f32(c0 + t + 4), b1 = vld1q_f32(c1 + t + 4);
		acc0 = vfmaq_f32(acc0, vld1q_f32(s + t), vfmaq_n_f32(a0, vsubq_f32(b0, a0), w));
		acc1 = vfmaq_f32(acc1, vld1q_f32(s + t + 4), vfmaq_n_f32(a1, vsubq_f32(b1, a1), w));
	}
	return vaddvq_f32(vaddq_f32(acc0, acc1));
#else
	float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
	for (int t = 0; t < TAPS; t += 4) {
		a0 += s[t] * (c0[t] + w * (c1[t] - c0[t]));
		a1 += s[t + 1] * (c0[t + 1] + w * (c1[t + 1] - c0[t + 1]));
		a2 += s[t + 2] * (c0[t + 2] + w * (c1[t + 2] - c0[t + 2]));
		a3 += s[t + 3] * (c0[t + 3] + w * (c1[t + 3] - c0[t + 3]));
	}
	return (a0 + a1) + (a2 + a3);
#endif
}

/** Largest float below 1 as an upper bound for interpolation fractions: a double fraction within 3e-8 of the next integer rounds to exactly 1.0f
    when converted to float, which would address the phase row one past the end of the kernel table. */
inline float clampFrac(float fr) { return fr < 0.99999994f ? fr : 0.99999994f; }

/** Kaiser-windowed sinc kernel bank: kPhases+1 rows (the extra row lets us interpolate linearly
    between neighbouring phases without a wrap test), TAPS coefficients per row, each row normalised
    to unit DC gain. The cut-off is 0.92 * Nyquist: a 16-tap kernel is a compromise between passband
    flatness (<0.1 dB up to ~0.35 fs) and image rejection (>60 dB for content below ~0.3 fs, >45 dB at 0.4 fs). */
template <int TAPS>
struct SincKernel {
	static const int kPhases = 512;
	std::vector<float> tab;

	explicit SincKernel(double cutoff = 0.92, double beta = 8.6) {
		tab.assign((kPhases + 1) * TAPS, 0.f);
		const int half = TAPS / 2;
		for (int p = 0; p <= kPhases; p++) {
			double fr = (double) p / kPhases;
			double sum = 0;
			for (int t = 0; t < TAPS; t++) {
				double u = (t - (half - 1)) - fr; // distance from the read position to tap t
				double s = (std::fabs(u) < 1e-12) ? cutoff : std::sin(kPi * cutoff * u) / (kPi * u);
				double w = kaiser(u + half, (double) TAPS, beta);
				double v = s * w;
				tab[p * TAPS + t] = (float) v;
				sum += v;
			}
			for (int t = 0; t < TAPS; t++)
				tab[p * TAPS + t] = (float) (tab[p * TAPS + t] / sum);
		}
	}

	/** Interpolate at position (i0 + fr), reading x[-(half-1)] .. x[half] relative to i0. */
	inline float read(const float* xi0, float fr) const {
		// fr can round up to exactly 1.0f (a double fraction within 3e-8 of the next integer); the phase row p+1 would then lie one row past the table
		fr = clampFrac(fr);
		float fp = fr * (float) kPhases;
		int p = (int) fp;
		float w = fp - (float) p;
		const float* c0 = &tab[p * TAPS];
		const float* c1 = c0 + TAPS;
		return kernelDot<TAPS>(xi0 - (TAPS / 2 - 1), c0, c1, w);
	}
};

template <int TAPS>
inline const SincKernel<TAPS>& sharedSincKernel() {
	static const SincKernel<TAPS> k;
	return k;
}

/** Power-of-two ring buffer with a mirrored guard so sinc reads never need per-tap wrapping.
    Sample k (absolute index) lives at buf[k & mask]; the first `guard` slots are mirrored after
    the end of the buffer. */
class MirrorRing {
public:
	static const int kGuard = 40; // >= widest kernel (32 taps) + margin

	void alloc(int sizePow2) {
		size_ = sizePow2;
		mask_ = sizePow2 - 1;
		buf_.alloc((size_t) size_ + kGuard);
		w_ = 0;
	}
	void reset() {
		buf_.clear();
		w_ = 0;
	}
	inline void push(float x) {
		int i = (int) (w_ & (uint64_t) mask_);
		buf_[i] = x;
		if (i < kGuard)
			buf_[size_ + i] = x;
		w_++;
	}
	/** Number of samples pushed so far. */
	uint64_t count() const { return w_; }
	int size() const { return size_; }

	/** Sample with absolute index k (must be within the last `size` samples). */
	inline float at(uint64_t k) const { return buf_[(int) (k & (uint64_t) mask_)]; }

	/** Windowed-sinc read at absolute fractional position pos. Requires pos + TAPS/2 <= count()-1
	    and pos - TAPS/2 >= count() - size. */
	template <int TAPS>
	inline float readSinc(double pos) const {
		double fl = std::floor(pos);
		uint64_t i0 = (uint64_t) (int64_t) fl;
		float fr = (float) (pos - fl);
		fr = clampFrac(fr); // see SincKernel::read
		// pointer to sample i0; the kernel reads i0-(TAPS/2-1) .. i0+TAPS/2 which is contiguous thanks to
		// the mirror as long as we index from the (i0-(TAPS/2-1)) slot.
		int base = (int) ((i0 - (uint64_t) (TAPS / 2 - 1)) & (uint64_t) mask_);
		const float* s = buf_.data() + base; // points to first tap
		const SincKernel<TAPS>& K = sharedSincKernel<TAPS>();
		float fp = fr * (float) SincKernel<TAPS>::kPhases;
		int p = (int) fp;
		float w = fp - (float) p;
		const float* c0 = &K.tab[p * TAPS];
		return kernelDot<TAPS>(s, c0, c0 + TAPS, w);
	}

	/** Linear read (cheap, used by non-critical paths). */
	inline float readLinear(double pos) const {
		double fl = std::floor(pos);
		uint64_t i0 = (uint64_t) (int64_t) fl;
		float fr = (float) (pos - fl);
		float a = at(i0), b = at(i0 + 1);
		return a + (b - a) * fr;
	}

private:
	int size_ = 0, mask_ = 0;
	uint64_t w_ = 0;
	AlignedBuffer<float> buf_;
};

} // namespace fc
