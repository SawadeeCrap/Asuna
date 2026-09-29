// FusionClone DSP core — common helpers.
//
// The DSP core is framework independent (no Rack headers) and written in C++11 so the
// exact same code is used by the VCV Rack module, the research harness, the CLI and the
// unit tests.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace fc {

static const double kPi = 3.14159265358979323846;
static const double kTwoPi = 6.28318530717958647692;

static const int kMaxVoices = 16;

template <typename T>
inline T clampT(T x, T lo, T hi) {
	return x < lo ? lo : (x > hi ? hi : x);
}

inline float lerpf(float a, float b, float t) {
	return a + (b - a) * t;
}

inline float dbToLin(float db) {
	return std::pow(10.f, db * 0.05f);
}

inline float linToDb(float lin) {
	return 20.f * std::log10(std::max(lin, 1e-12f));
}

/** Cents -> frequency ratio. */
inline double centsToRatio(double cents) {
	return std::exp2(cents / 1200.0);
}

inline int nextPow2(int x) {
	int p = 1;
	while (p < x)
		p <<= 1;
	return p;
}

inline int ilog2(int x) {
	int l = 0;
	while ((1 << (l + 1)) <= x)
		l++;
	return l;
}

/** Flush denormals in recursive filter states (portable, branch is cheap). */
inline float flushDenormal(float x) {
	return (std::fabs(x) < 1e-20f) ? 0.f : x;
}
inline double flushDenormal(double x) {
	return (std::fabs(x) < 1e-200) ? 0.0 : x;
}

/** Fractional part in [0,1). */
inline double frac(double x) {
	return x - std::floor(x);
}

// ---------------------------------------------------------------------------------------
// Aligned allocation (needed by pffft and useful for SIMD friendly buffers).
// ---------------------------------------------------------------------------------------
inline void* alignedAlloc(size_t bytes, size_t align = 64) {
#if defined(_WIN32)
	return _aligned_malloc(bytes, align);
#else
	void* p = nullptr;
	if (posix_memalign(&p, align, bytes) != 0)
		return nullptr;
	return p;
#endif
}

inline void alignedFree(void* p) {
#if defined(_WIN32)
	_aligned_free(p);
#else
	std::free(p);
#endif
}

/** Fixed-size aligned float/double buffer. Allocation happens only in alloc() which must be
    called from non-realtime context (constructor / prepare()). */
template <typename T>
class AlignedBuffer {
public:
	AlignedBuffer() : p_(nullptr), n_(0) {}
	~AlignedBuffer() { release(); }
	AlignedBuffer(const AlignedBuffer&) = delete;
	AlignedBuffer& operator=(const AlignedBuffer&) = delete;

	void alloc(size_t n) {
		release();
		n_ = n;
		p_ = static_cast<T*>(alignedAlloc(std::max<size_t>(n, 1) * sizeof(T), 64));
		if (!p_)
			throw std::bad_alloc();
		clear();
	}
	void release() {
		if (p_)
			alignedFree(p_);
		p_ = nullptr;
		n_ = 0;
	}
	void clear() {
		if (p_)
			std::memset(p_, 0, n_ * sizeof(T));
	}
	T* data() { return p_; }
	const T* data() const { return p_; }
	size_t size() const { return n_; }
	T& operator[](size_t i) { return p_[i]; }
	const T& operator[](size_t i) const { return p_[i]; }

private:
	T* p_;
	size_t n_;
};

// ---------------------------------------------------------------------------------------
// Deterministic random numbers. Voice seeds must be reproducible across sessions/platforms,
// so only integer arithmetic is used to produce raw bits.
// ---------------------------------------------------------------------------------------
inline uint32_t hash32(uint32_t x) {
	// lowbias32 (Chris Wellons)
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

inline uint32_t hashCombine(uint32_t a, uint32_t b) {
	return hash32(a ^ hash32(b + 0x9e3779b9U + (a << 6) + (a >> 2)));
}

struct Rng {
	uint32_t s;
	Rng() : s(0x12345678u) {}
	explicit Rng(uint32_t seed) { reseed(seed); }
	void reseed(uint32_t seed) {
		s = hash32(seed ^ 0xA5A5A5A5u);
		if (s == 0)
			s = 0x9e3779b9u;
	}
	uint32_t nextU32() {
		// xorshift32
		uint32_t x = s;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		s = x;
		return x;
	}
	/** Uniform in [0,1). */
	float uniform() { return (nextU32() >> 8) * (1.f / 16777216.f); }
	/** Uniform in [-1,1). */
	float bipolar() { return uniform() * 2.f - 1.f; }
	/** Approximate standard normal (Irwin-Hall n=6, scaled). Bounded to ±3*sqrt(...) — good
	    enough for tolerance sampling and needs no libm. */
	float gauss() {
		float a = 0.f;
		for (int i = 0; i < 6; i++)
			a += uniform();
		return (a - 3.f) * 1.41421356f; // var of sum of 6 U = 0.5 -> scale sqrt(2)
	}
};

/** Inverse of the standard normal CDF (Acklam approximation), used to build stratified
    Gaussian detune sets. */
inline double normalQuantile(double p) {
	static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
	                           1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
	static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
	                           6.680131188771972e+01, -1.328068155288572e+01};
	static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
	                           -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
	static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
	                           3.754408661907416e+00};
	const double plow = 0.02425, phigh = 1 - plow;
	p = clampT(p, 1e-9, 1.0 - 1e-9);
	if (p < plow) {
		double q = std::sqrt(-2 * std::log(p));
		return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
		       ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
	}
	if (p > phigh) {
		double q = std::sqrt(-2 * std::log(1 - p));
		return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
		       ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
	}
	double q = p - 0.5, r = q * q;
	return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
	       (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// ---------------------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------------------
/** Zeroth order modified Bessel function of the first kind. */
inline double besselI0(double x) {
	double sum = 1.0, term = 1.0, hx = 0.5 * x;
	for (int k = 1; k < 60; k++) {
		term *= (hx / k);
		double t2 = term * term;
		sum += t2;
		if (t2 < 1e-18 * sum)
			break;
	}
	return sum;
}

inline double kaiser(double n, double N, double beta) {
	// n in [0,N], symmetric
	double r = 2.0 * n / N - 1.0;
	double a = 1.0 - r * r;
	if (a < 0)
		a = 0;
	return besselI0(beta * std::sqrt(a)) / besselI0(beta);
}

/** Periodic Hann window value at index i of length N (DFT-even) — exact zeros at ±2 bins. */
inline float hannPeriodic(int i, int N) {
	return 0.5f - 0.5f * std::cos((float) (kTwoPi * i / N));
}

} // namespace fc
