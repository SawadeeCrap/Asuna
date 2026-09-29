// Measures the accuracy of the windowed-sinc fractional reader on sine waves and checks the SIMD dot product against a scalar reference.
#include "../src/dsp/SincInterp.hpp"
#include <cstdio>

int main() {
	fc::MirrorRing ring;
	ring.alloc(4096);
	int fails = 0;
	{
		// kernelDot (SSE2 / NEON / scalar, whichever this build selects) against a plain double precision loop on random data
		fc::Rng rng(99);
		double worst = 0.0;
		for (int trial = 0; trial < 2000; trial++) {
			float s[32], c0[32], c1[32];
			for (int t = 0; t < 32; t++) {
				s[t] = rng.bipolar();
				c0[t] = rng.bipolar();
				c1[t] = rng.bipolar();
			}
			const float w = rng.uniform();
			double ref16 = 0, ref32 = 0, mag = 1e-9;
			for (int t = 0; t < 32; t++) {
				const double term = (double) s[t] * ((double) c0[t] + (double) w * ((double) c1[t] - (double) c0[t]));
				if (t < 16) ref16 += term;
				ref32 += term;
				mag += std::fabs(term);
			}
			worst = std::max(worst, std::fabs(fc::kernelDot<16>(s, c0, c1, w) - ref16) / mag);
			worst = std::max(worst, std::fabs(fc::kernelDot<32>(s, c0, c1, w) - ref32) / mag);
		}
#if defined(FC_SIMD_SSE)
		const char* path = "SSE2";
#elif defined(FC_SIMD_NEON)
		const char* path = "NEON";
#else
		const char* path = "scalar";
#endif
		printf("kernelDot (%s) vs double precision reference: worst relative error %.2e\n", path, worst);
		if (worst > 1e-6) {
			printf("  FAIL\n");
			fails++;
		}
	}
	for (int TAPS : {16, 32}) {
		printf("TAPS=%d\n", TAPS);
		for (double f : {0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45}) {
			ring.reset();
			for (int n = 0; n < 4000; n++) ring.push((float) std::sin(fc::kTwoPi * f * n + 0.3));
			double maxErr = 0;
			for (int k = 0; k < 2000; k++) {
				double pos = 1000 + k * 0.37031;
				float v = TAPS == 16 ? ring.readSinc<16>(pos) : ring.readSinc<32>(pos);
				double ref = std::sin(fc::kTwoPi * f * pos + 0.3);
				maxErr = std::max(maxErr, std::fabs(v - ref));
			}
			double db = 20 * std::log10(maxErr + 1e-12);
			printf("  f=%.2f fs   max error %.1f dB\n", f, db);
			if (f <= 0.3 && db > (TAPS == 16 ? -55 : -70)) fails++;
		}
	}
	return fails;
}
