// Measures the accuracy of the windowed-sinc fractional reader on sine waves.
#include "../src/dsp/SincInterp.hpp"
#include <cstdio>

int main() {
	fc::MirrorRing ring;
	ring.alloc(4096);
	int fails = 0;
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
