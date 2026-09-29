// Verifies the Hilbert pair: 90 degree phase difference and single-sideband frequency shifting behaviour.
#include "../src/dsp/Filters.hpp"
#include <cstdio>
int main() {
	int fails = 0;
	// 1. phase difference over frequency, measured with sinusoids
	double worst = 0; double worstF = 0;
	for (double f : {0.003, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49}) {
		fc::HilbertPair h;
		double si = 0, sq = 0, ci = 0, cq = 0; int n = 20000;
		for (int k = 0; k < n; k++) {
			float x = (float) std::sin(fc::kTwoPi * f * k), i, q;
			h.process(x, i, q);
			if (k > 5000) { double c = std::cos(fc::kTwoPi * f * k), s = std::sin(fc::kTwoPi * f * k);
				si += i * s; ci += i * c; sq += q * s; cq += q * c; }
		}
		double phI = std::atan2(ci, si), phQ = std::atan2(cq, sq);
		double d = (phQ - phI) * 180 / fc::kPi; while (d > 180) d -= 360; while (d < -180) d += 360;
		double err = std::fabs(std::fabs(d) - 90);
		if (f >= 0.003) { if (err > worst) { worst = err; worstF = f; } }
		printf("f=%.3f fs  branch phase difference = %7.2f deg\n", f, d);
	}
	printf("worst deviation from 90 deg: %.2f deg (at %.3f fs)\n", worst, worstF);
	if (worst > 3.5) fails++;
	// 2. SSB shifting: shift a 1 kHz tone by +37 Hz -> energy at 1037 Hz, image at 963 Hz suppressed
	{
		double fs = 48000; fc::HilbertPair h; int n = 96000; double delta = 37.0;
		std::vector<float> up(n), dn(n);
		for (int k = 0; k < n; k++) {
			float x = (float) std::sin(fc::kTwoPi * 1000.0 * k / fs), i, q; h.process(x, i, q);
			double th = fc::kTwoPi * delta * k / fs;
			up[k] = (float) (i * std::cos(th) - q * std::sin(th));
			dn[k] = (float) (i * std::cos(th) + q * std::sin(th));
		}
		auto tone = [&](const std::vector<float>& y, double f) { double c = 0, s = 0; for (int k = 20000; k < n; k++) { c += y[k] * std::cos(fc::kTwoPi * f * k / fs); s += y[k] * std::sin(fc::kTwoPi * f * k / fs); } return std::sqrt(c * c + s * s) / (n - 20000) * 2; };
		double a1037 = tone(up, 1037), a963 = tone(up, 963), b963 = tone(dn, 963), b1037 = tone(dn, 1037);
		printf("SSB 'up' : 1037 Hz %.4f  963 Hz %.5f -> image rejection %.1f dB\n", a1037, a963, 20 * std::log10(a963 / a1037 + 1e-12));
		printf("SSB 'down': 963 Hz %.4f 1037 Hz %.5f -> image rejection %.1f dB\n", b963, b1037, 20 * std::log10(b1037 / b963 + 1e-12));
		bool upOk = a1037 > 10 * a963, dnOk = b963 > 10 * b1037;
		printf("direction convention: %s\n", upOk && dnOk ? "i*cos - q*sin shifts UP (as coded in the engine)" : "REVERSED (swap signs in the engine)");
		if (!(upOk && dnOk) && !(a963 > 10 * a1037 && b1037 > 10 * b963)) fails++;
	}
	return fails;
}
