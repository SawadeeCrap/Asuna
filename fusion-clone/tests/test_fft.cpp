// Verifies FFT backend against a naive DFT and checks the packed layout + inverse scaling.
#include "../src/dsp/FFT.hpp"
#include <cstdio>
#include <complex>

int main() {
	int fails = 0;
	printf("FFT backend: %s\n", fc::RealFFT::backendName());
	for (int N : {64, 128, 512, 4096}) {
		fc::AlignedBuffer<float> x, X, y;
		x.alloc(N); X.alloc(N); y.alloc(N);
		fc::Rng rng(N);
		for (int i = 0; i < N; i++) x[i] = rng.bipolar();
		fc::RealFFT fft(N);
		fft.forward(x.data(), X.data());
		// naive DFT
		double maxErr = 0, maxMag = 0;
		for (int k = 0; k <= N / 2; k++) {
			std::complex<double> s(0, 0);
			for (int n = 0; n < N; n++)
				s += (double) x[n] * std::polar(1.0, -2 * fc::kPi * k * n / N);
			double re, im;
			if (k == 0) { re = X[0]; im = 0; }
			else if (k == N / 2) { re = X[1]; im = 0; }
			else { re = X[2 * k]; im = X[2 * k + 1]; }
			maxErr = std::max(maxErr, std::abs(s - std::complex<double>(re, im)));
			maxMag = std::max(maxMag, std::abs(s));
		}
		fft.inverse(X.data(), y.data());
		double rtErr = 0;
		for (int i = 0; i < N; i++) rtErr = std::max(rtErr, (double) std::fabs(y[i] / N - x[i]));
		bool ok = maxErr < 1e-4 * maxMag && rtErr < 1e-5;
		printf("N=%5d  fwd maxErr=%.3e (rel %.2e)  roundtrip err=%.3e  %s\n", N, maxErr, maxErr / maxMag, rtErr, ok ? "OK" : "FAIL");
		if (!ok) fails++;
	}
	return fails;
}
