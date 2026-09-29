// Lock-robustness matrix: every waveform family the Fusion VCO2 can produce (and a few hostile ones) x the full pitch range x quality modes.
// For a steady note the engine must (a) lock, (b) lock onto the right repeating unit, (c) stay locked (no spurious drops) and (d) keep the
// clones' fundamental within 0.05 cent of the source (3 cents for sources with a DETUNE stage: their own sidebands beat against the main
// oscillator, so the composite has no single exact period and the loop follows its power-weighted mean). This is the regression net for tracker / analyser / novelty-detector interactions
// (it would have caught: a chirp-feedback instability at M=4 and low pitch, and a tracker fooled by ripple in the flat part of a narrow pulse).
//
//   test_lock                 BALANCED, every case
//   test_lock full            all four quality modes (slow)
//   test_lock eco high ...    the named quality modes (eco, balanced, high, ultra)
#include "../research/common/fusion_source.hpp"
#include "../src/dsp/Engine.hpp"
#include <cstdio>
#include <cstring>
#include <strings.h>

using namespace research;

/** Similarity of two signals that ignores a time shift (integer lag from the FFT cross-correlation, fractional lag by a phase-ramp search), DC and
    everything outside 5 Hz .. 16 kHz: the peak of the normalised cross-correlation of the segments [i0, i1). 1.0 = identical waveform shape. */
static double fidelity(const std::vector<float>& x, const std::vector<float>& y, size_t i0, size_t i1, double fs) {
	const int N = 65536;
	const int n = (int) (i1 - i0);
	static fc::RealFFT fft(N);
	std::vector<float> a(N, 0.f), b(N, 0.f), X(N), Y(N), S(N, 0.f), r(N);
	for (int i = 0; i < n; i++) {
		const float w = fc::hannPeriodic(i, n);
		a[i] = x[i0 + i] * w;
		b[i] = y[i0 + i] * w;
	}
	fft.forward(a.data(), X.data());
	fft.forward(b.data(), Y.data());
	const int kLo = (int) std::ceil(5.0 * N / fs), kHi = (int) std::floor(std::min(16000.0, 0.45 * fs) * N / fs);
	double ex = 0, ey = 0;
	for (int k = kLo; k <= kHi; k++) {
		const double xr = X[2 * k], xi = X[2 * k + 1], yr = Y[2 * k], yi = Y[2 * k + 1];
		ex += xr * xr + xi * xi;
		ey += yr * yr + yi * yi;
		S[2 * k] = (float) (xr * yr + xi * yi);
		S[2 * k + 1] = (float) (xi * yr - xr * yi);
	}
	fft.inverse(S.data(), r.data());
	int bestLag = 0;
	float bestV = -1e30f;
	for (int t = -100; t <= 100; t++) {
		const float v = r[(t + N) % N];
		if (v > bestV) {
			bestV = v;
			bestLag = t;
		}
	}
	double best = -1.0;
	for (double d = -1.0; d <= 1.0001; d += 0.05) {
		const double th = 2.0 * fc::kPi * (bestLag + d) / N;
		double cr = 1.0, ci = 0.0, sum = 0.0;
		const double dr = std::cos(th), di = std::sin(th);
		// rotate to bin kLo first
		cr = std::cos(th * kLo);
		ci = std::sin(th * kLo);
		for (int k = kLo; k <= kHi; k++) {
			sum += (double) S[2 * k] * cr - (double) S[2 * k + 1] * ci;
			const double nr = cr * dr - ci * di;
			ci = cr * di + ci * dr;
			cr = nr;
		}
		best = std::max(best, sum / std::sqrt(ex * ey + 1e-300));
	}
	return best;
}

struct Case {
	const char* name;
	double wSaw, wTri, wPulse, wSine, pw, sub;
	int det;
	double detK, tube;
};

int main(int argc, char** argv) {
	const double fs = 48000.0;
	const Case cases[] = {
	    {"saw", 1, 0, 0, 0, 0.5, 0.0, DETUNE_NONE, 0, 0},
	    {"tri", 0, 1, 0, 0, 0.5, 0.0, DETUNE_NONE, 0, 0},
	    {"sine", 0, 0, 0, 1, 0.5, 0.0, DETUNE_NONE, 0, 0},
	    {"pulse 5%", 0, 0, 1, 0, 0.05, 0.0, DETUNE_NONE, 0, 0},
	    {"pulse 10%", 0, 0, 1, 0, 0.10, 0.0, DETUNE_NONE, 0, 0},
	    {"pulse 30%", 0, 0, 1, 0, 0.30, 0.0, DETUNE_NONE, 0, 0},
	    {"pulse 50%", 0, 0, 1, 0, 0.50, 0.0, DETUNE_NONE, 0, 0},
	    {"pulse 92%", 0, 0, 1, 0, 0.92, 0.0, DETUNE_NONE, 0, 0},
	    {"saw+sub .5", 1, 0, 0, 0, 0.5, 0.5, DETUNE_NONE, 0, 0},
	    {"saw+sub 1", 1, 0, 0, 0, 0.5, 1.0, DETUNE_NONE, 0, 0},
	    {"pulse+sub+tube", 0, 0, 1, 0, 0.3, 0.6, DETUNE_NONE, 0, 0.5},
	    {"saw+tube", 1, 0, 0, 0, 0.5, 0.0, DETUNE_NONE, 0, 0.7},
	    {"saw doppler .3", 1, 0, 0, 0, 0.5, 0.0, DETUNE_DOPPLER, 0.3, 0},
	    {"saw ssb .3", 1, 0, 0, 0, 0.5, 0.0, DETUNE_SSB, 0.3, 0},
	};
	const double freqs[] = {20, 30, 41.2, 55, 82.4, 110, 220, 440, 880, 1760, 3520, 4186};
	static const char* qn[] = {"ECO", "BALANCED", "HIGH", "ULTRA"};
	std::vector<int> qualities;
	for (int a = 1; a < argc; a++) {
		if (!strcmp(argv[a], "full")) {
			qualities = {0, 1, 2, 3};
			break;
		}
		for (int q = 0; q < 4; q++)
			if (!strcasecmp(argv[a], qn[q]))
				qualities.push_back(q);
	}
	if (qualities.empty())
		qualities = {1};

	int failures = 0, total = 0;
	for (int q : qualities) {
		printf("\n=== quality %s ===\ncase \\ f0 (Hz)   ", qn[q]);
		for (double f : freqs)
			printf("%8.1f", f);
		printf("\n");
		for (const Case& c : cases) {
			printf("%-16s", c.name);
			for (double f0 : freqs) {
				// the unit the clones must run at: one repeating unit (two main cycles when the sub oscillator is on)
				const double expectUnit = c.sub > 0.0 ? f0 * 0.5 : f0;
				SourceSpec s;
				s.fs = fs; s.f0 = f0; s.seconds = 2.6; s.seed = 3; s.noiseDb = -80;
				s.wSaw = c.wSaw; s.wTri = c.wTri; s.wPulse = c.wPulse; s.wSine = c.wSine; s.pulseWidth = c.pw; s.sub = c.sub;
				s.detuneModel = c.det; s.detune = c.detK; s.tube = c.tube;
				std::vector<float> x = renderSource(s);
				for (float& v : x)
					v *= 4.f;
				fc::Engine e;
				e.prepare(fs);
				fc::EngineParams p;
				p.voices = 8;
				p.quality = q;
				e.setParams(p);
				const size_t judgeFrom = (size_t) (1.4 * fs);
				double lockedTime = 0.0;
				int dropsAtJudge = 0;
				// clone fundamental error over the last 0.3 s: mean (bias) and largest excursion (jitter)
				double errSum = 0.0, errMaxAbs = 0.0;
				long errN = 0;
				for (size_t i = 0; i < x.size(); i++) {
					float l, r;
					e.process(x[i], l, r);
					if (i == judgeFrom)
						dropsAtJudge = e.dropCount(0) + e.dropCount(1) + e.dropCount(2) + e.dropCount(3) + e.dropCount(4);
					if (i >= judgeFrom && e.lockedNow())
						lockedTime += 1.0 / fs;
					if (i + (size_t) (0.3 * fs) >= x.size() && (i & 15) == 0 && e.cloneUnitFreq() > 0.0) {
						const double err = 1200.0 * std::log2(e.cloneUnitFreq() / expectUnit);
						errSum += err;
						errMaxAbs = std::max(errMaxAbs, std::fabs(err));
						errN++;
					}
				}
				const double unitBias = errN ? errSum / errN : 1e9;
				// Sources with a DETUNE stage have no single exact period: the sidebands beat against the main line and the composite's phase (hence the
				// tracked frequency, which the clones follow like the original does) wobbles by a few cents at the LFO rate: 3 cents on average, 8 cents
				// peak. Otherwise the mean must be within 0.05 cent and the excursions within 0.3 cent + 0.03 cent per 100 Hz (measurement noise grows
				// with the pitch).
				const double jitterTol = c.det != DETUNE_NONE ? 8.0 : 0.3 + 0.0003 * f0;
				// (bias: 0.05 cent + 0.01 cent per 100 Hz — the interpolation error of the angular resampler grows with the harmonic frequency)
				const bool biasFail = std::fabs(unitBias) > (c.det != DETUNE_NONE ? 3.0 : 0.05 + 0.0001 * f0), jitterFail = errMaxAbs > jitterTol;
				const int drops = e.dropCount(0) + e.dropCount(1) + e.dropCount(2) + e.dropCount(3) + e.dropCount(4);
				const double lockedFrac = lockedTime / ((double) (x.size() - judgeFrom) / fs);
				// Fidelity: with every divergence control at zero the clones are exact copies of the source, so the output must be a scaled copy of the
				// input. Sources with a DETUNE stage are excluded (their sidebands are not periodic, so the periodic model cannot reproduce them).
				double corr = 1.0;
				if (c.det == DETUNE_NONE) {
					fc::Engine e2;
					e2.prepare(fs);
					fc::EngineParams p2;
					p2.voices = 8; p2.quality = q;
					p2.spread = 0.f; p2.drift = 0.f; p2.character = 0.f; p2.phase = 0.f; p2.harmonic = 0.f; p2.summing = 0.f;
					e2.setParams(p2);
					std::vector<float> y(x.size());
					for (size_t i = 0; i < x.size(); i++) {
						float l, r;
						e2.process(x[i], l, r);
						y[i] = l;
					}
					corr = fidelity(x, y, (size_t) (1.6 * fs), (size_t) (2.5 * fs), fs);
				}
				total++;
				char cell[16];
				if (lockedFrac < 0.98)
					snprintf(cell, sizeof cell, "L%.2f", lockedFrac);
				else if (biasFail)
					snprintf(cell, sizeof cell, "U%+.2f", unitBias);
				else if (jitterFail)
					snprintf(cell, sizeof cell, "J%.2f", errMaxAbs);
				else if (drops - dropsAtJudge > 0)
					snprintf(cell, sizeof cell, "D%d", drops - dropsAtJudge);
				else if (corr < 0.99)
					snprintf(cell, sizeof cell, "C%.3f", corr);
				else
					snprintf(cell, sizeof cell, "ok");
				if (strcmp(cell, "ok") != 0)
					failures++;
				printf("%8s", cell);
				fflush(stdout);
			}
			printf("\n");
		}
	}
	printf("\n(L = locked fraction over the last 1.2 s, U = mean clone fundamental error in cents over the last 0.3 s, J = largest excursion, D = drops after settling, C = shape correlation of the divergence-free clone output with the input < 0.99: the clones do not\n reproduce the source)\n");
	printf("%d / %d cases failed\n", failures, total);
	return failures ? 1 : 0;
}
