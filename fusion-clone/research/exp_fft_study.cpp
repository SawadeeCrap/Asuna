// FFT-size / analysis-window study for the period-synchronous analyser (docs/ARCHITECTURE.md section 7).
//
//   S1  accuracy of the extracted harmonic amplitudes against the analytic spectrum of a band-limited sawtooth (amplitude ~ 1/k),
//       for window length M in {2, 4} periods and grid size Nc_max in {1024 .. 8192}, from 20 Hz to C8. Harmonic j sits exactly on FFT bin
//       M*j, so with an integer number of periods in the window the error must vanish, except where the angle grid has fewer points per
//       period than the input has samples (Nc < P): then the resampler decimates without an anti-alias filter and content above the grid's
//       Nyquist folds into the harmonics. That is why ECO uses Nc_max = 4096 and not 2048.
//   S2  tracking of a 5 Hz +-15 cent vibrato by the frequency tracker alone (analyser, no engine): rms error of the tracker's frequency
//       prediction for "now" against the true instantaneous frequency, M = 2 versus M = 4.
//   S3  window lengths in milliseconds (the physical measurement latency of the analyser: M periods).
#include "common/fusion_source.hpp"
#include "../src/dsp/CycleAnalyzer.hpp"
#include <cstdio>

using namespace research;
static const double FS = 48000.0;

static double cents(double a, double b) { return 1200.0 * std::log2(a / b); }

int main() {
	printf("S1  error of the extracted harmonic amplitudes against the analytic sawtooth 1/j law, in dB, as  max(harmonics below 0.30 fs) / max(0.30 fs .. 0.46 fs);\n");
	printf("    1.5 s of tracking, spectral shape relative to the fundamental. '-' = the grid decimates the input (Nc_max < period in samples); J = harmonics extracted\n\n");
	const double f0s[] = {20.0, 30.0, 55.0, 110.0, 440.0, 1760.0, 3520.0, 4186.0};
	const int Ms[] = {2, 4};
	const int Ncs[] = {1024, 2048, 4096, 8192};
	printf("%-8s", "f0 (Hz)");
	for (int M : Ms)
		for (int Nc : Ncs)
			printf("  M=%d Nc<=%-5d   ", M, Nc);
	printf("\n");
	for (double f0 : f0s) {
		printf("%-8.1f", f0);
		// exact additive sawtooth with amplitude 1/j up to 0.46 fs (double precision phase recurrence): the analytic truth for the comparison
		const double P = FS / f0;
		const int nS = (int) (2.0 * FS);
		const int Jsrc = (int) std::floor(0.46 * FS / f0);
		std::vector<double> ar(Jsrc + 1), ai(Jsrc + 1), rr(Jsrc + 1), ri(Jsrc + 1);
		for (int j = 1; j <= Jsrc; j++) {
			const double w = 2.0 * fc::kPi * j * f0 / FS;
			ar[j] = 1.0; ai[j] = 0.0;
			rr[j] = std::cos(w); ri[j] = std::sin(w);
		}
		std::vector<float> x((size_t) nS);
		for (int n = 0; n < nS; n++) {
			double v = 0.0;
			for (int j = 1; j <= Jsrc; j++) {
				v += ai[j] / j;
				const double nr = ar[j] * rr[j] - ai[j] * ri[j];
				ai[j] = ar[j] * ri[j] + ai[j] * rr[j];
				ar[j] = nr;
				if ((n & 1023) == 1023) { const double g = 1.0 / std::sqrt(ar[j] * ar[j] + ai[j] * ai[j]); ar[j] *= g; ai[j] *= g; }
			}
			x[(size_t) n] = (float) (0.4 * v);
		}
		for (int M : Ms) {
			for (int Nc : Ncs) {
				fc::MirrorRing ring;
				ring.alloc(1 << 17);
				fc::CycleAnalyzer ca;
				fc::CycleAnalyzer::Config cfg;
				cfg.M = M; cfg.maxNc = Nc; cfg.taps = 32;
				fc::CycleAnalyzer::Config maxCfg = cfg;
				maxCfg.maxNc = 8192; // allocate for the biggest, run with the requested limit
				ca.prepare(FS, maxCfg);
				ca.configure(cfg);
				const size_t startAt = (size_t) (0.5 * FS);
				bool started = false;
				for (size_t i = 0; i < x.size(); i++) {
					ring.push(x[i]);
					if (i == startAt) {
						ca.start(P * fc::centsToRatio(-5.0), ring.count());
						started = true;
					}
					if (started)
						ca.step(ring);
				}
				const fc::HarmonicSet& hs = ca.set();
				// compare the spectral *shape* (each harmonic relative to the fundamental) with the 1/j law
				const double a1 = std::sqrt((double) hs.re[1] * hs.re[1] + (double) hs.im[1] * hs.im[1]);
				// below 0.30 fs the analysis resampler's 32-tap kernel is flat to < 0.001 dB; between 0.30 fs and the top of the band (0.46 fs) its
				// transition band droops by a few dB (the price of a compact kernel), so the two regions are reported separately
				double eLo = 0.0, eHi = 0.0;
				for (int j = 2; j <= hs.J; j++) {
					if (j > Jsrc) break; // the test signal is band-limited to 0.46 fs
					const double amp = std::sqrt((double) hs.re[j] * hs.re[j] + (double) hs.im[j] * hs.im[j]);
					const double e = std::fabs(20.0 * std::log10(std::max(amp / std::max(a1, 1e-30), 1e-12) * (double) j));
					if (j * f0 <= 0.30 * FS)
						eLo = std::max(eLo, e);
					else
						eHi = std::max(eHi, e);
				}
				char cell[48];
				const bool decimated = hs.Nc < P - 1.0 && hs.Nc >= cfg.maxNc;
				snprintf(cell, sizeof cell, "%s%.4f/%.1f J%d", decimated ? "- " : "", eLo, eHi, hs.J);
				printf("  %-16s", cell);
			}
		}
		printf("\n");
	}

	// S1b: the interpolation kernel of the analysis resampler (16 taps in ECO / BALANCED, 32 in HIGH / ULTRA) sets the top of the usable band
	printf("\nS1b harmonic amplitude error by analysis kernel (M = 2, Nc <= 4096), dB as max(below 0.30 fs) / max(0.30 fs .. 0.40 fs) / max(0.40 fs .. 0.46 fs)\n\n");
	printf("%-8s %-24s %-24s\n", "f0 (Hz)", "16 taps (ECO, BALANCED)", "32 taps (HIGH, ULTRA)");
	for (double f0 : {20.0, 55.0, 110.0, 440.0}) {
		printf("%-8.1f", f0);
		const double P = FS / f0;
		const int nS = (int) (2.0 * FS);
		const int Jsrc = (int) std::floor(0.46 * FS / f0);
		std::vector<double> ar(Jsrc + 1), ai(Jsrc + 1), rr(Jsrc + 1), ri(Jsrc + 1);
		for (int j = 1; j <= Jsrc; j++) {
			const double w = 2.0 * fc::kPi * j * f0 / FS;
			ar[j] = 1.0; ai[j] = 0.0;
			rr[j] = std::cos(w); ri[j] = std::sin(w);
		}
		std::vector<float> x((size_t) nS);
		for (int n = 0; n < nS; n++) {
			double v = 0.0;
			for (int j = 1; j <= Jsrc; j++) {
				v += ai[j] / j;
				const double nr = ar[j] * rr[j] - ai[j] * ri[j];
				ai[j] = ar[j] * ri[j] + ai[j] * rr[j];
				ar[j] = nr;
				if ((n & 1023) == 1023) { const double g = 1.0 / std::sqrt(ar[j] * ar[j] + ai[j] * ai[j]); ar[j] *= g; ai[j] *= g; }
			}
			x[(size_t) n] = (float) (0.4 * v);
		}
		for (int taps : {16, 32}) {
			fc::MirrorRing ring;
			ring.alloc(1 << 17);
			fc::CycleAnalyzer ca;
			fc::CycleAnalyzer::Config cfg;
			cfg.M = 2; cfg.maxNc = 4096; cfg.taps = taps;
			fc::CycleAnalyzer::Config maxCfg = cfg;
			maxCfg.M = 4; maxCfg.maxNc = 8192; maxCfg.taps = 32;
			ca.prepare(FS, maxCfg);
			ca.configure(cfg);
			for (int i = 0; i < nS; i++) {
				ring.push(x[(size_t) i]);
				if (i == (int) (0.5 * FS))
					ca.start(P * fc::centsToRatio(-5.0), ring.count());
				if (i > (int) (0.5 * FS))
					ca.step(ring);
			}
			const fc::HarmonicSet& hs = ca.set();
			const double a1 = std::sqrt((double) hs.re[1] * hs.re[1] + (double) hs.im[1] * hs.im[1]);
			double e[3] = {0, 0, 0};
			for (int j = 2; j <= hs.J && j <= Jsrc; j++) {
				const double amp = std::sqrt((double) hs.re[j] * hs.re[j] + (double) hs.im[j] * hs.im[j]);
				const double err = std::fabs(20.0 * std::log10(std::max(amp / std::max(a1, 1e-30), 1e-12) * (double) j));
				const double fr = j * f0 / FS;
				const int b = fr <= 0.30 ? 0 : (fr <= 0.40 ? 1 : 2);
				e[b] = std::max(e[b], err);
			}
			char cell[48];
			snprintf(cell, sizeof cell, "%.4f / %.3f / %.2f", e[0], e[1], e[2]);
			printf(" %-24s", cell);
		}
		printf("\n");
	}

	printf("\nS2  vibrato tracking (5 Hz, +-15 cents), tracker frequency prediction for 'now' vs truth: rms error in cents (analyser only)\n\n");
	printf("%-10s %10s %10s\n", "f0 (Hz)", "M=2", "M=4");
	for (double f0 : {41.2, 55.0, 110.0, 220.0, 440.0, 880.0}) {
		double res[2];
		for (int mi = 0; mi < 2; mi++) {
			const int M = Ms[mi];
			SourceSpec s;
			s.fs = FS; s.f0 = f0; s.seconds = 4.0; s.seed = 5; s.noiseDb = -80;
			s.f0Track.resize((size_t) (s.seconds * FS));
			std::vector<double> trueF(s.f0Track.size());
			for (size_t i = 0; i < s.f0Track.size(); i++) {
				trueF[i] = f0 * std::pow(2.0, 15.0 * std::sin(2.0 * fc::kPi * 5.0 * (double) i / FS) / 1200.0);
				s.f0Track[i] = trueF[i];
			}
			std::vector<float> x = renderSource(s);
			fc::MirrorRing ring;
			ring.alloc(1 << 17);
			fc::CycleAnalyzer ca;
			fc::CycleAnalyzer::Config cfg;
			cfg.M = M; cfg.maxNc = 8192; cfg.taps = 32;
			ca.prepare(FS, cfg);
			const size_t startAt = (size_t) (0.3 * FS);
			double sq = 0.0;
			long cnt = 0;
			for (size_t i = 0; i < x.size(); i++) {
				ring.push(x[i]);
				if (i == startAt)
					ca.start(FS / f0, ring.count());
				if (i > startAt) {
					ca.step(ring);
					if (i > (size_t) (2.0 * FS)) {
						const double e = cents(ca.omega() * FS, trueF[i]);
						sq += e * e;
						cnt++;
					}
				}
			}
			res[mi] = std::sqrt(sq / (double) std::max(1L, cnt));
		}
		printf("%-10.1f %10.2f %10.2f\n", f0, res[0], res[1]);
		fflush(stdout);
	}

	printf("\nS3  analysis window length M x period, in milliseconds (the analyser's measurement latency is about half a window plus one hop)\n\n");
	printf("%-10s %10s %10s\n", "f0 (Hz)", "M=2", "M=4");
	for (double f0 : {20.0, 41.2, 55.0, 110.0, 440.0, 1760.0})
		printf("%-10.1f %10.1f %10.1f\n", f0, 2000.0 / f0, 4000.0 / f0);
	return 0;
}
