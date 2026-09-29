// Beat-texture statistics: does a clone bank behave like N *independent* oscillators? Distribution over many random realisations of
// (a) envelope coefficient of variation and (b) envelope recurrence, in a high harmonic band where beats are fast enough to average.
// Compared: reference bank of independent oscillators, the engine (J), the rotating-delay chorus (F1), and an evenly-spaced "supersaw" bank.
#include "candidates/candidates_td.hpp"
#include <cstdio>
using namespace research;
static const double FS = 48000;

struct Stat { double m = 0, s = 0; int n = 0; std::vector<double> v; void add(double x) { v.push_back(x); }
	void fin() { n = v.size(); m = 0; for (double x : v) m += x; m /= std::max(1, n); s = 0; for (double x : v) s += (x - m) * (x - m); s = std::sqrt(s / std::max(1, n - 1)); } };

int main(int argc, char** argv) {
	int seeds = argc > 1 ? atoi(argv[1]) : 10;
	printf("%-30s %-6s %-40s %14s %14s\n", "scene", "N", "system", "CV mean+-sd", "recur mean+-sd");
	struct Sc { const char* n; double f0; int N; double sigma; };
	Sc scs[] = {{"saw 110 Hz, sigma 6c", 110, 8, 6}, {"saw 110 Hz, sigma 6c", 110, 16, 6}, {"saw 220 Hz, sigma 3c", 220, 8, 3}, {"saw 55 Hz, sigma 10c", 55, 8, 10}};
	for (auto& sc : scs) {
		int km = std::max(1, (int) std::ceil(3.0 / (sc.sigma * 5.78e-4 * sc.f0)));
		km = std::min(km, 40);
		double f1 = (km - 0.35) * sc.f0, f2 = (km + 0.35) * sc.f0;
		Stat cvRef, cvJ, cvF1, cvSS, rcRef, rcJ, rcF1, rcSS;
		for (int sd = 0; sd < seeds; sd++) {
			SourceSpec s; s.fs = FS; s.f0 = sc.f0; s.seconds = 20; s.seed = 100 + sd; s.noiseDb = -90;
			std::vector<double> offs;
			for (int i = 0; i < sc.N - 1; i++) { double p = fc::goldenSeq(300 + sd, i + 1); p = p < 0.5 ? p * 0.8 : 0.2 + p * 0.8; offs.push_back(sc.sigma * fc::normalQuantile(p)); }
			auto x = renderSource(s);
			for (auto& v : x) v *= 4.f;
			size_t a = (size_t) (5 * FS);
			auto ref = renderBank(s, sc.N, offs, 0.0, 900 + sd);
			auto yj = renderEngine(x, FS, offs, fc::QUALITY_BALANCED, 700 + sd);
			// F1: rotating delay chorus
			RotatingDelay f1c(40); CloneCtx ctx; ctx.fs = FS; ctx.f0 = sc.f0; ctx.unitPeriod = FS / sc.f0;
			std::vector<float> yc = x;
			for (int i = 0; i < sc.N - 1; i++) { ctx.seed = 50 + i + sd * 31; auto cl = f1c.renderClone(x, fc::centsToRatio(offs[i]), ctx); for (size_t k = 0; k < yc.size(); k++) yc[k] += cl[k]; }
			// supersaw: evenly spaced detunes, identical start phase-ish (retrigger), no drift
			std::vector<double> even; for (int i = 0; i < sc.N - 1; i++) even.push_back(sc.sigma * 2.4 * ((i + 1.0) / sc.N - 0.5) * 2.0);
			auto ss = renderBank(s, sc.N, even, 0.0, 900 + sd);
			cvRef.add(bandEnvelopeStats(ref, FS, f1, f2, a, ref.size()).cv);
			cvJ.add(bandEnvelopeStats(yj, FS, f1, f2, a, yj.size()).cv);
			cvF1.add(bandEnvelopeStats(yc, FS, f1, f2, a, yc.size()).cv);
			cvSS.add(bandEnvelopeStats(ss, FS, f1, f2, a, ss.size()).cv);
			rcRef.add(envRecurrence(ref, FS, f1, f2, a, ref.size()));
			rcJ.add(envRecurrence(yj, FS, f1, f2, a, yj.size()));
			rcF1.add(envRecurrence(yc, FS, f1, f2, a, yc.size()));
			rcSS.add(envRecurrence(ss, FS, f1, f2, a, ss.size()));
		}
		Stat* cv[4] = {&cvRef, &cvJ, &cvF1, &cvSS}; Stat* rc[4] = {&rcRef, &rcJ, &rcF1, &rcSS};
		const char* nm[4] = {"independent oscillators (reference)", "J  this work", "F1 rotating-delay chorus", "evenly spaced detunes (supersaw)"};
		for (int i = 0; i < 4; i++) { cv[i]->fin(); rc[i]->fin(); printf("%-30s %-6d %-40s %6.3f+-%5.3f %6.3f+-%5.3f   (band k=%d)\n", i == 0 ? sc.n : "", sc.N, nm[i], cv[i]->m, cv[i]->s, rc[i]->m, rc[i]->s, km); }
		printf("\n"); fflush(stdout);
	}
	return 0;
}
