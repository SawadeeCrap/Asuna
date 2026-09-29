// Transient experiments: (T1) gated note-on, (T2) pitch step. Each candidate clones one source into N voices; the reference is a bank of N
// independent oscillators that all receive the same gate / the same pitch step (as they would from a shared VCA / 1V-per-octave line).
#include "candidates/candidates_td.hpp"
#include <cstdio>

using namespace research;
static const double FS = 48000;

static std::vector<double> offsetsFor(int nClones, double sigmaCents, uint32_t seed) {
	std::vector<double> c;
	for (int i = 0; i < nClones; i++) {
		double p = fc::goldenSeq(seed, i + 1);
		p = p < 0.5 ? p * 0.8 : 0.2 + p * 0.8;
		c.push_back(sigmaCents * fc::normalQuantile(p));
	}
	return c;
}

// cycle-synchronous RMS envelope (window = one repeating unit) sampled every `hopMs`
static std::vector<float> cycleRms(const std::vector<float>& x, double P, double hopMs) {
	int win = std::max(8, (int) std::lround(P)), hop = std::max(1, (int) (hopMs * 1e-3 * FS));
	std::vector<float> e;
	std::vector<double> cs(x.size() + 1, 0.0);
	for (size_t i = 0; i < x.size(); i++) cs[i + 1] = cs[i] + (double) x[i] * x[i];
	for (size_t p = 0; p + win <= x.size(); p += hop) e.push_back((float) std::sqrt((cs[p + win] - cs[p]) / win));
	return e;
}

// fraction of the windowed energy NOT explained by a one-period-lag comb (0 = periodic with period P, ~2 = uncorrelated)
static double combResidual(const std::vector<float>& y, size_t t0, size_t t1, double P) {
	int lag = (int) std::lround(P);
	double num = 0, den = 1e-30;
	for (size_t n = std::max(t0, (size_t) lag); n < t1 && n < y.size(); n++) {
		double d = y[n] - y[n - lag];
		num += d * d;
		den += (double) y[n] * y[n];
	}
	return num / den;
}

int main() {
	const double sigma = 6.0;
	std::vector<std::unique_ptr<Candidate>> cands;
	cands.emplace_back(new PhaseVocoder(4096, 4, false, "A  phase vocoder N=4096"));
	cands.emplace_back(new PhaseVocoder(4096, 4, true, "B  PV + identity locking N=4096"));
	cands.emplace_back(new PhaseVocoder(16384, 4, true, "B' PV + identity locking N=16384"));
	cands.emplace_back(new SinusoidalMQ(4096));
	cands.emplace_back(new RotatingDelay(40));
	cands.emplace_back(new PeriodJump());
	cands.emplace_back(new Psola());
	cands.emplace_back(new MultiResPv());

	printf("=== T1: gated note-on (2 ms attack at t=1.0 s), N=8, clones start phase-coherent for offline candidates ===\n");
	printf("%-8s %-46s %9s %9s %9s %10s\n", "f0", "candidate", "preEcho dB", "rise ms", "overshoot", "envErr dB");
	for (double f0 : {110.0, 440.0, 41.2}) {
		const int N = 8;
		SourceSpec s; s.fs = FS; s.f0 = f0; s.seconds = 3; s.seed = 31; s.noiseDb = -85;
		s.ampEnv.resize((size_t) (3 * FS));
		for (size_t i = 0; i < s.ampEnv.size(); i++) { double t = i / FS; s.ampEnv[i] = (float) std::min(1.0, std::max(0.0, (t - 1.0) / 0.002)); }
		std::vector<double> offs = offsetsFor(N - 1, sigma, 5);
		auto x = renderSource(s); for (auto& v : x) v *= 4.f;
		auto ref = renderBank(s, N, offs, 0.0, 77); for (auto& v : ref) v *= 4.f;
		double P = FS / f0;
		auto eref = cycleRms(ref, P, 0.25);
		double steadyRef = rms(ref, (size_t) (2.0 * FS), (size_t) (2.9 * FS));
		auto stat = [&](const std::vector<float>& y, const char* nm, bool isRef) {
			double st = rms(y, (size_t) (2.0 * FS), (size_t) (2.9 * FS));
			auto e = cycleRms(y, P, 0.25);
			double eh = 0.25e-3;
			auto idx = [&](double t) { return (int) std::lround(t / eh); };
			double pre = 0; for (int i = idx(0.97); i < idx(0.999); i++) pre = std::max(pre, (double) e[i]);
			int t10 = -1, t90 = -1; double mx = 0;
			for (int i = idx(0.998); i < idx(1.25); i++) { double r = e[i] / st; mx = std::max(mx, r); if (t10 < 0 && r >= 0.1) t10 = i; if (t90 < 0 && r >= 0.9) t90 = i; }
			// envelope error vs reference over [-5, +60] ms, both normalised by their own steady RMS
			double s2 = 0; int c = 0;
			for (int i = idx(0.995); i < idx(1.06); i++) { double a = std::max(-60.0, db(e[i] / st)), b = std::max(-60.0, db(eref[i] / steadyRef)); s2 += (a - b) * (a - b); c++; }
			printf("%-8.1f %-46s %9.1f %9.2f %9.2f %10.2f\n", isRef ? f0 : f0, nm, db(pre / st), (t10 >= 0 && t90 >= 0) ? (t90 - t10) * 0.25 : -1.0, db(mx), std::sqrt(s2 / c));
		};
		stat(ref, "REFERENCE (independent bank)", true);
		stat(renderEngine(x, FS, offs, fc::QUALITY_BALANCED, 5, 1.0), "J  this work (bloom-in)", false);
		for (auto& c : cands) {
			if (f0 < 60 && std::string(c->name()).find("N=4096") != std::string::npos && std::string(c->name()).find("B'") == std::string::npos && false) continue;
			CloneCtx ctx; ctx.fs = FS; ctx.f0 = f0; ctx.unitPeriod = P; ctx.randomPhase = false; // best case for them: no extra lag
			std::vector<float> sum = x;
			for (size_t i = 0; i < offs.size(); i++) { ctx.seed = 1000 + (uint32_t) i; auto cl = c->renderClone(x, fc::centsToRatio(offs[i]), ctx); for (size_t k = 0; k < sum.size() && k < cl.size(); k++) sum[k] += cl[k]; }
			stat(sum, c->name(), false);
		}
		printf("\n");
	}

	printf("=== T2: pitch step (fifth up 110 -> 165 Hz, and semitone 110 -> 116.5 Hz at t=1.0 s), N=8 ===\n");
	printf("(settle = time until the output is periodic at the NEW period, comb residual < 0.25 over one period; ref = independent bank; lower is better)\n");
	printf("%-12s %-46s %11s %14s\n", "step", "candidate", "settle ms", "level dip dB");
	for (double f2 : {165.0, 116.5}) {
		const int N = 8; double f1 = 110.0;
		SourceSpec s; s.fs = FS; s.f0 = f1; s.seconds = 3; s.seed = 31; s.noiseDb = -85;
		s.f0Track.resize((size_t) (3 * FS)); for (size_t i = 0; i < s.f0Track.size(); i++) s.f0Track[i] = i < FS ? f1 : f2;
		std::vector<double> offs = offsetsFor(N - 1, sigma, 5);
		auto x = renderSource(s); for (auto& v : x) v *= 4.f;
		auto ref = renderBank(s, N, offs, 0.0, 77); for (auto& v : ref) v *= 4.f;
		double Pn = FS / f2;
		auto stat = [&](const std::vector<float>& y, const char* nm) {
			double st = rms(y, (size_t) (2.0 * FS), (size_t) (2.9 * FS));
			int win = (int) std::lround(2 * Pn);
			double settle = -1;
			for (size_t t = (size_t) FS; t < (size_t) (1.4 * FS); t += 24) {
				bool ok = true;
				for (size_t u = t; u < t + (size_t) (0.02 * FS) && ok; u += 96) if (combResidual(y, u, u + win, Pn) > 0.25) ok = false;
				if (ok) { settle = (double) (t - (size_t) FS) / FS * 1000.0; break; }
			}
			auto e = cycleRms(y, Pn, 0.25);
			double mn = 1e9; for (int i = (int) (1.0 / 0.25e-3); i < (int) (1.1 / 0.25e-3); i++) mn = std::min(mn, (double) e[i]);
			printf("%-12.1f %-46s %11.1f %14.1f\n", f2, nm, settle, db(mn / st));
		};
		stat(ref, "REFERENCE (independent bank)");
		stat(renderEngine(x, FS, offs, fc::QUALITY_BALANCED, 5, 1.0), "J  this work");
		for (auto& c : cands) {
			CloneCtx ctx; ctx.fs = FS; ctx.f0 = f1; ctx.unitPeriod = FS / f1; ctx.randomPhase = false;
			if (!c->supportsVarPitch() && std::string(c->name()).find("PSOLA") != std::string::npos) continue; // fixed-period marks cannot follow a step
			std::vector<float> sum = x;
			for (size_t i = 0; i < offs.size(); i++) { ctx.seed = 1000 + (uint32_t) i; auto cl = c->renderClone(x, fc::centsToRatio(offs[i]), ctx); for (size_t k = 0; k < sum.size() && k < cl.size(); k++) sum[k] += cl[k]; }
			stat(sum, c->name());
		}
		printf("\n");
	}
	return 0;
}
