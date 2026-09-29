// Architecture comparison: every candidate clones ONE synthetic Fusion-like source into N voices and is scored against a bank of N genuinely
// independent oscillators (same tuning offsets). See docs/ARCHITECTURE.md for the interpretation.
#include "candidates/candidates_td.hpp"
#include <cstdio>
#include <cstring>
#include <chrono>

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

struct Row { std::string cand; double lvlDev, spreadRatio, ihDb, ripple, cv, rec, cpuX; bool ok; };

struct Scene {
	const char* name; SourceSpec spec; int N;
};


static std::vector<float> normalizeTo(std::vector<float> y, double targetRms, size_t a) {
	double r = rms(y, a);
	float g = (float) (targetRms / std::max(r, 1e-12));
	for (auto& v : y) v *= g;
	return y;
}

int main(int argc, char** argv) {
	std::string only = argc > 1 ? argv[1] : "";
	double sigma = 6.0; // cents (one-sigma) of the clone tolerance distribution: ~ SPREAD 0.5 at the default 20-cent range
	std::vector<Scene> scenes;
	auto add = [&](const char* nm, double f0, int N, std::function<void(SourceSpec&)> f) {
		SourceSpec s; s.fs = FS; s.f0 = f0; s.seed = 21; s.noiseDb = -85; s.seconds = f0 < 60 ? 16 : 14;
		f(s);
		scenes.push_back({nm, s, N});
	};
	add("saw 110 Hz N=2", 110, 2, [](SourceSpec&) {});
	add("saw 110 Hz N=16", 110, 16, [](SourceSpec&) {});
	add("saw 41.2 Hz N=4", 41.2, 4, [](SourceSpec&) {});
	add("saw 20 Hz N=4", 20, 4, [](SourceSpec&) {});
	add("saw 440 Hz N=8", 440, 8, [](SourceSpec&) {});
	add("saw 1760 Hz N=8", 1760, 8, [](SourceSpec&) {});
	add("saw+sub(.5) 110 Hz N=8", 110, 8, [](SourceSpec& s) { s.sub = 0.5; });
	add("saw+Doppler detune .3, 110 Hz N=8", 110, 8, [](SourceSpec& s) { s.detuneModel = DETUNE_DOPPLER; s.detune = 0.3; });
	add("pulse10%+tube .4, 220 Hz N=8", 220, 8, [](SourceSpec& s) { s.wSaw = 0; s.wPulse = 1; s.pulseWidth = 0.1; s.tube = 0.4; });
	if (argc > 2) { std::string want = argv[2]; std::vector<Scene> k; for (auto& s : scenes) if (std::string(s.name).find(want) != std::string::npos) k.push_back(s); scenes = k; }

	std::vector<std::unique_ptr<Candidate>> cands;
	cands.emplace_back(new PhaseVocoder(4096, 4, false, "A  phase vocoder (smb) N=4096"));
	cands.emplace_back(new PhaseVocoder(16384, 4, false, "A' phase vocoder (smb) N=16384"));
	cands.emplace_back(new PhaseVocoder(4096, 4, true, "B  PV + identity locking N=4096"));
	cands.emplace_back(new PhaseVocoder(16384, 4, true, "B' PV + identity locking N=16384"));
	cands.emplace_back(new SinusoidalMQ(4096));
	cands.emplace_back(new HarmonicHeterodyne(4096));
	cands.emplace_back(new FreqShifter());
	cands.emplace_back(new RotatingDelay(40));
	cands.emplace_back(new PeriodJump());
	cands.emplace_back(new Psola());
	cands.emplace_back(new MultiResPv());

	printf("%-40s %-46s %8s %8s %8s %8s %6s %6s\n", "scene", "candidate", "lvl|dB|", "spread x", "IH dB", "ripple", "CV", "recur");
	printf("%-40s %-46s %8s %8s %8s %8s %6s %6s\n", "", "(reference: N independent oscillators)", "", "", "", "", "", "");
	for (auto& sc : scenes) {
		if (!only.empty() && only != "all" && std::string(sc.name).find(only) == std::string::npos) continue;
		const SourceSpec& src = sc.spec;
		const int N = sc.N;
		std::vector<double> offs = offsetsFor(N - 1, sigma, 5);
		std::vector<float> x = renderSource(src);
		for (auto& v : x) v *= 4.f;
		SourceSpec sr = src; // reference = bank with the same offsets, source scaled the same way
		std::vector<float> ref = renderBank(sr, N, offs, 0.0, 77);
		for (auto& v : ref) v *= 4.f;
		double f0 = src.f0, unitP = (src.sub > 0 ? 2.0 : 1.0) * FS / src.f0;
		size_t a = (size_t) (4.0 * FS);
		double tgt = rms(ref, a);
		int nfft = src.f0 < 60 ? (1 << 17) : (1 << 17);
		Psd pref = welch(ref, FS, nfft, a);
		int K = 8;
		Clusters cref = harmonicClusters(pref, f0, K, 0.03);
		double ihRef = interHarmonicDb(pref, f0, 12);
		std::vector<double> profRef = harmonicProfileDb(pref, f0, 20);
		EnvStats evR = bandEnvelopeStats(ref, FS, f0 * 0.85, f0 * 1.15, a, ref.size());
		double recR = envRecurrence(ref, FS, f0 * 0.85, f0 * 1.15, a, ref.size());
		Psd pin = welch(x, FS, nfft, a);
		std::vector<double> profIn = harmonicProfileDb(pin, f0, 20);
		printf("%-40s %-46s %8s %8.1f %8.1f %8.2f %6.3f %6.3f\n", sc.name, "REFERENCE (independent bank)", "-", 1.0, ihRef, combRippleDb(profRef, profIn), evR.cv, recR);

		auto report = [&](const std::string& nm, std::vector<float> y, double ms) {
			y = normalizeTo(y, tgt, a);
			Psd py = welch(y, FS, nfft, a);
			Clusters cy = harmonicClusters(py, f0, K, 0.03);
			double lv = 0, sp = 0; int cnt = 0;
			double meanD = 0;
			for (int k = 0; k < K; k++) meanD += dbPow(cy.energy[k] / std::max(cref.energy[k], 1e-30));
			meanD /= K;
			for (int k = 0; k < K; k++) lv += std::fabs(dbPow(cy.energy[k] / std::max(cref.energy[k], 1e-30)) - meanD);
			std::vector<double> ratios;
			for (int k = 1; k < K; k++) if (cref.spread[k] > 0.05) ratios.push_back(cy.spread[k] / cref.spread[k]);
			std::sort(ratios.begin(), ratios.end());
			sp = ratios.empty() ? 0 : ratios[ratios.size() / 2];
			EnvStats ev = bandEnvelopeStats(y, FS, f0 * 0.85, f0 * 1.15, a, y.size());
			double rc = envRecurrence(y, FS, f0 * 0.85, f0 * 1.15, a, y.size());
			printf("%-40s %-46s %8.2f %8.2f %8.1f %8.2f %6.3f %6.3f  [%.1fs]\n", "", nm.c_str(), lv / K, sp, interHarmonicDb(py, f0, 12), combRippleDb(harmonicProfileDb(py, f0, 20), profIn), ev.cv, rc, ms / 1000.0);
			fflush(stdout);
		};
		// J : production engine
		{
			auto t0 = std::chrono::steady_clock::now();
			auto y = renderEngine(x, FS, offs, fc::QUALITY_BALANCED, 5);
			double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
			report("J  harmonic-locked oscillator bank (this work)", y, ms);
		}
		for (auto& c : cands) {
			CloneCtx ctx; ctx.fs = FS; ctx.f0 = f0; ctx.unitPeriod = unitP;
			if (src.f0 < 60 && (std::string(c->name()).find("N=16384") == std::string::npos && std::string(c->name()).find("multi-res") == std::string::npos)) {
				// long windows only at very low pitch would be needed; still run them all so the failure is visible
			}
			auto t0 = std::chrono::steady_clock::now();
			std::vector<float> sum = x;
			for (size_t i = 0; i < offs.size(); i++) {
				ctx.seed = 1000 + (uint32_t) i;
				std::vector<float> cl = c->renderClone(x, fc::centsToRatio(offs[i]), ctx);
				for (size_t k = 0; k < sum.size() && k < cl.size(); k++) sum[k] += cl[k];
			}
			double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
			report(c->name(), sum, ms);
		}
		printf("\n");
	}
	return 0;
}
