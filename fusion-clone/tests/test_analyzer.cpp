// Cycle analyser: frequency tracker convergence from a deliberately wrong period, steady-state jitter.
#include "../research/common/fusion_source.hpp"
#include "../src/dsp/CycleAnalyzer.hpp"
#include <cstdio>

using namespace research;

static double cents(double a, double b) { return 1200.0 * std::log2(a / b); }

int main(int argc, char** argv) {
	const double fs = 48000;
	int M = argc > 1 ? atoi(argv[1]) : 2;
	printf("=== convergence from a +8 cent period error, M=%d ===\n", M);
	printf("%-18s %8s | %s\n", "signal", "unit Hz", "frequency error (cents) at t = 0.1 0.2 0.4 0.8 1.6 s  | steady-state jitter (cents rms, last 0.5s) | coherence");
	struct C { const char* name; double f0; double sub; double saw, tri, pulse; int det; double detK; };
	C cases[] = {
		{"saw 20Hz", 20, 0, 1,0,0, DETUNE_NONE, 0}, {"saw 41.2Hz", 41.2, 0, 1,0,0, DETUNE_NONE, 0},
		{"saw 110Hz", 110, 0, 1,0,0, DETUNE_NONE, 0}, {"saw 440Hz", 440, 0, 1,0,0, DETUNE_NONE, 0},
		{"saw 1760Hz", 1760, 0, 1,0,0, DETUNE_NONE, 0}, {"tri 110Hz", 110, 0, 0,1,0, DETUNE_NONE, 0},
		{"pulse20% 220Hz", 220, 0, 0,0,1, DETUNE_NONE, 0}, {"saw+sub 110Hz", 110, 0.6, 1,0,0, DETUNE_NONE, 0},
		{"saw dopplerDet .3", 110, 0, 1,0,0, DETUNE_DOPPLER, 0.3}, {"saw ssbDet .3", 110, 0, 1,0,0, DETUNE_SSB, 0.3},
	};
	for (const C& c : cases) {
		SourceSpec s; s.fs = fs; s.f0 = c.f0; s.seconds = 2.2; s.sub = c.sub; s.wSaw = c.saw; s.wTri = c.tri; s.wPulse = c.pulse; s.pulseWidth = 0.2;
		s.detuneModel = c.det; s.detune = c.detK; s.seed = 5; s.noiseDb = -80;
		auto x = renderSource(s);
		double unit = c.sub > 0 ? 2.0 : 1.0;
		double truePeriod = unit * fs / c.f0;
		fc::MirrorRing ring; ring.alloc(1 << 17);
		fc::CycleAnalyzer ca; fc::CycleAnalyzer::Config cfg; cfg.M = M; cfg.maxNc = 8192; cfg.taps = 32;
		ca.prepare(fs, cfg);
		size_t startAt = (size_t) (0.5 * fs); // give the ring some history first
		double cs[5] = {0}; int ci = 0; double jit = 0; int jn = 0; double cohAvg = 0;
		double times[5] = {0.1, 0.2, 0.4, 0.8, 1.6};
		for (size_t i = 0; i < x.size(); i++) {
			ring.push(x[i]);
			if (i == startAt) ca.start(truePeriod * fc::centsToRatio(-8.0), ring.count());
			if (i > startAt) {
				ca.step(ring);
				double t = (double) (i - startAt) / fs;
				if (ci < 5 && t >= times[ci]) { cs[ci++] = cents(ca.omega() * truePeriod, 1.0); }
				if (t > 1.2) { double e = cents(ca.omega() * truePeriod, 1.0); jit += e * e; jn++; cohAvg += ca.coherence(); }
			}
		}
		printf("%-18s %8.1f | %+6.2f %+6.2f %+6.2f %+6.2f %+6.2f | %6.3f | %.3f\n", c.name, fs / truePeriod, cs[0], cs[1], cs[2], cs[3], cs[4], jn ? std::sqrt(jit / jn) : 0, jn ? cohAvg / jn : 0);
	}
	return 0;
}
