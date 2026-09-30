// Pitch tracker: no confident wrong answers, anywhere in its range.
//
// The engine acts on the tracker's verdicts: two agreeing estimates start an acquisition, three confident estimates that disagree with the
// tracked period drop the lock (and with it all the clones). test_pitch checks the *mean* error at a dozen pitches, which a tracker that is right
// 95 % of the time and confidently wrong the rest passes; the engine does not: it flickers. This test counts the wrong answers.
//
// Every lane of the tracker searches a limited range of lags. Just below a lane's lowest frequency the true dip of the normalised difference lies
// beyond the last lag the lane looks at, and the value at the last lag - still falling towards the dip, 0.87 .. 0.93 "confident" - used to be
// reported as a period 5 - 10 % too short; it beat the lane that saw the true period ("shortest period wins"). For a triangle wave at 46.6, 166
// and 591.5 Hz that was every estimate: the engine locked and dropped the lock 30 - 300 times in 2.6 s. A saw or a pulse showed it only every
// now and then, which is why it went unnoticed. So: waveforms x (the neighbourhood of every lane edge + a quarter-octave grid).
//
// The verdict is judged the way the engine judges it (Engine::onTrackerEstimate): an estimate is *consistent* with the true repeating unit if its
// period is within 5 % of it, or of half of it (3 %) or of twice it (8 %); a confident (> 0.9) estimate that is not consistent counts as wrong.
//
//   test_tracker            the sweep (about half a minute)
//   test_tracker quick      lane edges only
#include "../research/common/fusion_source.hpp"
#include "../src/dsp/PitchTracker.hpp"
#include <cstdio>
#include <cstring>

using namespace research;

struct Case {
	const char* name;
	double wSaw, wTri, wPulse, pw, sub, tube;
};

int main(int argc, char** argv) {
	const double fs = 48000.0;
	const bool quick = argc > 1 && !strcmp(argv[1], "quick");
	const Case cases[] = {
	    {"saw", 1, 0, 0, 0.5, 0.0, 0.0},
	    {"tri", 0, 1, 0, 0.5, 0.0, 0.0},
	    {"pulse 30%", 0, 0, 1, 0.3, 0.0, 0.0},
	    {"saw+pulse+tube", 0.9, 0, 0.3, 0.3, 0.0, 0.2},
	    {"saw+sub", 1, 0, 0, 0.5, 0.5, 0.0},
	};
	// lowest frequencies of the lanes 1 .. 4 (PitchTracker::prepare) and the fractions of them just below which a lane cannot see its dip
	const double laneLo[] = {50.0, 180.0, 640.0, 2200.0};
	std::vector<double> freqs;
	for (double e : laneLo)
		for (double b = 0.87; b < 1.005; b += 0.01) // the bad band is a few percent wide: 1 % steps
			freqs.push_back(e * b);
	if (!quick)
		for (double f = 32.0; f < 4200.0; f *= std::pow(2.0, 0.25))
			freqs.push_back(f);

	printf("%zu pitches x %zu waveforms; a cell is the share of the estimates that were confident and wrong, or '-' when there were none\n\n", freqs.size(),
	       sizeof(cases) / sizeof(cases[0]));
	int failures = 0;
	for (const Case& c : cases) {
		printf("%-15s", c.name);
		int worstCase = 0;
		double worstWrong = 0.0, worstOk = 100.0, worstAt = 0.0;
		for (double f0 : freqs) {
			SourceSpec s;
			s.fs = fs; s.f0 = f0; s.seconds = 1.3; s.seed = 11; s.noiseDb = -70; s.driftCents = 4.0;
			s.wSaw = c.wSaw; s.wTri = c.wTri; s.wPulse = c.wPulse; s.pulseWidth = c.pw; s.sub = c.sub; s.tube = c.tube;
			const std::vector<float> x = renderSource(s);
			const double unit = (c.sub > 0.05 ? 2.0 : 1.0) * fs / f0; // the true repeating unit in samples
			fc::PitchTracker pt;
			pt.prepare(fs);
			int total = 0, wrong = 0, okAny = 0;
			for (size_t i = 0; i < x.size(); i++) {
				if (!pt.push(x[i] * 2.5f) || i < (size_t) (0.5 * fs))
					continue;
				const fc::PitchTracker::Estimate& e = pt.estimate();
				total++;
				if (!e.valid)
					continue;
				const double ratio = e.period / unit;
				const bool consistent = std::fabs(ratio - 1.0) < 0.05 || std::fabs(ratio - 0.5) < 0.03 || std::fabs(ratio - 2.0) < 0.08;
				if (consistent)
					okAny++;
				else if (e.conf > 0.9f)
					wrong++;
			}
			const double wrongPct = 100.0 * wrong / std::max(1, total), okPct = 100.0 * okAny / std::max(1, total);
			if (wrongPct > worstWrong) {
				worstWrong = wrongPct;
				worstAt = f0;
			}
			worstOk = std::min(worstOk, okPct);
			if (wrongPct > 0.5 || okPct < 95.0) {
				failures++;
				worstCase++;
				printf("\n  FAIL: %s at %.2f Hz: %.2f %% of the estimates confident and wrong, %.1f %% right", c.name, f0, wrongPct, okPct);
			}
		}
		if (worstCase == 0)
			printf("  ok   (worst: %.2f %% wrong at %.1f Hz, at least %.1f %% right everywhere)\n", worstWrong, worstAt, worstOk);
		else
			printf("\n");
	}
	printf("\n%s (%d failing case%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
	return failures ? 1 : 0;
}
