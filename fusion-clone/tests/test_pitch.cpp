// Pitch tracker accuracy sweep on the synthetic Fusion-like source.
// The tracker only has to be good enough to seed the cycle analyser (its result is refined to ~0.01 cent afterwards, and octave errors that come
// from the fractional-lag artefact at very high pitch are corrected by the engine's odd-harmonic test): up to 1760 Hz it must report the repeating
// unit within 60 cents on average, with no octave error and no dropout; above that the sweep is informational only (tests/test_lock.cpp covers the
// full engine there, up to C8 = 4186 Hz).
#include "../research/common/fusion_source.hpp"
#include "../src/dsp/PitchTracker.hpp"
#include <cstdio>

using namespace research;

struct Case { const char* name; double wSaw, wTri, wPulse, wSine, pw, sub; int det; double detK; double tube; };

int main() {
	const double fs = 48000;
	std::vector<Case> cases = {
		{"saw",        1,0,0,0, 0.5, 0.0, DETUNE_NONE, 0, 0},
		{"tri",        0,1,0,0, 0.5, 0.0, DETUNE_NONE, 0, 0},
		{"pulse10%",   0,0,1,0, 0.10,0.0, DETUNE_NONE, 0, 0},
		{"sine",       0,0,0,1, 0.5, 0.0, DETUNE_NONE, 0, 0},
		{"saw+sub.5",  1,0,0,0, 0.5, 0.5, DETUNE_NONE, 0, 0},
		{"saw+sub1.0", 1,0,0,0, 0.5, 1.0, DETUNE_NONE, 0, 0},
		{"pulse+sub+tube", 0,0,1,0, 0.3, 0.6, DETUNE_NONE, 0, 0.5},
		{"saw+dopplerDet", 1,0,0,0, 0.5, 0.0, DETUNE_DOPPLER, 0.3, 0},
	};
	double freqs[] = {16.35, 20, 30, 41.2, 55, 82.4, 110, 220, 440, 880, 1760, 3520, 4186};
	printf("%-16s", "case \\ f0 (Hz)");
	for (double f : freqs) printf("%8.1f", f);
	printf("\n");
	int bad = 0;
	for (const Case& c : cases) {
		printf("%-16s", c.name);
		for (double f0 : freqs) {
			SourceSpec s; s.fs = fs; s.f0 = f0; s.seconds = 1.6; s.seed = 7;
			s.wSaw = c.wSaw; s.wTri = c.wTri; s.wPulse = c.wPulse; s.wSine = c.wSine; s.pulseWidth = c.pw; s.sub = c.sub;
			s.detuneModel = c.det; s.detune = c.detK; s.tube = c.tube; s.noiseDb = -70;
			auto x = renderSource(s);
			fc::PitchTracker pt; pt.prepare(fs);
			// feed at a moderate level (Rack: 5 V -> 1.0)
			double sumErrCents = 0; int cnt = 0; int multWrong = 0; int invalid = 0; int total = 0;
			int expectMult = c.sub > 0 ? 2 : 1;
			for (size_t i = 0; i < x.size(); i++) {
				if (pt.push(x[i] * 2.5f) && i > (size_t) (0.9 * fs)) {
					const auto& e = pt.estimate();
					total++;
					if (!e.valid) { invalid++; continue; }
					double unitHz = fs / e.period;
					double fFund = unitHz * expectMult; // compare the repeating unit (2T with a sub) with the expectation
					// compare fundamental (allow detune sidebands: measure against f0)
					double err = 1200.0 * std::log2(fFund / f0);
					if (std::fabs(err) > 1200) err = 1200 * (err > 0 ? 1 : -1);
					sumErrCents += std::fabs(err); cnt++;
					if (std::fabs(1200.0 * std::log2(unitHz * expectMult / f0)) > 300) multWrong++;
				}
			}
			double mean = cnt ? sumErrCents / cnt : 999;
			char buf[32];
			if (total == 0 || invalid > total / 2) snprintf(buf, sizeof buf, "  ---  ");
			else snprintf(buf, sizeof buf, "%5.1f%c", mean, multWrong > cnt / 4 ? '*' : ' ');
			if (f0 <= 1800.0 && (total == 0 || invalid > total / 2 || mean > 60 || multWrong > cnt / 4)) {
				bad++;
				fprintf(stderr, "FAIL: %s at %.1f Hz (mean error %.1f cents, %d octave errors, %d/%d invalid)\n", c.name, f0, mean, multWrong, invalid, total);
			}
			printf("%8s", buf);
		}
		printf("\n");
	}
	printf("(values = mean |pitch error| in cents over the last 0.7 s; '*' = repeating unit off by >300 cents; '---' = no estimate)\n");
	printf("%s (%d failure%s)\n", bad ? "FAILED" : "ALL PASSED", bad, bad == 1 ? "" : "s");
	return bad ? 1 : 0;
}
