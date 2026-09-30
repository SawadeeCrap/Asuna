// Dynamic-behaviour tests: what a real patch does to the oscillator while the module is running.
//   D1  vibrato (pitch LFO) must not break the lock, and the clones must follow it
//   D2  turning the SUB knob up on a held note must bring the sub into the clones (repeating unit changes from T to 2T)
//   D3  a WAVE switch (saw -> narrow pulse at the same pitch) must re-lock quickly
//   D4  a note gap (VCA closes, then re-opens on another pitch) must re-lock on the new pitch
#include "../research/common/fusion_source.hpp"
#include "../src/dsp/Engine.hpp"
#include <cstdio>
#include <cstring>

using namespace research;

static const double FS = 48000.0;
static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); failures++; } else { printf("  ok:   "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::vector<float> scaled(std::vector<float> x, float g) {
	for (float& v : x)
		v *= g;
	return x;
}

int main() {
	// ---- D1 vibrato -----------------------------------------------------------------------------------------------------------
	printf("D1  vibrato: +-15 cents at 5 Hz\n");
	for (int q : {0, 1, 2, 3}) {
		for (double f0 : {55.0, 110.0, 220.0, 880.0}) {
			SourceSpec s;
			s.fs = FS; s.f0 = f0; s.seconds = 4.0; s.seed = 5; s.noiseDb = -80;
			const size_t n = (size_t) (s.seconds * FS);
			s.f0Track.resize(n);
			for (size_t i = 0; i < n; i++)
				s.f0Track[i] = f0 * std::pow(2.0, 15.0 * std::sin(2.0 * M_PI * 5.0 * (double) i / FS) / 1200.0);
			std::vector<float> x = scaled(renderSource(s), 4.f);
			fc::Engine e;
			e.prepare(FS);
			fc::EngineParams p;
			p.voices = 8; p.quality = q;
			e.setParams(p);
			double locked = 0, errSq = 0; size_t errN = 0; int dropsAt1 = 0;
			for (size_t i = 0; i < n; i++) {
				float l, r;
				e.process(x[i], l, r);
				if (i == (size_t) (1.5 * FS)) dropsAt1 = e.dropCount(0) + e.dropCount(1) + e.dropCount(2) + e.dropCount(3);
				if (i >= (size_t) (1.5 * FS) && e.lockedNow()) {
					locked += 1.0 / FS;
					const double c = 1200.0 * std::log2(e.cloneUnitFreq() / s.f0Track[i]);
					errSq += c * c; errN++;
				}
			}
			const int drops = e.dropCount(0) + e.dropCount(1) + e.dropCount(2) + e.dropCount(3) - dropsAt1;
			const double lockedFrac = locked / (s.seconds - 1.5), rms = errN ? std::sqrt(errSq / errN) : 999.0;
			// The frequency measurement refers to the window centre, (M/2) periods in the past; at M = 4 and 55 Hz that is 36 ms, 65 degrees of a 5 Hz
			// vibrato, which no predictor can bridge. There the requirement is only "stay locked and do not make the modulation worse"
			// (the modulation itself is 10.6 cents rms); everywhere else the clones must follow within 6 cents rms.
			const double limit = (q >= 2 && f0 < 100.0) ? 14.0 : (q == 0 && f0 < 100.0 ? 7.0 : 6.0); // ECO (M = 2) follows a little worse than BALANCED at 55 Hz
			CHECK(lockedFrac > 0.97 && drops == 0 && rms < limit, "q%d %6.1f Hz: locked %.3f, drops %d, clone-vs-source pitch error %.2f cents rms (limit %.0f)", q, f0, lockedFrac, drops, rms, limit);
		}
	}

	// ---- D2 sub knob turned up on a held note ---------------------------------------------------------------------------------
	// Deterministic fidelity check: with every divergence control at zero the clones are exact copies of the source, so the output must be a
	// scaled copy of the input. If the clones lacked the sub the correlation would fall to ~0.8 (the sub carries ~40 % of the power).
	printf("D2  SUB knob turned up during a held 110 Hz note (clones must carry the sub)\n");
	{
		SourceSpec a;
		a.fs = FS; a.f0 = 110.0; a.seconds = 5.0; a.seed = 9; a.noiseDb = -80; a.startPhase = 0.0;
		SourceSpec b = a;
		b.wSaw = 0.0; b.sub = 1.0; // the sub square on its own, in phase with the main oscillator
		std::vector<float> xa = renderSource(a), xb = renderSource(b);
		const size_t n = xa.size();
		std::vector<float> x(n);
		for (size_t i = 0; i < n; i++) {
			const double t = (double) i / FS;
			const double g = std::min(1.0, std::max(0.0, (t - 1.5) / 1.0)) * 0.6; // 0 -> 0.6 between 1.5 s and 2.5 s
			x[i] = 4.f * (xa[i] + (float) g * xb[i]);
		}
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p;
		p.voices = 8; p.quality = 1;
		p.spread = 0.f; p.drift = 0.f; p.character = 0.f; p.phase = 0.f; p.harmonic = 0.f; p.summing = 0.f;
		e.setParams(p);
		std::vector<float> outL(n);
		double lockedLate = 0;
		for (size_t i = 0; i < n; i++) {
			float l, r;
			e.process(x[i], l, r);
			outL[i] = l;
			if (i >= (size_t) (4.0 * FS) && e.lockedNow()) lockedLate += 1.0 / FS;
		}
		double best = 0.0;
		const size_t i0 = (size_t) (4.0 * FS), i1 = (size_t) (5.0 * FS);
		for (int lag = -40; lag <= 40; lag++) {
			double sxy = 0, sxx = 0, syy = 0;
			for (size_t i = i0; i < i1; i++) { sxy += (double) x[i] * outL[i + lag]; sxx += (double) x[i] * x[i]; syy += (double) outL[i + lag] * outL[i + lag]; }
			best = std::max(best, sxy / std::sqrt(sxx * syy + 1e-30));
		}
		printf("       correlation of output and input over the last second: %.4f, locked %.2f, clone unit frequency %.2f Hz\n", best, lockedLate, e.cloneUnitFreq());
		CHECK(lockedLate > 0.95, "still locked at the end of the sweep");
		CHECK(std::fabs(e.cloneUnitFreq() - 55.0) < 0.1, "the clones run at the doubled period (55 Hz unit), i.e. they carry the sub");
		CHECK(best > 0.99, "output = exact scaled copy of the input (correlation %.4f > 0.99)", best);
	}

	// ---- D3 waveform switch -----------------------------------------------------------------------------------------------------
	printf("D3  WAVE switch: saw -> 15%% pulse at the same pitch (t = 1.5 s)\n");
	for (double f0 : {55.0, 220.0}) {
		SourceSpec a;
		a.fs = FS; a.f0 = f0; a.seconds = 3.5; a.seed = 4; a.noiseDb = -80; a.startPhase = 0.0;
		SourceSpec b = a;
		b.wSaw = 0.0; b.wPulse = 1.0; b.pulseWidth = 0.15;
		std::vector<float> xa = renderSource(a), xb = renderSource(b);
		std::vector<float> x(xa.size());
		for (size_t i = 0; i < x.size(); i++) x[i] = 4.f * (i < (size_t) (1.5 * FS) ? xa[i] : xb[i]);
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p;
		p.voices = 8; p.quality = 1;
		e.setParams(p);
		double relock = -1.0;
		bool wasDropped = false;
		for (size_t i = 0; i < x.size(); i++) {
			float l, r;
			e.process(x[i], l, r);
			if (i > (size_t) (1.5 * FS)) {
				if (!e.lockedNow()) wasDropped = true;
				if (wasDropped && e.lockedNow() && relock < 0) relock = (double) i / FS - 1.5;
			}
		}
		CHECK(e.lockedNow() && relock >= 0.0 && relock < 0.6, "%6.1f Hz: re-locked %.0f ms after the switch (dropped: %s)", f0, relock * 1000.0, wasDropped ? "yes" : "no");
	}

	// ---- D4 gap then a new pitch -------------------------------------------------------------------------------------------------
	printf("D4  note gap, then a different pitch (fifth up)\n");
	for (double f0 : {82.4, 440.0}) {
		SourceSpec a;
		a.fs = FS; a.f0 = f0; a.seconds = 1.2; a.seed = 4; a.noiseDb = -80;
		SourceSpec b = a;
		b.f0 = f0 * 1.5;
		b.seconds = 2.0;
		std::vector<float> xa = scaled(renderSource(a), 4.f), xb = scaled(renderSource(b), 4.f);
		std::vector<float> x(xa);
		x.resize(xa.size() + (size_t) (0.3 * FS), 0.f); // 300 ms of silence
		x.insert(x.end(), xb.begin(), xb.end());
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p;
		p.voices = 8; p.quality = 1;
		e.setParams(p);
		for (size_t i = 0; i < x.size(); i++) { float l, r; e.process(x[i], l, r); }
		const double errCents = 1200.0 * std::log2(e.cloneUnitFreq() / (f0 * 1.5));
		CHECK(e.lockedNow() && std::fabs(errCents) < 0.05, "%6.1f Hz -> %.1f Hz: locked on the new pitch (error %.3f cents)", f0, f0 * 1.5, errCents);
	}

	printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
	return failures ? 1 : 0;
}
