// Artifact probes for the "zero-artifact goal" (clicks, discontinuities, zipper noise, aliasing), objective stand-ins for listening.
//
// The click detector uses a *pure tone* input. Every voice of the engine is then a plain (detuned) sine, so the output is a sum of at most
// K smooth sinusoids below f_max and its second difference is bounded by  w_max^2 * sum(A_i)  with  sum(A_i) <= RMS * sqrt(2K)  (Cauchy-Schwarz).
// A step in the output (a parameter that jumps, a voice that is switched on instantly, an unsmoothed crossfade) has a second difference of the
// order of the step itself, thousands of times larger than the bound; a 0.2 % step is already detected. The bound is evaluated on the RMS of the
// window under test.
//
//   A1  every control changed by a step at t = 2 s (and each at its worst-case direction): no click
//   A2  every continuous control swept in 10 ms steps (a knob turned at 100 Hz update rate): no zipper steps
//   A3  VOICES stepped through 1 .. 16 and back, quality switches, original-only toggle: no click
//   A4  note gate (2 ms attack), gate off, pitch step: the output never has a larger second difference than the input's own event shape allows
//   A5  aliasing: a sawtooth close to the top of the range, one detuned clone: energy that belongs to neither harmonic series
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wmissing-field-initializers" // Case entries without a `pre` hook
#endif
#include "../research/common/fusion_source.hpp"
#include "../research/common/metrics.hpp"
#include "../src/dsp/Engine.hpp"
#include <cstdio>
#include <functional>
#include <string>

using namespace research;
static const double FS = 48000.0;
static int fails = 0, total = 0;
#define CHECK(cond, ...) do { total++; if (!(cond)) { fails++; printf("  FAIL  "); printf(__VA_ARGS__); printf("\n"); } else { printf("  ok    "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static std::vector<float> tone(double f0, double seconds, double amp = 1.6) { // 8 V peak: a hot but realistic oscillator level
	std::vector<float> x((size_t) (seconds * FS));
	for (size_t i = 0; i < x.size(); i++)
		x[i] = (float) (amp * std::sin(2.0 * fc::kPi * f0 * (double) i / FS));
	return x;
}

/** max |y[n+1] - 2 y[n] + y[n-1]| over [i0, i1) */
static double maxD2(const std::vector<float>& y, size_t i0, size_t i1) {
	double m = 0;
	for (size_t i = std::max<size_t>(i0, 1); i + 1 < i1 && i + 1 < y.size(); i++)
		m = std::max(m, std::fabs((double) y[i + 1] - 2.0 * y[i] + y[i - 1]));
	return m;
}

static double rmsOf(const std::vector<float>& y, size_t i0, size_t i1) {
	double s = 0;
	for (size_t i = i0; i < i1; i++)
		s += (double) y[i] * y[i];
	return std::sqrt(s / (double) (i1 - i0));
}

/** click bound for a sum of at most K sinusoids below fmax (Hz) with the given RMS */
static double bound(double rms, int K, double fmax) {
	const double w = 2.0 * fc::kPi * fmax / FS;
	return w * w * rms * std::sqrt(2.0 * K);
}

/** the same, evaluated on the louder of the windows before and after the change (a level step makes either of them the relevant one) */
static double boundAround(const std::vector<float>& y, size_t i0, size_t i1, int K, double fmax) {
	const size_t pre0 = i0 > (size_t) (0.4 * FS) ? i0 - (size_t) (0.4 * FS) : 0;
	return bound(std::max(rmsOf(y, pre0, i0), rmsOf(y, i0, i1)), K, fmax);
}

struct Case {
	const char* name;
	std::function<void(fc::EngineParams&)> apply;
	int K; // number of sinusoids in the worst case
	std::function<void(fc::EngineParams&)> pre; // optional: changes the parameters the run starts with
};

static fc::EngineParams baseParams() {
	fc::EngineParams p;
	p.voices = 8; p.spread = 0.6f; p.drift = 0.3f; p.character = 0.f; p.harmonic = 0.f; p.phase = 0.6f; p.width = 0.f; p.mix = 1.f; p.summing = 0.f;
	return p;
}

int main() {
	const double f0 = 110.0;
	const auto x = tone(f0, 4.0);
	const size_t tEv = (size_t) (2.0 * FS), tEnd = tEv + (size_t) (0.4 * FS);

	printf("A1  step of every control at t = 2 s, pure 110 Hz tone: worst second difference / bound (must be < 1)\n");
	const Case cases[] = {
	    {"MIX 1 -> 0", [](fc::EngineParams& p) { p.mix = 0.f; }, 16},
	    {"MIX 0 -> 1 (from 0)", nullptr, 16, nullptr}, // handled below
	    {"VOICES 8 -> 16", [](fc::EngineParams& p) { p.voices = 16; }, 16},
	    {"VOICES 8 -> 1", [](fc::EngineParams& p) { p.voices = 1; }, 16},
	    {"VOICES 8 -> 2", [](fc::EngineParams& p) { p.voices = 2; }, 16},
	    {"SPREAD 0.6 -> 1", [](fc::EngineParams& p) { p.spread = 1.f; }, 16},
	    {"SPREAD 0.6 -> 0", [](fc::EngineParams& p) { p.spread = 0.f; }, 16},
	    {"DRIFT 0.3 -> 1", [](fc::EngineParams& p) { p.drift = 1.f; }, 16},
	    {"CHARACTER 0 -> 1 (HIGH)", [](fc::EngineParams& p) { p.character = 1.f; p.quality = fc::QUALITY_BALANCED; }, 16},
	    {"PHASE 0.6 -> 1", [](fc::EngineParams& p) { p.phase = 1.f; }, 16},
	    {"HARMONIC 0 -> 1", [](fc::EngineParams& p) { p.harmonic = 1.f; }, 16},
	    {"WIDTH 0 -> 1", [](fc::EngineParams& p) { p.width = 1.f; }, 16},
	    {"OUTPUT +12 dB", [](fc::EngineParams& p) { p.outputDb = 12.f; }, 16},
	    {"OUTPUT -24 dB", [](fc::EngineParams& p) { p.outputDb = -24.f; }, 16},
	    {"SUMMING 0 -> 1", [](fc::EngineParams& p) { p.summing = 1.f; }, 16},
	    {"LEVEL LAW 0.075 -> 0.5", [](fc::EngineParams& p) { p.levelLaw = 0.5f; }, 16},
	    {"ORIGINAL ONLY on", [](fc::EngineParams& p) { p.originalOnly = true; }, 16},
	    {"DETUNE RANGE 20 -> 60", [](fc::EngineParams& p) { p.detuneRangeCents = 60.f; }, 16},
	    {"FUSION layer on (RATIO)", [](fc::EngineParams& p) { p.algorithm = fc::ALGO_FUSION; p.fusionShift = 1.f; }, 48},
	    {"FUSION layer on (HZ)", [](fc::EngineParams& p) { p.algorithm = fc::ALGO_FUSION; p.fusionShift = 1.f; p.shiftMode = fc::SHIFT_HZ; }, 48},
	    {"FUSION shift model RATIO -> HZ", [](fc::EngineParams& p) { p.shiftMode = fc::SHIFT_HZ; }, 48,
	     [](fc::EngineParams& p) { p.algorithm = fc::ALGO_FUSION; p.fusionShift = 0.8f; p.shiftMode = fc::SHIFT_RATIO; }},
	    {"FUSION shift model HZ -> RATIO", [](fc::EngineParams& p) { p.shiftMode = fc::SHIFT_RATIO; }, 48,
	     [](fc::EngineParams& p) { p.algorithm = fc::ALGO_FUSION; p.fusionShift = 0.8f; p.shiftMode = fc::SHIFT_HZ; }},
	    {"FUSION layer off (was on)", [](fc::EngineParams& p) { p.algorithm = fc::ALGO_CLASSIC; }, 48,
	     [](fc::EngineParams& p) { p.algorithm = fc::ALGO_FUSION; p.fusionShift = 1.f; }},
	    {"FUSION layer off (was on, HZ)", [](fc::EngineParams& p) { p.algorithm = fc::ALGO_CLASSIC; }, 48,
	     [](fc::EngineParams& p) { p.algorithm = fc::ALGO_FUSION; p.fusionShift = 1.f; p.shiftMode = fc::SHIFT_HZ; }},
	};
	for (const Case& c : cases) {
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p = baseParams();
		bool fromZero = std::string(c.name).find("from 0") != std::string::npos;
		if (fromZero)
			p.mix = 0.f;
		if (c.pre)
			c.pre(p);
		e.setParams(p);
		std::vector<float> y(x.size());
		for (size_t i = 0; i < x.size(); i++) {
			if (i == tEv) {
				fc::EngineParams q = p;
				if (fromZero)
					q.mix = 1.f;
				else
					c.apply(q);
				e.setParams(q);
			}
			float l, r;
			e.process(x[i], l, r);
			y[i] = l;
		}
		double worst = maxD2(y, tEv, tEnd) / boundAround(y, tEv, tEnd, c.K, f0 * 1.10 + 9.0);
		worst = std::max(worst, 0.0);
		CHECK(worst < 1.0, "%-28s worst second difference = %.3f x bound", c.name, worst);
	}

	printf("A2  continuous controls swept in 10 ms steps (100 Hz update rate), 1 s ramps: worst second difference / bound\n");
	struct Sweep {
		const char* name;
		std::function<void(fc::EngineParams&, float)> apply; // u in 0..1
		int K;
	};
	const Sweep sweeps[] = {
	    {"MIX", [](fc::EngineParams& p, float u) { p.mix = 1.f - u; }, 16},
	    {"SPREAD", [](fc::EngineParams& p, float u) { p.spread = u; }, 16},
	    {"DRIFT", [](fc::EngineParams& p, float u) { p.drift = u; }, 16},
	    {"PHASE", [](fc::EngineParams& p, float u) { p.phase = u; }, 16},
	    {"HARMONIC", [](fc::EngineParams& p, float u) { p.harmonic = u; }, 16},
	    {"CHARACTER", [](fc::EngineParams& p, float u) { p.character = u; }, 16},
	    {"WIDTH", [](fc::EngineParams& p, float u) { p.width = u; }, 16},
	    {"OUTPUT", [](fc::EngineParams& p, float u) { p.outputDb = -24.f + 36.f * u; }, 16},
	    {"SUMMING", [](fc::EngineParams& p, float u) { p.summing = u; }, 16},
	    {"FUSION amount", [](fc::EngineParams& p, float u) { p.algorithm = fc::ALGO_FUSION; p.fusionShift = u; }, 48},
	};
	for (const Sweep& s : sweeps) {
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p = baseParams();
		e.setParams(p);
		std::vector<float> y(x.size());
		for (size_t i = 0; i < x.size(); i++) {
			if (i >= tEv && i < tEv + (size_t) FS && ((i - tEv) % 480) == 0) {
				fc::EngineParams q = p;
				s.apply(q, (float) (i - tEv) / (float) FS);
				e.setParams(q);
			}
			float l, r;
			e.process(x[i], l, r);
			y[i] = l;
		}
		const size_t t1 = tEv + (size_t) FS;
		double worst = maxD2(y, tEv, t1) / boundAround(y, tEv, t1, s.K, f0 * 1.10 + 9.0);
		CHECK(worst < 1.0, "%-28s worst second difference = %.3f x bound", s.name, worst);
	}

	printf("A3  VOICES stepped 1 -> 16 -> 1 (one step per 150 ms), quality switches, original-only toggles\n");
	{
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p = baseParams();
		e.setParams(p);
		const auto xl = tone(f0, 8.0);
		std::vector<float> y(xl.size());
		for (size_t i = 0; i < xl.size(); i++) {
			const size_t step = i / (size_t) (0.15 * FS);
			if (i % (size_t) (0.15 * FS) == 0 && step < 32) {
				fc::EngineParams q = p;
				q.voices = step < 16 ? 1 + (int) step : 16 - (int) (step - 16);
				e.setParams(q);
			}
			float l, r;
			e.process(xl[i], l, r);
			y[i] = l;
		}
		const size_t a = (size_t) (0.3 * FS), b = (size_t) (4.8 * FS);
		double worst = maxD2(y, a, b) / bound(rmsOf(y, a, b), 16, f0 * 1.10 + 9.0);
		CHECK(worst < 1.0, "VOICES 1..16..1 steps: worst second difference = %.3f x bound", worst);
	}
	{
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p = baseParams();
		e.setParams(p);
		const auto xl = tone(f0, 8.0);
		std::vector<float> y(xl.size());
		for (size_t i = 0; i < xl.size(); i++) {
			if (i == (size_t) (2.0 * FS) || i == (size_t) (4.0 * FS) || i == (size_t) (6.0 * FS)) {
				static const int seq[3] = {fc::QUALITY_HIGH, fc::QUALITY_ULTRA, fc::QUALITY_ECO};
				p.quality = seq[i / (size_t) (2.0 * FS) - 1];
				e.setParams(p);
			}
			float l, r;
			e.process(xl[i], l, r);
			y[i] = l;
		}
		double worst = maxD2(y, (size_t) (1.0 * FS), (size_t) (7.8 * FS)) / bound(rmsOf(y, (size_t) (1.0 * FS), (size_t) (7.8 * FS)), 16, f0 * 1.10 + 9.0);
		CHECK(worst < 1.0, "QUALITY switches (drop, re-lock, bloom): worst second difference = %.3f x bound", worst);
	}
	{
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p = baseParams();
		e.setParams(p);
		const auto xl = tone(f0, 6.0);
		std::vector<float> y(xl.size());
		for (size_t i = 0; i < xl.size(); i++) {
			if (i % (size_t) (0.5 * FS) == 0) {
				fc::EngineParams q = p;
				q.originalOnly = (i / (size_t) (0.5 * FS)) & 1;
				e.setParams(q);
			}
			float l, r;
			e.process(xl[i], l, r);
			y[i] = l;
		}
		double worst = maxD2(y, (size_t) (1.0 * FS), (size_t) (5.9 * FS)) / bound(rmsOf(y, (size_t) (1.0 * FS), (size_t) (5.9 * FS)), 16, f0 * 1.10 + 9.0);
		CHECK(worst < 1.0, "ORIGINAL ONLY toggled every 0.5 s: worst second difference = %.3f x bound", worst);
	}

	printf("A4  source events on a pure tone (the input's own second difference is the yardstick): output / input\n");
	{
		// gate with a 2 ms attack at 1.0 s, release (2 ms) at 2.5 s, then a phase-continuous pitch step (110 -> 165 Hz) at 3.5 s and a hard
		// note change (new note starts at phase 0, 165 -> 220 Hz: the input itself has a step there) at 4.5 s
		const size_t n = (size_t) (6.0 * FS);
		std::vector<float> xs(n);
		double ph = 0;
		for (size_t i = 0; i < n; i++) {
			const double t = (double) i / FS;
			const double f = t < 3.5 ? 110.0 : (t < 4.5 ? 165.0 : 220.0);
			ph += 2.0 * fc::kPi * f / FS;
			if (i == (size_t) (4.5 * FS))
				ph = 0.0;
			double g = 1.0;
			if (t < 1.0) g = 0.0;
			else if (t < 1.002) g = (t - 1.0) / 0.002;
			else if (t > 2.5 + 0.002 && t < 2.6) g = 0.0;
			else if (t > 2.5 && t < 2.6) g = 1.0 - (t - 2.5) / 0.002;
			else if (t >= 2.6 && t < 2.602) g = (t - 2.6) / 0.002;
			xs[i] = (float) (1.6 * g * std::sin(ph));
		}
		fc::Engine e;
		e.prepare(FS);
		e.setParams(baseParams());
		std::vector<float> y(n);
		for (size_t i = 0; i < n; i++) {
			float l, r;
			e.process(xs[i], l, r);
			y[i] = l;
		}
		struct Win { const char* name; double t0, t1; };
		const Win wins[] = {{"gate on  (2 ms attack)", 0.99, 1.30}, {"gate off (2 ms release)", 2.49, 2.60}, {"pitch step 110 -> 165 Hz", 3.49, 3.80},
		                    {"hard note change 165 -> 220 Hz", 4.49, 4.80}};
		for (const Win& w : wins) {
			const size_t i0 = (size_t) (w.t0 * FS), i1 = (size_t) (w.t1 * FS);
			const double din = maxD2(xs, i0, i1), dout = maxD2(y, i0, i1);
			// the original path contributes the input's own event shape at gain <= ~1.3 (level law), the clones fade over >= 3 ms:
			// the output's worst second difference may exceed the input's by the clones' smooth part only
			const double smooth = bound(rmsOf(y, i0, i1), 16, 220.0 * 1.1 + 9.0);
			CHECK(dout <= 1.6 * din + smooth, "%-30s input %.3e, output %.3e (allowed %.3e)", w.name, din, dout, 1.6 * din + smooth);
		}
	}

	printf("A5  aliasing: sawtooth 1700 Hz (harmonics to 22 kHz), original + one clone detuned +38 cents; energy off both harmonic series\n");
	{
		SourceSpec s;
		s.fs = FS; s.f0 = 1700.0; s.seconds = 6.0; s.seed = 5; s.noiseDb = -120; s.startPhase = 0.0;
		std::vector<float> xs = renderSource(s);
		for (auto& v : xs)
			v *= 4.f;
		fc::Engine e;
		e.prepare(FS);
		fc::EngineParams p;
		p.voices = 2; p.spread = 0.f; p.drift = 0.f; p.character = 0.f; p.harmonic = 0.f; p.phase = 1.f; p.width = 0.f; p.summing = 0.f;
		e.setParams(p);
		const float cents[2] = {0.f, 38.f};
		e.setDetuneOverrideCents(cents, 2); // voice 1 sits at +38 cents
		std::vector<float> y(xs.size());
		for (size_t i = 0; i < xs.size(); i++) {
			float l, r;
			e.process(xs[i], l, r);
			y[i] = l;
		}
		const size_t a = (size_t) (2.0 * FS);
		const int N = 1 << 17;
		Psd ps = welch(y, FS, N, a);
		const double bin = FS / N, r = std::pow(2.0, 38.0 / 1200.0);
		double eOn = 0, eOff = 0;
		for (size_t k = 1; k < ps.p.size(); k++) {
			const double f = (double) k * bin;
			if (f < 300.0 || f > 23500.0)
				continue;
			// distance (in bins) to the nearest harmonic of the original and of the clone
			const double h1 = std::fabs(f - std::round(f / 1700.0) * 1700.0) / bin;
			const double h2 = std::fabs(f - std::round(f / (1700.0 * r)) * 1700.0 * r) / bin;
			if (std::min(h1, h2) <= 40.0)
				eOn += ps.p[k];
			else
				eOff += ps.p[k];
		}
		const double dbOff = 10.0 * std::log10(eOff / std::max(eOn, 1e-30) + 1e-30);
		CHECK(dbOff < -60.0, "off-series energy (aliasing, intermodulation, noise) = %.1f dB re the harmonic energy (bound -60 dB)", dbOff);
	}

	printf("\n%d / %d checks passed\n", total - fails, total);
	return fails ? 1 : 0;
}
