// Soak test: MINUTES of audio through the engine, with realistic inputs, and invariants checked all along.
//
// Everything else in tests/ runs seconds. A real session runs for hours, and a state variable that only misbehaves after N million samples (a
// float counter that stops counting at 2^24, an accumulator that is never wrapped, a running sum that drifts, a filter whose covariance grows
// while nothing is measured, an index computed from a value that has become NaN) shows up here and nowhere else.
//
//   test_soak [minutes = 10] [sample rate = 48000] [scenario = all] [quality = 3] [voices = 16]
//
//   scenarios  steady  one saw + sub + tube oscillator with a slow pitch drift, fixed controls
//              seq     a sequencer-like pattern (16 notes, portamento now and then, gate envelope), fixed controls
//              knobs   the steady source while the controls are turned at random (every ~0.2 s on average)
//              notes   random notes 0.2 - 2 s with random waveforms, gaps, DC and noise bursts, controls fixed
//              hw      hardware-like: 20 s notes (55 Hz - 1 kHz, sub on/off, waveform mixes incl. triangle and sine, tube) with pitch drift, mains
//                      hum, DC offset and slow tremolo added; the inter-harmonic energy of the output is compared with that of the input
//              chaos   everything at once, for hunting crashes (run it under the sanitizers, see tests/README): notes with vibrato and glides, noise
//                      at all levels, DC steps, hard-clipped tones, impulse trains, sweeps through the whole range, sub-audio waves, tones up to
//                      Nyquist, denormal-level and huge signals, NaN / Inf bursts, silence - with EVERY control (the expert ones, the random seed,
//                      the quality) changed at random, sometimes several times within a millisecond
//
// Every 5 s of audio one line is printed: input / output peak and rms, the fraction of the time the engine was locked, the pitch it reports, the
// inter-harmonic energy of the output (steady scenario: how "metallic" it is), and the number of times the engine's NaN/Inf safety nets fired.
// FAIL: a non-finite or absurdly large output sample, any safety-net hit, and (steady) a one-minute average output level that drifts by more than
// 2 dB or one-minute inter-harmonic energy that rises by more than 6 dB against the first minute.
#include "../research/common/fusion_source.hpp"
#include "../research/common/metrics.hpp"
#include "../src/dsp/Engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <pthread.h>

using namespace research;

static int failures = 0;
static void fail(const char* what) {
	printf("  FAIL: %s\n", what);
	failures++;
}

struct Window {
	double inPeak = 0, outPeak = 0, inSq = 0, outSq = 0;
	long n = 0, lockedN = 0, ctrlN = 0;
	double freqSum = 0;
};

static bool g_skipRender = false; // SOAK_START_CHUNK: draw the random parameters of the skipped notes but do not render them

static std::vector<float> renderNote(double fs, double f0, double seconds, uint32_t seed, double gain, double sub, double tube, double noiseDb, double drift, double wSaw,
                                     double wPulse, const std::vector<double>* track, const std::vector<float>* env, double wTri = 0.0, double wSine = 0.0) {
	if (g_skipRender)
		return std::vector<float>(1, 0.f);
	SourceSpec s;
	s.fs = fs;
	s.f0 = f0;
	s.seconds = seconds;
	s.seed = seed;
	s.noiseDb = noiseDb;
	s.sub = sub;
	s.tube = tube;
	s.driftCents = drift;
	s.wSaw = wSaw;
	s.wPulse = wPulse;
	s.wTri = wTri;
	s.wSine = wSine;
	s.pulseWidth = 0.3;
	if (track)
		s.f0Track = *track;
	if (env)
		s.ampEnv = *env;
	std::vector<float> x = renderSource(s);
	for (float& v : x)
		v *= (float) gain;
	return x;
}

static double lastUnitHz = 0.0; // hw scenario: repeating-unit frequency of the newest chunk

static std::vector<float> chaosSegment(double fs, fc::Rng& rng);

// next chunk of the input for a scenario; `chunk` counts from 0
static std::vector<float> nextChunk(const std::string& sc, double fs, int chunk, fc::Rng& rng) {
	if (sc == "chaos") {
		std::vector<float> x;
		while (x.size() < (size_t) (fs * 8.0)) {
			const std::vector<float> seg = chaosSegment(fs, rng);
			x.insert(x.end(), seg.begin(), seg.end());
		}
		return x;
	}
	if (sc == "steady" || sc == "knobs")
		return renderNote(fs, 110.0, 30.0, 100u + (uint32_t) chunk, 1.0, 0.4, 0.3, -70.0, 3.0, 1.0, 0.0, nullptr, nullptr);
	if (sc == "seq") {
		// 16 steps of 250 ms in a minor pentatonic, C2 .. C4, gate 60 %, a 40 ms glide into every fourth step
		static const int scale[] = {0, 3, 5, 7, 10, 12, 15, 17, 19, 22, 24};
		const int steps = 16;
		const double stepSec = 0.25;
		const size_t per = (size_t) (fs * stepSec);
		const size_t n = per * (size_t) steps;
		std::vector<double> track(n);
		std::vector<float> env(n);
		double prev = 65.406;
		for (int s = 0; s < steps; s++) {
			const int deg = scale[(int) (rng.uniform() * 10.999f)];
			const double f = 65.406 * std::pow(2.0, deg / 12.0);
			for (size_t i = 0; i < per; i++) {
				const size_t k = (size_t) s * per + i;
				const double t = (double) i / fs;
				const double glide = (s % 4 == 3) ? std::min(1.0, t / 0.04) : 1.0;
				track[k] = prev * std::pow(f / prev, glide);
				const double att = std::min(1.0, t / 0.01);
				const double rel = t < stepSec * 0.6 ? 1.0 : std::exp(-(t - stepSec * 0.6) / 0.05);
				env[k] = (float) (att * rel);
			}
			prev = f;
		}
		return renderNote(fs, 65.406, (double) n / fs, 200u + (uint32_t) chunk, 1.0, 0.3, 0.2, -75.0, 2.0, 0.7, 0.3, &track, &env);
	}
	if (sc == "hw") {
		double f0 = 55.0 * std::pow(2.0, 4.2 * (double) rng.uniform());
		const double sub = rng.uniform() < 0.5f ? 0.3 + 0.5 * (double) rng.uniform() : 0.0;
		// a quarter of the notes put the repeating unit 5 - 10 % below the lower edge of a lane of the pitch tracker (50, 180, 640 Hz): the pitches at
		// which the tracker used to be confidently wrong (see tests/test_tracker.cpp); a uniformly drawn pitch hits those bands only now and then
		if (rng.uniform() < 0.25f) {
			static const double edges[] = {50.0, 180.0, 640.0};
			const double unit = edges[(int) (rng.uniform() * 2.999f)] * (0.90 + 0.05 * (double) rng.uniform());
			f0 = sub > 0.05 ? 2.0 * unit : unit;
		}
		const double wSaw = 0.4 + 0.6 * (double) rng.uniform(), wPulse = (double) rng.uniform() * 0.6;
		// a triangle (40 % of the notes; on its own a third of the time) and a sine (20 %): the triangle is the wave that showed the pitch tracker's
		// weakness at three pitches, the saw only every now and then
		const double wTri = rng.uniform() < 0.4f ? 0.3 + (double) rng.uniform() : 0.0, wSine = rng.uniform() < 0.2f ? 0.5 * (double) rng.uniform() : 0.0;
		const bool triOnly = wTri > 0.0 && rng.uniform() < 0.33f;
		std::vector<float> x = renderNote(fs, f0, 20.0, rng.nextU32(), 0.5 + 1.5 * (double) rng.uniform(), sub, (double) rng.uniform() * 0.5,
		                                  -50.0 - 20.0 * (double) rng.uniform(), 4.0 + 6.0 * (double) rng.uniform(), triOnly ? 0.0 : wSaw, triOnly ? 0.0 : wPulse, nullptr, nullptr,
		                                  wTri, wSine);
		// mains hum up to 0.0012 (about -41 dB against these notes). Stronger hum close to the fundamental modulates the clones through the level
		// follower and, at M = 2, through the leakage of the analysis window: 0.0039 at 50 Hz on a 94.5 Hz note gives sidebands at (n + 1/2) f0, -44 dB
		// against the harmonics (-52 dB at 0.001, gone at 0.0004, and gone for notes above ~200 Hz); see docs/ARCHITECTURE.md 9.7
		const double hum = 0.0012 * (double) rng.uniform(), dc = 0.03 * (double) rng.bipolar(), trem = 0.1 * (double) rng.uniform();
		printf("      [note %d: f0 %.2f Hz, sub %.2f, saw %.2f, pulse %.2f, tri %.2f, sine %.2f, hum %.4f, dc %.3f, tremolo %.2f]\n", chunk, f0, sub, triOnly ? 0.0 : wSaw,
		       triOnly ? 0.0 : wPulse, wTri, wSine, hum, dc, trem);
		for (size_t i = 0; i < x.size(); i++) {
			const double t = (double) i / fs;
			x[i] = (float) ((double) x[i] * (1.0 - trem + trem * std::sin(6.2831853 * 0.13 * t)) + hum * std::sin(6.2831853 * 50.0 * t) + dc);
		}
		lastUnitHz = sub > 0.05 ? f0 * 0.5 : f0;
		return x;
	}
	// notes
	std::vector<float> x;
	while (x.size() < (size_t) (fs * 20.0)) {
		const double f0 = 30.0 * std::pow(2.0, 6.0 * (double) rng.uniform());
		const double sec = 0.2 + 1.8 * (double) rng.uniform();
		double wSaw = (double) rng.uniform(), wPulse = (double) rng.uniform();
		if (wSaw < 0.05 && wPulse < 0.05)
			wSaw = 1.0;
		std::vector<float> seg = renderNote(fs, f0, sec, rng.nextU32(), 0.3 + 3.0 * (double) rng.uniform(), rng.uniform() < 0.4f ? (double) rng.uniform() : 0.0,
		                                    rng.uniform() < 0.3f ? (double) rng.uniform() : 0.0, -90.0 + 30.0 * (double) rng.uniform(), 0.0, wSaw, wPulse, nullptr, nullptr);
		const float dice = rng.uniform();
		if (dice < 0.08f)
			std::fill(seg.begin(), seg.end(), 0.f); // silence
		else if (dice < 0.14f)
			for (float& v : seg)
				v = 2.f * rng.bipolar(); // noise burst
		else if (dice < 0.17f)
			std::fill(seg.begin(), seg.end(), 1.5f); // DC
		x.insert(x.end(), seg.begin(), seg.end());
	}
	return x;
}

// One hostile or unusual stretch of input for the chaos scenario (0.05 s .. a few s).
static std::vector<float> chaosSegment(double fs, fc::Rng& rng) {
	const double pi2 = 6.283185307179586;
	const double sec = 0.05 + 3.0 * (double) rng.uniform() * (double) rng.uniform();
	const size_t n = std::max<size_t>(16, (size_t) (sec * fs));
	std::vector<float> x(n, 0.f);
	switch ((int) (rng.uniform() * 14.f)) {
	case 0: // silence
		break;
	case 1: // white noise
		for (float& v : x)
			v = 2.f * rng.bipolar();
		break;
	case 2: { // noise at a random low level (1e-3 .. 10)
		const double a = 1e-3 * std::pow(10.0, 4.0 * (double) rng.uniform());
		for (float& v : x)
			v = (float) (a * (double) rng.bipolar());
		break;
	}
	case 3: { // a DC step
		const float dc = 20.f * rng.bipolar();
		for (float& v : x)
			v = dc;
		break;
	}
	case 4: { // a tone through a hard clipper (a square-ish wave with a random duty)
		const double f = 20.0 * std::pow(2.0, 9.0 * (double) rng.uniform()), g = 1.0 + 200.0 * (double) rng.uniform(), off = 2.0 * (double) rng.bipolar();
		double ph = rng.uniform();
		for (float& v : x) {
			v = (float) std::max(-2.0, std::min(2.0, g * std::sin(pi2 * ph) + off));
			ph += f / fs;
			ph -= std::floor(ph);
		}
		break;
	}
	case 5: { // an impulse train
		const double f = 5.0 * std::pow(2.0, 11.0 * (double) rng.uniform());
		double ph = 0.0;
		for (float& v : x) {
			ph += f / fs;
			if (ph >= 1.0) {
				ph -= 1.0;
				v = 3.f;
			}
		}
		break;
	}
	case 6: { // an exponential sweep through the whole range, up or down
		const double f0 = 10.0, f1 = 0.45 * fs;
		const bool up = rng.uniform() < 0.5f;
		double ph = 0.0;
		for (size_t i = 0; i < n; i++) {
			const double t = (double) i / (double) n, f = up ? f0 * std::pow(f1 / f0, t) : f1 * std::pow(f0 / f1, t);
			ph += f / fs;
			ph -= std::floor(ph);
			x[i] = (float) (0.8 * std::sin(pi2 * ph));
		}
		break;
	}
	case 7: { // a sub-audio wave (an LFO patched by mistake), unipolar or bipolar
		const double f = 0.3 + 20.0 * (double) rng.uniform();
		const bool uni = rng.uniform() < 0.5f;
		double ph = 0.0;
		for (float& v : x) {
			v = (float) (uni ? 1.0 + std::sin(pi2 * ph) : 2.0 * std::sin(pi2 * ph));
			ph += f / fs;
			ph -= std::floor(ph);
		}
		break;
	}
	case 8: { // a tone close to Nyquist
		const double f = (0.30 + 0.195 * (double) rng.uniform()) * fs;
		double ph = 0.0;
		for (float& v : x) {
			v = (float) std::sin(pi2 * ph);
			ph += f / fs;
			ph -= std::floor(ph);
		}
		break;
	}
	case 9: { // denormal-level signal
		const double a = std::pow(10.0, -25.0 - 20.0 * (double) rng.uniform());
		for (float& v : x)
			v = (float) (a * (double) rng.bipolar());
		break;
	}
	case 10: { // a huge tone (a patch cable into the wrong output: 50 x nominal level)
		const double f = 30.0 * std::pow(2.0, 7.0 * (double) rng.uniform());
		double ph = 0.0;
		for (float& v : x) {
			v = (float) (50.0 * std::sin(pi2 * ph));
			ph += f / fs;
			ph -= std::floor(ph);
		}
		break;
	}
	case 11: { // a tone with NaN / Inf samples in it (an upstream module that misbehaves)
		const double f = 60.0 * std::pow(2.0, 5.0 * (double) rng.uniform());
		double ph = 0.0;
		for (float& v : x) {
			v = (float) std::sin(pi2 * ph);
			ph += f / fs;
			ph -= std::floor(ph);
			const float d = rng.uniform();
			if (d < 0.001f)
				v = std::nanf("");
			else if (d < 0.002f)
				v = INFINITY;
			else if (d < 0.003f)
				v = -INFINITY;
		}
		break;
	}
	default: { // a note: random waveform, sub, tube, noise, with vibrato or a glide (up to two octaves) or steps
		const double f0 = 25.0 * std::pow(2.0, 7.5 * (double) rng.uniform());
		fc::Rng r2(rng.nextU32());
		SourceSpec sp;
		sp.fs = fs;
		sp.f0 = f0;
		sp.seconds = (double) n / fs;
		sp.seed = rng.nextU32();
		sp.noiseDb = -40.0 - 40.0 * (double) rng.uniform();
		sp.sub = rng.uniform() < 0.4f ? (double) rng.uniform() : 0.0;
		sp.tube = rng.uniform() < 0.5f ? (double) rng.uniform() : 0.0;
		sp.driftCents = 20.0 * (double) rng.uniform();
		sp.wSaw = rng.uniform() < 0.6f ? (double) rng.uniform() : 0.0;
		sp.wTri = rng.uniform() < 0.4f ? (double) rng.uniform() : 0.0;
		sp.wPulse = rng.uniform() < 0.5f ? (double) rng.uniform() : 0.0;
		sp.wSine = rng.uniform() < 0.2f ? (double) rng.uniform() : 0.0;
		if (sp.wSaw + sp.wTri + sp.wPulse + sp.wSine < 0.05)
			sp.wSaw = 1.0;
		sp.pulseWidth = 0.02 + 0.96 * (double) rng.uniform();
		const int motion = (int) (rng.uniform() * 4.f);
		if (motion > 0) {
			sp.f0Track.resize(n);
			const double depth = 60.0 * (double) rng.uniform(), rate = 0.1 + 12.0 * (double) rng.uniform(), oct = 2.0 * (double) rng.bipolar();
			const double stepAt = (double) n * (0.2 + 0.6 * (double) rng.uniform());
			for (size_t i = 0; i < n; i++) {
				double cents = 0.0;
				if (motion == 1)
					cents = depth * std::sin(pi2 * rate * (double) i / fs); // vibrato
				else if (motion == 2)
					cents = 1200.0 * oct * (double) i / (double) n; // a glide
				else
					cents = (double) i < stepAt ? 0.0 : 1200.0 * oct; // a pitch step
				sp.f0Track[i] = std::min(0.45 * fs, std::max(5.0, f0 * std::pow(2.0, cents / 1200.0)));
			}
		}
		x = renderSource(sp);
		const float g = (float) (0.2 + 4.0 * (double) r2.uniform());
		for (float& v : x)
			v *= g;
		break;
	}
	}
	return x;
}

// `wild`: every control, including the expert ones and the random seed (what a user with a hand on the knobs and a sequencer on the CV inputs does)
static void mutateWild(fc::EngineParams& p, fc::Rng& rng) {
	const int k = 1 + (int) (rng.uniform() * 4.f);
	for (int j = 0; j < k; j++) {
		switch ((int) (rng.uniform() * 22.f)) {
		case 0: p.voices = 1 + (int) (rng.uniform() * 16.f); break;
		case 1: p.spread = rng.uniform(); break;
		case 2: p.drift = rng.uniform(); break;
		case 3: p.character = rng.uniform(); break;
		case 4: p.phase = rng.uniform(); break;
		case 5: p.harmonic = rng.uniform(); break;
		case 6: p.width = rng.uniform(); break;
		case 7: p.mix = rng.uniform() < 0.2f ? 0.f : rng.uniform(); break;
		case 8: p.outputDb = -24.f + 36.f * rng.uniform(); break;
		case 9: p.algorithm = rng.uniform() < 0.5f ? fc::ALGO_CLASSIC : fc::ALGO_FUSION; break;
		case 10: p.fusionShift = rng.uniform(); break;
		case 11: p.shiftMode = rng.uniform() < 0.5f ? fc::SHIFT_RATIO : fc::SHIFT_HZ; break;
		case 12: p.summing = rng.uniform(); break;
		case 13: p.quality = (int) (rng.uniform() * 4.f); break;
		case 14: p.detuneRangeCents = 5.f + 55.f * rng.uniform(); break;
		case 15: p.spreadCurve = 0.8f + 2.2f * rng.uniform(); break;
		case 16: p.driftRate = rng.uniform(); break;
		case 17: p.driftCorrelation = rng.uniform(); break;
		case 18: p.levelLaw = 0.5f * rng.uniform(); break;
		case 19: p.originalOnly = rng.uniform() < 0.3f; break;
		case 20: p.seed = rng.nextU32(); break;
		default: p.voices = rng.uniform() < 0.5f ? 1 : 16; break;
		}
	}
}

static void mutateControls(fc::EngineParams& p, fc::Rng& rng, int quality) {
	const int k = 1 + (int) (rng.uniform() * 3.f);
	for (int j = 0; j < k; j++) {
		switch ((int) (rng.uniform() * 14.f)) {
		case 0: p.voices = 1 + (int) (rng.uniform() * 16.f); break;
		case 1: p.spread = rng.uniform(); break;
		case 2: p.drift = rng.uniform(); break;
		case 3: p.character = rng.uniform(); break;
		case 4: p.phase = rng.uniform(); break;
		case 5: p.harmonic = rng.uniform(); break;
		case 6: p.width = rng.uniform(); break;
		case 7: p.mix = rng.uniform() < 0.2f ? 0.f : rng.uniform(); break;
		case 8: p.outputDb = -12.f + 18.f * rng.uniform(); break;
		case 9: p.algorithm = rng.uniform() < 0.5f ? fc::ALGO_CLASSIC : fc::ALGO_FUSION; break;
		case 10: p.fusionShift = rng.uniform(); break;
		case 11: p.shiftMode = rng.uniform() < 0.5f ? fc::SHIFT_RATIO : fc::SHIFT_HZ; break;
		case 12: p.summing = rng.uniform(); break;
		default: p.quality = rng.uniform() < 0.1f ? (int) (rng.uniform() * 4.f) : quality; break;
		}
	}
}

static int realMain(int argc, char** argv) {
	const double minutes = argc > 1 ? atof(argv[1]) : 10.0;
	const double fs = argc > 2 ? atof(argv[2]) : 48000.0;
	const std::string only = argc > 3 ? argv[3] : "all";
	const int quality = argc > 4 ? atoi(argv[4]) : 3;
	const int voices = argc > 5 ? atoi(argv[5]) : 16;
	const size_t total = (size_t) (minutes * 60.0 * fs);
	const size_t win = (size_t) (5.0 * fs);
	printf("soak test: %.1f min of audio at %.0f Hz, quality %d, %d voices (2^24 samples = %.2f min at this rate)\n", minutes, fs, quality, voices,
	       16777216.0 / fs / 60.0);

	std::vector<std::string> list;
	if (only == "all")
		list = {"steady", "seq", "knobs", "notes", "hw", "chaos"};
	else
		list.push_back(only);

	for (const std::string& sc : list) {
		printf("\n=== scenario %s\n", sc.c_str());
		printf("  t(min)  in-peak  out-peak  in-rms  out-rms  gain(dB)  locked  pitch(Hz)  IH(dB)  hits\n");
		fflush(stdout);
		fc::Rng rng(12345u + (uint32_t) sc.size() + (getenv("SOAK_SEED") ? (uint32_t) atoi(getenv("SOAK_SEED")) * 7919u : 0u)); // SOAK_SEED: a different draw of the random scenarios
		fc::Engine e;
		e.prepare(fs);
		fc::EngineParams p;
		p.voices = voices;
		p.quality = quality;
		p.algorithm = fc::ALGO_FUSION;
		p.seed = 0x5EED1234u;
		e.setParams(p);

		std::vector<float> chunk;
		size_t pos = 0, done = 0;
		int chunkIdx = 0;
		if (getenv("SOAK_START_CHUNK")) { // replay: skip to a chunk (one note per chunk in the hw scenario) without rendering the ones before it
			g_skipRender = true;
			while (chunkIdx < atoi(getenv("SOAK_START_CHUNK")))
				nextChunk(sc, fs, chunkIdx++, rng);
			g_skipRender = false;
		}
		// SOAK_TRACE_FROM / SOAK_TRACE_TO (seconds of audio): the engine's state is printed every 0.1 s in between, and with SOAK_DUMP=file the
		// input and the output of that interval are written as raw float32 triples (input, left, right) for analysis with other tools
		const double trFrom = getenv("SOAK_TRACE_FROM") ? atof(getenv("SOAK_TRACE_FROM")) : -1.0;
		const double trTo = getenv("SOAK_TRACE_TO") ? atof(getenv("SOAK_TRACE_TO")) : 1e18;
		FILE* dumpF = (trFrom >= 0.0 && getenv("SOAK_DUMP")) ? fopen(getenv("SOAK_DUMP"), "wb") : nullptr;
		uint32_t lastEstSeq = 0;
		int lastDrops = 0;
		float recentPeak = 0.f; // input peak over the last second or so
		if (getenv("SOAK_PRE_SILENCE")) { // let the engine's clock run (and every counter and schedule phase with it) before the scenario starts
			const size_t n = (size_t) (atof(getenv("SOAK_PRE_SILENCE")) * fs);
			float l = 0.f, r = 0.f;
			for (size_t i = 0; i < n; i++)
				e.process(0.f, l, r);
		}
		Window w;
		std::vector<float> ring((size_t) (4.0 * fs), 0.f); // last 4 s of the mono output for the inter-harmonic metric
		std::vector<float> inRing((size_t) (4.0 * fs), 0.f); // and of the input (hw scenario)
		size_t ringPos = 0;
		double unitHz = 0.0;
		double gain0 = 0, ih0 = 0, blkIn = 0, blkOut = 0, blkIh = 0;
		long blkN = 0;
		int windows = 0;
		bool bad = false;
		while (done < total && !bad) {
			if (pos >= chunk.size()) {
				chunk = nextChunk(sc, fs, chunkIdx++, rng);
				pos = 0;
				unitHz = lastUnitHz;
			}
			if (sc == "knobs" && rng.uniform() < 1.f / (0.2f * (float) fs)) {
				mutateControls(p, rng, quality);
				e.setParams(p);
			}
			if (sc == "chaos" && rng.uniform() < 1.f / (0.05f * (float) fs)) { // every 50 ms on average, in bursts now and then
				const int burst = rng.uniform() < 0.1f ? 1 + (int) (rng.uniform() * 8.f) : 1;
				for (int b = 0; b < burst; b++) {
					mutateWild(p, rng);
					e.setParams(p);
				}
			}
			const float xRaw = chunk[pos++];
			const float x = std::isfinite(xRaw) ? xRaw : 0.f; // (the engine is fed the raw value below; statistics use the finite one)
			float l = 0, r = 0;
			e.process(xRaw, l, r);
			done++;
			// chaos feeds inputs up to 50 (250 V) and trims the output up to +12 dB: there the bound is relative to the recent input peak (the largest
			// gain seen is 5, the limit 60 x); everywhere else 1000
			recentPeak = std::max(recentPeak * (1.f - 1.f / (float) fs), std::fabs(x));
			const float limit = sc == "chaos" ? 60.f * (recentPeak + 0.05f) : 1e3f;
			if (!(l == l) || !(r == r) || std::fabs(l) > limit || std::fabs(r) > limit) {
				printf("  t=%.3f min: non-finite or absurd output (%g, %g) after %zu samples (recent input peak %g)\n", (double) done / fs / 60.0, (double) l, (double) r, done,
				       (double) recentPeak);
				fail("output not finite / bounded");
				bad = true;
			}
			if (trFrom >= 0.0) {
				const double ts = (double) done / fs;
				if (ts >= trFrom && ts <= trTo) {
					if (dumpF) {
						const float trip[3] = {x, l, r};
						fwrite(trip, sizeof(float), 3, dumpF);
					}
					{ // every tracker estimate that disagrees with the period the analyser is locked to, and every drop of the lock
						const fc::PitchTracker::Estimate& te = e.tracker().estimate();
						if (te.seq != lastEstSeq) {
							lastEstSeq = te.seq;
							const double locked = e.analyzer().active() ? e.analyzer().period() : 0.0;
							const double ratio = locked > 0.0 ? te.period / locked : 0.0;
							if (te.valid && locked > 0.0 && (ratio < 0.97 || ratio > 1.03))
								printf("      [est %8.4f s] lane %d period %9.3f (x%d) conf %.3f  ratio to the locked period %.4f\n", ts, te.lane, te.period, te.mult, (double) te.conf, ratio);
						}
						int drops = 0;
						for (int k = 0; k < 5; k++)
							drops += e.dropCount(k);
						if (drops != lastDrops) {
							lastDrops = drops;
							printf("      [drop %8.4f s] reason %d, locked period was %.3f\n", ts, e.lastDropReason(), e.analyzer().active() ? e.analyzer().period() : 0.0);
						}
					}
					if (done % (size_t) (0.1 * fs) == 0) {
						const fc::EngineStatus& st = e.status();
						const fc::CycleAnalyzer& an = e.analyzer();
						printf("    [trace %7.2f s] %-5s unit %8.3f Hz  coh %.2f  per %.3f  lockW %.2f  Nc %4d  good %3d  rej %d  drops %d/%d/%d/%d/%d  guard %d\n", ts,
						       st.locked ? "LOCK" : (st.acquiring ? "acq" : "idle"), st.unitFreqHz, st.coherence, st.periodicity, st.lockWeight, an.currentNc(),
						       an.goodHops(), an.rejectedMeasurements(), e.dropCount(0), e.dropCount(1), e.dropCount(2), e.dropCount(3), e.dropCount(4), e.guardHits());
					}
				}
			}
			w.inPeak = std::max(w.inPeak, (double) std::fabs(x));
			w.outPeak = std::max(w.outPeak, (double) std::max(std::fabs(l), std::fabs(r)));
			w.inSq += (double) x * x;
			w.outSq += 0.5 * ((double) l * l + (double) r * r);
			w.n++;
			if ((done & 1023) == 0) {
				w.ctrlN++;
				if (e.status().locked)
					w.lockedN++;
				w.freqSum += e.status().unitFreqHz;
			}
			ring[ringPos] = 0.5f * (l + r);
			inRing[ringPos] = x;
			if (++ringPos == ring.size())
				ringPos = 0;
			if ((size_t) w.n >= win) {
				windows++;
				const double inRms = std::sqrt(w.inSq / (double) w.n), outRms = std::sqrt(w.outSq / (double) w.n);
				const double gdb = 20.0 * std::log10(std::max(outRms, 1e-9) / std::max(inRms, 1e-9));
				double ih = 0;
				if (sc == "steady") {
					std::vector<float> mono(ring.size());
					for (size_t i = 0; i < ring.size(); i++)
						mono[i] = ring[(ringPos + i) % ring.size()];
					Psd psd = welch(mono, fs, 32768);
					ih = interHarmonicDb(psd, 55.0, 20); // the repeating unit of the sub-carrying saw is 55 Hz
				}
				if (sc == "hw" && pos > (size_t) (6.0 * fs) && unitHz > 0) { // the last 4 s lie inside the current note, 2 s after its start
					std::vector<float> mono(ring.size()), mi(ring.size());
					for (size_t i = 0; i < ring.size(); i++) {
						mono[i] = ring[(ringPos + i) % ring.size()];
						mi[i] = inRing[(ringPos + i) % ring.size()];
					}
					const int K = (int) std::min(20.0, 0.3 * fs / unitHz);
					ih = interHarmonicDb(welch(mono, fs, 32768), unitHz, K);
					const double ihIn = interHarmonicDb(welch(mi, fs, 32768), unitHz, K);
					printf("      unit %.1f Hz: inter-harmonic energy output %.1f dB, input %.1f dB\n", unitHz, ih, ihIn);
					if (ih > ihIn + 12.0 && ih > -40.0) {
						printf("  t=%.2f min: the output is far more inharmonic than the input (%.1f dB against %.1f dB)\n", (double) done / fs / 60.0, ih, ihIn);
						fail("metallic (inharmonic) output for a harmonic input");
						bad = true;
					}
				}
				// hw: a steady note must stay locked. A window that starts at least 0.5 s after the note began and is locked less than 90 % of the time
				// is a lock that flickers (drop, re-acquire, drop ...): the clones stutter in and out
				if (sc == "hw" && pos > (size_t) (5.5 * fs) && (double) w.lockedN / (double) std::max(1L, w.ctrlN) < 0.90 && !bad) {
					printf("  t=%.2f min: locked only %.0f %% of the time within a steady note (the lock flickers; %d drops so far: novelty %d, tracker mismatch %d, coherence %d, doubling %d)\n",
					       (double) done / fs / 60.0, 100.0 * (double) w.lockedN / (double) std::max(1L, w.ctrlN), e.dropCount(0) + e.dropCount(1) + e.dropCount(2) + e.dropCount(3) + e.dropCount(4),
					       e.dropCount(1), e.dropCount(2), e.dropCount(3), e.dropCount(4));
					fail("the lock flickers within a steady note");
					bad = true;
				}
				printf("  %6.2f  %7.3f  %8.3f  %6.3f  %7.3f  %8.2f  %5.0f%%  %9.2f  %6.1f  %4d\n", (double) done / fs / 60.0, w.inPeak, w.outPeak, inRms, outRms, gdb,
				       100.0 * (double) w.lockedN / (double) std::max(1L, w.ctrlN), w.freqSum / (double) std::max(1L, w.ctrlN), ih, e.guardHits());
				fflush(stdout);
				if (e.guardHits() != 0 && !bad) {
					fail("the engine's NaN/Inf safety net fired");
					bad = true;
				}
				blkIn += w.inSq;
				blkOut += w.outSq;
				blkIh += ih;
				blkN += w.n;
				if (windows % 12 == 0) { // one minute: the level and the inter-harmonic energy are compared minute by minute (single 5 s windows beat)
					const double gmin = 10.0 * std::log10(std::max(blkOut, 1e-18) / std::max(blkIn, 1e-18));
					const double ihmin = blkIh / 12.0;
					printf("  ---- minute %d: level %.2f dB, inter-harmonic %.1f dB\n", windows / 12, gmin, ihmin);
					if (windows == 12) {
						gain0 = gmin;
						ih0 = ihmin;
					} else if (sc == "steady" && std::fabs(gmin - gain0) > 2.0) {
						printf("  t=%.2f min: level %.2f dB against %.2f dB in the first minute\n", (double) done / fs / 60.0, gmin, gain0);
						fail("the output level drifted by more than 2 dB against the first minute");
						bad = true;
					} else if (sc == "steady" && ihmin > ih0 + 6.0) {
						printf("  t=%.2f min: inter-harmonic energy %.1f dB against %.1f dB in the first minute\n", (double) done / fs / 60.0, ihmin, ih0);
						fail("inter-harmonic energy (metallic content) rose by more than 6 dB");
						bad = true;
					}
					blkIn = blkOut = blkIh = 0;
					blkN = 0;
				}
				w = Window();
			}
		}
		printf("  scenario %s: %s\n", sc.c_str(), bad ? "FAILED" : "ok");
		if (dumpF)
			fclose(dumpF);
	}
	printf("\n%s (%d failures)\n", failures ? "SOAK TEST FAILED" : "ALL PASSED", failures);
	return failures ? 1 : 0;
}

// Rack runs process() on threads with small stacks (a secondary thread on macOS has 512 KB by default). SOAK_STACK_KB=256 runs the whole test on
// a thread with such a stack, so that a frame that is too big for it crashes here instead of in the user's session.
struct MainArgs {
	int argc;
	char** argv;
	int result;
};
static void* mainThread(void* p) {
	MainArgs* a = (MainArgs*) p;
	a->result = realMain(a->argc, a->argv);
	return nullptr;
}

int main(int argc, char** argv) {
	const char* kb = getenv("SOAK_STACK_KB");
	if (!kb)
		return realMain(argc, argv);
	MainArgs a = {argc, argv, 1};
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setstacksize(&attr, (size_t) atoi(kb) * 1024);
	pthread_t t;
	printf("running on a thread with a %s KB stack\n", kb);
	if (pthread_create(&t, &attr, mainThread, &a) != 0) {
		printf("pthread_create failed\n");
		return 1;
	}
	pthread_join(t, nullptr);
	return a.result;
}
