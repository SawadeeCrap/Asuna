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

static std::vector<float> renderNote(double fs, double f0, double seconds, uint32_t seed, double gain, double sub, double tube, double noiseDb, double drift, double wSaw,
                                     double wPulse, const std::vector<double>* track, const std::vector<float>* env) {
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

// next chunk of the input for a scenario; `chunk` counts from 0
static std::vector<float> nextChunk(const std::string& sc, double fs, int chunk, fc::Rng& rng) {
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

int main(int argc, char** argv) {
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
		list = {"steady", "seq", "knobs", "notes"};
	else
		list.push_back(only);

	for (const std::string& sc : list) {
		printf("\n=== scenario %s\n", sc.c_str());
		printf("  t(min)  in-peak  out-peak  in-rms  out-rms  gain(dB)  locked  pitch(Hz)  IH(dB)  hits\n");
		fflush(stdout);
		fc::Rng rng(12345u + (uint32_t) sc.size());
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
		Window w;
		std::vector<float> ring((size_t) (4.0 * fs), 0.f); // last 4 s of the mono output for the inter-harmonic metric
		size_t ringPos = 0;
		double gain0 = 0, ih0 = 0, blkIn = 0, blkOut = 0, blkIh = 0;
		long blkN = 0;
		int windows = 0;
		bool bad = false;
		while (done < total && !bad) {
			if (pos >= chunk.size()) {
				chunk = nextChunk(sc, fs, chunkIdx++, rng);
				pos = 0;
			}
			if (sc == "knobs" && rng.uniform() < 1.f / (0.2f * (float) fs)) {
				mutateControls(p, rng, quality);
				e.setParams(p);
			}
			const float x = chunk[pos++];
			float l = 0, r = 0;
			e.process(x, l, r);
			done++;
			if (!(l == l) || !(r == r) || std::fabs(l) > 1e3f || std::fabs(r) > 1e3f) {
				printf("  t=%.3f min: non-finite or absurd output (%g, %g) after %zu samples\n", (double) done / fs / 60.0, (double) l, (double) r, done);
				fail("output not finite / bounded");
				bad = true;
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
	}
	printf("\n%s (%d failures)\n", failures ? "SOAK TEST FAILED" : "ALL PASSED", failures);
	return failures ? 1 : 0;
}
