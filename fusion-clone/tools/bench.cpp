// CPU and latency benchmark for the FusionClone engine.
//   CPU: wall-clock time per second of audio (= fraction of ONE core) for every quality x voice count at several pitches.
//   Latency: original path (always 0) and the "bloom" time: how long after a note onset / pitch step the clones are fully present.
// Build (see tools/Makefile):  make -C tools bench      Run:  ./tools/bench [seconds] [md]
#include "../research/common/fusion_source.hpp"
#include "../src/dsp/Engine.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>

using namespace research;

static std::vector<float> makeSaw(double f0, double fs, double seconds, double sub = 0.0) {
	SourceSpec s; s.fs = fs; s.f0 = f0; s.seconds = seconds; s.seed = 3; s.noiseDb = -80; s.sub = sub;
	auto x = renderSource(s);
	for (auto& v : x) v *= 4.f;
	return x;
}

int main(int argc, char** argv) {
	double seconds = argc > 1 ? atof(argv[1]) : 8.0;
	bool md = argc > 2 && std::strcmp(argv[2], "md") == 0;
	const double fs = 48000;
	const double pitches[] = {20.0, 110.0, 440.0, 3520.0};
	const int voices[] = {1, 2, 4, 8, 16};
	const char* qn[] = {"ECO", "BALANCED", "HIGH", "ULTRA"};
	printf("FFT backend: %s   sample rate %.0f Hz   %.0f s of audio per cell\n\n", fc::RealFFT::backendName(), fs, seconds);
	printf("CPU load, %% of one core (lower is better; 100%% = real-time limit of one core on THIS machine; fastest of 3 identical runs per cell)\n\n");
	if (md) printf("| quality | pitch | 1 voice | 2 | 4 | 8 | 16 |\n|---|---|---|---|---|---|---|\n");
	else printf("%-9s %-8s %8s %8s %8s %8s %8s\n", "quality", "pitch", "1 voice", "2", "4", "8", "16");
	for (int q = 0; q < 4; q++) {
		for (double f0 : pitches) {
			auto x = makeSaw(f0, fs, seconds);
			double cpu[5];
			for (int vi = 0; vi < 5; vi++) {
				cpu[vi] = 1e30;
				for (int run = 0; run < 3; run++) { // the engine is deterministic: the fastest of three identical runs is the compute cost without scheduler noise
					fc::Engine e; e.prepare(fs);
					fc::EngineParams p; p.voices = voices[vi]; p.quality = q; p.spread = 0.5f; p.drift = 0.3f; p.character = 0.3f; p.harmonic = 0.3f; p.phase = 0.7f; p.width = 0.3f;
					p.algorithm = fc::ALGO_CLASSIC; e.setParams(p);
					std::vector<float> l(x.size()), r(x.size());
					// warm up 1 s (lock + first tables), then time the rest
					size_t warm = (size_t) fs;
					for (size_t i = 0; i < warm; i++) e.process(x[i], l[i], r[i]);
					auto t0 = std::chrono::steady_clock::now();
					for (size_t i = warm; i < x.size(); i++) e.process(x[i], l[i], r[i]);
					double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
					cpu[vi] = std::min(cpu[vi], 100.0 * dt / ((double) (x.size() - warm) / fs));
				}
			}
			if (md) printf("| %s | %.0f Hz | %.2f | %.2f | %.2f | %.2f | %.2f |\n", qn[q], f0, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4]);
			else printf("%-9s %-8.0f %8.2f %8.2f %8.2f %8.2f %8.2f\n", qn[q], f0, cpu[0], cpu[1], cpu[2], cpu[3], cpu[4]);
			fflush(stdout);
		}
	}

	// ---- worst-case cost of one audio block ---------------------------------------------------------------------------------------
	// Rack calls process() once per sample inside the audio driver's block (typically 256 frames): what matters for dropouts is the most expensive block,
	// not the mean. The engine is deterministic, so identical runs are repeated and each block's cost is the minimum over the runs (scheduler noise out).
	{
		const int kBlock = 256, kRuns = 3;
		printf("\nWORST BLOCK at 16 voices: most expensive 256-sample block as %% of the block's real-time budget (%.2f ms), min of %d identical runs; steady state\n\n", 1000.0 * kBlock / fs, kRuns);
		if (md) printf("| quality | 20 Hz | 110 Hz | 440 Hz | 3520 Hz |\n|---|---|---|---|---|\n");
		else printf("%-9s %8s %8s %8s %8s\n", "quality", "20 Hz", "110 Hz", "440 Hz", "3520 Hz");
		for (int q = 0; q < 4; q++) {
			double worst[4];
			for (int pi = 0; pi < 4; pi++) {
				auto x = makeSaw(pitches[pi], fs, std::min(seconds, 5.0));
				const size_t warm = (size_t) fs;
				const size_t nBlocks = (x.size() - warm) / kBlock;
				std::vector<double> best(nBlocks, 1e30);
				for (int run = 0; run < kRuns; run++) {
					fc::Engine e; e.prepare(fs);
					fc::EngineParams p; p.voices = 16; p.quality = q; p.spread = 0.5f; p.drift = 0.3f; p.character = 0.3f; p.harmonic = 0.3f; p.phase = 0.7f; p.width = 0.3f;
					p.algorithm = fc::ALGO_CLASSIC; e.setParams(p);
					float l, r;
					for (size_t i = 0; i < warm; i++) e.process(x[i], l, r);
					for (size_t b = 0; b < nBlocks; b++) {
						auto t0 = std::chrono::steady_clock::now();
						for (int i = 0; i < kBlock; i++) e.process(x[warm + b * kBlock + i], l, r);
						best[b] = std::min(best[b], std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
					}
				}
				double mx = 0;
				for (double v : best) mx = std::max(mx, v);
				worst[pi] = 100.0 * mx / ((double) kBlock / fs);
			}
			if (md) printf("| %s | %.1f | %.1f | %.1f | %.1f |\n", qn[q], worst[0], worst[1], worst[2], worst[3]);
			else printf("%-9s %8.1f %8.1f %8.1f %8.1f\n", qn[q], worst[0], worst[1], worst[2], worst[3]);
			fflush(stdout);
		}
	}

	printf("\nFUSION algorithm layer (16 voices, BALANCED, 110 Hz): ");
	for (int mode = 0; mode < 2; mode++) {
		auto x = makeSaw(110, fs, seconds);
		fc::Engine e; e.prepare(fs);
		fc::EngineParams p; p.voices = 16; p.quality = 1; p.algorithm = fc::ALGO_FUSION; p.fusionShift = 0.5f; p.shiftMode = mode; e.setParams(p);
		std::vector<float> l(x.size()), r(x.size());
		size_t warm = (size_t) fs;
		for (size_t i = 0; i < warm; i++) e.process(x[i], l[i], r[i]);
		auto t0 = std::chrono::steady_clock::now();
		for (size_t i = warm; i < x.size(); i++) e.process(x[i], l[i], r[i]);
		double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		printf("%s %.2f %%   ", mode ? "HZ mode" : "RATIO mode", 100.0 * dt / ((double) (x.size() - warm) / fs));
	}
	printf("\n\n");

	// ---- latency ---------------------------------------------------------------------------------------------------------
	printf("LATENCY\n  original path: 0 samples (direct connection, no delay line).\n  clone 'bloom' time = time from note onset (gate at t = 1.0 s) until the clone weight exceeds 90%%:\n\n");
	if (md) printf("| pitch | ECO | BALANCED | HIGH | ULTRA |\n|---|---|---|---|---|\n");
	else printf("%-8s %10s %10s %10s %10s\n", "pitch", "ECO", "BALANCED", "HIGH", "ULTRA");
	for (double f0 : {20.0, 41.2, 110.0, 440.0, 1760.0}) {
		double bl[4];
		for (int q = 0; q < 4; q++) {
			SourceSpec s; s.fs = fs; s.f0 = f0; s.seconds = 2.5; s.seed = 3; s.noiseDb = -80;
			s.ampEnv.resize((size_t) (2.5 * fs));
			for (size_t i = 0; i < s.ampEnv.size(); i++) { double t = i / fs; s.ampEnv[i] = (float) std::min(1.0, std::max(0.0, (t - 1.0) / 0.002)); }
			auto x = renderSource(s); for (auto& v : x) v *= 4.f;
			fc::Engine e; e.prepare(fs);
			fc::EngineParams p; p.voices = 8; p.quality = q; e.setParams(p);
			double t90 = -1;
			for (size_t i = 0; i < x.size(); i++) {
				float l, r; e.process(x[i], l, r);
				if (i > (size_t) fs && t90 < 0 && e.status().lockWeight > 0.9f) t90 = (double) i / fs - 1.0;
			}
			bl[q] = t90 * 1000.0;
		}
		if (md) printf("| %.1f Hz | %.1f ms | %.1f ms | %.1f ms | %.1f ms |\n", f0, bl[0], bl[1], bl[2], bl[3]);
		else printf("%-8.1f %8.1f ms %8.1f ms %8.1f ms %8.1f ms\n", f0, bl[0], bl[1], bl[2], bl[3]);
	}
	return 0;
}
