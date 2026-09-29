// fusionclone_cli — offline renderer for the FusionClone engine (the exact same DSP core as the Rack module).
//
//   fusionclone_cli in.wav out.wav [options]
//   fusionclone_cli in.wav --ab outdir [options]     renders the A/B ladder: A = original, B = +1 clone, C = +3, D = +7, E = +15
//
// The input is interpreted with 1.0 = 5 V (Rack's nominal audio level), same as inside the module; use --gain-db to adapt a recording.
#include "../research/common/wav.hpp"
#include "../src/dsp/Engine.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage() {
	printf("usage: fusionclone_cli in.wav out.wav [options]\n"
	       "       fusionclone_cli in.wav --ab outdir [options]\n"
	       "options (defaults in brackets):\n"
	       "  --voices N [8]        1..16 total voices (1 original + N-1 clones)\n"
	       "  --spread x [0.35] --drift x [0.25] --character x [0.3] --phase x [0.6] --harmonic x [0.3] --width x [0] --mix x [1]   (0..1)\n"
	       "  --output-db dB [0]    --gain-db dB [0]  input trim before the engine\n"
	       "  --quality eco|balanced|high|ultra [balanced]\n"
	       "  --algo classic|fusion [classic]   --shift ratio|hz [ratio]   --fusion x [0.3]\n"
	       "  --detune-range cents [20] --spread-curve x [1.6] --drift-rate x [0.5] --drift-corr x [0.25] --summing x [0.35] --level-law x [0.075]\n"
	       "  --seed N              deterministic voice population\n"
	       "  --original-only       bypass all clones\n"
	       "  --stereo              write 2 channels (L/R) instead of the mono sum\n");
}

static bool render(const wav::Audio& in, const std::string& out, fc::EngineParams p, float gainDb, bool stereo) {
	int chans = in.channels;
	std::vector<float> x = in.channel(0);
	if (chans > 1) { // mono assumption: sum channels
		for (size_t i = 0; i < x.size(); i++) { float s = 0; for (int c = 0; c < chans; c++) s += in.data[i * chans + c]; x[i] = s / chans; }
	}
	const float g = std::pow(10.f, gainDb / 20.f);
	fc::Engine e;
	e.prepare(in.sampleRate);
	e.setParams(p);
	std::vector<float> o(x.size() * (stereo ? 2 : 1));
	for (size_t i = 0; i < x.size(); i++) {
		float l, r;
		e.process(x[i] * g, l, r);
		if (stereo) { o[2 * i] = l; o[2 * i + 1] = r; } else o[i] = 0.5f * (l + r);
	}
	return wav::writeFloat(out, o.data(), x.size(), stereo ? 2 : 1, in.sampleRate);
}

int main(int argc, char** argv) {
	if (argc < 3) { usage(); return 1; }
	std::string inPath = argv[1], outPath, abDir;
	fc::EngineParams p;
	float gainDb = 0.f;
	bool stereo = false;
	int i = 2;
	if (argv[2][0] != '-') { outPath = argv[2]; i = 3; }
	for (; i < argc; i++) {
		std::string a = argv[i];
		auto next = [&](double def) { return i + 1 < argc ? atof(argv[++i]) : def; };
		if (a == "--voices") p.voices = (int) next(8);
		else if (a == "--spread") p.spread = (float) next(0.35);
		else if (a == "--drift") p.drift = (float) next(0.25);
		else if (a == "--character") p.character = (float) next(0.3);
		else if (a == "--phase") p.phase = (float) next(0.6);
		else if (a == "--harmonic") p.harmonic = (float) next(0.3);
		else if (a == "--width") p.width = (float) next(0);
		else if (a == "--mix") p.mix = (float) next(1);
		else if (a == "--output-db") p.outputDb = (float) next(0);
		else if (a == "--gain-db") gainDb = (float) next(0);
		else if (a == "--quality" && i + 1 < argc) { std::string q = argv[++i]; p.quality = q == "eco" ? 0 : q == "high" ? 2 : q == "ultra" ? 3 : 1; }
		else if (a == "--algo" && i + 1 < argc) p.algorithm = std::string(argv[++i]) == "fusion" ? fc::ALGO_FUSION : fc::ALGO_CLASSIC;
		else if (a == "--shift" && i + 1 < argc) p.shiftMode = std::string(argv[++i]) == "hz" ? fc::SHIFT_HZ : fc::SHIFT_RATIO;
		else if (a == "--fusion") p.fusionShift = (float) next(0.3);
		else if (a == "--detune-range") p.detuneRangeCents = (float) next(20);
		else if (a == "--spread-curve") p.spreadCurve = (float) next(1.6);
		else if (a == "--drift-rate") p.driftRate = (float) next(0.5);
		else if (a == "--drift-corr") p.driftCorrelation = (float) next(0.25);
		else if (a == "--summing") p.summing = (float) next(0.35);
		else if (a == "--level-law") p.levelLaw = (float) next(0.075);
		else if (a == "--seed") p.seed = (uint32_t) strtoul(argv[++i], NULL, 0);
		else if (a == "--original-only") p.originalOnly = true;
		else if (a == "--stereo") stereo = true;
		else if (a == "--ab" && i + 1 < argc) abDir = argv[++i];
		else { usage(); return 1; }
	}
	wav::Audio in;
	if (!wav::read(inPath, in)) { fprintf(stderr, "cannot read %s\n", inPath.c_str()); return 2; }
	printf("input: %s  %d Hz, %d ch, %zu frames\n", inPath.c_str(), in.sampleRate, in.channels, in.frames());
	if (!abDir.empty()) {
		const int ladder[5] = {1, 2, 4, 8, 16};
		const char* names[5] = {"A_original", "B_plus1clone", "C_plus3clones", "D_plus7clones", "E_plus15clones"};
		for (int k = 0; k < 5; k++) {
			fc::EngineParams q = p; q.voices = ladder[k];
			std::string path = abDir + "/" + names[k] + ".wav";
			if (!render(in, path, q, gainDb, stereo)) { fprintf(stderr, "cannot write %s (does the directory exist?)\n", path.c_str()); return 3; }
			printf("wrote %s\n", path.c_str());
		}
		return 0;
	}
	if (outPath.empty()) { usage(); return 1; }
	if (!render(in, outPath, p, gainDb, stereo)) { fprintf(stderr, "cannot write %s\n", outPath.c_str()); return 3; }
	printf("wrote %s\n", outPath.c_str());
	return 0;
}
