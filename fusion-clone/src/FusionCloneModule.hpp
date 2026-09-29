// Fusion Clone — module (DSP glue). Header-only so the module can be instantiated in headless tests without any widget code.
#pragma once
#include "plugin.hpp"

#include <atomic>
#include <cstdio>

#include "dsp/Engine.hpp"

static const uint32_t kDefaultSeed = 0x5EED1234u;

struct FusionClone : Module {
	enum ParamId {
		VOICES_PARAM,
		SPREAD_PARAM,
		DRIFT_PARAM,
		CHARACTER_PARAM,
		PHASE_PARAM,
		HARMONIC_PARAM,
		WIDTH_PARAM,
		MIX_PARAM,
		OUTPUT_PARAM,
		QUALITY_PARAM,
		ALGO_PARAM,
		ORIG_PARAM,
		RANDOM_PARAM,
		// expert (context menu)
		DETUNE_RANGE_PARAM,
		SPREAD_CURVE_PARAM,
		DRIFT_RATE_PARAM,
		DRIFT_CORR_PARAM,
		FUSION_SHIFT_PARAM,
		SHIFT_MODE_PARAM,
		SUMMING_PARAM,
		LEVEL_LAW_PARAM,
		NUM_PARAMS
	};
	enum InputId { AUDIO_INPUT, VOICES_CV_INPUT, SPREAD_CV_INPUT, DRIFT_CV_INPUT, CHARACTER_CV_INPUT, MIX_CV_INPUT, NUM_INPUTS };
	enum OutputId { L_OUTPUT, R_OUTPUT, NUM_OUTPUTS };
	enum LightId { LOCK_LIGHT, ACQ_LIGHT, POLY_LIGHT, NUM_LIGHTS };

	fc::Engine engine;
	fc::EngineParams ep;
	bool prepared = false;
	uint32_t seed = kDefaultSeed;
	dsp::ClockDivider paramDivider, lightDivider;
	dsp::BooleanTrigger randomTrigger;
	bool polyInput = false;

	// GUI snapshot (single writer = audio thread, seqlock protected)
	struct Snapshot {
		std::atomic<uint32_t> seq;
		fc::EngineStatus st;
		int voices;
		int quality;
		int algorithm;
		float detuneRange;
		bool connected;
		bool poly;
		Snapshot() : seq(0), voices(8), quality(1), algorithm(0), detuneRange(20.f), connected(false), poly(false) {}
	} snap;

	FusionClone() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);

		configParam(VOICES_PARAM, 1.f, 16.f, 8.f, "Voices (1 original + N-1 clones)", " voices");
		paramQuantities[VOICES_PARAM]->snapEnabled = true;
		configParam(SPREAD_PARAM, 0.f, 1.f, 0.35f, "Spread (detune)", "%", 0.f, 100.f);
		configParam(DRIFT_PARAM, 0.f, 1.f, 0.25f, "Drift (slow analog instability)", "%", 0.f, 100.f);
		configParam(CHARACTER_PARAM, 0.f, 1.f, 0.3f, "Character (level tolerance, saturation asymmetry, pitch jitter)", "%", 0.f, 100.f);
		configParam(PHASE_PARAM, 0.f, 1.f, 0.6f, "Phase divergence", "%", 0.f, 100.f);
		configParam(HARMONIC_PARAM, 0.f, 1.f, 0.3f, "Harmonic divergence", "%", 0.f, 100.f);
		configParam(WIDTH_PARAM, 0.f, 1.f, 0.f, "Stereo width (0 = mono-compatible)", "%", 0.f, 100.f);
		configParam(MIX_PARAM, 0.f, 1.f, 1.f, "Mix (0 = original only)", "%", 0.f, 100.f);
		configParam(OUTPUT_PARAM, -24.f, 12.f, 0.f, "Output", " dB");
		configSwitch(QUALITY_PARAM, 0.f, 3.f, 1.f, "Quality", {"ECO", "BALANCED", "HIGH", "ULTRA"});
		configSwitch(ALGO_PARAM, 0.f, 1.f, 0.f, "Algorithm", {"CLASSIC (pure partial cloning)", "FUSION (adds per-voice detune-cluster layer)"});
		configSwitch(ORIG_PARAM, 0.f, 1.f, 0.f, "Original only (A/B, debugging)", {"Off", "On"});
		configButton(RANDOM_PARAM, "Randomize voice seeds");

		configParam(DETUNE_RANGE_PARAM, 5.f, 60.f, 20.f, "Detune range at SPREAD = 100%", " cents");
		configParam(SPREAD_CURVE_PARAM, 0.8f, 3.f, 1.6f, "Spread curve (exponent)");
		configParam(DRIFT_RATE_PARAM, 0.f, 1.f, 0.5f, "Drift rate", "%", 0.f, 100.f);
		configParam(DRIFT_CORR_PARAM, 0.f, 1.f, 0.25f, "Drift correlation between voices", "%", 0.f, 100.f);
		configParam(FUSION_SHIFT_PARAM, 0.f, 1.f, 0.3f, "Fusion detune-cluster amount (FUSION algorithm)", "%", 0.f, 100.f);
		configSwitch(SHIFT_MODE_PARAM, 0.f, 1.f, 0.f, "Fusion shift mode", {"RATIO (Doppler / BBD delay reading)", "HZ (single-sideband frequency shift)"});
		configParam(SUMMING_PARAM, 0.f, 1.f, 0.35f, "Analog-style summing saturation", "%", 0.f, 100.f);
		configParam(LEVEL_LAW_PARAM, 0.f, 0.5f, 0.075f, "Level law (0 = equal power, 0.5 = unity sum)");

		configInput(AUDIO_INPUT, "Audio (Fusion VCO2 out)");
		configInput(VOICES_CV_INPUT, "Voices CV (10 V = 16 voices)");
		configInput(SPREAD_CV_INPUT, "Spread CV");
		configInput(DRIFT_CV_INPUT, "Drift CV");
		configInput(CHARACTER_CV_INPUT, "Character CV");
		configInput(MIX_CV_INPUT, "Mix CV");
		configOutput(L_OUTPUT, "Left");
		configOutput(R_OUTPUT, "Right");
		configLight(LOCK_LIGHT, "Locked: periodic structure found, clones active");
		configLight(ACQ_LIGHT, "Acquiring / bypassing to the original");
		configLight(POLY_LIGHT, "Polyphonic input: only channel 1 is used");
		configBypass(AUDIO_INPUT, L_OUTPUT);
		configBypass(AUDIO_INPUT, R_OUTPUT);

		paramDivider.setDivision(16);
		lightDivider.setDivision(1024);
	}

	void onSampleRateChange(const SampleRateChangeEvent& e) override {
		// allocates: runs on the engine thread when the module is added / the rate changes, never inside process()
		engine.prepare(e.sampleRate);
		prepared = true;
		updateParams();
	}

	// One line each when the module enters / leaves the engine (never from process()). They make it visible in Rack's log.txt that the plugin was
	// instantiated and what the engine was doing when the module went away (used by tests/rack_smoke.sh and handy for bug reports).
	void onAdd(const AddEvent& e) override {
		Module::onAdd(e);
		INFO("Fusion Clone: module added");
	}

	void onRemove(const RemoveEvent& e) override {
		Module::onRemove(e);
		fc::EngineStatus st;
		int voices = 0, quality = 0, algorithm = 0;
		float detune = 0.f;
		bool connected = false, poly = false;
		if (readSnapshot(st, voices, quality, algorithm, detune, connected, poly))
			INFO("Fusion Clone: module removed; last state %s, repeating unit %.2f Hz, %d voice(s), input %.1f dB, safety-net hits %d",
			     st.locked ? "LOCKED" : (st.acquiring ? "ACQUIRING" : "PASS-THRU"), st.unitFreqHz, voices, (double) st.inputLevelDb, engine.guardHits());
	}

	void onReset(const ResetEvent& e) override {
		Module::onReset(e);
		seed = kDefaultSeed;
	}

	void onRandomize(const RandomizeEvent& e) override {
		// "Randomize" from the module menu only draws a new population of oscillators; it keeps VOICES / QUALITY / ALGORITHM.
		(void) e;
		seed = random::u32();
	}

	json_t* dataToJson() override {
		json_t* rootJ = json_object();
		json_object_set_new(rootJ, "version", json_integer(2));
		json_object_set_new(rootJ, "seed", json_integer((json_int_t) seed));
		return rootJ;
	}

	void dataFromJson(json_t* rootJ) override {
		json_t* s = json_object_get(rootJ, "seed");
		if (s)
			seed = (uint32_t) json_integer_value(s);
	}

	float cvOrZero(int input, float scale) {
		Input& in = inputs[input];
		return in.isConnected() ? in.getVoltage() * scale : 0.f;
	}

	void updateParams() {
		fc::EngineParams p;
		float vf = params[VOICES_PARAM].getValue() + cvOrZero(VOICES_CV_INPUT, 1.5f); // 10 V spans 15 steps
		p.voices = clamp((int) std::floor(vf + 0.5f), 1, 16);
		p.spread = clamp(params[SPREAD_PARAM].getValue() + cvOrZero(SPREAD_CV_INPUT, 0.1f), 0.f, 1.f);
		p.drift = clamp(params[DRIFT_PARAM].getValue() + cvOrZero(DRIFT_CV_INPUT, 0.1f), 0.f, 1.f);
		p.character = clamp(params[CHARACTER_PARAM].getValue() + cvOrZero(CHARACTER_CV_INPUT, 0.1f), 0.f, 1.f);
		p.mix = clamp(params[MIX_PARAM].getValue() + cvOrZero(MIX_CV_INPUT, 0.1f), 0.f, 1.f);
		p.phase = params[PHASE_PARAM].getValue();
		p.harmonic = params[HARMONIC_PARAM].getValue();
		p.width = params[WIDTH_PARAM].getValue();
		p.outputDb = params[OUTPUT_PARAM].getValue();
		p.quality = clamp((int) std::floor(params[QUALITY_PARAM].getValue() + 0.5f), 0, 3);
		p.algorithm = params[ALGO_PARAM].getValue() > 0.5f ? fc::ALGO_FUSION : fc::ALGO_CLASSIC;
		p.originalOnly = params[ORIG_PARAM].getValue() > 0.5f;
		p.detuneRangeCents = params[DETUNE_RANGE_PARAM].getValue();
		p.spreadCurve = params[SPREAD_CURVE_PARAM].getValue();
		p.driftRate = params[DRIFT_RATE_PARAM].getValue();
		p.driftCorrelation = params[DRIFT_CORR_PARAM].getValue();
		p.fusionShift = params[FUSION_SHIFT_PARAM].getValue();
		p.shiftMode = params[SHIFT_MODE_PARAM].getValue() > 0.5f ? fc::SHIFT_HZ : fc::SHIFT_RATIO;
		p.summing = params[SUMMING_PARAM].getValue();
		p.levelLaw = params[LEVEL_LAW_PARAM].getValue();
		p.seed = seed;
		ep = p;
		if (prepared)
			engine.setParams(p);
	}

	void process(const ProcessArgs& args) override {
		if (!prepared) {
			outputs[L_OUTPUT].setVoltage(0.f);
			outputs[R_OUTPUT].setVoltage(0.f);
			return;
		}
		if (paramDivider.process()) {
			if (randomTrigger.process(params[RANDOM_PARAM].getValue() > 0.5f))
				seed = random::u32();
			updateParams();
		}
		// Nominal Rack audio level is 5 V; the DSP core works in units where 1.0 = 5 V.
		Input& in = inputs[AUDIO_INPUT];
		const bool connected = in.isConnected();
		polyInput = in.getChannels() > 1;
		const float x = connected ? in.getVoltage(0) * 0.2f : 0.f;
		float l = 0.f, r = 0.f;
		engine.process(x, l, r);
		outputs[L_OUTPUT].setVoltage(l * 5.f);
		outputs[R_OUTPUT].setVoltage(r * 5.f);

		if (lightDivider.process()) {
			const fc::EngineStatus& st = engine.status();
			const float dt = args.sampleTime * 1024.f;
			lights[LOCK_LIGHT].setBrightnessSmooth(st.locked ? st.lockWeight : 0.f, dt);
			lights[ACQ_LIGHT].setBrightnessSmooth(connected && !st.locked && st.inputLevelDb > -70.f ? 1.f : 0.f, dt);
			lights[POLY_LIGHT].setBrightnessSmooth(polyInput ? 1.f : 0.f, dt);
			// publish to the GUI (seqlock: odd = write in progress)
			uint32_t s0 = snap.seq.load(std::memory_order_relaxed);
			snap.seq.store(s0 + 1, std::memory_order_release);
			snap.st = st;
			snap.voices = ep.voices;
			snap.quality = ep.quality;
			snap.algorithm = ep.algorithm;
			snap.detuneRange = ep.detuneRangeCents;
			snap.connected = connected;
			snap.poly = polyInput;
			snap.seq.store(s0 + 2, std::memory_order_release);
		}
	}

	/** Called from the GUI thread. */
	bool readSnapshot(fc::EngineStatus& st, int& voices, int& quality, int& algorithm, float& detuneRange, bool& connected, bool& poly) {
		for (int tries = 0; tries < 4; tries++) {
			uint32_t a = snap.seq.load(std::memory_order_acquire);
			if (a & 1u)
				continue;
			st = snap.st;
			voices = snap.voices;
			quality = snap.quality;
			algorithm = snap.algorithm;
			detuneRange = snap.detuneRange;
			connected = snap.connected;
			poly = snap.poly;
			uint32_t b = snap.seq.load(std::memory_order_acquire);
			if (a == b)
				return true;
		}
		return false;
	}
};

