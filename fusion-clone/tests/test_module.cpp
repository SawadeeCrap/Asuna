// Headless module test: instantiates the real FusionClone Module class against Rack's own engine classes (Module, ParamQuantity, jansson).
// Covers: parameter/CV mapping (VOICES quantisation), process() output, bypass semantics, and PATCH SAVE/LOAD reproducibility.
#include "../src/FusionCloneModule.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace rack;

namespace rack { void headlessInstallContext(); namespace plugin { extern Model* g_testModel; } }

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } else printf("ok:   %s\n", msg); } while (0)

static const float SR = 48000.f;

// Module::toJson() reads model->plugin->slug/version; provide a stand-in Model (the real one needs the widget code).
static plugin::Plugin gPlugin;
struct TestModel : plugin::Model {
	engine::Module* createModule() override { return new FusionClone; }
	app::ModuleWidget* createModuleWidget(engine::Module*) override { return nullptr; }
};
static TestModel gModel;
static void bind(FusionClone& m) {
	gPlugin.slug = "FusionClone";
	gPlugin.version = "2.0.0";
	gModel.plugin = &gPlugin;
	gModel.slug = "FusionClone";
	m.model = &gModel;
	plugin::g_testModel = &gModel;
}

static void prepare(FusionClone& m) {
	bind(m);
	Module::SampleRateChangeEvent e;
	e.sampleRate = SR;
	e.sampleTime = 1.f / SR;
	m.onSampleRateChange(e);
}

static Module::ProcessArgs args() {
	Module::ProcessArgs a;
	a.sampleRate = SR;
	a.sampleTime = 1.f / SR;
	a.frame = 0;
	return a;
}

/** Render `n` samples of a 110 Hz band-limited-ish saw (peak ~4 V) through the module; returns the left output. */
static std::vector<float> render(FusionClone& m, int n, double f0 = 110.0, float amp = 4.f) {
	std::vector<float> out(n);
	double ph = 0.0;
	for (int i = 0; i < n; i++) {
		float v = 0.f;
		for (int k = 1; k <= 60 && k * f0 < 20000.0; k++)
			v += (float) (std::sin(2 * M_PI * k * ph) / k);
		v *= amp * 0.6366f * 0.5f;
		ph += f0 / SR; ph -= std::floor(ph);
		m.inputs[FusionClone::AUDIO_INPUT].channels = 1;
		m.inputs[FusionClone::AUDIO_INPUT].setVoltage(v, 0);
		m.process(args());
		out[i] = m.outputs[FusionClone::L_OUTPUT].getVoltage();
	}
	return out;
}

int main() {
	setvbuf(stdout, NULL, _IONBF, 0);
	rack::headlessInstallContext();
	// ---- 1. parameter / CV mapping ------------------------------------------------------------------------------------------
	{
		FusionClone m; prepare(m);
		m.params[FusionClone::VOICES_PARAM].setValue(16.f);
		m.updateParams();
		CHECK(m.ep.voices == 16, "VOICES knob 16 -> 16 voices (1 original + 15 clones)");
		m.params[FusionClone::VOICES_PARAM].setValue(1.f);
		m.updateParams();
		CHECK(m.ep.voices == 1, "VOICES knob 1 -> original only");
		m.params[FusionClone::VOICES_PARAM].setValue(1.f);
		m.inputs[FusionClone::VOICES_CV_INPUT].channels = 1;
		int ok = 1;
		for (int v = 1; v <= 16; v++) {
			m.inputs[FusionClone::VOICES_CV_INPUT].setVoltage((v - 1) * (10.f / 15.f), 0);
			m.updateParams();
			if (m.ep.voices != v) ok = 0;
		}
		CHECK(ok, "VOICES CV 0..10 V quantises cleanly onto 1..16 (10/15 V per step)");
		m.inputs[FusionClone::VOICES_CV_INPUT].setVoltage(-5.f, 0); m.updateParams();
		bool lo = m.ep.voices == 1;
		m.inputs[FusionClone::VOICES_CV_INPUT].setVoltage(30.f, 0); m.updateParams();
		CHECK(lo && m.ep.voices == 16, "VOICES CV clamps to 1..16");
		m.inputs[FusionClone::VOICES_CV_INPUT].channels = 0;
		m.params[FusionClone::SPREAD_PARAM].setValue(0.5f);
		m.inputs[FusionClone::SPREAD_CV_INPUT].channels = 1;
		m.inputs[FusionClone::SPREAD_CV_INPUT].setVoltage(3.f, 0); m.updateParams();
		CHECK(std::fabs(m.ep.spread - 0.8f) < 1e-5f, "SPREAD CV adds 0.1 per volt");
		m.inputs[FusionClone::SPREAD_CV_INPUT].setVoltage(10.f, 0); m.updateParams();
		CHECK(m.ep.spread == 1.f, "SPREAD CV clamps at 100 %");
		m.params[FusionClone::QUALITY_PARAM].setValue(3.f); m.updateParams();
		CHECK(m.ep.quality == fc::QUALITY_ULTRA, "QUALITY knob -> ULTRA");
	}
	// ---- 2. process(): silence, unconnected, MIX=0 identity, VOICES=1 identity ---------------------------------------------------
	{
		FusionClone m; prepare(m);
		m.process(args());
		CHECK(m.outputs[FusionClone::L_OUTPUT].getVoltage() == 0.f, "no input -> silent output");
		FusionClone m2;
		m2.params[FusionClone::MIX_PARAM].setValue(0.f);
		m2.params[FusionClone::VOICES_PARAM].setValue(16.f);
		prepare(m2);
		auto out = render(m2, 48000);
		// compare with the input regenerated
		double err = 0, ref = 0, ph = 0.0;
		for (int i = 0; i < 48000; i++) {
			float v = 0.f; for (int k = 1; k <= 60 && k * 110.0 < 20000.0; k++) v += (float) (std::sin(2 * M_PI * k * ph) / k);
			v *= 4.f * 0.6366f * 0.5f; ph += 110.0 / SR; ph -= std::floor(ph);
			if (i > 4800) { err += (out[i] - v) * (out[i] - v); ref += v * v; }
		}
		CHECK(err / ref < 1e-8, "TEST 14: MIX = 0 outputs exactly the original");
	}
	{
		FusionClone m;
		m.params[FusionClone::VOICES_PARAM].setValue(1.f); // a patch is loaded (params restored) before the module is added to the engine
		m.params[FusionClone::MIX_PARAM].setValue(1.f);
		prepare(m);
		auto out = render(m, 48000);
		double err = 0, ref = 0, ph = 0.0;
		for (int i = 0; i < 48000; i++) {
			float v = 0.f; for (int k = 1; k <= 60 && k * 110.0 < 20000.0; k++) v += (float) (std::sin(2 * M_PI * k * ph) / k);
			v *= 4.f * 0.6366f * 0.5f; ph += 110.0 / SR; ph -= std::floor(ph);
			if (i > 4800) { err += (out[i] - v) * (out[i] - v); ref += v * v; }
		}
		CHECK(err / ref < 1e-8, "TEST 1: VOICES = 1 output is the input (no processing, zero latency)");
	}
	// ---- 3. clones produce density: VOICES = 8 differs from the input and stays bounded ------------------------------------------
	{
		FusionClone m; prepare(m);
		m.params[FusionClone::VOICES_PARAM].setValue(8.f);
		auto out = render(m, 96000);
		double pk = 0, r = 0, ri = 0; int n = 0;
		for (int i = 48000; i < 96000; i++) { pk = std::max(pk, (double) std::fabs(out[i])); r += out[i] * out[i]; n++; }
		r = std::sqrt(r / n);
		CHECK(pk < 12.0 && r > 0.5, "VOICES = 8 yields a bounded, non-silent output");
		CHECK(m.engine.lockedNow(), "engine locked onto the 110 Hz oscillator");
		(void) ri;
	}
	// ---- 4. PATCH SAVE / LOAD reproducibility ---------------------------------------------------------------------------------------
	{
		FusionClone a; prepare(a);
		a.params[FusionClone::VOICES_PARAM].setValue(12.f);
		a.params[FusionClone::SPREAD_PARAM].setValue(0.62f);
		a.params[FusionClone::DRIFT_PARAM].setValue(0.4f);
		a.params[FusionClone::HARMONIC_PARAM].setValue(0.55f);
		a.params[FusionClone::WIDTH_PARAM].setValue(0.3f);
		a.params[FusionClone::QUALITY_PARAM].setValue(2.f);
		a.params[FusionClone::ALGO_PARAM].setValue(1.f);
		a.params[FusionClone::DETUNE_RANGE_PARAM].setValue(33.f);
		a.seed = 0xC0FFEE42u;
		json_t* j = a.toJson();
		FusionClone b; prepare(b);
		b.fromJson(j);
		json_decref(j);
		CHECK(b.seed == 0xC0FFEE42u, "seed survives the patch JSON");
		CHECK(b.params[FusionClone::VOICES_PARAM].getValue() == 12.f && std::fabs(b.params[FusionClone::SPREAD_PARAM].getValue() - 0.62f) < 1e-6f &&
		      b.params[FusionClone::QUALITY_PARAM].getValue() == 2.f && b.params[FusionClone::DETUNE_RANGE_PARAM].getValue() == 33.f, "all parameters (primary + expert) survive the patch JSON");
		auto oa = render(a, 96000), ob = render(b, 96000);
		int same = 1; double maxd = 0;
		for (size_t i = 0; i < oa.size(); i++) { maxd = std::max(maxd, (double) std::fabs(oa[i] - ob[i])); if (oa[i] != ob[i]) same = 0; }
		CHECK(same, "TEST (patch load): reloaded module reproduces the identical output bit-for-bit");
		printf("      max |difference| = %g\n", maxd);
		// a different seed must give a different population
		FusionClone c; prepare(c);
		c.params[FusionClone::VOICES_PARAM].setValue(12.f); c.params[FusionClone::SPREAD_PARAM].setValue(0.62f); c.params[FusionClone::DRIFT_PARAM].setValue(0.4f);
		c.params[FusionClone::HARMONIC_PARAM].setValue(0.55f); c.params[FusionClone::WIDTH_PARAM].setValue(0.3f); c.params[FusionClone::QUALITY_PARAM].setValue(2.f);
		c.params[FusionClone::ALGO_PARAM].setValue(1.f); c.params[FusionClone::DETUNE_RANGE_PARAM].setValue(33.f);
		c.seed = 0x12345678u;
		auto oc = render(c, 96000);
		double d = 0; for (size_t i = 48000; i < oa.size(); i++) d += std::fabs(oa[i] - oc[i]);
		CHECK(d > 1.0, "a different seed produces a different oscillator population");
	}
	// ---- 5. RANDOMIZE: new seed, same parameters ---------------------------------------------------------------------------------------
	{
		FusionClone m; prepare(m);
		uint32_t s0 = m.seed;
		m.params[FusionClone::RANDOM_PARAM].setValue(0.f);
		for (int i = 0; i < 64; i++) { m.inputs[FusionClone::AUDIO_INPUT].channels = 1; m.process(args()); } // idle: trigger sees the released button
		CHECK(m.seed == s0, "seed unchanged while the button is not pressed");
		m.params[FusionClone::RANDOM_PARAM].setValue(1.f);
		for (int i = 0; i < 64; i++) { m.inputs[FusionClone::AUDIO_INPUT].channels = 1; m.process(args()); }
		CHECK(m.seed != s0, "RANDOMIZE button draws a new seed");
		uint32_t s1 = m.seed;
		for (int i = 0; i < 64; i++) m.process(args());
		CHECK(m.seed == s1, "held button does not keep re-randomising (edge triggered)");
	}
	printf("\n%s (%d failure%s)\n", fails ? "FAILED" : "ALL PASSED", fails, fails == 1 ? "" : "s");
	return fails;
}
