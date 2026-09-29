// Fusion Clone — VCV Rack 2 module.
//
// One audio input (the output of ONE hardware Erica Synths Fusion VCO2, or any monophonic oscillator) is analysed period-synchronously and
// re-synthesised as a population of 1..16 independent virtual oscillators (voice 0 is always the untouched original).
#include "plugin.hpp"

#include <cstdio>

#include "FusionCloneModule.hpp"
#include "PanelLayout.hpp"

// ------------------------------------------------------------------------------------------------------------------------------
// GUI
// ------------------------------------------------------------------------------------------------------------------------------
namespace {

const NVGcolor kAmber = nvgRGB(0xff, 0xb0, 0x30);
const NVGcolor kAmberDim = nvgRGBA(0xff, 0xb0, 0x30, 70);
const NVGcolor kTeal = nvgRGB(0x40, 0xd8, 0xc8);
const NVGcolor kInk = nvgRGB(0xd8, 0xdc, 0xe2);

std::shared_ptr<Font> uiFont() {
	return APP->window->loadFont(asset::system("res/fonts/ShareTechMono-Regular.ttf"));
}

/** Static labels (Rack's SVG loader ignores <text>). */
struct PanelLabels : TransparentWidget {
	void draw(const DrawArgs& args) override {
		std::shared_ptr<Font> font = uiFont();
		if (!font)
			return;
		nvgFontFaceId(args.vg, font->handle);
		nvgTextAlign(args.vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);
		nvgFillColor(args.vg, kInk);
		for (int i = 0; i < layout::kNumLabels; i++) {
			nvgFontSize(args.vg, layout::kLabels[i].size);
			nvgText(args.vg, layout::kLabels[i].x, layout::kLabels[i].y, layout::kLabels[i].text, NULL);
		}
		// title
		nvgFontSize(args.vg, layout::kTitleSize);
		nvgFillColor(args.vg, kAmber);
		nvgTextAlign(args.vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
		nvgText(args.vg, layout::kTitleX, layout::kTitleY, "FUSION CLONE", NULL);
	}
};

/** Voice count, lock state, source spectrum and detune density strip. */
struct CloneDisplay : TransparentWidget {
	FusionClone* module = NULL;

	void drawLayer(const DrawArgs& args, int layer) override {
		if (layer == 1) {
			NVGcontext* vg = args.vg;
			fc::EngineStatus st;
			int voices = 8, quality = 1, algorithm = 0;
			float range = 20.f;
			bool connected = true, poly = false;
			if (module) {
				if (!module->readSnapshot(st, voices, quality, algorithm, range, connected, poly))
					return;
			} else {
				// module browser preview: a plausible saw spectrum
				for (int i = 0; i < fc::EngineStatus::kSpecBins; i++)
					st.spectrumDb[i] = -20.f * std::log10((float) (i + 1)) * 1.0f;
				st.locked = true;
				st.unitFreqHz = 110.0;
				for (int v = 1; v < fc::kMaxVoices; v++)
					st.voiceCents[v] = 9.f * std::sin(1.7f * v) * (1.f + 0.2f * v);
			}
			std::shared_ptr<Font> font = uiFont();
			if (!font)
				return;
			const float W = box.size.x, H = box.size.y;
			// big voice count
			char buf[32];
			nvgFontFaceId(vg, font->handle);
			nvgFontSize(vg, 46.f);
			nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
			nvgFillColor(vg, kAmber);
			snprintf(buf, sizeof buf, "%d", voices);
			nvgText(vg, 8.f, 46.f, buf, NULL);
			nvgFontSize(vg, 7.f);
			nvgFillColor(vg, kAmberDim);
			nvgText(vg, 9.f, 58.f, voices == 1 ? "VOICE" : "VOICES", NULL);

			// state line
			nvgFontSize(vg, 8.f);
			nvgTextAlign(vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BASELINE);
			static const char* qn[] = {"ECO", "BAL", "HIGH", "ULTRA"};
			const char* stateTxt = "NO INPUT";
			if (connected && st.locked) {
				snprintf(buf, sizeof buf, "LOCK %.1f Hz", st.unitFreqHz);
				stateTxt = buf;
			} else if (connected && st.inputLevelDb > -70.f)
				stateTxt = st.acquiring ? "ACQUIRING" : "PASS-THRU";
			nvgFillColor(vg, st.locked ? kTeal : kAmberDim);
			nvgText(vg, W - 6.f, 12.f, stateTxt, NULL);
			nvgFillColor(vg, kAmberDim);
			char b2[48];
			snprintf(b2, sizeof b2, "%s  %s", qn[clamp(quality, 0, 3)], algorithm ? "FUSION" : "CLASSIC");
			nvgText(vg, W - 6.f, 22.f, b2, NULL);

			// source spectrum bars (harmonics 1..48, 0..-60 dB)
			const float x0 = 78.f, x1 = W - 6.f, yb = 56.f, hmax = 26.f;
			const int nb = fc::EngineStatus::kSpecBins;
			const float bw = (x1 - x0) / nb;
			for (int i = 0; i < nb; i++) {
				float d = clamp(st.spectrumDb[i], -60.f, 0.f);
				float h = (d + 60.f) / 60.f * hmax;
				nvgBeginPath(vg);
				nvgRect(vg, x0 + i * bw + 0.5f, yb - h, std::max(1.f, bw - 1.f), std::max(h, 0.5f));
				nvgFillColor(vg, i == 0 ? kAmber : nvgRGBA(0xff, 0xb0, 0x30, 150));
				nvgFill(vg);
			}
			nvgFontSize(vg, 6.5f);
			nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
			nvgFillColor(vg, kAmberDim);
			nvgText(vg, x0, 12.f, "PARTIALS", NULL);

			// detune density strip: original at 0 (bright), clones as dots at their static+drift offset
			const float ys = H - 8.f;
			const float cx = W * 0.5f;
			const float rng = std::max(10.f, range * 0.6f);
			const float scale = (W * 0.5f - 12.f) / rng;
			nvgBeginPath(vg);
			nvgMoveTo(vg, 8.f, ys);
			nvgLineTo(vg, W - 8.f, ys);
			nvgStrokeColor(vg, kAmberDim);
			nvgStrokeWidth(vg, 0.7f);
			nvgStroke(vg);
			for (int v = 1; v < voices && v < fc::kMaxVoices; v++) {
				float x = cx + clamp(st.voiceCents[v], -rng * 1.05f, rng * 1.05f) * scale;
				nvgBeginPath(vg);
				nvgCircle(vg, x, ys, 2.6f);
				nvgFillColor(vg, nvgRGBA(0x40, 0xd8, 0xc8, 200));
				nvgFill(vg);
			}
			nvgBeginPath(vg);
			nvgRect(vg, cx - 1.5f, ys - 6.f, 3.f, 12.f);
			nvgFillColor(vg, kAmber);
			nvgFill(vg);
			nvgFontSize(vg, 6.5f);
			nvgTextAlign(vg, NVG_ALIGN_LEFT | NVG_ALIGN_BASELINE);
			nvgFillColor(vg, kAmberDim);
			snprintf(buf, sizeof buf, "-%.0fc", rng);
			nvgText(vg, 8.f, ys - 6.f, buf, NULL);
			nvgTextAlign(vg, NVG_ALIGN_RIGHT | NVG_ALIGN_BASELINE);
			snprintf(buf, sizeof buf, "+%.0fc", rng);
			nvgText(vg, W - 8.f, ys - 6.f, buf, NULL);
			nvgTextAlign(vg, NVG_ALIGN_CENTER | NVG_ALIGN_BASELINE);
			nvgText(vg, cx, ys - 8.f, "REAL VCO2", NULL);
		}
		TransparentWidget::drawLayer(args, layer);
	}

	void draw(const DrawArgs& args) override {
		nvgBeginPath(args.vg);
		nvgRoundedRect(args.vg, 0.f, 0.f, box.size.x, box.size.y, 3.f);
		nvgFillColor(args.vg, nvgRGB(0x0b, 0x0e, 0x11));
		nvgFill(args.vg);
		nvgStrokeColor(args.vg, nvgRGB(0x2a, 0x30, 0x38));
		nvgStrokeWidth(args.vg, 1.f);
		nvgStroke(args.vg);
		TransparentWidget::draw(args);
	}
};

/** Slider bound to a ParamQuantity for the expert section of the context menu. */
struct ExpertSlider : ui::Slider {
	explicit ExpertSlider(ParamQuantity* q) {
		quantity = q;
		box.size.x = 220.f;
	}
};

} // namespace

struct FusionCloneWidget : ModuleWidget {
	FusionCloneWidget(FusionClone* module) {
		setModule(module);
		setPanel(createPanel(asset::plugin(pluginInstance, "res/FusionClone.svg")));

		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, 0)));
		addChild(createWidget<ScrewSilver>(Vec(RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));
		addChild(createWidget<ScrewSilver>(Vec(box.size.x - 2 * RACK_GRID_WIDTH, RACK_GRID_HEIGHT - RACK_GRID_WIDTH)));

		PanelLabels* labels = new PanelLabels();
		labels->box.size = box.size;
		addChild(labels);

		CloneDisplay* disp = new CloneDisplay();
		disp->module = module;
		disp->box.pos = Vec(layout::kDispX, layout::kDispY);
		disp->box.size = Vec(layout::kDispW, layout::kDispH);
		addChild(disp);

		addParam(createParamCentered<RoundBigBlackKnob>(Vec(POS_VOICES), module, FusionClone::VOICES_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(Vec(POS_SPREAD), module, FusionClone::SPREAD_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(Vec(POS_DRIFT), module, FusionClone::DRIFT_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(Vec(POS_CHARACTER), module, FusionClone::CHARACTER_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(Vec(POS_PHASE), module, FusionClone::PHASE_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(Vec(POS_HARMONIC), module, FusionClone::HARMONIC_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(Vec(POS_WIDTH), module, FusionClone::WIDTH_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(Vec(POS_MIX), module, FusionClone::MIX_PARAM));
		addParam(createParamCentered<RoundBlackKnob>(Vec(POS_OUTPUT), module, FusionClone::OUTPUT_PARAM));
		addParam(createParamCentered<RoundSmallBlackKnob>(Vec(POS_QUALITY), module, FusionClone::QUALITY_PARAM));
		addParam(createParamCentered<CKSS>(Vec(POS_ALGO), module, FusionClone::ALGO_PARAM));
		addParam(createParamCentered<CKSS>(Vec(POS_ORIG), module, FusionClone::ORIG_PARAM));
		addParam(createParamCentered<TL1105>(Vec(POS_RANDOM), module, FusionClone::RANDOM_PARAM));

		addChild(createLightCentered<SmallLight<GreenLight>>(Vec(POS_LIGHT_LOCK), module, FusionClone::LOCK_LIGHT));
		addChild(createLightCentered<SmallLight<YellowLight>>(Vec(POS_LIGHT_ACQ), module, FusionClone::ACQ_LIGHT));
		addChild(createLightCentered<SmallLight<RedLight>>(Vec(POS_LIGHT_POLY), module, FusionClone::POLY_LIGHT));

		addInput(createInputCentered<PJ301MPort>(Vec(POS_CV_VOICES), module, FusionClone::VOICES_CV_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(POS_CV_SPREAD), module, FusionClone::SPREAD_CV_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(POS_CV_DRIFT), module, FusionClone::DRIFT_CV_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(POS_CV_CHARACTER), module, FusionClone::CHARACTER_CV_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(POS_CV_MIX), module, FusionClone::MIX_CV_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(POS_IN), module, FusionClone::AUDIO_INPUT));
		addOutput(createOutputCentered<PJ301MPort>(Vec(POS_OUT_L), module, FusionClone::L_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(Vec(POS_OUT_R), module, FusionClone::R_OUTPUT));
	}

	void appendContextMenu(ui::Menu* menu) override {
		FusionClone* m = dynamic_cast<FusionClone*>(module);
		if (!m)
			return;
		menu->addChild(new ui::MenuSeparator);
		menu->addChild(createMenuLabel("Expert"));
		struct Entry {
			int id;
			const char* name;
		};
		static const Entry sliders[] = {
		    {FusionClone::DETUNE_RANGE_PARAM, "Detune range"}, {FusionClone::SPREAD_CURVE_PARAM, "Spread curve"},
		    {FusionClone::DRIFT_RATE_PARAM, "Drift rate"},     {FusionClone::DRIFT_CORR_PARAM, "Drift correlation"},
		    {FusionClone::FUSION_SHIFT_PARAM, "Fusion layer"}, {FusionClone::SUMMING_PARAM, "Analog summing"},
		    {FusionClone::LEVEL_LAW_PARAM, "Level law"},
		};
		for (size_t i = 0; i < sizeof(sliders) / sizeof(sliders[0]); i++)
			menu->addChild(new ExpertSlider(m->paramQuantities[sliders[i].id]));
		menu->addChild(createIndexSubmenuItem(
		    "Fusion layer shift model", {"RATIO (Doppler / BBD delay)", "HZ (single-sideband)"},
		    [=]() { return (size_t) (m->params[FusionClone::SHIFT_MODE_PARAM].getValue() > 0.5f ? 1 : 0); },
		    [=](size_t v) { m->params[FusionClone::SHIFT_MODE_PARAM].setValue(v ? 1.f : 0.f); }));
		menu->addChild(new ui::MenuSeparator);
		menu->addChild(createMenuItem("Randomize voice seeds", "", [=]() { m->seed = random::u32(); }));
		menu->addChild(createMenuLabel("Original path: direct (0 samples). Clones bloom in a few periods after an onset/step."));
		char buf[96];
		fc::EngineStatus st;
		int voices = 0, quality = 0, algorithm = 0;
		float range = 0.f;
		bool connected = false, poly = false;
		if (m->readSnapshot(st, voices, quality, algorithm, range, connected, poly)) {
			if (st.locked)
				snprintf(buf, sizeof buf, "Locked: repeating unit %.2f Hz, bloom time %.0f ms at this pitch", st.unitFreqHz, (double) st.bloomMs);
			else
				snprintf(buf, sizeof buf, "Not locked (%s)", connected ? (st.acquiring ? "acquiring" : "no periodic input") : "no input");
			menu->addChild(createMenuLabel(buf));
		}
		snprintf(buf, sizeof buf, "Voice seed: %08X", (unsigned) m->seed);
		menu->addChild(createMenuLabel(buf));
		menu->addChild(createMenuLabel("The clones are perceptually, not electrically, equivalent to real oscillators."));
	}
};

Model* modelFusionClone = createModel<FusionClone, FusionCloneWidget>("FusionClone");
