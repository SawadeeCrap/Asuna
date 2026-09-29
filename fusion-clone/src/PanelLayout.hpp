// Panel layout (pixels; Rack: 1 HP = 15 px, panel height 380 px). 16 HP = 240 px wide.
// `#define POS_xxx x, y` lines are also parsed by tools/preview_panel.py to render a layout preview — keep that exact format.
#pragma once

namespace layout {

static const float kWidth = 240.f;
static const float kHeight = 380.f;

// display window
static const float kDispX = 15.f, kDispY = 30.f, kDispW = 210.f, kDispH = 78.f;

// row 1
#define POS_VOICES 34, 146
#define POS_SPREAD 94, 144
#define POS_DRIFT 152, 144
#define POS_CHARACTER 210, 144
// row 2
#define POS_PHASE 34, 204
#define POS_HARMONIC 94, 204
#define POS_WIDTH 152, 204
#define POS_MIX 210, 204
// row 3
#define POS_OUTPUT 34, 260
#define POS_QUALITY 94, 260
#define POS_ALGO 143, 260
#define POS_ORIG 181, 260
#define POS_RANDOM 216, 260
// lights (under the display)
#define POS_LIGHT_LOCK 96, 116
#define POS_LIGHT_ACQ 120, 116
#define POS_LIGHT_POLY 144, 116
// CV inputs
#define POS_CV_VOICES 29, 320
#define POS_CV_SPREAD 72, 320
#define POS_CV_DRIFT 115, 320
#define POS_CV_CHARACTER 158, 320
#define POS_CV_MIX 201, 320
// audio
#define POS_IN 29, 358
#define POS_OUT_L 172, 358
#define POS_OUT_R 210, 358

struct LabelSpec {
	const char* text;
	float x, y;
	float size;
};

// Static text (drawn by code because Rack's SVG loader ignores <text>).
static const LabelSpec kLabels[] = {
    {"VOICES", 34, 176, 9.f},   {"SPREAD", 94, 172, 9.f},    {"DRIFT", 152, 172, 9.f},    {"CHARACTER", 210, 172, 8.f},
    {"PHASE", 34, 234, 9.f},    {"HARMONIC", 94, 230, 8.f},  {"WIDTH", 152, 230, 9.f},    {"MIX", 210, 230, 9.f},
    {"OUTPUT", 34, 290, 9.f},   {"QUALITY", 94, 286, 9.f},   {"MODE", 143, 286, 9.f},     {"ORIG", 181, 286, 9.f},
    {"RAND", 216, 286, 9.f},    {"VOICES", 29, 302, 7.f},    {"SPREAD", 72, 302, 7.f},    {"DRIFT", 115, 302, 7.f},
    {"CHAR", 158, 302, 7.f},    {"MIX", 201, 302, 7.f},      {"AUDIO IN", 29, 340, 7.f},  {"OUT L", 172, 340, 7.f},
    {"OUT R", 210, 340, 7.f},   {"LOCK", 96, 128, 6.5f},     {"ACQ", 120, 128, 6.5f},     {"POLY", 144, 128, 6.5f},
    {"CV IN", 115, 312, 6.5f},
};
static const int kNumLabels = (int) (sizeof(kLabels) / sizeof(kLabels[0]));

} // namespace layout
