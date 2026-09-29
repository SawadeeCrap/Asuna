// Panel layout (pixels; Rack: 1 HP = 15 px, panel height 380 px). 16 HP = 240 px wide.
// `#define POS_xxx x, y` lines are also parsed by tools/preview_panel.py to render a layout preview — keep that exact format.
#pragma once

namespace layout {

static const float kWidth = 240.f;
static const float kHeight = 380.f;

// title (drawn by code at the left of the title band, clear of the corner screw at x 15..30)
static const float kTitleX = 34.f, kTitleY = 21.f, kTitleSize = 13.f;

// display window
static const float kDispX = 15.f, kDispY = 30.f, kDispW = 210.f, kDispH = 78.f;

// row 1 (section frame y 112..172)
#define POS_VOICES 34, 139
#define POS_SPREAD 94, 137
#define POS_DRIFT 152, 137
#define POS_CHARACTER 210, 137
// row 2 (frame y 172..232)
#define POS_PHASE 34, 197
#define POS_HARMONIC 94, 197
#define POS_WIDTH 152, 197
#define POS_MIX 210, 197
// row 3 (frame y 232..292)
#define POS_OUTPUT 34, 257
#define POS_QUALITY 94, 257
#define POS_ALGO 143, 257
#define POS_ORIG 181, 257
#define POS_RANDOM 216, 257
// status lights (title band, right of the name)
#define POS_LIGHT_LOCK 150, 9
#define POS_LIGHT_ACQ 172, 9
#define POS_LIGHT_POLY 194, 9
// CV inputs (frame y 292..328)
#define POS_CV_VOICES 29, 314
#define POS_CV_SPREAD 72, 314
#define POS_CV_DRIFT 115, 314
#define POS_CV_CHARACTER 158, 314
#define POS_CV_MIX 201, 314
// audio (frame y 328..372; the corner screws occupy y >= 365 at x 15..30 and 210..225)
#define POS_IN 29, 352
#define POS_OUT_L 172, 352
#define POS_OUT_R 210, 352

struct LabelSpec {
	const char* text;
	float x, y;
	float size;
};

// Static text (drawn by code because Rack's SVG loader ignores <text>).
static const LabelSpec kLabels[] = {
    {"VOICES", 34, 170.5f, 9.f},   {"SPREAD", 94, 167, 9.f},     {"DRIFT", 152, 167, 9.f},    {"CHARACTER", 210, 167, 7.5f},
    {"PHASE", 34, 227, 9.f},       {"HARMONIC", 94, 227, 8.f},   {"WIDTH", 152, 227, 9.f},    {"MIX", 210, 227, 9.f},
    {"OUTPUT", 34, 287, 9.f},      {"QUALITY", 94, 287, 9.f},    {"MODE", 143, 287, 9.f},     {"ORIG", 181, 287, 9.f},
    {"RAND", 216, 287, 9.f},       {"CV VOICES", 29, 300.5f, 6.5f}, {"CV SPREAD", 72, 300.5f, 6.5f}, {"CV DRIFT", 115, 300.5f, 6.5f},
    {"CV CHAR", 158, 300.5f, 6.5f}, {"CV MIX", 201, 300.5f, 6.5f}, {"AUDIO IN", 29, 338.5f, 7.f}, {"OUT L", 172, 338.5f, 7.f},
    {"OUT R", 210, 338.5f, 7.f},   {"LOCK", 150, 22, 6.f},       {"ACQ", 172, 22, 6.f},       {"POLY", 194, 22, 6.f},
    {"1", 13, 169, 7.f},           {"16", 56, 169, 7.f}, // ends of the VOICES scale
};
static const int kNumLabels = (int) (sizeof(kLabels) / sizeof(kLabels[0]));

} // namespace layout
