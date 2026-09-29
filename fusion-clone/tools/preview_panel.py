#!/usr/bin/env python3
"""Render a layout preview of the module panel from src/PanelLayout.hpp and check it for overlaps.

    python3 tools/preview_panel.py [--out docs/figures/panel_preview.png] [--svg docs/figures/panel_preview.svg]

Parses the `#define POS_xxx x, y` lines and the label table of src/PanelLayout.hpp (positions in Rack pixels, 15 px per HP, 380 px high),
draws each widget with the size of its Rack counterpart, and reports overlapping widget/label boxes. The PNG is rendered with Playwright's
Chromium if available (otherwise only the SVG is written). This is a layout aid, not a screenshot of Rack: the real look comes from
Rack's widgets and the fonts loaded at run time.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT = os.path.join(HERE, "..", "src", "PanelLayout.hpp")

# widget kind by name: (shape, width, height) in px as drawn by Rack's default widgets used in src/FusionClone.cpp
KINDS = [
    ("VOICES", ("knob", 46, 46)),           # RoundBigBlackKnob
    ("QUALITY", ("knob", 26, 26)),          # RoundSmallBlackKnob
    ("LIGHT_", ("light", 8, 8)),            # SmallLight
    ("ALGO", ("switch", 14, 26)),           # CKSS
    ("ORIG", ("switch", 14, 26)),
    ("RANDOM", ("button", 14, 14)),         # TL1105
    ("CV_", ("port", 24, 24)),              # PJ301MPort
    ("IN", ("port", 24, 24)),
    ("OUT_", ("port", 24, 24)),
]
DEFAULT_KIND = ("knob", 38, 38)             # RoundBlackKnob


def parse():
    src = open(LAYOUT).read()
    pos = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in re.finditer(r"#define\s+POS_(\w+)\s+([\d.]+)\s*,\s*([\d.]+)", src)}
    labels = [(m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)))
              for m in re.finditer(r'\{"([^"]+)"\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)f?\s*\}', src)]
    disp = re.search(r"kDispX\s*=\s*([\d.]+)f?,\s*kDispY\s*=\s*([\d.]+)f?,\s*kDispW\s*=\s*([\d.]+)f?,\s*kDispH\s*=\s*([\d.]+)f?", src)
    width = float(re.search(r"kWidth\s*=\s*([\d.]+)", src).group(1))
    height = float(re.search(r"kHeight\s*=\s*([\d.]+)", src).group(1))
    return pos, labels, tuple(float(v) for v in disp.groups()), width, height


def kind_of(name):
    for key, k in KINDS:
        if name == key or (key.endswith("_") and name.startswith(key)):
            return k
    return DEFAULT_KIND


def boxes(pos, labels):
    out = []
    for name, (x, y) in pos.items():
        shape, w, h = kind_of(name)
        out.append(("widget", name, x - w / 2, y - h / 2, x + w / 2, y + h / 2, shape))
    for text, x, y, size in labels:
        w = 0.62 * size * len(text)
        out.append(("label", text, x - w / 2, y - size * 0.8, x + w / 2, y + size * 0.2, "text"))
    return out


def overlaps(bxs, disp):
    found = []
    dx, dy, dw, dh = disp
    for i in range(len(bxs)):
        a = bxs[i]
        if a[0] == "widget" and not (a[4] <= dx or a[2] >= dx + dw or a[5] <= dy or a[3] >= dy + dh):
            pass  # lights sit under the display window on purpose (y > dy + dh), widgets must not overlap it
        for j in range(i + 1, len(bxs)):
            b = bxs[j]
            if a[0] == "label" and b[0] == "label":
                pass
            ox = min(a[4], b[4]) - max(a[2], b[2])
            oy = min(a[5], b[5]) - max(a[3], b[3])
            if ox > 1.0 and oy > 1.0:
                found.append((a[1], b[1], ox, oy))
    return found


def svg(pos, labels, disp, width, height, bxs):
    dx, dy, dw, dh = disp
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%g" height="%g" viewBox="0 0 %g %g" font-family="monospace">' % (width * 2, height * 2, width, height),
         '<rect width="%g" height="%g" fill="#1b1e24"/>' % (width, height),
         '<rect x="%g" y="%g" width="%g" height="%g" rx="3" fill="#0b0d10" stroke="#ffb030" stroke-width="0.6"/>' % (dx, dy, dw, dh),
         '<text x="15" y="22" font-size="15" fill="#ffb030">FUSION CLONE</text>']
    for kind, name, x0, y0, x1, y1, shape in bxs:
        if kind != "widget":
            continue
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if shape in ("knob", "port", "light"):
            fill = {"knob": "#3a3f48", "port": "#252a31", "light": "#40d8c8"}[shape]
            s.append('<circle cx="%g" cy="%g" r="%g" fill="%s" stroke="#8a929e" stroke-width="0.5"/>' % (cx, cy, w / 2, fill))
        else:
            s.append('<rect x="%g" y="%g" width="%g" height="%g" rx="2" fill="#3a3f48" stroke="#8a929e" stroke-width="0.5"/>' % (x0, y0, w, h))
    for text, x, y, size in labels:
        s.append('<text x="%g" y="%g" font-size="%g" fill="#d8dce2" text-anchor="middle">%s</text>' % (x, y, size, text))
    s.append("</svg>")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "figures", "panel_preview.png"))
    ap.add_argument("--svg", default=os.path.join(HERE, "..", "docs", "figures", "panel_preview.svg"))
    a = ap.parse_args()
    pos, labels, disp, width, height = parse()
    bxs = boxes(pos, labels)
    ov = overlaps(bxs, disp)
    print("%d widgets, %d labels, panel %g x %g px (%g HP)" % (len(pos), len(labels), width, height, width / 15))
    if ov:
        print("OVERLAPS:")
        for n1, n2, ox, oy in ov:
            print("  %-10s %-10s overlap %.1f x %.1f px" % (n1, n2, ox, oy))
    else:
        print("no overlapping widget / label boxes")
    for kind, name, x0, y0, x1, y1, shape in bxs:
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            print("OUTSIDE PANEL: %s %s (%.0f,%.0f)-(%.0f,%.0f)" % (kind, name, x0, y0, x1, y1))
    os.makedirs(os.path.dirname(a.svg), exist_ok=True)
    doc = svg(pos, labels, disp, width, height, bxs)
    with open(a.svg, "w") as f:
        f.write(doc)
    print("wrote", a.svg)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": int(width * 2), "height": int(height * 2)})
            pg.set_content("<body style='margin:0;background:#000'>%s</body>" % doc)
            pg.screenshot(path=a.out)
            b.close()
        print("wrote", a.out)
    except Exception as e:  # noqa: BLE001
        print("PNG not rendered (%s); the SVG is the preview" % e, file=sys.stderr)
    return 1 if ov else 0


if __name__ == "__main__":
    sys.exit(main())
