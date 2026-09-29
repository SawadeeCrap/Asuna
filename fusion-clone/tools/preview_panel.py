#!/usr/bin/env python3
"""Render a layout preview of the module panel from src/PanelLayout.hpp and check it for overlaps.

    python3 tools/preview_panel.py [--out docs/figures/panel_preview.png] [--svg docs/figures/panel_preview.svg]

Parses the `#define POS_xxx x, y` lines and the label table of src/PanelLayout.hpp (positions in Rack pixels, 15 px per HP, 380 px high),
draws each widget with the size of its Rack counterpart on top of the panel background res/FusionClone.svg, and reports (a) overlapping
widget / label / corner-screw boxes and (b) widgets or labels that straddle a section frame of the panel. The PNG is rendered with
Playwright's Chromium if available, else with a Chromium / Chrome binary found on the machine (otherwise only the SVG is written). The display
window shows a *mock* drawn with the geometry and font sizes of CloneDisplay (module-browser preview data). This is a layout aid, not a
screenshot of Rack: the real look comes from Rack's widgets and the fonts loaded at run time.
"""
import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT = os.path.join(HERE, "..", "src", "PanelLayout.hpp")
PANEL = os.path.join(HERE, "..", "res", "FusionClone.svg")
# Rack draws ScrewSilver widgets (15 x 15) at the four corners (src/FusionClone.cpp)
TITLE = (15.0, 22.0, 15.0)  # x, baseline y, font size of "FUSION CLONE" (overridden from PanelLayout.hpp)
SCREWS = [("SCREW_TL", 15, 0), ("SCREW_TR", 210, 0), ("SCREW_BL", 15, 365), ("SCREW_BR", 210, 365)]

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
              for m in re.finditer(r'\{"([^"]+)"\s*,\s*([\d.]+)f?\s*,\s*([\d.]+)f?\s*,\s*([\d.]+)f?\s*\}', src)]
    disp = re.search(r"kDispX\s*=\s*([\d.]+)f?,\s*kDispY\s*=\s*([\d.]+)f?,\s*kDispW\s*=\s*([\d.]+)f?,\s*kDispH\s*=\s*([\d.]+)f?", src)
    global TITLE
    t = re.search(r"kTitleX\s*=\s*([\d.]+)f?,\s*kTitleY\s*=\s*([\d.]+)f?,\s*kTitleSize\s*=\s*([\d.]+)f?", src)
    if t:
        TITLE = tuple(float(v) for v in t.groups())
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
    for name, x, y in SCREWS:
        out.append(("widget", name, x, y, x + 15, y + 15, "screw"))
    tx, ty, tsz = TITLE
    out.append(("label", "TITLE", tx, ty - tsz * 0.75, tx + 0.62 * tsz * len("FUSION CLONE"), ty + tsz * 0.2, "text"))
    for text, x, y, size in labels:
        w = 0.62 * size * len(text)
        out.append(("label", text, x - w / 2, y - size * 0.8, x + w / 2, y + size * 0.2, "text"))
    return out


def frames():
    """Section frames of the panel background (rounded rectangles with rx=4 in res/FusionClone.svg)."""
    out = []
    try:
        src = open(PANEL).read()
    except OSError:
        return out
    for m in re.finditer(r"<rect\b([^>]*)>", src):
        attrs = dict(re.findall(r'(?<![\w-])([\w-]+)="([^"]*)"', m.group(1)))
        if attrs.get("rx") == "4":
            x, y, w, h = (float(attrs[k]) for k in ("x", "y", "width", "height"))
            out.append((x, y, x + w, y + h))
    return out


def straddling(bxs, frs, disp):
    """Widgets / labels that cross the border of a frame (or sit in none, outside the title band and the display)."""
    dx, dy, dw, dh = disp
    found = []
    for kind, name, x0, y0, x1, y1, shape in bxs:
        if shape == "screw" or y1 <= 28.5 or (x0 >= dx and x1 <= dx + dw and y0 >= dy and y1 <= dy + dh):
            continue
        inside = [f for f in frs if x0 >= f[0] - 0.5 and x1 <= f[2] + 0.5 and y0 >= f[1] - 0.5 and y1 <= f[3] + 0.5]
        if not inside:
            found.append((kind, name))
    return found


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


def display_mock(disp):
    """SVG group that mimics CloneDisplay::drawLayer (src/FusionClone.cpp) with the module-browser preview data: same geometry, same
    font sizes. A mock for the figure, not a capture of what Rack draws."""
    import math
    dx, dy, W, H = disp
    amber, dim, teal = "#ffb030", "rgba(255,176,48,0.27)", "#40d8c8"
    o = ['<g transform="translate(%g,%g)" font-family="monospace">' % (dx, dy)]
    o.append('<text x="8" y="46" font-size="46" fill="%s">8</text>' % amber)
    o.append('<text x="9" y="58" font-size="7" fill="%s">VOICES</text>' % dim)
    o.append('<text x="%g" y="12" font-size="8" fill="%s" text-anchor="end">LOCK 110.0 Hz</text>' % (W - 6, teal))
    o.append('<text x="%g" y="22" font-size="8" fill="%s" text-anchor="end">BAL  CLASSIC</text>' % (W - 6, dim))
    x0, x1, yb, hmax, nb = 78.0, W - 6.0, 56.0, 26.0, 48
    bw = (x1 - x0) / nb
    for i in range(nb):
        d = max(-60.0, min(0.0, -20.0 * math.log10(i + 1)))
        h = (d + 60.0) / 60.0 * hmax
        o.append('<rect x="%g" y="%g" width="%g" height="%g" fill="%s"/>' % (x0 + i * bw + 0.5, yb - h, max(1.0, bw - 1.0), max(h, 0.5), amber if i == 0 else "rgba(255,176,48,0.59)"))
    o.append('<text x="%g" y="12" font-size="6.5" fill="%s">PARTIALS</text>' % (x0, dim))
    ys, cx, rng = H - 8.0, W * 0.5, max(10.0, 20.0 * 0.6)
    scale = (W * 0.5 - 12.0) / rng
    o.append('<line x1="8" y1="%g" x2="%g" y2="%g" stroke="%s" stroke-width="0.7"/>' % (ys, W - 8, ys, dim))
    for v in range(1, 8):
        c = 9.0 * math.sin(1.7 * v) * (1.0 + 0.2 * v)
        x = cx + max(-rng * 1.05, min(rng * 1.05, c)) * scale
        o.append('<circle cx="%g" cy="%g" r="2.6" fill="rgba(64,216,200,0.78)"/>' % (x, ys))
    o.append('<rect x="%g" y="%g" width="3" height="12" fill="%s"/>' % (cx - 1.5, ys - 6, amber))
    o.append('<text x="8" y="%g" font-size="6.5" fill="%s">-%.0fc</text>' % (ys - 6, dim, rng))
    o.append('<text x="%g" y="%g" font-size="6.5" fill="%s" text-anchor="end">+%.0fc</text>' % (W - 8, ys - 6, dim, rng))
    o.append('<text x="%g" y="%g" font-size="6.5" fill="%s" text-anchor="middle">REAL VCO2</text>' % (cx, ys - 8, dim))
    o.append("</g>")
    return "\n".join(o)


def svg(pos, labels, disp, width, height, bxs):
    dx, dy, dw, dh = disp
    s = ['<svg xmlns="http://www.w3.org/2000/svg" width="%g" height="%g" viewBox="0 0 %g %g" font-family="monospace">' % (width * 2, height * 2, width, height),
         '<rect width="%g" height="%g" fill="#1b1e24"/>' % (width, height)]
    try:  # the panel background (section frames, title band) as it ships in res/
        body = open(PANEL).read()
        body = body[body.index("<defs>"):body.rindex("</svg>")]
        s.append(body)
    except (OSError, ValueError):
        pass
    s += ['<rect x="%g" y="%g" width="%g" height="%g" rx="3" fill="#0b0d10" stroke="#ffb030" stroke-width="0.6"/>' % (dx, dy, dw, dh),
          display_mock(disp),
          '<text x="%g" y="%g" font-size="%g" fill="#ffb030">FUSION CLONE</text>' % TITLE]
    for kind, name, x0, y0, x1, y1, shape in bxs:
        if kind != "widget":
            continue
        cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if shape in ("knob", "port", "light"):
            fill = {"knob": "#3a3f48", "port": "#252a31", "light": "#40d8c8"}[shape]
            s.append('<circle cx="%g" cy="%g" r="%g" fill="%s" stroke="#8a929e" stroke-width="0.5"/>' % (cx, cy, w / 2, fill))
        elif shape == "screw":
            s.append('<circle cx="%g" cy="%g" r="6" fill="#b8bcc4" stroke="#6c727c" stroke-width="0.6"/>' % (cx, cy))
        else:
            s.append('<rect x="%g" y="%g" width="%g" height="%g" rx="2" fill="#3a3f48" stroke="#8a929e" stroke-width="0.5"/>' % (x0, y0, w, h))
    for text, x, y, size in labels:
        s.append('<text x="%g" y="%g" font-size="%g" fill="#d8dce2" text-anchor="middle">%s</text>' % (x, y, size, text))
    s.append("</svg>")
    return "\n".join(s)


def crop_png(path, out_w, out_h):
    """Crop an 8-bit RGB/RGBA PNG to its top-left out_w x out_h corner (pure Python, no imaging library needed)."""
    import struct
    import zlib
    data = open(path, "rb").read()
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        n, typ = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif typ == b"IDAT":
            idat += chunk
        pos += 12 + n
    w, h, depth, ctype, _, _, interlace = ihdr
    if depth != 8 or ctype not in (2, 6) or interlace:
        return False
    bpp = 3 if ctype == 2 else 4
    raw = zlib.decompress(idat)
    stride = w * bpp
    rows, prev = [], bytearray(stride)
    for y in range(min(h, out_h)):
        f = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if f == 1:
                line[i] = (line[i] + a) & 255
            elif f == 2:
                line[i] = (line[i] + b) & 255
            elif f == 3:
                line[i] = (line[i] + ((a + b) >> 1)) & 255
            elif f == 4:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 255
        prev = line
        rows.append(bytes([0]) + bytes(line[:min(w, out_w) * bpp]))
    ow, oh = min(w, out_w), len(rows)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", ow, oh, 8, ctype, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)
    return True


def render_with_browser(svg_path, png_path, width, height):
    """Fallback renderer: any Chromium / Chrome binary in PATH or in a Playwright browser cache."""
    import glob
    import shutil
    import subprocess
    cands = [shutil.which(n) for n in ("chromium", "chromium-browser", "google-chrome", "chrome")]
    cands += sorted(glob.glob(os.path.expanduser("/opt/pw-browsers/chromium-*/chrome-linux/chrome")))
    cands += sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")))
    for c in cands:
        if not c:
            continue
        try:
            html = os.path.abspath(svg_path) + ".html"
            with open(html, "w") as f:
                f.write("<body style='margin:0;background:#1b1e24'>" + open(svg_path).read() + "</body>")
            subprocess.run([c, "--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", "--screenshot=" + os.path.abspath(png_path),
                            "--window-size=%d,%d" % (width * 2, height * 2 + 86), "file://" + html],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            os.remove(html)
            crop_png(png_path, int(width * 2), int(height * 2))
            print("wrote", png_path, "(rendered with %s)" % os.path.basename(c))
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


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
    strad = straddling(bxs, frames(), disp)
    for kind, name in strad:
        print("NOT INSIDE ONE SECTION FRAME: %s %s" % (kind, name))
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
        if not render_with_browser(a.svg, a.out, width, height):
            print("PNG not rendered (%s); the SVG is the preview" % e, file=sys.stderr)
    return 1 if (ov or strad) else 0


if __name__ == "__main__":
    sys.exit(main())
