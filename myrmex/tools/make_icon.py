"""Draw the Myrmex app icon (vector, every size) -> src/myrmex/app/assets/{Myrmex.icns, icon_*.png}.

    QT_QPA_PLATFORM=offscreen python tools/make_icon.py

An ant (myrmex) of glossy black liquid metal, seen from above, on a deep warm graphite squircle with a
soft coral glow behind it - the organisms' material and the app's accent.  Drawn directly at every
size (crisp small icons), packed into an .icns (PNG entries) for the bundle.
"""
from __future__ import annotations

import math
import os
import struct
import sys

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen,
                           QRadialGradient)

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "src", "myrmex", "app", "assets")
CORAL = QColor(217, 119, 87)


def _blob(p: QPainter, cx: float, cy: float, rx: float, ry: float, S: float, rim: float = 1.0) -> None:
    """A glossy black liquid-metal ellipse: dark body, soft top-left light, a sharp specular, a coral rim."""
    r = QRectF((cx - rx) * S, (cy - ry) * S, 2 * rx * S, 2 * ry * S)
    g = QRadialGradient(QPointF((cx - 0.35 * rx) * S, (cy - 0.45 * ry) * S), max(rx, ry) * 1.25 * S)
    g.setColorAt(0.0, QColor(96, 96, 104))
    g.setColorAt(0.22, QColor(38, 38, 42))
    g.setColorAt(0.65, QColor(12, 12, 14))
    g.setColorAt(1.0, QColor(4, 4, 5))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(g)
    p.drawEllipse(r)
    rim_g = QRadialGradient(QPointF((cx + 0.55 * rx) * S, (cy + 0.6 * ry) * S), max(rx, ry) * 0.9 * S)
    c0 = QColor(CORAL)
    c0.setAlpha(int(110 * rim))
    c1 = QColor(CORAL)
    c1.setAlpha(0)
    rim_g.setColorAt(0.0, c0)
    rim_g.setColorAt(1.0, c1)
    p.setBrush(rim_g)
    clip = QPainterPath()
    clip.addEllipse(r)
    p.save()
    p.setClipPath(clip)
    p.drawEllipse(r)
    p.restore()
    sg = QRadialGradient(QPointF((cx - 0.38 * rx) * S, (cy - 0.5 * ry) * S), 0.42 * min(rx, ry) * S)
    sg.setColorAt(0.0, QColor(255, 255, 255, 230))
    sg.setColorAt(0.5, QColor(255, 255, 255, 60))
    sg.setColorAt(1.0, QColor(255, 255, 255, 0))
    p.setBrush(sg)
    p.drawEllipse(QRectF((cx - 0.68 * rx) * S, (cy - 0.78 * ry) * S, 0.62 * rx * S, 0.5 * ry * S))


def _limb(p: QPainter, pts, S: float, w: float) -> None:
    path = QPainterPath(QPointF(pts[0][0] * S, pts[0][1] * S))
    for (x0, y0), (x1, y1) in zip(pts[1:-1], pts[2:]):                 # smooth: quad through the midpoints
        path.quadTo(QPointF(x0 * S, y0 * S), QPointF((x0 + x1) / 2 * S, (y0 + y1) / 2 * S))
    path.lineTo(QPointF(pts[-1][0] * S, pts[-1][1] * S))
    p.setBrush(Qt.BrushStyle.NoBrush)
    pen = QPen(QColor(8, 8, 10), max(1.0, w * S), Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
               Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawPath(path)
    if S >= 48:                                                          # a glint along the limb
        hl = QPen(QColor(150, 150, 160, 120), max(0.6, w * 0.28 * S), Qt.PenStyle.SolidLine,
                  Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(hl)
        p.drawPath(path.translated(-0.25 * w * S, -0.3 * w * S))


def render(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    S = float(size)
    m = 0.1 * S if size >= 64 else 0.04 * S                              # the macOS icon grid (small: fill)
    body = QRectF(m, m, S - 2 * m, S - 2 * m)
    rad = body.width() * 0.225
    sq = QPainterPath()
    sq.addRoundedRect(body, rad, rad)
    if size >= 64:                                                       # a soft drop shadow
        for i in range(10):
            sh = QPainterPath()
            e = (i + 1) * S * 0.004
            sh.addRoundedRect(body.adjusted(-e, -e + S * 0.012, e, e + S * 0.012), rad + e, rad + e)
            p.fillPath(sh, QColor(0, 0, 0, 10))
    bg = QRadialGradient(QPointF(0.42 * S, 0.3 * S), 0.8 * S)
    bg.setColorAt(0.0, QColor(58, 54, 50))
    bg.setColorAt(0.55, QColor(28, 26, 25))
    bg.setColorAt(1.0, QColor(12, 11, 11))
    p.fillPath(sq, bg)
    p.save()
    p.setClipPath(sq)
    glow = QRadialGradient(QPointF(0.5 * S, 0.53 * S), 0.43 * S)
    c = QColor(CORAL)
    c.setAlpha(125)
    glow.setColorAt(0.0, c)
    c2 = QColor(CORAL)
    c2.setAlpha(40)
    glow.setColorAt(0.45, c2)
    glow.setColorAt(1.0, QColor(217, 119, 87, 0))
    p.fillRect(body, glow)
    # the ant, top-down, head up
    w = 0.02 if size >= 64 else 0.03
    legs = [((0.46, 0.38), (0.34, 0.31), (0.26, 0.2)), ((0.445, 0.43), (0.3, 0.46), (0.2, 0.56)),
            ((0.46, 0.48), (0.34, 0.6), (0.27, 0.8))]
    for leg in legs:
        _limb(p, [leg[0], leg[1], leg[2]], S, w)
        _limb(p, [(1 - x, y) for x, y in leg], S, w)
    ant = [(0.475, 0.21), (0.43, 0.145), (0.355, 0.155)]
    _limb(p, ant, S, w * 0.7)
    _limb(p, [(1 - x, y) for x, y in ant], S, w * 0.7)
    _blob(p, 0.5, 0.69, 0.135, 0.16, S)                                 # gaster
    _blob(p, 0.5, 0.535, 0.03, 0.03, S, rim=2.0)                       # the waist, lit from inside
    _blob(p, 0.5, 0.42, 0.065, 0.095, S)                               # thorax
    _blob(p, 0.5, 0.255, 0.085, 0.075, S)                              # head
    if size >= 32:
        for x in (0.355, 0.645):                                        # antenna tips: two coral sparks
            sp = QRadialGradient(QPointF(x * S, 0.155 * S), 0.03 * S)
            sp.setColorAt(0.0, QColor(255, 170, 140, 255))
            sp.setColorAt(0.4, QColor(217, 119, 87, 200))
            sp.setColorAt(1.0, QColor(217, 119, 87, 0))
            p.setBrush(sp)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(x * S, 0.155 * S), 0.03 * S, 0.03 * S)
    p.restore()
    if size >= 64:                                                       # a hairline of light along the top edge
        lg = QLinearGradient(QPointF(0, m), QPointF(0, m + 0.3 * S))
        lg.setColorAt(0.0, QColor(255, 255, 255, 60))
        lg.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.setPen(QPen(lg, max(1.0, S * 0.003)))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(sq)
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba.data())


ICNS_TYPES = (("icp4", 16), ("icp5", 32), ("icp6", 64), ("ic07", 128), ("ic08", 256), ("ic09", 512), ("ic10", 1024),
              ("ic11", 32), ("ic12", 64), ("ic13", 256), ("ic14", 512))


def write_icns(path: str) -> None:
    cache = {}
    entries = b""
    for kind, size in ICNS_TYPES:
        if size not in cache:
            cache[size] = png_bytes(render(size))
        data = cache[size]
        entries += kind.encode("ascii") + struct.pack(">I", len(data) + 8) + data
    with open(path, "wb") as f:
        f.write(b"icns" + struct.pack(">I", len(entries) + 8) + entries)


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    os.makedirs(ASSETS, exist_ok=True)
    write_icns(os.path.join(ASSETS, "Myrmex.icns"))
    for size in (64, 256, 1024):
        render(size).save(os.path.join(ASSETS, f"icon_{size}.png"))
    del app
    print("icon written:", os.path.abspath(ASSETS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
