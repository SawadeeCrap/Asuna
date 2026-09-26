"""The sidebar's line icons (painted per theme, no image files) and the app icon.

Icons are drawn in a 24-unit box with round 1.7-unit strokes, like the rest of the interface; each has a
normal (secondary text colour) and a selected (accent) version.
"""
from __future__ import annotations

import math
import os

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _live(p: QPainter) -> None:
    path = QPainterPath(QPointF(2, 12))
    for x, y in ((6, 12), (8.5, 6), (11.5, 18), (14.5, 4), (17, 14), (19, 12), (22, 12)):
        path.lineTo(x, y)
    p.drawPath(path)


def _character(p: QPainter) -> None:
    hexa = QPainterPath()
    for k in range(7):
        a = math.pi / 6 + k * math.pi / 3
        pt = QPointF(12 + 8.8 * math.cos(a), 12 + 8.8 * math.sin(a))
        hexa.moveTo(pt) if k == 0 else hexa.lineTo(pt)
    p.drawPath(hexa)
    p.drawEllipse(QPointF(12, 10.2), 2.6, 2.6)
    arc = QPainterPath(QPointF(7.6, 17.2))
    arc.quadTo(12, 12.6, 16.4, 17.2)
    p.drawPath(arc)


def _creature(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 3.8, 3.8)
    for (x0, y0), (cx, cy), (x1, y1) in (((12, 8.2), (10, 3.5), (5.5, 3.5)), ((15.4, 13.8), (20.5, 14.5), (21, 19.5)),
                                         ((8.6, 13.8), (5.5, 17), (3, 20.5))):
        path = QPainterPath(QPointF(x0, y0))
        path.quadTo(cx, cy, x1, y1)
        p.drawPath(path)


def _camera(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(2.5, 7, 13, 10), 2.2, 2.2)
    path = QPainterPath(QPointF(15.5, 10.5))
    for x, y in ((21.5, 7.5), (21.5, 16.5), (15.5, 13.5)):
        path.lineTo(x, y)
    path.closeSubpath()
    p.drawPath(path)


def _glove(p: QPainter) -> None:
    for x, y0, y1 in ((8, 13, 6.5), (11, 12, 4), (14, 12, 4.5), (17, 13, 7)):
        p.drawLine(QPointF(x, y0), QPointF(x, y1))
    palm = QPainterPath(QPointF(6, 12.5))
    palm.quadTo(5.5, 21, 12, 21)
    palm.quadTo(18.5, 21, 18, 12.5)
    p.drawPath(palm)
    p.drawLine(QPointF(6, 14.5), QPointF(3.5, 10.8))


def _inputs(p: QPainter) -> None:
    for x, knob in ((6, 15), (12, 8), (18, 12.5)):
        p.drawLine(QPointF(x, 3.5), QPointF(x, 20.5))
        p.save()
        p.setBrush(p.pen().color())
        p.drawEllipse(QPointF(x, knob), 2.2, 2.2)
        p.restore()


def _midi(p: QPainter) -> None:
    p.drawEllipse(QPointF(12, 12), 9, 9)
    p.save()
    p.setBrush(p.pen().color())
    p.setPen(Qt.PenStyle.NoPen)
    for a in (180, 225, 270, 315, 0):
        r = math.radians(a)
        p.drawEllipse(QPointF(12 + 5.2 * math.cos(r), 12.6 + 5.2 * math.sin(r)), 1.25, 1.25)
    p.restore()
    p.drawLine(QPointF(10.6, 18.4), QPointF(13.4, 18.4))


def _takes(p: QPainter) -> None:
    p.drawRoundedRect(QRectF(3, 6, 18, 12), 3, 3)
    p.save()
    p.setBrush(p.pen().color())
    p.drawEllipse(QPointF(12, 12), 2.8, 2.8)
    p.restore()


def _log(p: QPainter) -> None:
    for y in (6, 10, 14, 18):
        p.drawLine(QPointF(8.5, y), QPointF(20.5, y))
        p.save()
        p.setBrush(p.pen().color())
        p.drawEllipse(QPointF(4.8, y), 0.9, 0.9)
        p.restore()


def _touch(p: QPainter) -> None:
    """TouchDesigner: a screen with a wave of light running through it."""
    p.drawRoundedRect(QRectF(2.5, 4.5, 19, 13), 2.4, 2.4)
    path = QPainterPath(QPointF(5.5, 11.5))
    for x, y in ((8, 8.5), (10.5, 14), (13, 7.5), (15.5, 13), (18.5, 10)):
        path.lineTo(x, y)
    p.drawPath(path)
    p.drawLine(QPointF(9, 20.5), QPointF(15, 20.5))


DRAW = {"Live": _live, "Character": _character, "Creature": _creature, "Camera": _camera, "Glove": _glove,
        "TouchDesigner": _touch, "Inputs": _inputs, "MIDI": _midi, "Takes": _takes, "Log": _log}


def _pixmap(draw, color: str, size: int = 22, ratio: float = 2.0) -> QPixmap:
    pm = QPixmap(int(size * ratio), int(size * ratio))
    pm.fill(Qt.GlobalColor.transparent)
    pm.setDevicePixelRatio(ratio)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / 24.0, size / 24.0)
    p.setPen(QPen(QColor(color), 1.7, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)
    draw(p)
    p.end()
    return pm


def nav_icon(name: str, tok: dict, size: int = 22) -> QIcon:
    draw = DRAW.get(name, _log)
    icon = QIcon()
    icon.addPixmap(_pixmap(draw, tok["text2"], size), QIcon.Mode.Normal)
    icon.addPixmap(_pixmap(draw, tok["text"], size), QIcon.Mode.Active)
    icon.addPixmap(_pixmap(draw, tok["accent"], size), QIcon.Mode.Selected)
    return icon


def app_icon() -> QIcon:
    icon = QIcon()
    for s in (64, 256, 1024):
        f = os.path.join(ASSETS, f"icon_{s}.png")
        if os.path.exists(f):
            icon.addFile(f, QSize(s, s))
    return icon


def logo(size: int = 34) -> QPixmap:
    f = os.path.join(ASSETS, "icon_256.png")
    pm = QPixmap(f) if os.path.exists(f) else QPixmap()
    if pm.isNull():
        return pm
    pm = pm.scaled(size * 2, size * 2, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    pm.setDevicePixelRatio(2.0)
    return pm


__all__ = ["nav_icon", "app_icon", "logo", "DRAW"]
