"""Myrmex look: calm warm neutrals, one coral accent, soft rounded cards (light and dark, follows macOS).

Pure Qt style sheet on the Fusion style - no images or effects to load, nothing heavy to draw.
The few glyphs Qt needs as images (chevrons, the check mark) are painted once at start-up.
"""
from __future__ import annotations

import os
import tempfile

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QImage, QPainter, QPainterPath, QPalette, QPen
from PySide6.QtWidgets import QApplication

DARK = dict(bg="#262624", side="#1f1e1d", card="#2d2c2a", field="#1f1e1d", hover="#34332f", navsel="#3a3935",
            border="#3a3935", border2="#4a4944", text="#f5f4ee", text2="#b8b5aa", text3="#85827a",
            accent="#d97757", accent_h="#e08a6d", accent_p="#c4674a", on_accent="#ffffff", tint="#3d2e28",
            ok="#8fbf7a", warn="#e0b050", err="#e8705f", handle="#f5f4ee")
LIGHT = dict(bg="#faf9f5", side="#f1efe8", card="#ffffff", field="#ffffff", hover="#ebe8df", navsel="#e5e1d6",
             border="#e6e3da", border2="#d4d0c5", text="#1f1e1d", text2="#5c5a55", text3="#8e8b83",
             accent="#c96442", accent_h="#b85a3a", accent_p="#a44f33", on_accent="#ffffff", tint="#f7e6de",
             ok="#4f8a3a", warn="#b7791f", err="#c0392b", handle="#ffffff")
T: dict = dict(DARK)          # the tokens in use (widgets that paint themselves read them)

QSS = """
* { outline: 0; }
QWidget { color: %(text)s; font-size: 13px; }
QMainWindow, QWidget#page, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget, QDialog, QMessageBox {
    background: %(bg)s; }
QWidget#sidebar { background: %(side)s; border-right: 1px solid %(border)s; }
QLabel#brand { font-size: 17px; font-weight: 600; }
QLabel#brandsub { color: %(text3)s; font-size: 11px; }
QLabel#pagetitle { font-size: 22px; font-weight: 600; }
QLabel#pagesub, QLabel[muted="true"] { color: %(text3)s; }
QLabel#statlabel { color: %(text3)s; font-size: 11px; }
QLabel#statvalue { font-size: 15px; }
QLabel#enginestate { color: %(text2)s; font-size: 12px; }
QLabel#badge { color: %(text2)s; background: %(hover)s; border-radius: 9px; padding: 2px 9px; font-size: 11px; }

QListWidget#nav { background: transparent; border: none; padding: 0; }
QListWidget#nav::item { color: %(text2)s; padding: 0; margin: 2px 9px; border-radius: 10px; }
QListWidget#nav::item:hover { background: %(hover)s; color: %(text)s; }
QListWidget#nav::item:selected { background: %(navsel)s; color: %(text)s; }

QGroupBox { background: %(card)s; border: 1px solid %(border)s; border-radius: 12px;
            margin-top: 24px; padding: 16px 14px 12px 14px; font-weight: 600; font-size: 12px; }
QGroupBox QWidget { font-weight: 400; font-size: 13px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 4px; top: 3px;
                   color: %(text2)s; }

QPushButton { background: %(card)s; border: 1px solid %(border2)s; border-radius: 8px; padding: 6px 14px;
              min-height: 18px; }
QPushButton:hover { background: %(hover)s; }
QPushButton:pressed { background: %(navsel)s; }
QPushButton:disabled { color: %(text3)s; border-color: %(border)s; }
QPushButton#primary { background: %(accent)s; border: 1px solid %(accent)s; color: %(on_accent)s;
                      font-weight: 600; padding: 7px 18px; }
QPushButton#primary:hover { background: %(accent_h)s; border-color: %(accent_h)s; }
QPushButton#primary:pressed { background: %(accent_p)s; border-color: %(accent_p)s; }
QPushButton#primary[running="true"] { background: %(card)s; color: %(text)s; border: 1px solid %(border2)s; }
QPushButton#primary[running="true"]:hover { background: %(hover)s; }
QPushButton#pill { background: transparent; border: 1px solid %(border2)s; border-radius: 10px;
                   padding: 2px 10px; min-height: 14px; font-size: 11px; color: %(text3)s; }
QPushButton#pill:checked { color: %(accent)s; border-color: %(accent)s; background: %(tint)s; }
QPushButton#shot { padding: 7px 10px; text-align: left; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: %(field)s; border: 1px solid %(border2)s; border-radius: 7px; padding: 4px 8px; min-height: 20px;
    selection-background-color: %(accent)s; selection-color: %(on_accent)s; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QComboBox:on { border-color: %(accent)s; }
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled { color: %(text3)s; border-color: %(border)s; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: url("%(chev)s"); width: 10px; height: 10px; }
QComboBox QAbstractItemView { background: %(card)s; border: 1px solid %(border2)s; border-radius: 8px; padding: 4px;
                              selection-background-color: %(navsel)s; selection-color: %(text)s; outline: 0; }
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button {
    border: none; background: transparent; width: 18px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url("%(up)s"); width: 8px; height: 8px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: url("%(chev)s"); width: 8px; height: 8px; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 5px; border: 1px solid %(border2)s;
                       background: %(field)s; }
QCheckBox::indicator:hover { border-color: %(accent)s; }
QCheckBox::indicator:checked { background: %(accent)s; border-color: %(accent)s; image: url("%(check)s"); }

QSlider { min-height: 20px; }
QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: %(border)s; }
QSlider::sub-page:horizontal { height: 4px; border-radius: 2px; background: %(accent)s; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
                             background: %(handle)s; border: 1px solid %(border2)s; }
QSlider::sub-page:horizontal:disabled { background: %(border2)s; }
QSlider::handle:horizontal:disabled { background: %(border2)s; border-color: %(border2)s; }

QTableWidget, QListWidget, QPlainTextEdit {
    background: %(field)s; border: 1px solid %(border)s; border-radius: 10px; gridline-color: %(border)s;
    selection-background-color: %(navsel)s; selection-color: %(text)s; }
QTableWidget::item { padding: 3px 6px; }
QHeaderView { background: transparent; }
QHeaderView::section { background: transparent; color: %(text3)s; border: none; border-bottom: 1px solid %(border)s;
                       padding: 6px 8px; font-size: 11px; }
QTableCornerButton::section { background: transparent; border: none; }
QPlainTextEdit#log { padding: 10px; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 3px 2px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px 3px; }
QScrollBar::handle { background: %(border2)s; border-radius: 3px; min-height: 28px; min-width: 28px; }
QScrollBar::handle:hover { background: %(text3)s; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QProgressBar { background: %(border)s; border: none; border-radius: 4px; }
QProgressBar::chunk { background: %(accent)s; border-radius: 4px; }
QToolTip { background: %(card)s; color: %(text)s; border: 1px solid %(border2)s; border-radius: 6px; padding: 6px 8px; }
"""


def _glyphs(tok: dict, tag: str) -> dict:
    """Chevrons and a check mark, painted once (2x for Retina) - PNG needs no image plug-in."""
    d = os.path.join(tempfile.gettempdir(), f"myrmex-theme-{os.getuid() if hasattr(os, 'getuid') else 0}")
    os.makedirs(d, exist_ok=True)
    out = {}

    def paint(name, pts, color, width=2.2, size=20):
        img = QImage(size, size, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color), width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        path = QPainterPath(QPointF(*pts[0]))
        for q in pts[1:]:
            path.lineTo(QPointF(*q))
        p.drawPath(path)
        p.end()
        f = os.path.join(d, f"{name}-{tag}.png")
        img.save(f)
        out[name] = f.replace("\\", "/")
    paint("chev", [(5, 8), (10, 13), (15, 8)], tok["text2"])
    paint("up", [(5, 12), (10, 7), (15, 12)], tok["text2"])
    paint("check", [(5, 10.5), (8.5, 14), (15, 6.5)], tok["on_accent"], 2.6)
    return out


def scheme_is_dark(app: QGuiApplication) -> bool:
    try:
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:                                   # Qt < 6.5: ask the palette
        return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def apply(app: QApplication, dark: bool | None = None) -> None:
    dark = scheme_is_dark(app) if dark is None else dark
    tok = DARK if dark else LIGHT
    T.clear()
    T.update(tok)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, key in ((QPalette.ColorRole.Window, "bg"), (QPalette.ColorRole.WindowText, "text"),
                      (QPalette.ColorRole.Base, "field"), (QPalette.ColorRole.AlternateBase, "card"),
                      (QPalette.ColorRole.Text, "text"), (QPalette.ColorRole.Button, "card"),
                      (QPalette.ColorRole.ButtonText, "text"), (QPalette.ColorRole.Highlight, "accent"),
                      (QPalette.ColorRole.HighlightedText, "on_accent"), (QPalette.ColorRole.ToolTipBase, "card"),
                      (QPalette.ColorRole.ToolTipText, "text"), (QPalette.ColorRole.PlaceholderText, "text3"),
                      (QPalette.ColorRole.Mid, "border2"), (QPalette.ColorRole.Dark, "border2"),
                      (QPalette.ColorRole.Light, "hover"), (QPalette.ColorRole.Link, "accent")):
        pal.setColor(role, QColor(tok[key]))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(tok["text3"]))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(tok["text3"]))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(tok["text3"]))
    app.setPalette(pal)
    app.setStyleSheet(QSS % dict(tok, **_glyphs(tok, "dark" if dark else "light")))


def follow_system(app: QApplication, on_change=None) -> None:
    """Apply now and again whenever macOS switches between light and dark."""
    apply(app)
    try:
        def changed(*_):
            apply(app)
            if on_change is not None:
                on_change()
        app.styleHints().colorSchemeChanged.connect(changed)
    except AttributeError:
        pass


def mono_font(size: int = 12) -> QFont:
    f = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    f.setPointSize(size)
    return f


__all__ = ["apply", "follow_system", "mono_font", "T", "DARK", "LIGHT"]
