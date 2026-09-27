"""A slim window (like a side panel, Claude Desktop style): every page fits, nothing is cut off; grids reflow,
long checkbox texts wrap and still click; the rail's narrow / wide toggle; the window remembers its size."""
import os

import pytest

pytest.importorskip("PySide6")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from myrmex.app import theme
    app = QApplication.instance() or QApplication([])
    theme.follow_system(app)                                # (as the app itself looks)
    return app


def _window():
    from myrmex.app import window as W
    from myrmex.app.settings import AppSettings
    s = AppSettings()
    s.start_engine_on_launch = s.open_blender_on_start = False
    w = W.MainWindow(s)
    w.timer.stop()
    return w


def test_every_page_fits_a_side_panel(app):
    from PySide6.QtWidgets import QBoxLayout, QScrollArea
    w = _window()
    w.resize(w.MIN_W, 820)
    w.show()
    app.processEvents()
    assert w.width() == w.MIN_W <= 400 and w.minimumSizeHint().width() <= w.MIN_W
    for name, idx in w.page_index.items():
        w.stack.setCurrentIndex(idx)
        app.processEvents()
        sa = w.stack.widget(idx).findChild(QScrollArea)
        if sa is None:                                      # (the log)
            continue
        assert sa.widget().minimumSizeHint().width() <= sa.viewport().width(), name
    assert all(head.direction() == QBoxLayout.Direction.TopToBottom for _, head in w._heads)
    w.resize(1100, 820)
    app.processEvents()
    assert all(head.direction() == QBoxLayout.Direction.LeftToRight for _, head in w._heads)
    w.close()


def test_grids_reflow(app):
    w = _window()
    w.show()
    tiles = [w.st[k].parentWidget() for k in ("clock", "bpm", "beat", "transport", "state")]
    w.nav.setCurrentRow(w.page_index["Live"])
    w.resize(1100, 820)
    app.processEvents()
    ys = [t.geometry().y() for t in tiles]
    assert ys[0] == ys[1] == ys[2] == ys[3] and ys[4] > ys[0]           # 4 across when wide
    w.resize(w.MIN_W, 820)
    app.processEvents()
    ys = [t.geometry().y() for t in tiles]
    assert ys[0] == ys[1] and ys[2] > ys[1]                             # 2 across in a side panel
    w.close()


def test_long_checkbox_text_wraps_and_still_clicks(app):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QCheckBox, QFormLayout, QLabel, QVBoxLayout, QWidget

    from myrmex.app import responsive as R
    root = QWidget()
    v = QVBoxLayout(root)
    long_one = QCheckBox("Show the picture effects live in Blender (off: afterimages and ribbons only - lightest)")
    short = QCheckBox("Short")
    v.addWidget(long_one)
    form = QFormLayout()
    in_form = QCheckBox("Transparent background: only the organism (TouchDesigner draws the rest)")
    form.addRow(in_form)
    v.addLayout(form)
    v.addWidget(short)
    assert R.wrap_long_buttons(root) == 2
    assert long_one.text() == "" and short.text() == "Short" and long_one.property("wrapped")
    lab = long_one.parentWidget().findChild(QLabel)
    assert lab.wordWrap() and lab.text().startswith("Show the picture")
    ev = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(2, 2), QPointF(2, 2), Qt.MouseButton.LeftButton,
                     Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    lab.mousePressEvent(ev)
    assert long_one.isChecked()                                         # the text clicks the box
    long_one.setEnabled(False)
    assert not lab.isEnabled()
    root.show()
    long_one.setVisible(False)
    assert not long_one.parentWidget().isVisible()                     # hiding the box hides its text
    assert in_form.parentWidget().findChild(QLabel).text().startswith("Transparent")
    assert R.wrap_long_buttons(root) == 0                               # (once)
    root.close()


def test_narrow_toggle_and_remembered_size(app):
    w = _window()
    w.resize(900, 800)
    w.show()
    app.processEvents()
    w.toggle_narrow()
    app.processEvents()
    assert w.width() == w.NARROW and w.btn_narrow.text() == "‹›"
    w.toggle_narrow()
    app.processEvents()
    assert w.width() == 900 and w.btn_narrow.text() == "›‹"
    w.resize(430, 700)
    app.processEvents()
    w.close()                                               # (saves where and how wide it was)
    assert w.s.window
    from myrmex.app import window as W
    w2 = W.MainWindow(w.s)
    w2.timer.stop()
    assert w2.width() == 430 and w2.height() == 700
    w2.close()


def test_autogrid_columns():
    from PySide6.QtWidgets import QApplication, QPushButton

    from myrmex.app import responsive as R
    QApplication.instance() or QApplication([])
    box = R.grid_box([QPushButton(f"b{i}") for i in range(6)], min_cell=100, spacing=10, max_cols=4)
    g = box.layout()
    assert g.columns(1000) == 4 and g.columns(330) == 3 and g.columns(210) == 2 and g.columns(80) == 1
    assert g.heightForWidth(1000) < g.heightForWidth(210) < g.heightForWidth(80)
