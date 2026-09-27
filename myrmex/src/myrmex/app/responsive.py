"""Narrow windows: Myrmex can be as slim as a side panel (like Claude Desktop) - every page reflows.

* ``AutoGrid``: cells (buttons, knobs, label + slider pairs, status tiles) in as many equal columns as the
  width allows - one under the other in a slim window, side by side in a wide one.
* ``adapt(root)`` makes whatever a page built narrow-friendly: long checkbox / radio texts wrap (a checkbox
  cannot wrap its own text), rows of buttons flow onto more lines, forms put the label above the field when a
  row does not fit, combo boxes can shrink (their lists still show the whole names).
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QRect, QSize, Qt
from PySide6.QtWidgets import (QAbstractButton, QApplication, QBoxLayout, QCheckBox, QComboBox, QFormLayout, QGridLayout,
                               QHBoxLayout, QLabel, QLayout, QRadioButton, QSizePolicy, QWidget)


class AutoGrid(QLayout):
    """Equal cells, as many columns as fit (each at least ``min_cell`` wide, at most ``max_cols``)."""

    def __init__(self, parent: QWidget | None = None, min_cell: int = 160, spacing: int = 8,
                 max_cols: int | None = None):
        super().__init__(parent)
        self._items: list = []
        self.min_cell = int(min_cell)
        self.max_cols = max_cols
        self._sp = int(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # --- QLayout
    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation.Horizontal

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._place(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def minimumSize(self) -> QSize:
        m = self.contentsMargins()
        w = max([it.minimumSize().width() for it in self._shown()] or [0])
        return QSize(w + m.left() + m.right(), self.heightForWidth(w + m.left() + m.right()))

    def sizeHint(self) -> QSize:
        m = self.contentsMargins()
        n = len(self._shown())
        cols = min(n, self.max_cols or n) or 1
        w = max([it.sizeHint().width() for it in self._shown()] + [self.min_cell])
        width = cols * w + (cols - 1) * self._sp + m.left() + m.right()
        return QSize(width, self.heightForWidth(width))

    # --- placing
    def _shown(self) -> list:
        return [it for it in self._items if not it.isEmpty()]

    def columns(self, width: int) -> int:
        items = self._shown()
        cols = max(1, (width + self._sp) // (self.min_cell + self._sp))
        widest = max([it.minimumSize().width() for it in items] or [0])
        while cols > 1 and (width - (cols - 1) * self._sp) / cols < widest:
            cols -= 1                                    # (no cell narrower than what it needs)
        if self.max_cols:
            cols = min(cols, self.max_cols)
        return max(1, min(cols, len(items) or 1))

    def _place(self, rect: QRect, apply: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        items = self._shown()
        if not items:
            return m.top() + m.bottom()
        cols = self.columns(r.width())
        cw = (r.width() - (cols - 1) * self._sp) / cols
        y = r.y()
        for start in range(0, len(items), cols):
            row = items[start:start + cols]
            h = max(it.heightForWidth(int(cw)) if it.hasHeightForWidth() else it.sizeHint().height() for it in row)
            if apply:
                for k, it in enumerate(row):
                    x = r.x() + k * (cw + self._sp)
                    it.setGeometry(QRect(int(round(x)), y, int(round(cw)), h))
            y += h + self._sp
        return y - self._sp - rect.y() + m.bottom()


def grid_box(widgets, min_cell: int = 160, spacing: int = 8, max_cols: int | None = None) -> QWidget:
    """A widget holding ``widgets`` in an AutoGrid."""
    box = QWidget()
    g = AutoGrid(box, min_cell, spacing, max_cols)
    for w in widgets:
        g.addWidget(w)
    return box


def pair(label, widget: QWidget, label_width: int = 0) -> QWidget:
    """``label`` and ``widget`` side by side, as one cell."""
    box = QWidget()
    h = QHBoxLayout(box)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(10)
    lab = label if isinstance(label, QWidget) else QLabel(str(label))
    if label_width:
        lab.setMinimumWidth(label_width)
    h.addWidget(lab)
    h.addWidget(widget, 1)
    return box


# ---------------------------------------------------------------------------- adapting what a page built
class _Follow(QObject):
    """A wrapped checkbox's text follows it: enabled, shown / hidden."""

    def __init__(self, button: QAbstractButton, box: QWidget, label: QLabel):
        super().__init__(button)
        self.box, self.label = box, label

    def eventFilter(self, obj, ev) -> bool:
        t = ev.type()
        if t == QEvent.Type.EnabledChange:
            self.label.setEnabled(obj.isEnabled())
        elif t == QEvent.Type.HideToParent:
            self.box.hide()
        elif t == QEvent.Type.ShowToParent:
            self.box.show()
        elif t == QEvent.Type.ToolTipChange:
            self.label.setToolTip(obj.toolTip())
        return False


class _ClickLabel(QLabel):
    def __init__(self, text: str, button: QAbstractButton):
        super().__init__(text)
        self.button = button
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def minimumSizeHint(self) -> QSize:
        return QSize(60, super().minimumSizeHint().height())

    def mousePressEvent(self, ev) -> None:
        if self.button.isEnabled():
            self.button.click()


def _holder(root: QWidget, target) -> QLayout | None:
    """The layout (anywhere under ``root``) that holds ``target`` (a widget or a layout) directly."""
    def find(lay: QLayout | None):
        if lay is None:
            return None
        for i in range(lay.count()):
            it = lay.itemAt(i)
            if it.widget() is target or it.layout() is target:
                return lay
            if it.layout() is not None:
                hit = find(it.layout())
                if hit is not None:
                    return hit
        return None
    for w in [root] + root.findChildren(QWidget):
        hit = find(w.layout())
        if hit is not None:
            return hit
    return None


def _swap(holder: QLayout, old, new: QWidget) -> bool:
    """Put ``new`` where ``old`` (a widget or a layout) is in ``holder``."""
    idx = next((i for i in range(holder.count()) if holder.itemAt(i).widget() is old or
                holder.itemAt(i).layout() is old), -1)
    if idx < 0:
        return False
    if isinstance(holder, QFormLayout):
        row, role = holder.getItemPosition(idx)
        holder.takeAt(idx)
        holder.setWidget(row, role, new)
    elif isinstance(holder, QGridLayout):
        r, c, rs, cs = holder.getItemPosition(idx)
        holder.takeAt(idx)
        holder.addWidget(new, r, c, rs, cs)
    elif isinstance(holder, QBoxLayout):
        it = holder.itemAt(idx)
        stretch = holder.stretch(idx)
        holder.takeAt(idx)
        holder.insertWidget(idx, new, stretch)
        del it
    else:
        return False
    return True


def wrap_long_buttons(root: QWidget, limit: int = 30) -> int:
    """Checkboxes / radio buttons with a long text: the text wraps next to the box (and still clicks it)."""
    n = 0
    for b in root.findChildren(QAbstractButton):
        if not isinstance(b, (QCheckBox, QRadioButton)) or b.property("wrapped") or len(b.text()) <= limit:
            continue
        holder = _holder(root, b)
        if holder is None:
            continue
        box = QWidget()
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        if not _swap(holder, b, box):
            continue
        box.setLayout(h)
        lab = _ClickLabel(b.text(), b)
        lab.setEnabled(b.isEnabled())
        lab.setToolTip(b.toolTip())
        b.setText("")
        b.setProperty("wrapped", True)
        b.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        h.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)
        h.addWidget(lab, 1)
        if b.isHidden() and not b.isWindow():
            box.hide()
            b.show()
        b.installEventFilter(_Follow(b, box, lab))
        n += 1
    return n


_KEEP_IN_ROW = ("QLineEdit", "QSlider", "QSpinBox", "QDoubleSpinBox", "QProgressBar", "Knob", "QPlainTextEdit")


def flow_button_rows(root: QWidget, min_width: int = 200) -> int:
    """Rows of buttons / choices (two or more things, a button or a combo among them, nothing that shrinks by
    itself like a text field or a slider) flow onto more lines when narrow."""
    n = 0
    for w in [root] + root.findChildren(QWidget):
        lay = w.layout()
        stack = [lay] if lay is not None else []
        while stack:
            cur = stack.pop()
            for i in range(cur.count()):
                sub = cur.itemAt(i).layout()
                if sub is None:
                    continue
                widgets = [sub.itemAt(k).widget() for k in range(sub.count()) if sub.itemAt(k).widget() is not None]
                choices = [x for x in widgets if isinstance(x, (QAbstractButton, QComboBox))]
                if type(sub) is not QHBoxLayout or len(widgets) < 2 or not choices or \
                        any(type(x).__name__ in _KEEP_IN_ROW for x in widgets) or sub.minimumSize().width() < min_width:
                    stack.append(sub)
                    continue
                box = QWidget()
                if not _swap(cur, sub, box):
                    stack.append(sub)
                    continue
                g = AutoGrid(box, min_cell=max(x.sizeHint().width() for x in widgets), spacing=6)
                for x in widgets:
                    sub.removeWidget(x)
                    g.addWidget(x)
                sub.setParent(None)
                n += 1
    return n


def relax_combos(root: QWidget, chars: int = 8) -> None:
    """Combo boxes can shrink; their lists open as wide as the longest name."""
    for c in root.findChildren(QComboBox):
        if c.property("relaxed"):
            continue
        c.setProperty("relaxed", True)
        c.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        c.setMinimumContentsLength(chars)
        QApplication.sendEvent(c, QEvent(QEvent.Type.StyleChange))  # (Qt keeps the old minimum size otherwise)
        c.updateGeometry()

        def fit(*_, c=c):
            fm = c.view().fontMetrics()
            widest = max([fm.horizontalAdvance(c.itemText(i)) for i in range(c.count())] or [0])
            c.view().setMinimumWidth(widest + 40 if widest else 0)
        c.model().rowsInserted.connect(fit)
        c.model().modelReset.connect(fit)
        fit()


def wrap_labels(root: QWidget, longer_than: int = 24) -> None:
    """Labels with more than a few words wrap (status lines too: they get longer later)."""
    for lab in root.findChildren(QLabel):
        if lab.wordWrap() or lab.objectName() in ("pagetitle", "statvalue", "statlabel", "rectime", "brand") or \
                lab.pixmap() is not None and not lab.pixmap().isNull():
            continue
        if len(lab.text()) > longer_than:
            lab.setWordWrap(True)


def relax_forms(root: QWidget) -> None:
    for f in root.findChildren(QFormLayout):
        f.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)


def adapt(root: QWidget) -> None:
    """Everything above, on whatever ``root`` holds."""
    wrap_long_buttons(root)
    flow_button_rows(root)
    relax_combos(root)
    relax_forms(root)
    wrap_labels(root)


__all__ = ["AutoGrid", "grid_box", "pair", "adapt", "wrap_long_buttons", "flow_button_rows", "relax_combos",
           "relax_forms", "wrap_labels"]
