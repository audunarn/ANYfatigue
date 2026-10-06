"""A small QPainter plot widget with linear/log axes, in ANYfem's plot style.

Missing samples (NaN) stay visible gaps: a line is never drawn across absent
data.  Hovering shows the nearest sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QToolTip, QWidget

__all__ = ["PlotSeries", "PlotWidget", "nice_ticks"]

PALETTE = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2")


@dataclass
class PlotSeries:
    name: str
    x: np.ndarray
    y: np.ndarray
    kind: str = "line"          # line | points | bars
    color: Optional[str] = None
    width: float = 1.8

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError(f"series {self.name!r}: x and y differ in length")


def nice_ticks(low: float, high: float, target: int = 6) -> List[float]:
    """Round tick positions covering ``[low, high]``."""

    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return [low]
    raw = (high - low) / max(target - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = factor * magnitude
        if raw <= step:
            break
    first = math.ceil(low / step - 1e-9) * step
    ticks = []
    value = first
    while value <= high + 1e-9 * step:
        ticks.append(round(value, 12))
        value += step
    return ticks


def _label(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1e5 or abs(value) < 1e-3:
        return f"{value:.0e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")
    return f"{value:.4g}"


class PlotWidget(QWidget):
    """Plots a handful of series; set them with :meth:`set_plot`."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.series: List[PlotSeries] = []
        self.x_label = ""
        self.y_label = ""
        self.title = ""
        self.x_log = False
        self.y_log = False
        self.empty_text = "Nothing to plot"
        self._samples: List[Tuple[QPointF, PlotSeries, int]] = []
        self.setMinimumSize(280, 220)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    def set_plot(
        self,
        series: Sequence[PlotSeries],
        *,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        x_log: bool = False,
        y_log: bool = False,
        empty_text: str = "Nothing to plot",
    ) -> None:
        self.series = list(series)
        self.x_label, self.y_label, self.title = x_label, y_label, title
        self.x_log, self.y_log = x_log, y_log
        self.empty_text = empty_text
        self.update()

    def clear(self, text: str = "Nothing to plot") -> None:
        self.set_plot([], empty_text=text)

    def has_data(self) -> bool:
        return any(self._finite(s).any() for s in self.series)

    # ------------------------------------------------------------------
    def _finite(self, s: PlotSeries) -> np.ndarray:
        mask = np.isfinite(s.x) & np.isfinite(s.y)
        if self.x_log:
            mask &= s.x > 0.0
        if self.y_log:
            mask &= s.y > 0.0
        return mask

    def _limits(self, curves):
        def span(values, log):
            low, high = float(values.min()), float(values.max())
            if log:
                low, high = math.log10(low), math.log10(high)
                if high - low < 1e-9:
                    low, high = low - 0.5, high + 0.5
                low, high = math.floor(low), math.ceil(high)
                if high == low:
                    high += 1
                return low, high
            pad = (high - low) * 0.06 or max(abs(low) * 0.05, 1.0)
            return low - pad, high + pad

        xs = np.concatenate([s.x[m] for s, m in curves])
        ys = np.concatenate([s.y[m] for s, m in curves])
        if any(s.kind == "bars" for s, _ in curves) and not self.y_log:
            ys = np.concatenate([ys, [0.0]])
        return span(xs, self.x_log), span(ys, self.y_log)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        painter.setRenderHint(QPainter.Antialiasing)
        self._samples = []
        curves = [(s, self._finite(s)) for s in self.series]
        curves = [(s, m) for s, m in curves if m.any()]
        if not curves:
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect().adjusted(12, 12, -12, -12),
                             Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)
            return
        (xmin, xmax), (ymin, ymax) = self._limits(curves)
        legend_rows = (len(curves) + 1) // 2
        left, top = 72, 30 + (18 * legend_rows if len(curves) > 1 else 0) + (18 if self.title else 0)
        right, bottom = self.width() - 18, self.height() - 48
        if right <= left + 20 or bottom <= top + 20:
            return

        def fx(x):
            return math.log10(x) if self.x_log else x

        def fy(y):
            return math.log10(y) if self.y_log else y

        def project(x, y):
            return QPointF(left + (fx(x) - xmin) / (xmax - xmin) * (right - left),
                           bottom - (fy(y) - ymin) / (ymax - ymin) * (bottom - top))

        self._draw_axes(painter, left, top, right, bottom, xmin, xmax, ymin, ymax)
        if self.title:
            painter.setPen(QColor("#253247"))
            painter.drawText(QRectF(left, 4, right - left, 18), Qt.AlignLeft | Qt.AlignVCenter, self.title)
        painter.setClipRect(QRectF(left, top, right - left, bottom - top))
        for index, (s, mask) in enumerate(curves):
            color = QColor(s.color or PALETTE[index % len(PALETTE)])
            idx = np.flatnonzero(mask)
            if s.kind == "bars":
                self._draw_bars(painter, s, idx, color, project, ymin)
            elif s.kind == "points":
                painter.setPen(QPen(color, 1.0))
                painter.setBrush(color)
                for i in idx:
                    painter.drawEllipse(project(s.x[i], s.y[i]), 3.0, 3.0)
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setPen(QPen(color, s.width))
                for chunk in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                    pts = [project(s.x[i], s.y[i]) for i in chunk]
                    if len(pts) > 1:
                        painter.drawPolyline(QPolygonF(pts))
                    elif pts:
                        painter.drawEllipse(pts[0], 2.5, 2.5)
            step = max(1, len(idx) // 400)
            self._samples.extend((project(s.x[i], s.y[i]), s, int(i)) for i in idx[::step])
        painter.setClipping(False)
        painter.setPen(QColor("#253247"))
        if len(curves) > 1:
            width = (right - left) / 2
            for index, (s, _m) in enumerate(curves):
                color = QColor(s.color or PALETTE[index % len(PALETTE)])
                x0 = left + (index % 2) * width
                y0 = 4 + (18 if self.title else 0) + (index // 2) * 18
                painter.setPen(QPen(color, 2.4))
                painter.drawLine(QPointF(x0, y0 + 9), QPointF(x0 + 14, y0 + 9))
                painter.setPen(QColor("#253247"))
                painter.drawText(QRectF(x0 + 18, y0, width - 24, 18), Qt.AlignLeft | Qt.AlignVCenter,
                                 painter.fontMetrics().elidedText(s.name, Qt.ElideRight, int(width - 24)))
        painter.setPen(QColor("#475569"))
        painter.drawText(QRectF(left, bottom + 26, right - left, 20), Qt.AlignCenter, self.x_label)
        painter.save()
        painter.translate(14, (top + bottom) / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-(bottom - top) / 2, -10, bottom - top, 18), Qt.AlignCenter, self.y_label)
        painter.restore()

    def _draw_axes(self, painter, left, top, right, bottom, xmin, xmax, ymin, ymax) -> None:
        def ticks(low, high, log):
            if log:
                return [(d, f"1e{int(d)}" if abs(d) > 2 else _label(10.0 ** d)) for d in range(int(low), int(high) + 1)]
            return [(t, _label(t)) for t in nice_ticks(low, high)]

        for value, text in ticks(xmin, xmax, self.x_log):
            x = left + (value - xmin) / (xmax - xmin) * (right - left)
            if left - 1 <= x <= right + 1:
                painter.setPen(QPen(QColor("#e8edf4"), 1))
                painter.drawLine(QPointF(x, top), QPointF(x, bottom))
                painter.setPen(QColor("#64748b"))
                painter.drawText(QRectF(x - 32, bottom + 4, 64, 18), Qt.AlignCenter, text)
        for value, text in ticks(ymin, ymax, self.y_log):
            y = bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
            if top - 1 <= y <= bottom + 1:
                painter.setPen(QPen(QColor("#e8edf4"), 1))
                painter.drawLine(QPointF(left, y), QPointF(right, y))
                painter.setPen(QColor("#64748b"))
                painter.drawText(QRectF(2, y - 9, 64, 18), Qt.AlignRight | Qt.AlignVCenter, text)
        painter.setPen(QPen(QColor("#94a3b8"), 1))
        painter.drawLine(QPointF(left, bottom), QPointF(right, bottom))
        painter.drawLine(QPointF(left, top), QPointF(left, bottom))

    def _draw_bars(self, painter, s, idx, color, project, ymin) -> None:
        xs = s.x[idx]
        gaps = np.diff(np.sort(xs))
        gap = float(gaps[gaps > 0].min()) if gaps.size and (gaps > 0).any() else 1.0
        half = 0.4 * gap
        base = 1e-300 if self.y_log else 0.0
        painter.setPen(QPen(color.darker(120), 1))
        painter.setBrush(color)
        for i in idx:
            a = project(max(s.x[i] - half, 1e-300) if self.x_log else s.x[i] - half, max(s.y[i], base))
            b = project(s.x[i] + half, base if not self.y_log else 10.0 ** ymin)
            painter.drawRect(QRectF(a.x(), min(a.y(), b.y()), max(b.x() - a.x(), 1.0), abs(b.y() - a.y())))
        painter.setBrush(Qt.NoBrush)

    # ------------------------------------------------------------------
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._samples:
            return
        pos = event.position()
        point, series, index = min(
            self._samples, key=lambda row: (row[0].x() - pos.x()) ** 2 + (row[0].y() - pos.y()) ** 2)
        if (point.x() - pos.x()) ** 2 + (point.y() - pos.y()) ** 2 > 144:
            QToolTip.hideText()
            return
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{series.name}\n{self.x_label}: {series.x[index]:.6g}\n{self.y_label}: {series.y[index]:.6g}",
            self,
        )

    def leaveEvent(self, event) -> None:  # noqa: N802
        QToolTip.hideText()
        super().leaveEvent(event)
