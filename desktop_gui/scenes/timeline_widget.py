from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from desktop_gui.data_loader import Scenario
from desktop_gui.theme.tokens import AMBER, CYAN, GRID, MUTED, RED


def time_fraction(minute: float, start_minute: float, end_minute: float) -> float:
    """Map canonical simulation minutes to the visible graph interval."""
    span = max(float(end_minute) - float(start_minute), 1e-9)
    return min(max((float(minute) - float(start_minute)) / span, 0.0), 1.0)


def normalized_series_points(scenario: Scenario) -> tuple[tuple[float, float], ...]:
    """Return physical-time X and normalized-distance Y coordinates for tests/painting."""
    points = scenario.distance_series
    start, end = points[0].minute, points[-1].minute
    distances = [point.distance_km for point in points]
    low, high = min(distances), max(distances)
    distance_span = max(high - low, 1e-9)
    return tuple(
        (
            time_fraction(point.minute, start, end),
            (point.distance_km - low) / distance_span,
        )
        for point in points
    )


class TimelineWidget(QWidget):
    seek_requested = Signal(float)

    def __init__(self, coverage_percent: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(92)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._scenario: Scenario | None = None
        self._fraction = 0.0
        self._current_minute = 0.0
        self._coverage_percent = float(coverage_percent)

    def set_scenario(self, scenario: Scenario) -> None:
        self._scenario = scenario
        self._current_minute = scenario.distance_series[0].minute
        self._fraction = 0.0
        self.update()

    def set_fraction(self, fraction: float) -> None:
        self._fraction = min(max(float(fraction), 0.0), 1.0)
        if self._scenario:
            start = self._scenario.distance_series[0].minute
            end = self._scenario.distance_series[-1].minute
            self._current_minute = start + self._fraction * (end - start)
        self.update()

    def set_current_minute(self, minute: float) -> None:
        self._current_minute = float(minute)
        if self._scenario:
            points = self._scenario.distance_series
            self._fraction = time_fraction(minute, points[0].minute, points[-1].minute)
        self.update()

    @property
    def current_fraction(self) -> float:
        return self._fraction

    @property
    def tca_fraction(self) -> float | None:
        if not self._scenario:
            return None
        points = self._scenario.distance_series
        return time_fraction(
            float(self._scenario.value("time_to_tca_min")),
            points[0].minute,
            points[-1].minute,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._seek(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek(event.position().x())

    def _seek(self, x: float) -> None:
        margin = 8.0
        fraction = (x - margin) / max(self.width() - margin * 2, 1)
        self.seek_requested.emit(min(max(fraction, 0.0), 1.0))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(8, 8, self.width() - 16, self.height() - 43)
        painter.setPen(QPen(QColor(GRID), 1))
        painter.drawRoundedRect(rect, 3, 3)
        if self._scenario:
            points = self._scenario.distance_series
            distances = [point.distance_km for point in points]
            low, high = min(distances), max(distances)
            span = max(high - low, 0.000001)
            painter.setPen(QPen(QColor(CYAN), 2))
            path_points = []
            for point in points:
                x = rect.left() + rect.width() * time_fraction(
                    point.minute, points[0].minute, points[-1].minute
                )
                y = rect.bottom() - rect.height() * (point.distance_km - low) / span
                path_points.append((x, y))
            for first, second in zip(path_points, path_points[1:]):
                painter.drawLine(first[0], first[1], second[0], second[1])
            minimum = min(points, key=lambda point: point.distance_km)
            minimum_x = rect.left() + rect.width() * time_fraction(
                minimum.minute, points[0].minute, points[-1].minute
            )
            minimum_y = rect.bottom() - rect.height() * (minimum.distance_km - low) / span
            painter.setPen(QPen(QColor(AMBER), 2))
            painter.setBrush(QColor(AMBER))
            painter.drawEllipse(QRectF(minimum_x - 3, minimum_y - 3, 6, 6))
            tca_fraction = self.tca_fraction or 0.0
            tca_x = rect.left() + rect.width() * tca_fraction
            painter.setPen(QPen(QColor(RED), 1, Qt.PenStyle.DashLine))
            painter.drawLine(tca_x, rect.top(), tca_x, rect.bottom())
        cursor_x = rect.left() + rect.width() * self._fraction
        painter.setPen(QPen(QColor(AMBER), 2))
        painter.drawLine(cursor_x, rect.top() - 2, cursor_x, rect.bottom() + 2)
        bin_count = 120
        filled_count = round(self._coverage_percent * bin_count / 100.0)
        gap = 1.0
        bin_width = (rect.width() - gap * (bin_count - 1)) / bin_count
        for index in range(bin_count):
            color = QColor(CYAN if index < filled_count else GRID)
            painter.fillRect(
                QRectF(rect.left() + index * (bin_width + gap), rect.bottom() + 7, bin_width, 7),
                color,
            )
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRectF(rect.left(), rect.bottom() + 18, rect.width(), 14),
            (
                f"T+{self._current_minute:.3f} min  |  "
                f"TCA {float(self._scenario.value('time_to_tca_min')):.3f} min  |  "
                f"COVERAGE {self._coverage_percent:g}%"
                if self._scenario
                else f"COVERAGE {self._coverage_percent:g}%"
            ),
        )
