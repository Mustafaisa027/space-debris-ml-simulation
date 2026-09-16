from __future__ import annotations

from PySide6.QtCore import QObject, QElapsedTimer, QTimer, Signal


class TimelineController(QObject):
    ticked = Signal(float)
    state_changed = Signal(bool)

    def __init__(self, duration_minutes: float, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.duration_minutes = max(float(duration_minutes), 1.0)
        self.current_minute = 0.0
        self.speed = 18.0
        self._playing = False
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance_realtime)

    @property
    def is_playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        if self._playing:
            return
        self._playing = True
        self._clock.restart()
        self._timer.start()
        self.state_changed.emit(True)

    def pause(self) -> None:
        if not self._playing:
            return
        self._playing = False
        self._timer.stop()
        self.state_changed.emit(False)

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def seek_fraction(self, fraction: float) -> None:
        self.current_minute = self.duration_minutes * min(max(float(fraction), 0.0), 1.0)
        self.ticked.emit(self.current_minute)

    def set_speed(self, speed: float) -> None:
        self.speed = max(float(speed), 0.1)

    def step_fixed(self, seconds: float = 1.0 / 60.0) -> float:
        self.current_minute = (self.current_minute + seconds * self.speed) % self.duration_minutes
        self.ticked.emit(self.current_minute)
        return self.current_minute

    def _advance_realtime(self) -> None:
        elapsed_seconds = self._clock.restart() / 1000.0
        self.step_fixed(elapsed_seconds)

