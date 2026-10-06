"""Run a callable off the GUI thread and report back through signals.

The signals carry the task itself as a token and must be connected to a *method
of a QObject that lives in the GUI thread*, so Qt delivers them there (a queued
connection).  A bare lambda has no thread context and could run on the worker
thread, where touching widgets is unsafe.
"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal


class TaskSignals(QObject):
    finished = Signal(object, object)       # task, result
    failed = Signal(object, str)            # task, message


class Task(QRunnable):
    """``fn()`` on the global thread pool; ``signals.finished`` or ``signals.failed``."""

    def __init__(self, fn: Callable[[], Any], done: Callable[[Any], None]) -> None:
        super().__init__()
        self.fn = fn
        self.done = done
        self.signals = TaskSignals()
        self.setAutoDelete(False)          # the window owns it until the signal arrives

    def run(self) -> None:  # noqa: D102 (Qt API)
        try:
            result = self.fn()
        except BaseException as error:   # reported to the user, never lost in the pool
            self.signals.failed.emit(self, f"{type(error).__name__}: {error}")
        else:
            self.signals.finished.emit(self, result)
