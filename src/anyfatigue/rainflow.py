"""Rainflow cycle counting, consumed from ANYtimeseries.

Counting itself is owned by ANYtimeseries (``anyqats.fatigue.rainflow``,
ASTM E1049-85 section 5.4.4).  This module is the single boundary: it checks the
input, delegates, and returns a typed table.  If ANYtimeseries is not
installed it fails with :class:`MissingDependencyError` rather than counting
with some other routine.

End points
----------
ANYtimeseries' ``count_cycles`` drops the first and last sample unless told
otherwise.  For a quasi-static load series those samples often *are* the
extremes (a series that starts at its peak), so dropping them can silently
discard the largest cycle.  Here they are kept as reversals by default
(``endpoints=True``): the first and last samples start and close half cycles,
as for any non-repeating history.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import InputError, MissingDependencyError

__all__ = ["CycleTable", "count_rainflow_cycles", "rainflow_backend_version"]


@dataclass(frozen=True)
class CycleTable:
    """Rainflow cycles: range, mean and count (1.0 full, 0.5 half cycle)."""

    ranges: np.ndarray
    means: np.ndarray
    counts: np.ndarray

    def __len__(self) -> int:
        return int(self.ranges.size)

    @property
    def total_cycles(self) -> float:
        return float(self.counts.sum())

    @property
    def max_range(self) -> float:
        return float(self.ranges.max()) if self.ranges.size else 0.0

    @property
    def maxima(self) -> np.ndarray:
        return self.means + 0.5 * self.ranges

    @property
    def minima(self) -> np.ndarray:
        return self.means - 0.5 * self.ranges

    @classmethod
    def empty(cls) -> "CycleTable":
        zero = np.zeros(0)
        return cls(zero, zero.copy(), zero.copy())


def _backend():
    try:
        from anyqats.fatigue import rainflow as backend
    except ImportError as error:
        raise MissingDependencyError(
            "rainflow counting is provided by ANYtimeseries; install it with "
            "`pip install anytimes` (or put the ANYtimeseries checkout on "
            "PYTHONPATH)"
        ) from error
    return backend


def rainflow_backend_version() -> str:
    """Version of the ANYtimeseries distribution that does the counting."""

    _backend()
    try:
        from importlib import metadata

        return metadata.version("anytimes")
    except Exception:  # source checkout without dist-info
        return "unknown (source checkout)"


def count_rainflow_cycles(series, *, endpoints: bool = True) -> CycleTable:
    """Count the rainflow cycles of a 1-D series of signed stress.

    Zero-range cycles (a flat series, repeated samples) are removed because
    they carry no damage and only inflate cycle counts.
    """

    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise InputError("a stress series must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise InputError("stress series contains non-finite samples")
    if values.size < 2:
        return CycleTable.empty()
    table = _backend().count_cycles(values, endpoints=endpoints)
    if len(table) == 0:
        return CycleTable.empty()
    keep = table[:, 0] > 0.0
    return CycleTable(
        ranges=np.ascontiguousarray(table[keep, 0]),
        means=np.ascontiguousarray(table[keep, 1]),
        counts=np.ascontiguousarray(table[keep, 2]),
    )
