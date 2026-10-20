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


@dataclass(frozen=True)
class IndexedCycleTable:
    """Normal cycles plus owner indices, in detection order (never range-sorted)."""
    cycles: CycleTable
    start_indices: np.ndarray
    end_indices: np.ndarray
    closing_indices: np.ndarray  # -1 for final residual halves
    kinds: tuple

    def select(self, keep):
        return IndexedCycleTable(
            CycleTable(self.cycles.ranges[keep], self.cycles.means[keep], self.cycles.counts[keep]),
            self.start_indices[keep], self.end_indices[keep], self.closing_indices[keep],
            tuple(k for k, selected in zip(self.kinds, keep) if selected),
        )


def require_indexed_backend():
    """Fail closed rather than falling back to a global shear envelope."""
    backend = _backend()
    if getattr(backend, "INDEXED_CYCLES_VERSION", None) != 1 or not callable(
        getattr(backend, "count_cycles_indexed", None)
    ):
        raise MissingDependencyError(
            "cycle_associated requires ANYtimeseries indexed rainflow contract v1; "
            "install the indexed-rainflow owner build described in docs/CYCLE_ASSOCIATION.md. "
            "The published anytimes 1.1.0 does not provide it. No fallback was used."
        )
    return backend


def count_indexed_rainflow_cycles(series, *, endpoints=True, chunk_size=None) -> IndexedCycleTable:
    """Delegate indexed counting and validate the public owner contract."""
    backend = require_indexed_backend()
    try:
        if np.iscomplexobj(series):
            raise ValueError("complex history")
        values = np.asarray(series, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("not a finite vector")
        records = backend.count_cycles_indexed(values, endpoints=endpoints, chunk_size=chunk_size)
    except (ValueError, TypeError, OverflowError) as error:
        raise InputError(f"invalid indexed rainflow history/options: {error}") from error
    r, m, c, starts, ends, closes, kinds = [], [], [], [], [], [], []
    for record in records:
        try:
            a, b, closing = record.start_index, record.end_index, record.closing_index
            if not all(isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_)) for i in (a, b)):
                raise ValueError("non-integer indices")
            if not 0 <= a < b < len(values):
                raise ValueError("reversal indices outside the original record")
            if record.kind not in ("full", "initial_half", "residual_half"):
                raise ValueError("unknown cycle kind")
            if record.count != (1.0 if record.kind == "full" else 0.5):
                raise ValueError("cycle kind/count mismatch")
            if record.kind == "residual_half":
                if closing is not None:
                    raise ValueError("residual has a closing reversal")
            elif (not isinstance(closing, (int, np.integer)) or isinstance(closing, (bool, np.bool_))
                  or not b < closing < len(values)):
                raise ValueError("missing or invalid closing reversal")
            expected_range = abs(float(values[b])-float(values[a]))
            total = float(values[b])+float(values[a])
            expected_mean = 0.5*total if np.isfinite(total) else 0.5*values[b]+0.5*values[a]
            if not np.isfinite(record.range) or not np.isfinite(record.mean):
                raise ValueError("non-finite cycle")
            if record.range != expected_range or record.mean != expected_mean:
                raise ValueError("cycle values disagree with the indexed input samples")
        except (AttributeError, TypeError, ValueError, OverflowError) as error:
            raise InputError(f"ANYtimeseries indexed contract violation: {error}") from error
        if record.range <= 0.0:
            continue
        r.append(record.range); m.append(record.mean); c.append(record.count)
        starts.append(a); ends.append(b); closes.append(-1 if closing is None else closing)
        kinds.append(record.kind)
    return IndexedCycleTable(
        CycleTable(np.asarray(r), np.asarray(m), np.asarray(c)),
        np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64),
        np.asarray(closes, dtype=np.int64), tuple(kinds),
    )


__all__ += ["IndexedCycleTable", "count_indexed_rainflow_cycles", "require_indexed_backend"]


def indexed_backend_provenance():
    """Record code identity as well as version (development builds may share a version)."""
    import hashlib
    import inspect
    from pathlib import Path

    backend = require_indexed_backend()
    try:
        source = inspect.getsourcefile(backend.count_cycles_indexed)
        digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    except (OSError, TypeError):
        digest = "unavailable (source bytes not accessible)"
    return {"indexed_counter_contract": 1, "indexed_counter_source_sha256": digest}
