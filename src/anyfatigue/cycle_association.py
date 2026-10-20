"""Chronological first-return shear envelope, an explicit normal-master policy.

Full cycles use the original chronological window through first return to the
starting normal stress after the turning reversal. Exactly matching adjacent
half cycles share a complete window, retaining their two 0.5 counts. Unmatched
halves use their own reversal-to-reversal interval. Nested excursions remain in
parent windows. This is not a unique multiaxial decomposition or a DNV approval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .errors import InputError
from .rainflow import IndexedCycleTable

ASSOCIATION_POLICY = "chronological_first_return_paired_halves_v1"
__all__ = ["ASSOCIATION_POLICY", "CycleAssociation", "associate_shear_cycles"]


@dataclass(frozen=True)
class CycleAssociation:
    """Audit of the normal cycles, their temporal support and applied shear ranges.

    All indices are zero-based within the ordered load-block cases. Closure is
    at ``closure_left + closure_fraction`` in sample-index coordinates, with
    linear interpolation on that raw-sample segment (not between reversals).
    Actual timestamps are interpolated separately when available. Paired halves
    retain their individual cycle indices but share temporal support.
    """
    start_indices: np.ndarray
    turn_indices: np.ndarray
    closing_reversal_indices: np.ndarray
    support_start_indices: np.ndarray
    closure_left_indices: np.ndarray
    closure_fractions: np.ndarray
    paired_cycle_indices: np.ndarray
    kinds: tuple
    support_kinds: tuple
    shear_minima: np.ndarray
    shear_maxima: np.ndarray
    shear_ranges: np.ndarray
    start_times: Optional[np.ndarray] = None
    turn_times: Optional[np.ndarray] = None
    support_start_times: Optional[np.ndarray] = None
    closure_times: Optional[np.ndarray] = None

    def to_dict(self):
        out = {"policy": ASSOCIATION_POLICY, "index_basis": "zero-based ordered load-block cases"}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                out[name] = list(value) if isinstance(value, tuple) else value.tolist()
        return out


class _RangeTree:
    """Linear-storage min/max tree, O(log N) interval and first-crossing queries."""
    def __init__(self, values):
        self.size = 1 << (len(values)-1).bit_length()
        self.minimum = np.full(2*self.size, np.inf)
        self.maximum = np.full(2*self.size, -np.inf)
        self.minimum[self.size:self.size+len(values)] = values
        self.maximum[self.size:self.size+len(values)] = values
        level = self.size//2
        while level:
            self.minimum[level:2*level] = np.minimum(
                self.minimum[2*level:4*level:2], self.minimum[2*level+1:4*level:2])
            self.maximum[level:2*level] = np.maximum(
                self.maximum[2*level:4*level:2], self.maximum[2*level+1:4*level:2])
            level //= 2

    def extrema(self, first, last):
        low, high = np.inf, -np.inf
        left, right = int(first)+self.size, int(last)+self.size+1
        while left < right:
            if left & 1:
                low = min(low, self.minimum[left]); high = max(high, self.maximum[left]); left += 1
            if right & 1:
                right -= 1; low = min(low, self.minimum[right]); high = max(high, self.maximum[right])
            left //= 2; right //= 2
        return float(low), float(high)

    def first_crossing(self, first, last, target, above):
        bounds = self.maximum if above else self.minimum
        def visit(node, left, right):
            if right <= first or left > last:
                return None
            if (bounds[node] < target if above else bounds[node] > target):
                return None
            if right-left == 1:
                return left
            mid = (left+right)//2
            found = visit(2*node, left, mid)
            return found if found is not None else visit(2*node+1, mid, right)
        return visit(1, 0, self.size)


def _history(value, name):
    try:
        if np.iscomplexobj(value):
            raise ValueError
        out = np.asarray(value, dtype=float)
        if out.ndim != 1 or not np.all(np.isfinite(out)):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise InputError(f"{name} must be a finite real vector") from None
    return out


def associate_shear_cycles(normal, shear, indexed: IndexedCycleTable, *, times=None, resolution=0.0):
    """Resolve shear extrema without O(N) rescans of each nested cycle window.

    Pair only adjacent half cycles that return to EXACTLY the same sampled
    normal level. No stress tolerance, guessed phase or additional cycles are
    introduced. Pairing is greedy in chronological order and never crosses a
    record boundary. Unmatched halves are explicitly labelled in the audit.
    """
    normal, shear = _history(normal, "normal history"), _history(shear, "shear history")
    if normal.shape != shear.shape or len(normal) < 2:
        raise InputError("normal and shear histories must have equal length >= 2")
    if not np.isfinite(resolution) or resolution < 0.0:
        raise InputError("range resolution must be finite and non-negative")
    if times is not None:
        times = _history(times, "times")
        if times.shape != normal.shape or np.any(np.diff(times) <= 0.0):
            raise InputError("times must match the history and be strictly increasing")
    if not isinstance(indexed, IndexedCycleTable):
        raise InputError("indexed must be an IndexedCycleTable from the owner adapter")
    n = len(indexed.cycles)
    arrays = (indexed.start_indices, indexed.end_indices, indexed.closing_indices,
              indexed.cycles.means, indexed.cycles.counts)
    if any(a.shape != (n,) for a in arrays) or len(indexed.kinds) != n:
        raise InputError("indexed cycle columns have inconsistent lengths")
    if any(a.dtype.kind not in "iu" for a in (indexed.start_indices,indexed.end_indices,indexed.closing_indices)):
        raise InputError("cycle indices must be integer arrays")
    if not np.all(np.isfinite(indexed.cycles.ranges)) or not np.all(indexed.cycles.ranges > 0):
        raise InputError("associated cycles require finite positive normal ranges")
    if (np.any(indexed.start_indices < 0) or np.any(indexed.end_indices >= len(normal))
        or np.any(indexed.start_indices >= indexed.end_indices)):
        raise InputError("cycle indices are outside the supplied history")
    if not np.array_equal(indexed.cycles.ranges, np.abs(normal[indexed.end_indices]-normal[indexed.start_indices])):
        raise InputError("indexed cycles do not belong to the supplied normal history")
    support_start = indexed.start_indices.copy()
    closure_left = indexed.end_indices.copy()
    fractions = np.zeros(n)
    partners = np.full(n, -1, dtype=np.int64)
    support_kinds = ["half_interval"]*n
    lows, highs = np.empty(n), np.empty(n)

    halves = sorted((i for i in range(n) if indexed.cycles.counts[i] == 0.5),
                    key=lambda i: int(indexed.start_indices[i]))
    by_start = {int(indexed.start_indices[i]): i for i in halves}
    if len(by_start) != len(halves):
        raise InputError("duplicate half-cycle starts in indexed contract")
    for i in halves:
        if partners[i] != -1:
            continue
        j = by_start.get(int(indexed.end_indices[i]))
        if (j is not None and partners[j] == -1
            and normal[indexed.start_indices[i]] == normal[indexed.end_indices[j]]):
            partners[i], partners[j] = j, i
            for k in (i, j):
                support_start[k] = indexed.start_indices[i]
                closure_left[k] = indexed.end_indices[j]
                support_kinds[k] = "paired_halves"
    if n:
        normal_tree, shear_tree = _RangeTree(normal), _RangeTree(shear)
    for i in range(n):
        if indexed.cycles.counts[i] == 1.0:
            a, b, c = int(indexed.start_indices[i]), int(indexed.end_indices[i]), int(indexed.closing_indices[i])
            if indexed.kinds[i] != "full" or not b < c < len(normal):
                raise InputError("full cycle has no valid closing reversal")
            target = normal[a]
            j = normal_tree.first_crossing(b+1, c, target, target > normal[b])
            if j is None:
                raise InputError("normal cycle does not return to its start before the closing reversal")
            if normal[j] == target:
                closure_left[i] = j
            else:
                closure_left[i] = j-1
                fraction = (target-normal[j-1])/(normal[j]-normal[j-1])
                if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                    raise InputError("invalid fractional normal-cycle closure")
                fractions[i] = fraction
            support_kinds[i] = "full_return"
        elif indexed.cycles.counts[i] != 0.5:
            raise InputError("unsupported indexed cycle count")
        low, high = shear_tree.extrema(support_start[i], closure_left[i])
        f, left = fractions[i], closure_left[i]
        if f:
            endpoint = float((1-f)*shear[left] + f*shear[left+1])
            low, high = min(low, endpoint), max(high, endpoint)
        lows[i], highs[i] = low, high
    with np.errstate(over="ignore", invalid="ignore"):
        ranges = highs-lows
    if not np.all(np.isfinite(ranges)):
        raise InputError("associated shear range overflowed")
    ranges[ranges <= resolution] = 0.0
    start_t = turn_t = support_t = closure_t = None
    if times is not None:
        start_t, turn_t = times[indexed.start_indices], times[indexed.end_indices]
        support_t = times[support_start]
        closure_t = times[closure_left].copy()
        fractional = fractions > 0
        left, f = closure_left[fractional], fractions[fractional]
        closure_t[fractional] = (1-f)*times[left] + f*times[left+1]
    return CycleAssociation(
        indexed.start_indices.copy(), indexed.end_indices.copy(), indexed.closing_indices.copy(),
        support_start, closure_left, fractions, partners, indexed.kinds, tuple(support_kinds),
        lows, highs, ranges, start_t, turn_t, support_t, closure_t,
    )
