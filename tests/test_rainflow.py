"""Rainflow adapter around ANYtimeseries.

The counting algorithm belongs to ANYtimeseries; these tests check the contract
ANYfatigue relies on (end points, zero-range removal, fail-closed input) against
a published example and invariants that do not depend on the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyfatigue.errors import InputError
from anyfatigue.rainflow import CycleTable, count_rainflow_cycles


def _rows(table: CycleTable):
    return sorted(zip(table.ranges.tolist(), table.means.tolist(), table.counts.tolist()))


def test_astm_e1049_example_series():
    # ASTM E1049-85 section 5.4.4 example, as reproduced in ANYtimeseries'
    # documentation: one full cycle of range 4 and six half cycles.
    series = [0, -2, 1, -3, 5, -1, 3, -4, 4, -2, 0]
    table = count_rainflow_cycles(series, endpoints=False)
    assert _rows(table) == sorted([
        (3.0, -0.5, 0.5), (4.0, -1.0, 0.5), (4.0, 1.0, 1.0),
        (6.0, 1.0, 0.5), (8.0, 0.0, 0.5), (8.0, 1.0, 0.5), (9.0, 0.5, 0.5),
    ])


def test_total_variation_invariant():
    """sum(2 * count * range) equals the total variation of the reversals."""

    rng = np.random.default_rng(7)
    for _ in range(25):
        series = np.cumsum(rng.normal(size=400))
        table = count_rainflow_cycles(series, endpoints=True)
        reversals = _reversals(series)
        variation = np.abs(np.diff(reversals)).sum()
        assert 2.0 * np.sum(table.counts * table.ranges) == pytest.approx(variation)


def _reversals(series):
    keep = [series[0]]
    for i in range(1, len(series) - 1):
        if (series[i] - series[i - 1]) * (series[i + 1] - series[i]) < 0.0:
            keep.append(series[i])
    keep.append(series[-1])
    return np.asarray(keep)


def test_constant_amplitude_sine_counts_whole_cycles():
    n_cycles = 10
    t = np.linspace(0.0, n_cycles, n_cycles * 8 + 1)
    series = 25.0 * np.sin(2.0 * np.pi * (t + 0.25))   # starts at the crest
    table = count_rainflow_cycles(series, endpoints=True)
    assert table.max_range == pytest.approx(50.0)
    assert np.allclose(table.ranges, 50.0)
    assert table.total_cycles == pytest.approx(n_cycles)


def test_end_points_matter_for_a_series_that_starts_at_its_peak():
    series = [10.0, 0.0, 6.0, 1.0, 4.0]
    kept = count_rainflow_cycles(series, endpoints=True)
    dropped = count_rainflow_cycles(series, endpoints=False)
    assert kept.max_range == 10.0
    assert dropped.max_range < kept.max_range


def test_flat_and_short_series_have_no_cycles():
    assert len(count_rainflow_cycles([3.0, 3.0, 3.0, 3.0])) == 0
    assert len(count_rainflow_cycles([1.0])) == 0
    assert len(count_rainflow_cycles([])) == 0


def test_repeated_samples_do_not_create_cycles():
    a = count_rainflow_cycles([0, 4, 4, 4, -2, -2, 5], endpoints=True)
    b = count_rainflow_cycles([0, 4, -2, 5], endpoints=True)
    assert _rows(a) == _rows(b)


def test_means_follow_the_range_midpoint():
    table = count_rainflow_cycles([0.0, 10.0, 2.0, 8.0, 0.0], endpoints=True)
    assert np.allclose(table.maxima - table.minima, table.ranges)


@pytest.mark.parametrize("bad", [[1.0, float("nan"), 2.0], [[1.0, 2.0], [3.0, 4.0]]])
def test_malformed_series_fail_closed(bad):
    with pytest.raises(InputError):
        count_rainflow_cycles(bad)
