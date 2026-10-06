"""Rainflow time-series fatigue from a load-case series."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyfatigue import damage as dm
from anyfatigue import sn_curves as sn
from anyfatigue.errors import InputError
from anyfatigue.extract import ElementLocation
from anyfatigue.measures import StressMeasure
from anyfatigue.simplified import LoadCondition, SimplifiedSettings, assess_simplified
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings, assess_time_series
from helpers import plate_mesh, source_from_fields, tensor_field

LIFE = 20.0
LOC = ElementLocation((1,), surfaces=("top",), weld_direction=(0.0, 1.0, 0.0), name="plate")


def uniform_xx(value):
    return tensor_field(np.array([[value, 0, 0, 0, 0, 0], [0.0] * 6, [0.0] * 6]))


def series_source(values, dt=1.0, labels=None):
    mesh = plate_mesh(2, 2)
    labels = labels or [f"t{k}" for k in range(len(values))]
    return source_from_fields(mesh, [uniform_xx(v) for v in values], labels=labels,
                              times=[k * dt for k in range(len(values))]), labels


def settings(labels, **overrides):
    base = dict(curve=sn.get_curve("D", "air"), thickness_mm=25.0, design_life_years=LIFE,
                dff=3.0, blocks=(LoadBlock("sea", tuple(labels)),))
    base.update(overrides)
    return TimeSeriesSettings(**base)


def sine_values(amplitude, n_cycles, per_cycle=8, mean=0.0):
    t = np.arange(n_cycles * per_cycle + 1) / per_cycle
    return mean + amplitude * np.sin(2.0 * np.pi * (t + 0.25))     # starts and ends at a crest


def test_regular_wave_damage_equals_n_over_N_times_the_exposure_scale():
    n_cycles, amplitude, dt = 12, 30.0, 0.5
    values = sine_values(amplitude, n_cycles)
    source, labels = series_source(values, dt=dt)
    duration = (len(values) - 1) * dt
    result = assess_time_series(source, [LOC], settings(labels))
    curve = sn.get_curve("D", "air")
    n_fail = curve.cycles_to_failure(2.0 * amplitude)
    expected = (LIFE * dm.SECONDS_PER_YEAR / duration) * n_cycles / n_fail
    p = result.points[0]
    assert p.damage == pytest.approx(expected, rel=1e-12)
    assert p.parts[0].values["cycles_in_record"] == pytest.approx(n_cycles)
    assert p.parts[0].values["max_range_mpa"] == pytest.approx(2.0 * amplitude)
    assert p.usage == pytest.approx(3.0 * expected)


def test_damage_scales_with_stress_to_the_m_for_a_one_slope_curve():
    free = sn.get_curve("D", "free_corrosion")
    rng = np.random.default_rng(11)
    walk = np.cumsum(rng.normal(size=300)) * 4.0
    s1, labels = series_source(walk)
    s2, _ = series_source(1.5 * walk)
    d1 = assess_time_series(s1, [LOC], settings(labels, curve=free)).points[0].damage
    d2 = assess_time_series(s2, [LOC], settings(labels, curve=free)).points[0].damage
    assert d2 / d1 == pytest.approx(1.5 ** 3, rel=1e-12)


def test_damage_is_invariant_to_time_reversal():
    rng = np.random.default_rng(5)
    walk = np.cumsum(rng.normal(size=250)) * 5.0
    fwd, labels = series_source(walk)
    rev, rlabels = series_source(walk[::-1])
    a = assess_time_series(fwd, [LOC], settings(labels)).points[0].damage
    b = assess_time_series(rev, [LOC], settings(rlabels)).points[0].damage
    assert a == pytest.approx(b, rel=1e-12)


def test_damage_is_invariant_to_a_constant_stress_offset_without_mean_correction():
    values = sine_values(25.0, 6)
    a, la = series_source(values)
    b, lb = series_source(values + 140.0)
    assert assess_time_series(a, [LOC], settings(la)).points[0].damage == pytest.approx(
        assess_time_series(b, [LOC], settings(lb)).points[0].damage, rel=1e-12)


def test_non_welded_mean_stress_correction_reduces_damage_for_compression():
    values = sine_values(40.0, 6, mean=-20.0)       # swings between +20 and -60
    source, labels = series_source(values)
    plain = assess_time_series(source, [LOC], settings(labels)).points[0]
    corrected = assess_time_series(source, [LOC], settings(labels, mean_stress="non_welded")).points[0]
    f_m = (20.0 + 0.6 * 60.0) / 80.0
    assert corrected.parts[0].values["mean_stress_factor"] == pytest.approx(f_m)
    assert corrected.damage < plain.damage


def test_end_points_are_counted_as_half_cycles():
    # 0 -> 100 -> 0 -> 100: one full cycle (0..100) and a half cycle of the same range.
    source, labels = series_source([0.0, 100.0, 0.0, 100.0])
    p = assess_time_series(source, [LOC], settings(labels)).points[0]
    assert p.parts[0].values["cycles_in_record"] == pytest.approx(1.5)
    assert p.parts[0].values["max_range_mpa"] == pytest.approx(100.0)


def test_blocks_add_and_exposure_scales_linearly():
    values = sine_values(30.0, 5)
    source, labels = series_source(values)
    one = assess_time_series(source, [LOC], settings(labels, blocks=(
        LoadBlock("a", tuple(labels), exposure_fraction=1.0),))).points[0].damage
    half = assess_time_series(source, [LOC], settings(labels, blocks=(
        LoadBlock("a", tuple(labels), exposure_fraction=0.5),))).points[0]
    assert half.damage == pytest.approx(0.5 * one, rel=1e-12)

    labs = [f"t{k}" for k in range(len(values))]
    two = assess_time_series(source, [LOC], settings(labs, blocks=(
        LoadBlock("a", tuple(labs), exposure_fraction=0.25),
        LoadBlock("b", tuple(labs), exposure_fraction=0.25)))).points[0]
    assert two.damage == pytest.approx(0.5 * one, rel=1e-12)
    assert [part.name for part in two.parts] == ["a", "b"]


def test_explicit_duration_overrides_case_times():
    values = sine_values(30.0, 5)
    source, labels = series_source(values, dt=1.0)
    auto = assess_time_series(source, [LOC], settings(labels)).points[0].damage
    explicit = assess_time_series(source, [LOC], settings(labels, blocks=(
        LoadBlock("a", tuple(labels), duration_s=2.0 * (len(values) - 1)),))).points[0].damage
    assert explicit == pytest.approx(0.5 * auto, rel=1e-12)


def test_duration_without_times_or_explicit_value_fails_closed():
    mesh = plate_mesh(2, 2)
    source = source_from_fields(mesh, [uniform_xx(v) for v in (0.0, 10.0, 0.0)],
                                labels=["a", "b", "c"])       # no times
    with pytest.raises(InputError, match="duration"):
        assess_time_series(source, [LOC], settings(["a", "b", "c"]))
    ok = assess_time_series(source, [LOC], settings(["a", "b", "c"], blocks=(
        LoadBlock("x", ("a", "b", "c"), duration_s=10.0),)))
    assert ok.points[0].damage > 0.0


def test_partial_coverage_of_the_design_life_is_reported():
    source, labels = series_source(sine_values(20.0, 3))
    result = assess_time_series(source, [LOC], settings(labels, blocks=(
        LoadBlock("a", tuple(labels), exposure_fraction=0.3),)))
    assert any("30.0%" in w for w in result.warnings)


def test_series_is_kept_for_plotting_unless_disabled():
    values = sine_values(20.0, 3)
    source, labels = series_source(values)
    kept = assess_time_series(source, [LOC], settings(labels)).points[0].parts[0]
    assert kept.series == pytest.approx(values) and kept.series_times is not None
    dropped = assess_time_series(source, [LOC], settings(labels), keep_series=False).points[0].parts[0]
    assert dropped.series is None


@pytest.mark.parametrize("make", [
    lambda: LoadBlock("a", ("only",)),
    lambda: LoadBlock("a", ("x", "y"), exposure_fraction=1.5),
    lambda: LoadBlock("a", ("x", "y"), duration_s=0.0),
    lambda: settings(["x", "y"], mean_stress="goodman"),
    lambda: settings(["x", "y"], blocks=(LoadBlock("a", ("x", "y"), 0.7),
                                         LoadBlock("b", ("x", "y"), 0.7))),
    lambda: settings(["x", "y"], measure=StressMeasure("dnv_effective")),
    lambda: settings(["x", "y"], blocks=(LoadBlock("a", ("x", "y")), LoadBlock("a", ("x", "y"), 0.0))),
])
def test_invalid_settings_fail_closed(make):
    with pytest.raises(InputError):
        make()


def test_time_order_and_duplicates_fail_closed():
    source, labels = series_source([0.0, 10.0, 0.0, 10.0])
    with pytest.raises(InputError, match="increasing time"):
        assess_time_series(source, [LOC], settings(labels[::-1]))
    with pytest.raises(InputError, match="twice"):
        assess_time_series(source, [LOC], settings(["t0", "t1", "t1"]))
    with pytest.raises(InputError, match="no load case"):
        assess_time_series(source, [LOC], settings(["t0", "zz"]))


def test_settings_round_trip():
    s = settings(["a", "b"], mean_stress="non_welded", scf=1.2,
                 measure=StressMeasure("component", "yy"))
    assert TimeSeriesSettings.from_dict(s.to_dict()) == s


def test_agrees_with_the_simplified_method_for_constant_amplitude():
    """A regular wave and a Weibull distribution with h -> infinity are the same load."""

    amplitude, n_cycles, dt = 22.0, 40, 0.25
    values = sine_values(amplitude, n_cycles)
    source, labels = series_source(values, dt=dt)
    curve = sn.get_curve("D", "air")
    ts = assess_time_series(source, [LOC], settings(labels, curve=curve)).points[0].damage

    period = dt * 8                                        # 8 samples per cycle
    cond = LoadCondition("c", labels[0], weibull_h=1.0e6, period_s=period)
    simp_source, _ = series_source([amplitude] + [0.0] * 2, dt=1.0, labels=[labels[0], "z1", "z2"])
    simplified = assess_simplified(simp_source, [LOC], SimplifiedSettings(
        curve=curve, thickness_mm=25.0, design_life_years=LIFE, dff=3.0, n0=math.e,
        conditions=(cond,))).points[0].damage
    # the record spans n_cycles periods plus nothing: duration = n_cycles * period
    assert ts == pytest.approx(simplified, rel=2e-5)


def test_time_series_results_and_reports_state_the_edition_status():
    from anyfatigue.report import report_markdown

    source, labels = series_source(sine_values(20.0, 3))
    result = assess_time_series(source, [LOC], settings(labels, blocks=(LoadBlock("a", tuple(labels)),)))
    assert any("2024-10" in w for w in result.warnings)
    assert "Warning: S-N data are transcribed from DNV-RP-C203, April 2010" in report_markdown(result)
