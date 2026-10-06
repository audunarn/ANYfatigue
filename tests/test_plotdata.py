"""Plot data: S-N line, design-life spectrum, damage along a line."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyfatigue import examples, plotdata
from anyfatigue import sn_curves as sn
from anyfatigue.errors import InputError


@pytest.fixture(scope="module")
def synthetic():
    source, definition = examples.synthetic_plate(steps=80)
    return source, definition


def test_sn_line_follows_the_curve_and_validates_limits():
    d = sn.get_curve("D")
    n, s = plotdata.sn_curve_line(d, 10.0, 400.0)
    assert np.all(np.diff(n) < 0.0) and s[0] == pytest.approx(10.0)
    assert n == pytest.approx(d.cycles_to_failure(s))
    with pytest.raises(InputError):
        plotdata.sn_curve_line(d, 0.0, 10.0)


def test_time_series_spectrum_is_a_cumulative_exceedance_curve(synthetic):
    source, definition = synthetic
    result = definition.run(source)
    part = result.critical.parts[0]
    cycles, eff = plotdata.design_life_spectrum(result, part)
    assert np.all(np.diff(eff) <= 0.0) and np.all(np.diff(cycles) > 0.0)
    scale = part.values["exposure_scale"]
    assert cycles[-1] == pytest.approx(part.cycles.total_cycles * scale)
    curve = sn.SNCurve.from_dict(result.settings["curve"])
    assert eff[0] == pytest.approx(float(curve.effective_range(
        part.cycles.max_range, scf=result.settings["scf"], thickness_mm=result.settings["thickness_mm"])))


def test_simplified_spectrum_passes_through_one_over_n0_at_the_n0_range(synthetic):
    source, definition = synthetic
    simplified = examples.simplified_for(source, definition).run(source)
    part = simplified.critical.parts[0]
    cycles, eff = plotdata.design_life_spectrum(simplified, part)
    n0, n_total = simplified.settings["n0"], part.values["cycles_in_design_life"]
    curve = sn.SNCurve.from_dict(simplified.settings["curve"])
    s0 = float(curve.effective_range(part.values["stress_range_n0_mpa"],
                                     scf=simplified.settings["scf"],
                                     thickness_mm=simplified.settings["thickness_mm"]))
    at_s0 = float(np.interp(math.log(s0), np.log(eff), np.log(cycles)))
    assert math.exp(at_s0) == pytest.approx(n_total / n0, rel=2e-3)
    assert np.all(np.diff(cycles) < 0.0)           # larger ranges are exceeded less often


def test_area_between_spectrum_and_curve_is_the_damage(synthetic):
    """Damage = integral of dn / N(S) over the spectrum: recomputes the Miner sum."""

    source, definition = synthetic
    result = definition.run(source)
    part = result.critical.parts[0]
    cycles, eff = plotdata.design_life_spectrum(result, part)
    curve = sn.SNCurve.from_dict(result.settings["curve"])
    increments = np.diff(np.concatenate(([0.0], cycles)))
    assert np.sum(increments / curve.cycles_to_failure(eff)) == pytest.approx(part.damage, rel=1e-9)


def test_damage_along_line_groups_by_location_and_surface(synthetic):
    source, definition = synthetic
    result = definition.run(source)
    grouped = plotdata.damage_along_line(result)
    assert list(grouped) == ["weld top"]
    chainage, damage = grouped["weld top"]
    assert chainage[0] == 0.0 and np.all(np.diff(chainage) > 0.0) and len(damage) == 9


def test_empty_spectrum_for_a_zero_stress_point():
    source, definition = examples.synthetic_plate(steps=20)
    zero = definition.run(source)
    part = zero.points[0].parts[0]
    from dataclasses import replace

    empty = replace(part, cycles=None)
    assert plotdata.design_life_spectrum(zero, empty) is None


def test_sn_curve_between_cycles_spans_the_requested_cycle_range():
    curve = sn.get_curve("F", "seawater_cp")
    n, s = plotdata.sn_curve_between_cycles(curve, 1.0e3, 1.0e9)
    assert n.max() == pytest.approx(1.0e9, rel=1e-6) and n.min() == pytest.approx(1.0e3, rel=1e-6)
    assert np.all(np.diff(s) > 0.0)
    assert n == pytest.approx(curve.cycles_to_failure(s))
    for bad in ((0.0, 10.0), (10.0, 10.0), (100.0, 10.0)):
        with pytest.raises(InputError):
            plotdata.sn_curve_between_cycles(curve, *bad)
