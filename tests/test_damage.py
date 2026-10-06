"""Miner sums and the Weibull closed forms.

Independent references:

* hand-evaluated ``n / N`` for constant amplitude;
* the gamma-function column of DNV-RP-C203 Table 5-1 (``Gamma(1 + 3/h)``);
* direct numerical integration of the Weibull density (D.13-2), which shares no
  code with the incomplete-gamma form (D.13-1);
* the exceedance identity ``Q(dS0) = 1/n0`` behind equation (5.1.2).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyfatigue import damage as dm
from anyfatigue import sn_curves as sn
from anyfatigue.errors import InputError

# DNV-RP-C203 (April 2010) Table 5-1, Gamma(1 + m/h) for m = 3.0.
TABLE_5_1 = {0.60: 120.000, 0.70: 37.234, 0.80: 16.586, 0.90: 9.261,
             1.00: 6.000, 1.10: 4.306}


def test_miner_constant_amplitude_matches_hand_calculation():
    d = sn.get_curve("D", "air")
    result = dm.miner_damage([100.0], [2.0e5], d)
    n_fail = 10 ** (12.164 - 3 * math.log10(100.0))
    assert result.total == pytest.approx(2.0e5 / n_fail)
    assert result.cycles_to_failure[0] == pytest.approx(n_fail)


def test_miner_applies_scf_and_thickness_once():
    d = sn.get_curve("D", "air")
    scf, t = 1.4, 40.0
    result = dm.miner_damage([50.0], [1.0e6], d, scf=scf, thickness_mm=t)
    effective = 50.0 * scf * (t / 25.0) ** 0.20
    assert result.effective_ranges[0] == pytest.approx(effective)
    assert result.total == pytest.approx(1.0e6 / d.cycles_to_failure(effective))


def test_miner_scale_extrapolates_a_record_to_its_exposure():
    d = sn.get_curve("D")
    base = dm.miner_damage([80.0, 40.0], [3.0, 10.0], d).total
    assert dm.miner_damage([80.0, 40.0], [3.0, 10.0], d, scale=7.5).total == pytest.approx(
        7.5 * base
    )


def test_miner_empty_and_zero_range():
    d = sn.get_curve("D")
    assert dm.miner_damage([], [], d).total == 0.0
    assert dm.miner_damage([0.0], [5.0], d).total == 0.0


@pytest.mark.parametrize("bad", [
    ([1.0, 2.0], [1.0]), ([[1.0]], [[1.0]]), ([1.0], [-1.0]), ([float("nan")], [1.0]),
])
def test_miner_rejects_malformed_input(bad):
    with pytest.raises(InputError):
        dm.miner_damage(*bad, sn.get_curve("D"))


@pytest.mark.parametrize("h", sorted(TABLE_5_1))
def test_one_slope_weibull_damage_uses_the_printed_gamma_values(h):
    free = sn.get_curve("D", "free_corrosion")  # single slope, m = 3
    q, n = 12.5, 1.0e8
    expected = n * q ** 3 / free.a1 * TABLE_5_1[h]
    got = dm.weibull_damage(free, scale_q=q, shape_h=h, n_cycles=n)
    assert got == pytest.approx(expected, rel=2.0e-4)  # table has 3-5 digits


def test_weibull_scale_gives_exceedance_probability_one_over_n0():
    h, n0, s0 = 0.85, 1.0e4, 70.0
    q = dm.weibull_scale(s0, h, n0)
    assert math.exp(-((s0 / q) ** h)) == pytest.approx(1.0 / n0)


@pytest.mark.parametrize("name,env", [
    ("D", "air"), ("B1", "air"), ("F3", "seawater_cp"), ("W3", "air"), ("C2", "seawater_cp"),
])
@pytest.mark.parametrize("h", [0.6, 0.8, 1.0, 1.2])
@pytest.mark.parametrize("q", [0.5, 6.0, 25.0, 90.0])
def test_closed_form_agrees_with_direct_integration(name, env, h, q):
    curve = sn.get_curve(name, env)
    n = 2.0e7
    closed = dm.weibull_damage(curve, scale_q=q, shape_h=h, n_cycles=n, scf=1.2, thickness_mm=30.0)
    numeric = dm.weibull_damage_by_integration(
        curve, scale_q=q, shape_h=h, n_cycles=n, scf=1.2, thickness_mm=30.0
    )
    assert closed == pytest.approx(numeric, rel=1.0e-6)


def test_two_slope_collapses_to_the_one_slope_form_for_large_ranges():
    two = sn.get_curve("D", "air")
    one = sn.SNCurve.custom("D1", m1=3.0, log_a1=two.log_a1)
    # Mathematical limit q >> S1 = 52.6 MPa.  The relative difference is of the
    # order P(1 + m1/h, x) with x = (S1/q)^h; at q = 5000 MPa, h = 0.9 that is
    # x = 0.02 and P ~ 1e-9, far inside the tolerance.
    q, h, n = 5000.0, 0.9, 1.0e6
    a = dm.weibull_damage(two, scale_q=q, shape_h=h, n_cycles=n)
    b = dm.weibull_damage(one, scale_q=q, shape_h=h, n_cycles=n)
    assert a == pytest.approx(b, rel=1e-6)


def test_two_slope_collapses_to_slope_two_for_small_ranges():
    two = sn.get_curve("D", "air")
    lower = sn.SNCurve.custom("D2", m1=5.0, log_a1=two.log_a2)
    q, h, n = 0.4, 0.9, 1.0e9   # q << S1: essentially all damage on slope 2
    a = dm.weibull_damage(two, scale_q=q, shape_h=h, n_cycles=n)
    b = dm.weibull_damage(lower, scale_q=q, shape_h=h, n_cycles=n)
    assert a == pytest.approx(b, rel=1e-6)


def test_scf_scales_the_weibull_scale_so_damage_goes_as_scf_to_the_m():
    free = sn.get_curve("D", "free_corrosion")
    d1 = dm.weibull_damage(free, scale_q=10.0, shape_h=0.9, n_cycles=1e7)
    d2 = dm.weibull_damage(free, scale_q=10.0, shape_h=0.9, n_cycles=1e7, scf=2.0)
    assert d2 / d1 == pytest.approx(2.0 ** 3)


def test_truncation_at_the_largest_range_lowers_damage():
    curve = sn.get_curve("D", "air")
    q = dm.weibull_scale(120.0, 0.8, 1e4)
    full = dm.weibull_damage_by_integration(curve, scale_q=q, shape_h=0.8, n_cycles=1e6)
    cut = dm.weibull_damage_by_integration(
        curve, scale_q=q, shape_h=0.8, n_cycles=1e6, upper_range=120.0
    )
    assert cut < full


def test_zero_stress_and_zero_cycles_give_zero_damage():
    curve = sn.get_curve("D")
    assert dm.weibull_damage(curve, scale_q=0.0, shape_h=1.0, n_cycles=1e6) == 0.0
    assert dm.weibull_damage(curve, scale_q=5.0, shape_h=1.0, n_cycles=0.0) == 0.0


@pytest.mark.parametrize("args", [
    {"scale_q": -1.0, "shape_h": 1.0, "n_cycles": 1.0},
    {"scale_q": 1.0, "shape_h": 0.0, "n_cycles": 1.0},
    {"scale_q": 1.0, "shape_h": 1.0, "n_cycles": -1.0},
])
def test_weibull_rejects_invalid_parameters(args):
    with pytest.raises(InputError):
        dm.weibull_damage(sn.get_curve("D"), **args)


def test_weibull_scale_rejects_n0_not_above_one():
    with pytest.raises(InputError):
        dm.weibull_scale(10.0, 1.0, 1.0)


def test_non_welded_mean_stress_factor_equation_2_5_1():
    assert dm.non_welded_mean_stress_factor(100.0, -50.0) == pytest.approx(
        (100.0 + 0.6 * 50.0) / 150.0
    )
    assert dm.non_welded_mean_stress_factor(100.0, 20.0) == 1.0   # no compression
    assert dm.non_welded_mean_stress_factor(-20.0, -100.0) == pytest.approx(0.6)
    assert dm.non_welded_mean_stress_factor(0.0, 0.0) == 1.0


def test_design_check_is_damage_times_dff():
    ok = dm.design_check(0.4, 2.0)
    assert ok["usage"] == pytest.approx(0.8) and ok["passed"]
    assert not dm.design_check(0.6, 2.0)["passed"]
    with pytest.raises(InputError):
        dm.design_check(0.1, 0.0)
