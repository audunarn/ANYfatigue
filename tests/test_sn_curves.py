"""S-N curve data and thickness correction.

References are the printed columns of DNV-RP-C203 (April 2010) that are *not*
used to build the curves: the "fatigue limit at 10^7 cycles" column of Tables
2-1 and 2-2 is derived here from ``log a1`` (air) or ``log a2`` (cathodic
protection), so it independently checks the transcription of those columns.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyfatigue import sn_curves as sn
from anyfatigue.errors import InputError

# Table 2-1 / 2-2, "Fatigue limit at 10^7 cycles" column (MPa), April 2010.
FATIGUE_LIMIT_1E7 = {
    "B1": 106.97, "B2": 93.59, "C": 73.10, "C1": 65.50, "C2": 58.48,
    "D": 52.63, "E": 46.78, "F": 41.52, "F1": 36.84, "F3": 32.75,
    "G": 29.24, "W1": 26.32, "W2": 23.39, "W3": 21.05, "T": 52.63,
}


def _rounding_bound(m: float, printed_limit: float) -> float:
    """Relative error bound from the table's own rounding.

    ``log a`` is printed to three decimals (+-0.0005) and the limit to two
    (+-0.005 MPa); a stress ``10**((log a - 7)/m)`` therefore carries at most
    ``0.0005 * ln(10) / m`` relative error from ``log a`` plus
    ``0.005 / limit`` from the printed value.
    """

    return 0.0005 * math.log(10.0) / m + 0.005 / printed_limit


@pytest.mark.parametrize("name", sorted(FATIGUE_LIMIT_1E7))
def test_air_curve_reproduces_the_printed_fatigue_limit(name):
    printed = FATIGUE_LIMIT_1E7[name]
    curve = sn.get_curve(name, "air")
    # Slope 1 reaches the printed limit at 10^7 cycles within table rounding.
    assert curve.stress_switch == pytest.approx(
        printed, rel=_rounding_bound(curve.m1, printed)
    )
    # So does slope 2 (the limit is printed from it).
    from_slope_2 = 10.0 ** ((curve.log_a2 - 7.0) / curve.m2)
    assert from_slope_2 == pytest.approx(
        printed, rel=_rounding_bound(curve.m2, printed)
    )


@pytest.mark.parametrize("name", sorted(FATIGUE_LIMIT_1E7))
def test_cathodic_protection_curve_joins_the_air_curve_beyond_the_knee(name):
    cp = sn.get_curve(name, "seawater_cp")
    air = sn.get_curve(name, "air")
    assert cp.n_switch == 1.0e6
    assert cp.log_a2 == air.log_a2 and cp.m2 == air.m2
    # Slope 2 evaluated at 10^7 cycles is the printed fatigue limit.
    printed = FATIGUE_LIMIT_1E7[name]
    assert 10.0 ** ((cp.log_a2 - 7.0) / cp.m2) == pytest.approx(
        printed, rel=_rounding_bound(cp.m2, printed)
    )
    # Slope 1 and slope 2 meet at the knee: same stress range at 10^6 cycles,
    # to within the rounding of both intercepts.
    knee_from_slope_1 = cp.stress_switch
    knee_from_slope_2 = 10.0 ** ((cp.log_a2 - 6.0) / cp.m2)
    bound = 0.0005 * math.log(10.0) * (1.0 / cp.m1 + 1.0 / cp.m2)
    assert knee_from_slope_1 == pytest.approx(knee_from_slope_2, rel=bound)


def test_free_corrosion_is_single_slope_m3_for_every_curve():
    for name in sn.curve_names():
        curve = sn.get_curve(name, "free_corrosion")
        assert not curve.bilinear
        assert curve.m1 == 3.0
        assert math.isinf(float(curve.cycles_to_failure(0.0)))


def test_free_corrosion_values_transcribed_from_table_2_3():
    expected = {"B1": 12.436, "D": 11.687, "F3": 11.068, "W3": 10.493, "T": 11.687}
    for name, log_a in expected.items():
        assert sn.get_curve(name, "free_corrosion").log_a1 == log_a


def test_thickness_exponent_by_curve_class():
    for name, k in {"B1": 0.0, "B2": 0.0, "C": 0.15, "D": 0.20, "E": 0.20,
                    "F": 0.25, "W3": 0.25}.items():
        assert sn.get_curve(name).k == k


def test_cycles_to_failure_follows_log_n_equals_log_a_minus_m_log_s():
    d = sn.get_curve("D", "air")
    s = 100.0
    assert d.cycles_to_failure(s) == pytest.approx(10 ** (12.164 - 3 * math.log10(s)))
    low = 30.0  # below the 52.63 MPa knee: slope 2
    assert d.cycles_to_failure(low) == pytest.approx(10 ** (15.606 - 5 * math.log10(low)))


def test_cycles_to_failure_is_monotonic_and_vectorised():
    d = sn.get_curve("D", "air")
    s = np.geomspace(5.0, 400.0, 300)
    n = d.cycles_to_failure(s)
    assert n.shape == s.shape
    assert np.all(np.diff(n) < 0.0)


def test_thickness_factor_uses_reference_thickness_below_t_ref():
    d = sn.get_curve("D")
    assert d.thickness_factor(10.0) == 1.0
    assert d.thickness_factor(25.0) == 1.0
    assert d.thickness_factor(50.0) == pytest.approx(2.0 ** 0.20)
    assert d.thickness_factor(None) == 1.0
    with pytest.raises(InputError):
        d.thickness_factor(0.0)


def test_thickness_correction_shortens_life_as_dnv_2_4_3():
    d = sn.get_curve("D")
    base = d.cycles_to_failure(d.effective_range(100.0, thickness_mm=25.0))
    thick = d.cycles_to_failure(d.effective_range(100.0, thickness_mm=50.0))
    # log N = log a - m log(dS (t/tref)^k)  ->  N ratio = (t/tref)^(-m k)
    assert thick / base == pytest.approx(2.0 ** (-3.0 * 0.20))


def test_t_curve_exponent_depends_on_scf_and_reference_is_32_mm():
    t = sn.get_curve("T")
    assert t.t_ref_mm == 32.0
    assert t.thickness_exponent(scf=5.0) == 0.25
    assert t.thickness_exponent(scf=10.0) == 0.25
    assert t.thickness_exponent(scf=10.5) == 0.30
    assert t.thickness_factor(64.0, scf=12.0) == pytest.approx(2.0 ** 0.30)


def test_allowable_range_inverts_cycles_to_failure():
    d = sn.get_curve("F", "seawater_cp")
    for cycles in (1e4, 1e6, 5e6, 1e7, 3e8):
        s = d.allowable_range(cycles, scf=1.5, thickness_mm=40.0)
        eff = d.effective_range(s, scf=1.5, thickness_mm=40.0)
        assert d.cycles_to_failure(eff) == pytest.approx(cycles, rel=1e-9)


def test_custom_curve_and_round_trip():
    c = sn.SNCurve.custom("X", m1=3.0, log_a1=12.0, m2=5.0, log_a2=15.0,
                          n_switch=1e7, k=0.1)
    assert sn.SNCurve.from_dict(c.to_dict()) == c
    assert "user-defined" in c.source


@pytest.mark.parametrize("kwargs", [
    {"m1": 0.0, "log_a1": 12.0},
    {"m1": 3.0, "log_a1": float("nan")},
    {"m1": 3.0, "log_a1": 12.0, "m2": 5.0},            # incomplete bi-linear
    {"m1": 3.0, "log_a1": 12.0, "t_ref_mm": -1.0},
])
def test_invalid_curves_fail_closed(kwargs):
    with pytest.raises(InputError):
        sn.SNCurve.custom("bad", **kwargs)


def test_unknown_curve_and_environment_fail_closed():
    with pytest.raises(InputError):
        sn.get_curve("Z9")
    with pytest.raises(InputError):
        sn.get_curve("D", "vacuum")


def test_negative_stress_range_is_rejected():
    with pytest.raises(InputError):
        sn.get_curve("D").cycles_to_failure(-1.0)


def test_anystructure_names_map_to_environments():
    assert sn.parse_anystructure_name("Ec") == sn.get_curve("E", "seawater_cp")
    assert sn.parse_anystructure_name("B1c") == sn.get_curve("B1", "seawater_cp")
    assert sn.parse_anystructure_name("C") == sn.get_curve("C", "air")
    with pytest.raises(InputError):
        sn.parse_anystructure_name("Q")


def test_builtin_curves_carry_an_edition_warning_but_custom_curves_do_not():
    warnings = sn.edition_warnings(sn.get_curve("D", "air"))
    assert len(warnings) == 1
    assert "April 2010" in warnings[0] and "2024-10" in warnings[0] and "has not been checked" in warnings[0]
    assert sn.edition_warnings(sn.SNCurve.custom("mine", m1=3.0, log_a1=12.0)) == ()


def test_the_t_curve_gets_an_extra_reference_thickness_warning():
    warnings = sn.edition_warnings(sn.get_curve("T", "seawater_cp"))
    assert len(warnings) == 2 and "32 mm" in warnings[1]
