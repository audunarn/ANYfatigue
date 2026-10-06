"""Simplified (Weibull) fatigue, end to end from a stress source.

Two kinds of reference:

* **ANYstructure parity.**  The constants below were produced by
  ``ANYstructure`` (commit dc977b7c, ``anystruct.calc_structure.CalcFatigue``,
  ``example_data.obj_dict`` with ``fat_obj_dict``: curve ``Ec``, ``h = 0.8``,
  ``T = 9 s``, ``n0 = 10^4``, 20 years, fraction 1, external pressure 50 kPa,
  plate thickness varied) as ``get_damage_slope1 + get_damage_slope2``, together
  with the stress range ``dS0`` it derived.  ANYfatigue must reproduce its own
  predecessor on the same inputs.
* **Invariants** that need no second implementation: damage scales with the
  fraction of life, with ``SCF**m`` for a one-slope curve, and so on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyfatigue import damage as dm
from anyfatigue import sn_curves as sn
from anyfatigue.errors import InputError
from anyfatigue.extract import ElementLocation, LineLocation, PointLocation, ReadOut
from anyfatigue.measures import StressMeasure
from anyfatigue.simplified import LoadCondition, SimplifiedSettings, assess_simplified
from helpers import plate_mesh, scaled, source_from_fields, tensor_field

# (thickness mm, dS0 MPa, damage)  -- ANYstructure dc977b7c, see module docstring
ANYSTRUCTURE_REFERENCE = [
    (25.0, 36.992, 0.0023537087192241858),
    (50.0, 9.248, 4.601139711827479e-06),
    (12.5, 147.968, 1.3778324651440195),
    (40.0, 14.45, 3.4281162120751996e-05),
]
UNIFORM_XX = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0.0]])


def uniform(sxx=0.0, syy=0.0, sxy=0.0):
    return tensor_field(np.array([[sxx, syy, 0.0, sxy, 0.0, 0.0],
                                  [0.0] * 6, [0.0] * 6]))


@pytest.fixture
def mesh():
    return plate_mesh(6, 4)


ELEMENT = ElementLocation((1,), surfaces=("top",), weld_direction=(0.0, 1.0, 0.0), name="plate")


def settings(**overrides):
    base = dict(
        curve=sn.parse_anystructure_name("Ec"), thickness_mm=25.0, design_life_years=20.0,
        dff=2.0, n0=1.0e4,
        conditions=(LoadCondition("loaded", "dyn", weibull_h=0.8, period_s=9.0),),
    )
    base.update(overrides)
    return SimplifiedSettings(**base)


@pytest.mark.parametrize("thickness,range0,expected", ANYSTRUCTURE_REFERENCE)
def test_reproduces_anystructure_damage(mesh, thickness, range0, expected):
    # The load case holds the stress *amplitude*; the default range factor 2 gives dS0.
    source = source_from_fields(mesh, [uniform(sxx=range0 / 2.0)], labels=["dyn"])
    result = assess_simplified(source, [ELEMENT], settings(thickness_mm=thickness))
    assert result.points[0].damage == pytest.approx(expected, rel=1e-12)
    assert result.points[0].parts[0].values["stress_range_n0_mpa"] == pytest.approx(range0)


def test_usage_life_and_pass_flag(mesh):
    source = source_from_fields(mesh, [uniform(sxx=36.992 / 2.0)], labels=["dyn"])
    p = assess_simplified(source, [ELEMENT], settings(dff=2.0)).points[0]
    assert p.usage == pytest.approx(2.0 * p.damage)
    assert p.life_years == pytest.approx(20.0 / p.damage)
    assert p.passed
    thick = assess_simplified(
        source_from_fields(mesh, [uniform(sxx=147.968 / 2.0)], labels=["dyn"]),
        [ELEMENT], settings(thickness_mm=12.5)).points[0]
    assert thick.damage > 1.0 and not thick.passed


def test_fraction_of_life_weights_each_condition(mesh):
    source = source_from_fields(mesh, [uniform(sxx=20.0), uniform(sxx=35.0)], labels=["a", "b"])
    both = assess_simplified(source, [ELEMENT], settings(conditions=(
        LoadCondition("loaded", "a", 0.8, 9.0, fraction=0.3),
        LoadCondition("ballast", "b", 0.9, 7.0, fraction=0.5),
    ))).points[0]
    only_a = assess_simplified(source, [ELEMENT], settings(conditions=(
        LoadCondition("loaded", "a", 0.8, 9.0, fraction=1.0),))).points[0]
    only_b = assess_simplified(source, [ELEMENT], settings(conditions=(
        LoadCondition("ballast", "b", 0.9, 7.0, fraction=1.0),))).points[0]
    assert both.parts[0].damage == pytest.approx(0.3 * only_a.damage, rel=1e-12)
    assert both.parts[1].damage == pytest.approx(0.5 * only_b.damage, rel=1e-12)
    assert both.damage == pytest.approx(0.3 * only_a.damage + 0.5 * only_b.damage, rel=1e-12)


def test_one_slope_damage_scales_with_scf_to_the_m_and_range_to_the_m(mesh):
    free = sn.get_curve("D", "free_corrosion")
    base = assess_simplified(source_from_fields(mesh, [uniform(sxx=10.0)], labels=["dyn"]),
                             [ELEMENT], settings(curve=free)).points[0].damage
    scf = assess_simplified(source_from_fields(mesh, [uniform(sxx=10.0)], labels=["dyn"]),
                            [ELEMENT], settings(curve=free, scf=1.7)).points[0].damage
    double = assess_simplified(source_from_fields(mesh, [uniform(sxx=20.0)], labels=["dyn"]),
                               [ELEMENT], settings(curve=free)).points[0].damage
    assert scf / base == pytest.approx(1.7 ** 3, rel=1e-12)
    assert double / base == pytest.approx(2.0 ** 3, rel=1e-12)


def test_reference_case_defines_the_range_when_given(mesh):
    source = source_from_fields(mesh, [uniform(sxx=30.0), uniform(sxx=-10.0)],
                                labels=["max", "min"])
    cond = LoadCondition("op", "max", 0.9, 8.0, reference_case="min", range_factor=99.0)
    paired = assess_simplified(source, [ELEMENT], settings(conditions=(cond,))).points[0]
    assert paired.parts[0].values["stress_range_n0_mpa"] == pytest.approx(40.0)   # not 99 x
    explicit = assess_simplified(
        source_from_fields(mesh, [uniform(sxx=20.0)], labels=["max"]), [ELEMENT],
        settings(conditions=(LoadCondition("op", "max", 0.9, 8.0),))).points[0]
    assert paired.damage == pytest.approx(explicit.damage, rel=1e-12)   # 2 x 20 = 40


def test_stress_measure_resolves_against_the_weld(mesh):
    source = source_from_fields(mesh, [uniform(sxx=30.0, syy=10.0)], labels=["dyn"])
    across = assess_simplified(source, [ELEMENT], settings()).points[0]       # weld along y
    along = assess_simplified(
        source, [ElementLocation((1,), surfaces=("top",), weld_direction=(1.0, 0.0, 0.0))],
        settings()).points[0]
    assert across.parts[0].values["stress_range_n0_mpa"] == pytest.approx(60.0)   # 2 x sxx
    assert along.parts[0].values["stress_range_n0_mpa"] == pytest.approx(20.0)    # 2 x syy
    principal = assess_simplified(
        source, [ElementLocation((1,), surfaces=("top",))],
        settings(measure=StressMeasure("principal_abs_max"))).points[0]
    assert principal.parts[0].values["stress_range_n0_mpa"] == pytest.approx(60.0)


def test_dnv_effective_range_uses_the_hot_spot_formula(mesh):
    source = source_from_fields(mesh, [uniform(sxx=40.0, syy=30.0, sxy=10.0)], labels=["dyn"])
    result = assess_simplified(
        source, [ELEMENT], settings(measure=StressMeasure("dnv_effective", alpha=0.72)))
    # ranges are 2 x the case stress; weld along y so perp = xx, par = yy
    from anyfatigue.measures import effective_hot_spot_range
    expected = effective_hot_spot_range(80.0, 60.0, 20.0, alpha=0.72)
    assert result.points[0].parts[0].values["stress_range_n0_mpa"] == pytest.approx(expected)
    assert any("directly read-out" in w for w in result.warnings)


def test_hot_spot_method_b_carries_the_112_factor(mesh):
    mesh = plate_mesh(12, 8)
    source = source_from_fields(mesh, [uniform(sxx=40.0)], labels=["dyn"])
    def run(method):
        loc = PointLocation(((0.3, 0.2, 0.0),), surfaces=("top",), weld_direction=(0, 1, 0),
                            readout=ReadOut(method, thickness_mm=20.0))
        return assess_simplified(source, [loc], settings(
            measure=StressMeasure("dnv_effective"))).points[0].parts[0].values["stress_range_n0_mpa"]
    assert run("hotspot_a") == pytest.approx(80.0)
    assert run("hotspot_b") == pytest.approx(1.12 * 80.0)


def test_line_location_finds_the_critical_node():
    mesh = plate_mesh(6, 4)
    # stress grows with y; the weld line runs along y at x = 0.3
    source = source_from_fields(mesh, [tensor_field(np.array(
        [[10.0, 0, 0, 0, 0, 0], [0.0] * 6, [200.0, 0, 0, 0, 0, 0]]))], labels=["dyn"])
    ids = tuple(100 + j * 7 + 3 for j in range(5))
    result = assess_simplified(source, [LineLocation(ids, surfaces=("top", "bottom"))], settings())
    assert len(result.points) == 10
    assert result.critical.xyz[1] == pytest.approx(0.4)
    damages = [p.damage for p in result.points if p.surface == "top"]
    assert damages == sorted(damages)
    top = {p.chainage: p.damage for p in result.points if p.surface == "top"}
    bottom = {p.chainage: p.damage for p in result.points if p.surface == "bottom"}
    assert top == pytest.approx(bottom)       # the bottom field is the mirror image


def test_zero_stress_has_infinite_life(mesh):
    source = source_from_fields(mesh, [uniform()], labels=["dyn"])
    p = assess_simplified(source, [ELEMENT], settings()).points[0]
    assert p.damage == 0.0 and math.isinf(p.life_years) and p.passed


def test_provenance_records_the_standard_and_the_units(mesh):
    source = source_from_fields(mesh, [uniform(sxx=10.0)], labels=["dyn"])
    result = assess_simplified(source, [ELEMENT], settings())
    assert "DNV-RP-C203" in result.provenance["standard"]
    assert result.provenance["stress_unit"] == "MPa"
    assert result.provenance["nodal_stress"]
    assert result.settings["curve"]["source"].startswith("DNV-RP-C203")


def test_settings_round_trip():
    s = settings(measure=StressMeasure("dnv_effective", alpha=0.8), scf=1.3)
    again = SimplifiedSettings.from_dict(s.to_dict())
    assert again == s


@pytest.mark.parametrize("make", [
    lambda: settings(conditions=()),
    lambda: settings(n0=1.0),
    lambda: settings(dff=0.0),
    lambda: settings(thickness_mm=-3.0),
    lambda: settings(conditions=(LoadCondition("a", "x", 0.8, 9.0, fraction=0.7),
                                 LoadCondition("b", "y", 0.8, 9.0, fraction=0.7))),
    lambda: settings(conditions=(LoadCondition("a", "x", 0.8, 9.0),
                                 LoadCondition("a", "y", 0.8, 9.0, fraction=0.0))),
    lambda: LoadCondition("a", "x", 0.0, 9.0),
    lambda: LoadCondition("a", "x", 0.8, 0.0),
])
def test_invalid_settings_fail_closed(make):
    with pytest.raises(InputError):
        make()


def test_unknown_case_and_missing_weld_direction_fail_closed(mesh):
    source = source_from_fields(mesh, [uniform(sxx=10.0)], labels=["dyn"])
    with pytest.raises(InputError, match="no load case"):
        assess_simplified(source, [ELEMENT], settings(
            conditions=(LoadCondition("a", "nope", 0.8, 9.0),)))
    with pytest.raises(InputError, match="weld direction"):
        assess_simplified(source, [ElementLocation((1,), surfaces=("top",))], settings())
    with pytest.raises(InputError):
        assess_simplified(source, [], settings())


def test_large_shape_parameter_approaches_constant_amplitude_damage(mesh):
    """h -> infinity collapses the Weibull distribution to one stress range.

    With n0 = e, ln n0 = 1 and q = dS0 for every h, so the damage is
    N * q^m / a * Gamma(1 + m/h) -> N q^m / a: the constant-amplitude Miner sum.
    """

    curve = sn.get_curve("D", "air")
    n_cycles = 2.0e7
    source = source_from_fields(mesh, [uniform(sxx=45.0)], labels=["dyn"])
    cond = LoadCondition("c", "dyn", weibull_h=1.0e6, period_s=20.0 * dm.SECONDS_PER_YEAR / n_cycles)
    got = assess_simplified(source, [ELEMENT], settings(
        curve=curve, n0=math.e, conditions=(cond,))).points[0].damage
    expected = dm.miner_damage([90.0], [n_cycles], curve, thickness_mm=25.0).total
    assert got == pytest.approx(expected, rel=1e-5)


def test_results_state_the_edition_status_of_builtin_curves(mesh):
    source = source_from_fields(mesh, [uniform(sxx=10.0)], labels=["dyn"])
    builtin = assess_simplified(source, [ELEMENT], settings())
    assert any("April 2010" in w and "2024-10" in w for w in builtin.warnings)
    custom = assess_simplified(source, [ELEMENT], settings(
        curve=sn.SNCurve.custom("mine", m1=3.0, log_a1=12.0, m2=5.0, log_a2=15.4, n_switch=1e7)))
    assert not any("April 2010" in w for w in custom.warnings)
