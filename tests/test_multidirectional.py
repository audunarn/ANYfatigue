"""Analytical/contract regressions, not experimental fatigue qualification."""

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from anyfatigue import FatiguePlane, MultidirectionalSettings, get_curve
from anyfatigue.cli import main
from anyfatigue.damage import SECONDS_PER_YEAR
from anyfatigue.errors import InputError
from anyfatigue.extract import ElementLocation, PointLocation, ReadOut, SamplePoint
from anyfatigue.measures import StressMeasure
from anyfatigue.multidirectional import directional_stress_history, multidirectional_equivalent_range
from anyfatigue.plotdata import design_life_spectrum
from anyfatigue.project import AnalysisDefinition
from anyfatigue.report import report_markdown, result_to_dict
from anyfatigue.stress_io import write_stress_npz
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings, _assess_point, assess_time_series
from helpers import plate_mesh, source_from_fields, tensor_field

CURVE = get_curve("D", "free_corrosion")  # single slope, m=3, analytical checks
LOC = ElementLocation((1,), surfaces=("top",), weld_direction=(0, 1, 0))


def config(curve=CURVE, *, gamma=1.0, step=45.0):
    return MultidirectionalSettings.uniform(
        curve, gamma=gamma, gamma_basis="Analytical test input, not a design recommendation",
        curve_basis="Same curve explicitly assigned for this analytical test", step_deg=step,
    )


def source_for(tensors):
    tensors = np.asarray(tensors, dtype=float)
    fields = [tensor_field(np.vstack((v, np.zeros((2, 6))))) for v in tensors]
    labels = tuple(f"t{i}" for i in range(len(fields)))
    return source_from_fields(plate_mesh(2, 2), fields, labels=labels,
                              times=np.arange(len(fields))), labels


def settings(labels, **overrides):
    options = dict(curve=CURVE, thickness_mm=25.0, design_life_years=2.0 / SECONDS_PER_YEAR,
                   dff=3.0, blocks=(LoadBlock("record", tuple(labels)),), multidirectional=config())
    options.update(overrides)
    return TimeSeriesSettings(**options)


def uniaxial(values):
    tensor = np.zeros((len(values), 6))
    tensor[:, 0] = values
    return tensor


def run(tensors, **overrides):
    source, labels = source_for(tensors)
    return assess_time_series(source, [LOC], settings(labels, **overrides))


def at(point, angle):
    return next(d for d in point.directions if d.angle_deg == angle)


def test_projection_matches_independent_traction_formula_and_physical_shear():
    tensor = np.array([[100., 40., 0., 30., 0., 0.], [-30., 60., 0., -20., 0., 0.]])
    a = math.radians(30.0)
    c, s = math.cos(a), math.sin(a)
    normal, shear = directional_stress_history(tensor, (0, 0, 1), (0, 1, 0), 30.0)
    # cross(z,y) = -x, so weld-relative shear is -global_xy.
    assert normal == pytest.approx(tensor[:, 0]*c*c + tensor[:, 1]*s*s - 2*tensor[:, 3]*s*c)
    assert shear == pytest.approx((tensor[:, 1]-tensor[:, 0])*s*c - tensor[:, 3]*(c*c-s*s))


def test_equivalent_range_matches_hand_values_and_multiplies_gamma_outside_root():
    actual = multidirectional_equivalent_range([3, 0, 4], [40/9, 100, 0], gamma=1.2)
    assert actual == pytest.approx([6, 108, 4.8])
    assert multidirectional_equivalent_range(3, 40/9, gamma=1.0) == pytest.approx(5)


def test_uniaxial_limit_matches_existing_scalar_assessment():
    source, labels = source_for(uniaxial([30, -30, 30]))
    md = settings(labels)
    multi = assess_time_series(source, [LOC], md).critical
    scalar = assess_time_series(source, [LOC], replace(md, multidirectional=None)).critical
    assert multi.damage == pytest.approx(1 / CURVE.cycles_to_failure(60), rel=1e-12)
    assert multi.damage == pytest.approx(scalar.damage, rel=1e-12)
    assert multi.normal_to_weld_damage == scalar.damage
    assert multi.critical_angle_deg == 0.0
    assert multi.usage == pytest.approx(3 * multi.damage)
    assert multi.life_years == pytest.approx(md.design_life_years / multi.damage)


def test_gamma_scf_and_thickness_apply_once():
    c = CURVE
    p = run(uniaxial([30, -30, 30]), scf=1.3, thickness_mm=50,
            multidirectional=config(gamma=1.4)).critical
    expected_range = 60 * 1.4 * 1.3 * (50 / 25) ** c.k
    assert p.damage == pytest.approx(1 / c.cycles_to_failure(expected_range), rel=1e-12)
    assert p.parts[0].cycles.ranges == pytest.approx([60, 60])
    assert p.parts[0].equivalent_ranges == pytest.approx([84, 84])
    assert p.parts[0].cycles.means == pytest.approx([0, 0])


def test_rotating_constant_principal_stress_is_not_a_constant_fatigue_history():
    phi = np.deg2rad([0, 45, 90, 45, 0])
    tensors = np.zeros((len(phi), 6))
    tensors[:, 0] = 100 * np.cos(phi)**2
    tensors[:, 1] = 100 * np.sin(phi)**2
    tensors[:, 3] = 100 * np.cos(phi)*np.sin(phi)
    p = run(tensors).critical
    assert np.linalg.eigvalsh(np.array([[[x, t], [t, y]] for x,y,_,t,_,_ in tensors]))[:, 1] == pytest.approx(100)
    # At angle zero: one cycle, normal range=100 and whole-record shear range=50.
    expected = 0.5 / CURVE.cycles_to_failure(math.sqrt(100**2 + .81*50**2))
    assert at(p, 0).damage == pytest.approx(expected, rel=1e-12)
    assert p.damage > 0
    assert p.normal_to_weld_damage > 0


def test_shear_extrema_between_normal_extrema_are_not_lost():
    tensors = uniaxial([30, 15, 0, -15, -30, -15, 0, 15, 30])
    tensors[:, 3] = [0, 50, 0, -50, 0, 50, 0, -50, 0]
    p = run(tensors, multidirectional=config(gamma=1.2)).critical
    # One normal cycle with range 60; shear range is 100, not zero at the normal extrema.
    expected = .25 / CURVE.cycles_to_failure(1.2 * math.sqrt(60**2 + .81*100**2))
    assert at(p, 0).damage == pytest.approx(expected, rel=1e-12)
    assert at(p, 0).damage > p.normal_to_weld_damage


def test_pure_shear_is_seen_on_rotated_planes_not_silently_zero_overall():
    tensors = np.zeros((3, 6))
    tensors[:, 3] = [40, -40, 40]
    p = run(tensors).critical
    assert p.normal_to_weld_damage == 0.0
    assert p.damage == pytest.approx(1 / CURVE.cycles_to_failure(80), rel=1e-12)
    assert p.critical_angle_deg == 45.0
    assert "no resolved normal-master cycle" in p.notes


def test_constant_tensor_has_no_cycles_even_with_nonzero_constant_shear():
    tensors = np.tile([20, -30, 0, 40, 0, 0], (4, 1))
    p = run(tensors).critical
    assert p.damage == 0.0 and math.isinf(p.life_years)
    assert all(d.damage == 0 for d in p.directions)


def test_half_cycles_use_the_owner_counting_contract():
    p = run(uniaxial([0, 100, 0, 100])).critical
    assert p.parts[0].cycles.total_cycles == 1.5
    assert p.damage == pytest.approx(1 / CURVE.cycles_to_failure(100), rel=1e-12)  # scale 2/3


def test_blocks_accumulate_on_the_same_plane_before_selecting_the_maximum():
    tensors = np.zeros((6, 6))
    tensors[:3, 0] = [30, -30, 30]
    tensors[3:, 1] = [30, -30, 30]
    source, labels = source_for(tensors)
    blocks = (LoadBlock("x", labels[:3], .5), LoadBlock("y", labels[3:], .5))
    p = assess_time_series(source, [LOC], settings(labels, blocks=blocks)).critical
    assert p.damage == pytest.approx(.5 / CURVE.cycles_to_failure(60), rel=1e-12)
    assert p.damage == max(d.damage for d in p.directions)
    assert p.damage == pytest.approx(sum(x.damage for x in p.parts))
    wrong_sum_of_maxima = sum(max(d.block_damages[i] for d in p.directions) for i in range(2))
    assert p.damage < wrong_sum_of_maxima


def test_directional_curves_and_their_thickness_exponents_are_used_in_damage_and_plots():
    weak = get_curve("F", "free_corrosion")
    md = MultidirectionalSettings(
        (FatiguePlane(0, CURVE), FatiguePlane(45, CURVE), FatiguePlane(90, weak), FatiguePlane(135, CURVE)),
        1.0, "test", "test directional assignment",
    )
    tensors = np.zeros((3, 6))
    tensors[:, 1] = [30, -30, 30]
    result = run(tensors, multidirectional=md, thickness_mm=50)
    p = result.critical
    assert p.critical_angle_deg == 90
    effective = 60 * 2 ** weak.k
    assert p.damage == pytest.approx(1 / weak.cycles_to_failure(effective), rel=1e-12)
    n, ranges = design_life_spectrum(result, p.parts[0])
    assert ranges == pytest.approx([effective, effective]) and n == pytest.approx([.5, 1])
    assert p.parts[0].curve == weak


def test_three_dimensional_rigid_frame_rotation_preserves_damage_and_plane_angle():
    tensors = np.array([[30, 10, 0, 15, 0, 0], [-20, 15, 0, -10, 0, 0], [30, 10, 0, 15, 0, 0.]])
    point = SamplePoint("p", "point", "top", np.zeros(3), np.array([0.,0,1]),
                        np.array([0.,1,0]), tensors)
    st = settings(("a", "b", "c"), multidirectional=config(step=5))
    geometry = {"record": ([0, 1, 2], np.arange(3), 2.0)}
    base = _assess_point(point, st, geometry, True)
    # A fixed proper rotation, built independently via orthogonal QR decomposition.
    q, _ = np.linalg.qr(np.array([[1.,2,3],[2,-1,1],[1,1,-1]]))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    rotated = []
    for xx,yy,zz,xy,yz,xz in tensors:
        matrix = q @ np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]) @ q.T
        rotated.append(matrix[[0,1,2,0,1,0], [0,1,2,1,2,2]])
    other = _assess_point(replace(point, tensors=np.array(rotated), normal=q@point.normal,
                                 weld_direction=q@point.weld_direction), st, geometry, True)
    assert other.damage == pytest.approx(base.damage, rel=1e-12)
    assert other.critical_angle_deg == base.critical_angle_deg
    assert [d.damage for d in other.directions] == pytest.approx([d.damage for d in base.directions], rel=1e-12, abs=1e-25)


def test_angular_refinement_converges_to_the_analytical_uniaxial_direction():
    a = math.radians(17)
    tensors = np.zeros((3, 6))
    values = np.array([30, -30, 30])
    tensors[:, 0] = values * math.cos(a)**2
    tensors[:, 1] = values * math.sin(a)**2
    tensors[:, 3] = values * math.cos(a)*math.sin(a)
    coarse = run(tensors, multidirectional=config(step=10)).critical
    fine = run(tensors, multidirectional=config(step=1)).critical
    assert coarse.damage <= fine.damage
    assert fine.critical_angle_deg == 163.0
    assert fine.damage == pytest.approx(1 / CURVE.cycles_to_failure(60), rel=1e-12)


def test_roundoff_normal_cycles_do_not_receive_the_full_shear_range():
    tensors = np.zeros((5, 6))
    tensors[:, 0] = np.array([1,-1,1,-1,1]) * 1e-13
    tensors[:, 3] = [50,0,-50,0,50]
    p = run(tensors).critical
    assert at(p, 0).damage == 0
    assert "roundoff-scale" in p.notes
    assert p.damage > 0  # real cycles remain on the rotated planes


@pytest.mark.parametrize("method,curve_name,readout_value", [
    ("hotspot_a", "D", 140.0), ("hotspot_b", "E", 142.0),
])
def test_extraction_paths_are_componentwise_and_have_no_hidden_112_factor(method, curve_name, readout_value):
    field = np.zeros((3, 6))
    field[0, 0], field[1, 0] = 100., 200.  # xx = 100 + 200*x, x in metres
    source = source_from_fields(plate_mesh(), [tensor_field(f*field) for f in (1,-1,1)],
                                labels=("a","b","c"), times=(0,1,2))
    loc = PointLocation(((.2,.2,0),), surfaces=("top","bottom"), weld_direction=(0,1,0),
                        readout=ReadOut(method, thickness_mm=20, direction=(1,0,0)))
    curve = get_curve(curve_name, "free_corrosion")
    result = assess_time_series(source, [loc], settings(source.case_labels, curve=curve,
                                 thickness_mm=20, multidirectional=config(curve)))
    for p in result.points:
        assert p.damage == pytest.approx(1 / curve.cycles_to_failure(2*readout_value), rel=1e-12)
        assert p.parts[0].cycles.max_range == pytest.approx(2*readout_value)


def test_project_roundtrip_cli_and_reports_preserve_method_curves_and_gamma(tmp_path):
    source, labels = source_for(uniaxial([30,-30,30]))
    definition = AnalysisDefinition("time_series", settings(labels), (LOC,))
    path = definition.save(tmp_path / "md.json")
    loaded = AnalysisDefinition.load(path)
    assert loaded == definition
    result = loaded.run(source, keep_series=False)
    assert result.critical.parts[0].series is None
    assert result.critical.parts[0].series_times is None
    data = result_to_dict(result, include_cycles=True)
    json.dumps(data, allow_nan=False)
    point = data["points"][0]
    assert point["critical_angle_deg"] == 0
    assert len(point["directions"]) == 4
    assert point["parts"][0]["cycles"]["range_kind"] == "normal_master"
    assert point["parts"][0]["cycles"]["equivalent_range_mpa"] == pytest.approx([60, 60])
    text = report_markdown(result)
    assert "Multidirectional fixed-plane" in text and "not an independently qualified" in text
    assert "Analytical test input" in text
    stress_path = write_stress_npz(source, tmp_path / "stress.npz")
    assert main(["run", str(path), "--stress", str(stress_path), "--out", str(tmp_path)]) == 0
    cli_data = json.loads((tmp_path / "md-result.json").read_text())
    assert cli_data["points"][0]["directions"] == point["directions"]


def test_old_settings_roundtrip_has_no_new_key_or_changed_scalar_results():
    original = settings(("a","b"), multidirectional=None)
    assert "multidirectional" not in original.to_dict()
    assert TimeSeriesSettings.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("overrides", [
    {"mean_stress":"non_welded"}, {"measure":StressMeasure("principal_abs_max")},
    {"measure":StressMeasure("component", "xx")}, {"multidirectional":"yes"},
    {"curve":get_curve("E", "air")},
])
def test_incompatible_settings_fail_closed(overrides):
    with pytest.raises(InputError):
        settings(("a","b"), **overrides)


@pytest.mark.parametrize("gamma", [0, .9, -1, float("nan"), float("inf"), True, "bad"])
def test_invalid_gamma_is_rejected(gamma):
    with pytest.raises(InputError):
        config(gamma=gamma)


@pytest.mark.parametrize("angle", [-1, 180, float("inf"), float("nan"), True, "bad"])
def test_invalid_angles_are_rejected(angle):
    with pytest.raises(InputError):
        FatiguePlane(angle, CURVE)


@pytest.mark.parametrize("overrides", [
    {"planes":()}, {"planes":(FatiguePlane(0,CURVE), FatiguePlane(90,CURVE))},
    {"planes":(FatiguePlane(0,CURVE), FatiguePlane(0,CURVE), FatiguePlane(90,CURVE))},
    {"planes":(FatiguePlane(10,CURVE), FatiguePlane(50,CURVE), FatiguePlane(90,CURVE))},
    {"planes":None}, {"planes":(1,2,3)}, {"gamma_basis":" "}, {"curve_basis":""},
    {"shear_association":"cycle_interval"},
])
def test_invalid_configuration_fails_closed(overrides):
    with pytest.raises(InputError):
        replace(config(), **overrides)


@pytest.mark.parametrize("step", [0, -1, 90, float("inf"), 1e-9])
def test_invalid_grid_step_is_rejected(step):
    with pytest.raises(InputError):
        config(step=step)


@pytest.mark.parametrize("tensors,normal,weld", [
    ([1,2,3], (0,0,1), (0,1,0)), (np.zeros((1,6)), (0,0,1), (0,1,0)),
    (np.full((2,6), np.nan), (0,0,1), (0,1,0)),
    (np.ones((2,6), dtype=complex), (0,0,1), (0,1,0)),
    (np.zeros((2,6)), (0,0,0), (0,1,0)),
    (np.zeros((2,6)), (0,0,1), None), (np.zeros((2,6)), (0,0,1), (0,0,1)),
])
def test_bad_surface_history_or_frame_is_rejected(tensors, normal, weld):
    with pytest.raises(InputError):
        directional_stress_history(tensors, normal, weld, 0)


@pytest.mark.parametrize("normal,shear,gamma", [
    (-1,0,1), (1,-1,1), (np.nan,1,1), (1,np.inf,1),
    ([1,2],[1,2,3],1), (1e308,1e308,2), (1+2j, 3, 1),
])
def test_invalid_equivalent_ranges_fail_closed(normal, shear, gamma):
    with pytest.raises(InputError):
        multidirectional_equivalent_range(normal, shear, gamma=gamma)
