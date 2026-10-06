"""ANYfem results as a stress source, checked against ANYfem itself and against
closed-form plate theory.

The model is ANYfem's own postprocessing fixture: a 1 m square plate, 10 mm
thick, simply supported on all four edges, under uniform pressure.  The load
cases form a batch (one factorisation) scaled by a factor each, which is also
what a load-case series produces.

References that are not ANYfatigue code:

* ANYfem's own field route, ``evaluate_field(shape, "top_xx")``, which forms
  the surface stress as membrane + bending in the element frame;
* Timoshenko, *Theory of Plates and Shells*: a simply supported square plate
  under uniform ``q`` has centre moment ``M = 0.0479 q a^2`` and surface stress
  ``6 M / t^2``;
* superposition: the response is linear in the load factor.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.anyfem

from anyfatigue import damage as dm
from anyfatigue import sn_curves as sn
from anyfatigue.errors import ExtractionError, UnsupportedInputError
from anyfatigue.extract import ElementLocation, LineLocation, PointLocation, ReadOut, extract
from anyfatigue.simplified import LoadCondition, SimplifiedSettings, assess_simplified
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings, assess_time_series

SIDE, THICKNESS, PRESSURE = 1.0, 0.010, 10_000.0
FACTORS = [0.0, 1.0, -1.0, 0.5, 0.0, 1.0, -1.0, 0.5, 0.0]       # one 4-step wave, twice, closed


@pytest.fixture(scope="module")
def model():
    from anyfem import Project, pinned, steel
    from anyfem.solve.run import solve_linear_static_many

    project = Project(name="deck")
    project.add_material(steel("S355", THICKNESS))
    project.add_plate_section("plate", thickness=THICKNESS, material="S355")
    geometry = project.geometry
    points = geometry.add_points([(0, 0, 0), (SIDE, 0, 0), (SIDE, SIDE, 0), (0, SIDE, 0)])
    edges = geometry.add_polyline(points, close=True)
    face = geometry.add_face(edges)
    project.assign_plate(face, "plate")
    for edge in edges:
        project.add_support(pinned(project.edge(edge)))
    names = []
    for k, factor in enumerate(FACTORS):
        names.append(f"step{k}")
        project.load_case(names[-1]).add_pressure(project.face(face), PRESSURE * factor)
    mesh = project.generate_mesh(SIDE / 16)
    batch = solve_linear_static_many(project, mesh=mesh, load_cases=names)
    return project, face, edges, batch


@pytest.fixture(scope="module")
def source(model):
    from anyfatigue.anyfem_adapter import from_solutions

    project, _face, _edges, batch = model
    return from_solutions(batch, times=[0.5 * k for k in range(len(FACTORS))], project=project)


def centre_element(source):
    centroids = source.mesh.centroids()
    distance = np.linalg.norm(centroids - np.array([0.5, 0.5, 0.0]), axis=1)
    return int(source.mesh.element_ids[np.argmin(distance)])


def test_labels_times_and_provenance(source):
    assert source.case_labels == tuple(f"step{k}" for k in range(len(FACTORS)))
    assert source.case_times[3] == pytest.approx(1.5)
    assert source.provenance["unit_conversion"].startswith("Pa -> MPa")
    assert source.provenance["producer"]["ANYfem"]
    assert source.nodal_origin.startswith("ANYfem patch recovery")


def test_element_stress_equals_anyfem_membrane_plus_bending(model, source):
    from anyfem.post import evaluate_field

    _project, _face, _edges, batch = model
    shape = batch.case("step1")
    rows = np.arange(source.mesh.n_elements)
    top = source.element_stress(rows, "top")[1]
    bottom = source.element_stress(rows, "bottom")[1]
    for column, name in ((0, "xx"), (1, "yy"), (3, "xy")):
        top_field = evaluate_field(shape, f"top_{name}", reduction="mean")
        bottom_field = evaluate_field(shape, f"bottom_{name}", reduction="mean")
        expected_top = np.array([top_field.element_values[int(e)] for e in source.mesh.element_ids])
        expected_bottom = np.array([bottom_field.element_values[int(e)] for e in source.mesh.element_ids])
        # ANYfem forms this in the element frame, the source reads it in global
        # axes; the plate lies in the global xy plane so they must agree.
        assert top[:, column] == pytest.approx(expected_top * 1e-6, rel=1e-9, abs=1e-9)
        assert bottom[:, column] == pytest.approx(expected_bottom * 1e-6, rel=1e-9, abs=1e-9)


def test_top_and_bottom_are_mirror_images_for_pure_bending(source):
    rows = np.arange(source.mesh.n_elements)
    top = source.element_stress(rows, "top")[1]
    bottom = source.element_stress(rows, "bottom")[1]
    in_plane = [0, 1, 2, 3]                                     # xx yy zz xy
    assert bottom[:, in_plane] == pytest.approx(-top[:, in_plane], abs=1e-9)   # no membrane
    # ANYfem reports the same transverse shear (yz, xz) on both faces: it is a
    # through-thickness average, not a surface stress, and the in-plane
    # projection in anyfatigue.measures never reads it.
    assert bottom[:, [4, 5]] == pytest.approx(top[:, [4, 5]], abs=1e-9)


def test_response_is_linear_in_the_load_factor(source):
    rows = np.arange(source.mesh.n_elements)
    base = source.element_stress(rows, "top")[1]
    for k, factor in enumerate(FACTORS):
        assert source.element_stress(rows, "top")[k] == pytest.approx(factor * base, abs=1e-8)
    nodes = np.arange(source.mesh.n_nodes)
    nodal = source.node_stress(nodes, "top")
    for k, factor in enumerate(FACTORS):
        assert nodal[k] == pytest.approx(factor * nodal[1], abs=1e-8)


def test_centre_stress_matches_timoshenko(source):
    expected = 6.0 * 0.0479 * PRESSURE * SIDE ** 2 / THICKNESS ** 2 * 1e-6      # MPa
    element = centre_element(source)
    row = source.mesh.element_row(element)
    sxx = source.element_stress([row], "top")[1, 0, 0]
    assert abs(sxx) == pytest.approx(expected, rel=0.03)
    # patch-recovered nodal stress at the centre node, interpolated at the centre point
    point = PointLocation(((0.5, 0.5, 0.0),), surfaces=("top",))
    interpolated = extract(source, point)[0].tensors[1]
    assert abs(interpolated[0]) == pytest.approx(expected, rel=0.03)
    assert abs(interpolated[1]) == pytest.approx(expected, rel=0.03)    # square plate: syy = sxx
    assert interpolated[2] == pytest.approx(0.0, abs=1e-6)               # no through-thickness stress


def test_interpolation_at_a_node_returns_the_nodal_value(source):
    mesh = source.mesh
    node = mesh.node_ids[mesh.n_nodes // 2]
    xyz = mesh.node_xyz[mesh.node_row(node)]
    got = extract(source, PointLocation((tuple(xyz),), surfaces=("top",)))[0].tensors
    assert got == pytest.approx(source.node_stress([mesh.node_row(node)], "top")[:, 0, :], abs=1e-9)


def test_normals_come_from_anyfem_and_are_unit(source):
    assert np.allclose(np.linalg.norm(source.mesh.normals(), axis=1), 1.0)
    assert np.allclose(np.abs(source.mesh.normals()[:, 2]), 1.0)          # plate in xy


def test_line_and_face_helpers(model, source):
    from anyfatigue.anyfem_adapter import element_location, line_location

    project, face, edges, batch = model
    line = line_location(batch, project.edge(edges[0]), surfaces=("top",))
    assert len(line.node_ids) == 17                                         # 16 divisions
    samples = extract(source, line)
    assert samples[0].chainage == 0.0 and samples[-1].chainage == pytest.approx(SIDE)
    assert all(abs(abs(s.weld_direction[0]) - 1.0) < 1e-9 for s in samples)   # edge along x
    elements = element_location(batch, project.face(face))
    assert len(elements.element_ids) == 256


def test_simplified_fatigue_from_the_centre_stress(source):
    element = centre_element(source)
    loc = ElementLocation((element,), surfaces=("top",), weld_direction=(0.0, 1.0, 0.0), name="centre")
    settings = SimplifiedSettings(
        curve=sn.get_curve("D", "air"), thickness_mm=10.0, design_life_years=20.0, dff=3.0,
        n0=1.0e4, conditions=(LoadCondition("sea", "step1", 0.9, 8.0),))
    point = assess_simplified(source, [loc], settings).points[0]
    amplitude = abs(source.element_stress([source.mesh.element_row(element)], "top")[1, 0, 0])
    q = dm.weibull_scale(2.0 * amplitude, 0.9, 1.0e4)
    expected = dm.weibull_damage(
        sn.get_curve("D", "air"), scale_q=q, shape_h=0.9,
        n_cycles=20.0 * dm.SECONDS_PER_YEAR / 8.0, thickness_mm=10.0)
    assert point.damage == pytest.approx(expected, rel=1e-12)
    assert point.parts[0].values["stress_range_n0_mpa"] == pytest.approx(2.0 * amplitude)


def test_time_series_fatigue_of_a_regular_pressure_wave(source):
    """Two 4-step waves 0, +1, -1, +0.5 (closed): hand-countable rainflow."""

    element = centre_element(source)
    loc = ElementLocation((element,), surfaces=("top",), weld_direction=(0.0, 1.0, 0.0), name="centre")
    curve = sn.get_curve("E", "air")
    settings = TimeSeriesSettings(
        curve=curve, thickness_mm=10.0, design_life_years=20.0, dff=3.0,
        blocks=(LoadBlock("wave", tuple(source.case_labels)),))
    point = assess_time_series(source, [loc], settings).points[0]

    s1 = abs(source.element_stress([source.mesh.element_row(element)], "top")[1, 0, 0])
    # load factors f = 0, 1, -1, .5, 0, 1, -1, .5, 0 (signs follow the plate: stress ~ f)
    # rainflow: full cycles of range 2 s1 and 1.5 s1 ... counted by hand below.
    from anyfatigue.rainflow import count_rainflow_cycles

    series = np.array([f for f in FACTORS]) * source.element_stress(
        [source.mesh.element_row(element)], "top")[1, 0, 0]
    cycles = count_rainflow_cycles(series)
    duration = 0.5 * (len(FACTORS) - 1)
    expected = dm.miner_damage(cycles.ranges, cycles.counts, curve, thickness_mm=10.0,
                               scale=20.0 * dm.SECONDS_PER_YEAR / duration).total
    assert point.damage == pytest.approx(expected, rel=1e-12)
    assert point.parts[0].values["max_range_mpa"] == pytest.approx(2.0 * s1)
    # largest range is the +1 to -1 swing: 2 x the +1 stress
    assert cycles.max_range == pytest.approx(2.0 * s1)


def test_hot_spot_on_the_anyfem_plate_reads_inside_the_plate(source):
    toe = (0.5, 0.0, 0.0)           # on the supported edge y = 0, weld along x
    loc = PointLocation((toe,), surfaces=("top",), weld_direction=(1.0, 0.0, 0.0),
                        readout=ReadOut("hotspot_a", thickness_mm=10.0, side=1))
    sample = extract(source, loc)[0]
    assert sample.tensors.shape == (len(FACTORS), 6)
    bad = PointLocation(
        ((0.5, 0.0, 0.0),), surfaces=("top",), weld_direction=(1.0, 0.0, 0.0),
        readout=ReadOut("hotspot_a", thickness_mm=10.0, side=-1))      # reads outside the plate
    with pytest.raises(ExtractionError, match="not on the mesh"):
        extract(source, bad)


def test_missing_global_surface_stresses_fail_closed():
    from anyfatigue.anyfem_adapter import _element_tensors

    class Recovery:
        element_stresses = {1: {"von_mises": np.zeros(4), "local_xx_top": np.zeros(4)}}

    with pytest.raises(UnsupportedInputError, match="return_global"):
        _element_tensors(Recovery(), np.array([1]), "top", "case")
    with pytest.raises(ExtractionError, match="no recovered stress"):
        _element_tensors(Recovery(), np.array([7]), "top", "case")


def test_input_validation_for_from_solutions(model):
    from anyfatigue.anyfem_adapter import from_solutions
    from anyfatigue.errors import InputError

    _project, _face, _edges, batch = model
    with pytest.raises(InputError):
        from_solutions(batch, labels=["only-one"])
    with pytest.raises(InputError):
        from_solutions(object())
