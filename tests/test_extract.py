"""Locating points, interpolating nodal stress and reading hot-spot stress.

The references are closed-form fields.  A linear field is reproduced exactly by
isoparametric interpolation on any convex quad or triangle, so interpolation
errors can only be implementation errors.  Hot-spot extrapolation has an exact
answer when the read-out points coincide with mesh nodes.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyfatigue import extract as ex
from anyfatigue.errors import ExtractionError, InputError
from anyfatigue.model import ArrayStressSource, SurfaceMesh
from helpers import plate_mesh, scaled, source_from_fields, tensor_field

# S = c0 + c1 x + c2 y for each of the six components (xx yy zz xy yz xz)
COEFF = np.array([
    [20.0, -10.0, 5.0, 8.0, 0.0, 0.0],      # constant part
    [300.0, 40.0, 0.0, -120.0, 0.0, 0.0],   # d/dx  (MPa per metre)
    [-60.0, 250.0, 0.0, 90.0, 0.0, 0.0],    # d/dy
])
FIELD = tensor_field(COEFF)


@pytest.fixture(params=["quads", "triangles", "skewed"])
def mesh(request):
    if request.param == "quads":
        return plate_mesh(6, 4)
    if request.param == "triangles":
        return plate_mesh(6, 4, triangles=True)
    return plate_mesh(6, 4, skew=0.25)


def test_locate_returns_weights_that_reproduce_the_point(mesh):
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(0.02, 0.58, 40), rng.uniform(0.02, 0.38, 40), np.zeros(40)])
    pts[:, 0] += 0.25 * pts[:, 1] if mesh.node_xyz[:, 0].max() > 0.6 + 1e-9 else 0.0
    rows, weights, distance = mesh.locate(pts)
    for p, e, w in zip(pts, rows, weights):
        corners = mesh.corner_xyz(int(e))
        assert w[: len(corners)] @ corners == pytest.approx(p, abs=1e-12)
        assert w.sum() == pytest.approx(1.0)
        assert np.all(w >= -1e-9)
    assert np.allclose(distance, 0.0)


def test_locate_refuses_a_point_off_the_mesh():
    mesh = plate_mesh()
    with pytest.raises(ExtractionError, match="no element"):
        mesh.locate([[2.0, 0.1, 0.0]])
    with pytest.raises(ExtractionError, match="no element"):
        mesh.locate([[0.1, 0.1, 0.5]])           # 0.5 m above the plate


def test_locate_tolerance_accepts_a_small_offset_from_the_plate():
    mesh = plate_mesh()
    _rows, _w, distance = mesh.locate([[0.1, 0.1, 0.004]], tolerance=0.01)
    assert distance[0] == pytest.approx(0.004)


def test_a_point_on_a_shared_edge_is_resolved_deterministically(mesh):
    p = mesh.node_xyz[mesh.node_row(100 + 7 + 3)] # an interior node
    rows, weights, _ = mesh.locate([p])
    again, _, _ = mesh.locate([p])
    assert rows[0] == again[0]
    assert weights[0].max() == pytest.approx(1.0)  # exactly at a corner


def test_interpolated_point_stress_is_exact_for_a_linear_field(mesh):
    source = source_from_fields(mesh, [FIELD, scaled(FIELD, 0.5)])
    pts = np.array([[0.123, 0.071, 0.0], [0.431, 0.302, 0.0], [0.05, 0.35, 0.0]])
    if mesh.node_xyz[:, 0].max() > 0.6 + 1e-9:       # skewed mesh: shift into it
        pts[:, 0] += 0.25 * pts[:, 1]
    samples = ex.extract(source, ex.PointLocation(tuple(map(tuple, pts)), surfaces=("top",)))
    assert len(samples) == 3
    for s, p in zip(samples, pts):
        assert s.tensors.shape == (2, 6)
        assert s.tensors[0] == pytest.approx(FIELD(p[None, :])[0], abs=1e-9)
        assert s.tensors[1] == pytest.approx(0.5 * FIELD(p[None, :])[0], abs=1e-9)
        assert s.normal == pytest.approx([0.0, 0.0, 1.0])


def test_bottom_surface_uses_the_bottom_field():
    mesh = plate_mesh()
    source = source_from_fields(mesh, [FIELD], bottom_factor=-0.8)
    p = ((0.2, 0.15, 0.0),)
    top = ex.extract(source, ex.PointLocation(p, surfaces=("top",)))[0].tensors
    bottom = ex.extract(source, ex.PointLocation(p, surfaces=("bottom",)))[0].tensors
    assert bottom == pytest.approx(-0.8 * top)


def test_hot_spot_a_is_exact_for_a_linear_field_off_the_nodes():
    mesh = plate_mesh(12, 8, 0.6, 0.4)
    source = source_from_fields(mesh, [FIELD])
    toe = (0.30, 0.20, 0.0)
    loc = ex.PointLocation(
        (toe,), surfaces=("top",),
        readout=ex.ReadOut("hotspot_a", thickness_mm=17.0, direction=(1.0, 0.0, 0.0)),
    )
    sample = ex.extract(source, loc)[0]
    # linear extrapolation of a linear field recovers the field at the toe
    assert sample.tensors[0] == pytest.approx(FIELD(np.array([toe]))[0], abs=1e-9)


def test_hot_spot_extrapolation_formulas_on_a_quadratic_field():
    # Mesh nodes every 10 mm along x so the read-out points (t = 20 mm: 10 mm and
    # 30 mm from the toe) coincide with nodes and interpolation is exact.
    mesh = plate_mesh(60, 4, 0.60, 0.04)
    def quad_field(p):
        p = np.atleast_2d(p)
        out = np.zeros((len(p), 6))
        out[:, 0] = (p[:, 0] - 0.20) ** 2 * 1.0e4        # sigma_xx = k (x - x_toe)^2
        return out
    source = source_from_fields(mesh, [quad_field])
    toe = (0.20, 0.02, 0.0)
    t_mm = 20.0
    a = ex.extract(source, ex.PointLocation((toe,), surfaces=("top",),
        readout=ex.ReadOut("hotspot_a", thickness_mm=t_mm, direction=(1, 0, 0))))[0]
    b = ex.extract(source, ex.PointLocation((toe,), surfaces=("top",),
        readout=ex.ReadOut("hotspot_b", thickness_mm=t_mm, direction=(1, 0, 0))))[0]
    t = t_mm / 1000.0
    s05, s15 = 1.0e4 * (0.5 * t) ** 2, 1.0e4 * (1.5 * t) ** 2
    assert a.tensors[0, 0] == pytest.approx(1.5 * s05 - 0.5 * s15, rel=1e-9)
    assert b.tensors[0, 0] == pytest.approx(s05, rel=1e-9)


def test_hot_spot_direction_follows_side_and_weld_direction():
    mesh = plate_mesh(12, 8)
    source = source_from_fields(mesh, [FIELD])
    toe = (0.30, 0.20, 0.0)
    for side, expected in ((1, -1.0), (-1, 1.0)):
        # weld along y: n x t = z x y = -x, so side=+1 reads towards -x
        loc = ex.PointLocation((toe,), surfaces=("top",), weld_direction=(0.0, 1.0, 0.0),
                               readout=ex.ReadOut("hotspot_b", thickness_mm=20.0, side=side))
        got = ex.extract(source, loc)[0].tensors[0]
        probe = np.array([[0.30 + expected * 0.010, 0.20, 0.0]])
        assert got == pytest.approx(FIELD(probe)[0], abs=1e-9)


def test_hot_spot_read_out_off_the_mesh_fails_closed_with_context():
    mesh = plate_mesh(12, 8)
    source = source_from_fields(mesh, [FIELD])
    loc = ex.PointLocation(((0.02, 0.20, 0.0),), surfaces=("top",),
        readout=ex.ReadOut("hotspot_a", thickness_mm=40.0, direction=(-1.0, 0.0, 0.0)))
    with pytest.raises(ExtractionError, match="read-out points"):
        ex.extract(source, loc)


def test_hot_spot_without_a_direction_fails_closed():
    mesh = plate_mesh()
    source = source_from_fields(mesh, [FIELD])
    loc = ex.PointLocation(((0.3, 0.2, 0.0),), surfaces=("top",),
                           readout=ex.ReadOut("hotspot_a", thickness_mm=10.0))
    with pytest.raises(InputError, match="weld direction"):
        ex.extract(source, loc)


def test_line_location_samples_every_node_with_the_tangent_as_weld_direction():
    mesh = plate_mesh(6, 4)
    source = source_from_fields(mesh, [FIELD])
    # nodes along x = 0.3 (i = 3): ids 100 + j * 7 + 3
    ids = tuple(100 + j * 7 + 3 for j in range(5))
    samples = ex.extract(source, ex.LineLocation(ids, surfaces=("top",)))
    assert [s.chainage for s in samples] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    for s in samples:
        assert abs(s.weld_direction[1]) == pytest.approx(1.0)      # along y
        assert s.tensors[0] == pytest.approx(FIELD(s.xyz[None, :])[0], abs=1e-9)
        assert s.normal == pytest.approx([0.0, 0.0, 1.0])


def test_line_hot_spot_reads_into_the_plate_on_the_chosen_side():
    mesh = plate_mesh(12, 8)
    source = source_from_fields(mesh, [FIELD])
    ids = tuple(100 + j * 13 + 6 for j in range(2, 7))      # x = 0.30, y = 0.10 .. 0.30
    for side in (1, -1):
        loc = ex.LineLocation(ids, surfaces=("top",),
                              readout=ex.ReadOut("hotspot_a", thickness_mm=20.0, side=side))
        samples = ex.extract(source, loc)
        # a linear field extrapolates exactly to the toe regardless of side
        for s in samples:
            assert s.tensors[0] == pytest.approx(FIELD(s.xyz[None, :])[0], abs=1e-9)


def test_line_hot_spot_side_selects_which_side_of_the_weld_is_read():
    # Weld along +y at x = 0.20 (node ids ascend in y, so the tangent is +y).
    # n x t = z x y = -x, hence side = +1 reads towards -x and side = -1 towards +x.
    # The field is cubic on purpose: linear extrapolation of a quadratic field from
    # +-(0.5 t, 1.5 t) gives x0^2 - 3 a^2, which is even in the offset and cannot tell
    # the two sides apart; the odd a^3 term of a cubic can.  With nodes every 10 mm
    # the read-out points (t = 20 mm: 10 and 30 mm from the toe) fall on nodes, so
    # the expected extrapolation is exact.
    k = 1.0e4
    mesh = plate_mesh(60, 4, 0.60, 0.04)

    def cubic(p):
        p = np.atleast_2d(p)
        out = np.zeros((len(p), 6))
        out[:, 0] = k * p[:, 0] ** 3
        return out

    source = source_from_fields(mesh, [cubic])
    ids = tuple(100 + j * 61 + 20 for j in range(1, 4))
    values = {}
    for side in (1, -1):
        loc = ex.LineLocation(ids, surfaces=("top",),
                              readout=ex.ReadOut("hotspot_a", thickness_mm=20.0, side=side))
        x1, x2 = 0.20 - side * 0.010, 0.20 - side * 0.030
        expected = 1.5 * k * x1 ** 3 - 0.5 * k * x2 ** 3
        for s in ex.extract(source, loc):
            assert s.tensors[0, 0] == pytest.approx(expected, rel=1e-9)
        values[side] = expected
    assert values[1] != pytest.approx(values[-1], rel=1e-3)      # the sides really differ
    on_weld = ex.extract(source, ex.LineLocation(ids, surfaces=("top",)))[0]
    assert on_weld.tensors[0, 0] == pytest.approx(k * 0.20 ** 3, rel=1e-9)


def test_point_hot_spot_side_selects_the_side_with_a_weld_direction():
    k = 1.0e4
    mesh = plate_mesh(60, 4, 0.60, 0.04)

    def cubic(p):
        p = np.atleast_2d(p)
        out = np.zeros((len(p), 6))
        out[:, 0] = k * p[:, 0] ** 3
        return out

    source = source_from_fields(mesh, [cubic])
    for side in (1, -1):
        loc = ex.PointLocation(((0.20, 0.02, 0.0),), surfaces=("top",), weld_direction=(0, 1, 0),
                               readout=ex.ReadOut("hotspot_a", thickness_mm=20.0, side=side))
        x1, x2 = 0.20 - side * 0.010, 0.20 - side * 0.030
        got = ex.extract(source, loc)[0].tensors[0, 0]
        assert got == pytest.approx(1.5 * k * x1 ** 3 - 0.5 * k * x2 ** 3, rel=1e-9)


def test_element_location_reads_element_values_at_the_centroid():
    mesh = plate_mesh(6, 4)
    source = source_from_fields(mesh, [FIELD, scaled(FIELD, 2.0)])
    samples = ex.extract(source, ex.ElementLocation((1, 8, 24), surfaces=("top", "bottom")))
    assert len(samples) == 6
    centroids = mesh.centroids()
    for s in samples:
        eid = int(s.label.split("element ")[1].split()[0])
        row = mesh.element_row(eid)
        assert s.xyz == pytest.approx(centroids[row])
        sign = 1.0 if s.surface == "top" else -1.0
        assert s.tensors[0] == pytest.approx(sign * FIELD(centroids[row][None, :])[0])


def test_element_location_projects_a_given_weld_direction_into_the_surface():
    mesh = plate_mesh(2, 2)
    source = source_from_fields(mesh, [FIELD])
    s = ex.extract(source, ex.ElementLocation((1,), surfaces=("top",),
                                              weld_direction=(1.0, 0.0, 3.0)))[0]
    assert s.weld_direction == pytest.approx([1.0, 0.0, 0.0])


def test_nodal_values_default_to_the_element_average():
    mesh = plate_mesh(2, 2)
    n_el = mesh.n_elements
    base = np.arange(n_el, dtype=float)[:, None] * np.ones((1, 6))
    source = ArrayStressSource(mesh, ["a"], base[None], base[None])
    assert source.nodal_origin == "element average"
    centre = mesh.node_row(100 + 4)                      # the shared middle node
    assert source.node_stress([centre], "top")[0, 0, 0] == pytest.approx(np.mean(np.arange(4.0)))
    corner = mesh.node_row(100)                          # a corner: one element only
    assert source.node_stress([corner], "top")[0, 0, 0] == pytest.approx(0.0)


def test_line_location_rejects_unknown_nodes_and_duplicates():
    mesh = plate_mesh()
    source = source_from_fields(mesh, [FIELD])
    with pytest.raises(ExtractionError, match="not in the mesh"):
        ex.extract(source, ex.LineLocation((100, 99999)))
    with pytest.raises(InputError):
        ex.LineLocation((100, 100))
    with pytest.raises(ExtractionError, match="element"):
        ex.extract(source, ex.ElementLocation((9999,)))


def test_location_validation():
    with pytest.raises(InputError):
        ex.ElementLocation(())
    with pytest.raises(InputError):
        ex.ElementLocation((1,), surfaces=("middle",))
    with pytest.raises(InputError):
        ex.ElementLocation((1,), surfaces=("top", "top"))
    with pytest.raises(InputError):
        ex.ReadOut("hotspot_a")                   # thickness missing
    with pytest.raises(InputError):
        ex.ReadOut("hotspot_a", thickness_mm=10.0, side=0)
    with pytest.raises(InputError):
        ex.PointLocation(((0.0, 0.0),))


def test_locations_round_trip_through_dicts():
    for loc in (
        ex.ElementLocation((3, 4), surfaces=("top",), weld_direction=(1, 0, 0), name="e"),
        ex.LineLocation((100, 101, 102), readout=ex.ReadOut("hotspot_a", thickness_mm=12.0, side=-1),
                        name="l"),
        ex.PointLocation(((0.1, 0.2, 0.0),), readout=ex.ReadOut(
            "hotspot_b", thickness_mm=8.0, direction=(0, 1, 0)), weld_direction=(1, 0, 0)),
    ):
        assert ex.location_from_dict(loc.to_dict()) == loc
    with pytest.raises(InputError):
        ex.location_from_dict({"type": "volume"})


def test_surface_mesh_validation():
    ids, xyz = np.array([1, 2, 3]), np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    with pytest.raises(InputError, match="unknown node"):
        SurfaceMesh(ids, xyz, np.array([1]), np.array([[1, 2, 9, -1]]))
    with pytest.raises(InputError, match="fourth corner"):
        SurfaceMesh(ids, xyz, np.array([1]), np.array([[1, -1, 3, 2]]))
    with pytest.raises(InputError, match="unique"):
        SurfaceMesh(np.array([1, 1, 3]), xyz, np.array([1]), np.array([[1, 1, 3, -1]]))
    with pytest.raises(InputError):
        SurfaceMesh(ids, xyz[:2], np.array([1]), np.array([[1, 2, 3, -1]]))


def test_normals_follow_the_corner_order_unless_supplied():
    up = plate_mesh(2, 2)
    down = plate_mesh(2, 2, flip=True)
    assert up.normals()[0] == pytest.approx([0, 0, 1])
    assert down.normals()[0] == pytest.approx([0, 0, -1])
    tri = plate_mesh(2, 2, triangles=True)
    assert np.allclose(tri.normals(), [0, 0, 1])
    supplied = SurfaceMesh(up.node_ids, up.node_xyz, up.element_ids, up.element_corner_ids,
                           element_normals=np.tile([0.0, 0.0, -2.0], (up.n_elements, 1)))
    assert supplied.normals()[0] == pytest.approx([0, 0, -1])


def test_node_normals_align_signs_across_incident_elements():
    mesh = plate_mesh(3, 3)
    rows = [mesh.node_row(100 + 5)]
    assert mesh.node_normals(rows)[0] == pytest.approx([0, 0, 1])
