"""Plate selection, sampling positions and physical shell stress separation."""

import numpy as np
import pytest

from anyfatigue.errors import ExtractionError, InputError
from anyfatigue.extract import LineLocation, PointLocation, ReadOut, extract, readout_preview
from anyfatigue.model import ArrayStressSource, StressSource, SurfaceMesh
from anyfatigue.plate import plate_stress_source
from helpers import plate_mesh, scaled, source_from_fields, tensor_field


class TracedSource(ArrayStressSource):
    def element_stress(self, rows, surface):
        self.reads.append(("element", tuple(self.mesh.element_ids[rows]), surface))
        return super().element_stress(rows, surface)

    def node_stress(self, rows, surface):
        self.reads.append(("node", tuple(self.mesh.node_ids[rows]), surface))
        return super().node_stress(rows, surface)


def junction():
    # Plate 41 lies in xy, plate 89 lies in yz. Node IDs 101 and 104
    # form the common weld line. The producer stores the web first.
    mesh = SurfaceMesh(
        [101, 102, 103, 104, 111, 112],
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
        [89, 41], [[101, 104, 111, 112], [101, 102, 103, 104]],
    )
    elements = np.zeros((2, 2, 6))
    elements[:, :, 0] = [[90, 10], [180, 20]]
    nodal = np.zeros((2, 6, 6))
    nodal[:, :, 0] = [[50, 10, 10, 50, 90, 90], [100, 20, 20, 100, 180, 180]]
    source = TracedSource(
        mesh, ["A", "B"], elements, -3 * elements,
        node_top=nodal, node_bottom=-3 * nodal, times=(0.0, 8.0),
        nodal_origin="producer nodal", provenance={"result": "two-sided"},
    )
    source.reads = []
    return source


@pytest.mark.parametrize("element_id,side,expected,normal,direction", [
    (41, -1, 10, (0, 0, 1), (1, 0, 0)),
    (89, 1, 90, (1, 0, 0), (0, 0, 1)),
])
def test_preview_and_extraction_keep_weld_junction_plate_and_side(
    element_id, side, expected, normal, direction
):
    source = junction()
    plate = plate_stress_source(source, [element_id])
    assert plate.case_labels == source.case_labels and plate.case_times == source.case_times
    assert plate.provenance["result"] == "two-sided"
    assert plate.provenance["plate_selection"]["shared_node_count"] == 2
    assert "selected-plate element average" in plate.nodal_origin
    assert "plate_selection" not in source.provenance
    assert list(plate.mesh.element_ids) == [element_id]
    assert plate.mesh.n_nodes == 4
    location = LineLocation((101, 104), readout=ReadOut("hotspot_a", 20, side))
    preview = readout_preview(plate, location)
    assert len(preview) == 2
    for station, toe in zip(preview, [(0, 0, 0), (0, 1, 0)]):
        assert station.xyz == toe and station.normal == normal and station.direction == direction
        assert np.array(station.probes) == pytest.approx(
            np.array(toe) + np.outer([0.010, 0.030], direction)
        )
    assert source.reads == []

    for result in extract(plate, location):
        factor = 1 if result.surface == "top" else -3
        assert result.tensors[:, 0] == pytest.approx(np.array([1, 2]) * factor * expected)
        assert not np.any(result.tensors[:, 1:])
    for kind, ids, _ in source.reads:
        if kind == "element":
            assert ids == (element_id,)
        else:
            assert not set(ids).intersection([101, 104])


@pytest.mark.parametrize("method", ["direct", "hotspot_a", "hotspot_b"])
def test_plate_keeps_analytic_interior_tensor_field_and_physical_directors(method):
    mesh = plate_mesh(4, 2, triangles=True, flip=True)
    intercept, x_slope, y_slope = np.arange(18).reshape(3, 6) + 1
    field = tensor_field([intercept, x_slope, y_slope])
    source = source_from_fields(mesh, [field, scaled(field, -2.5)], bottom_factor=-0.6)
    view = plate_stress_source(source, mesh.element_ids[::-1])
    toe = (0.25, 0.11, 0.0)
    location = PointLocation((toe,), readout=ReadOut(method, 40, direction=(1, 0, 0)))
    station, = readout_preview(view, location)
    assert station.normal == (0, 0, -1)
    offsets = {"direct": [0.0], "hotspot_a": [0.02, 0.06], "hotspot_b": [0.02]}[method]
    assert np.array(station.probes) == pytest.approx(np.array(toe) + np.outer(offsets, [1, 0, 0]))
    assert station.direction == (None if method == "direct" else (1, 0, 0))
    x = toe[0] + (0.02 if method == "hotspot_b" else 0.0)
    reference = intercept + x_slope * x + y_slope * toe[1]
    expected = np.stack([reference, -2.5 * reference])
    top, bottom = extract(view, location)
    assert top.tensors == pytest.approx(expected)
    assert bottom.tensors == pytest.approx(-0.6 * expected)


def test_probe_only_supported_by_neighbour_is_refused():
    mesh = plate_mesh(2, 1)
    source = source_from_fields(mesh, [tensor_field(np.ones((3, 6)))])
    location = PointLocation(
        ((0.28, 0.2, 0.0),), readout=ReadOut("hotspot_a", 20, direction=(1, 0, 0)),
    )
    assert len(readout_preview(source, location)[0].probes) == 2
    view = plate_stress_source(source, [1])
    for operation in (extract, readout_preview):
        with pytest.raises(ExtractionError, match="read-out points.*30 mm"):
            operation(view, location)
        with pytest.raises(ExtractionError, match="not in the mesh"):
            operation(view, LineLocation((102,)))


def test_preview_reports_actual_surface_positions_for_in_tolerance_points():
    mesh = plate_mesh()
    source = source_from_fields(mesh, [tensor_field(np.ones((3, 6)))])
    location = PointLocation(((0.21, 0.12, 0.003),), tolerance=0.004,
                             readout=ReadOut("hotspot_b", 10, direction=(1, 0, 0)))
    station, = readout_preview(source, location)
    assert station.xyz == pytest.approx((0.21, 0.12, 0.0))
    assert station.probes[0] == pytest.approx((0.215, 0.12, 0.0))


def test_element_only_source_fetches_only_the_selected_plate():
    full = junction()
    class ElementSource(StressSource):
        mesh = full.mesh
        case_labels = full.case_labels
        case_times = full.case_times
        provenance = {}

        def element_stress(self, rows, surface):
            return full.element_stress(rows, surface)

    view = plate_stress_source(ElementSource(), [41])
    assert "selected-plate element average" in view.nodal_origin
    assert full.reads == []
    tensor = view.node_stress([2, 0, 2], "bottom")
    assert tensor[:, :, 0] == pytest.approx(np.array([[-30] * 3, [-60] * 3]))
    assert full.reads == [("element", (41,), "bottom")]


@pytest.mark.parametrize("ids", [[], [41, 41], [41.5], [True], ["41"]])
def test_invalid_ids_are_rejected(ids):
    with pytest.raises(InputError):
        plate_stress_source(junction(), ids)


def test_unknown_shell_and_inconsistent_directors_fail_before_sampling():
    source = junction()
    with pytest.raises(ExtractionError, match="not in the mesh"):
        plate_stress_source(source, [1])
    with pytest.raises(ExtractionError, match="inconsistent surface directors"):
        plate_stress_source(source, [41, 89])
    mesh = plate_mesh(2, 1)
    flipped = SurfaceMesh(mesh.node_ids, mesh.node_xyz, mesh.element_ids,
                          mesh.element_corner_ids, [[0, 0, 1], [0, 0, -1]])
    source = source_from_fields(flipped, [tensor_field(np.ones((3, 6)))])
    with pytest.raises(ExtractionError, match="inconsistent surface directors"):
        plate_stress_source(source, mesh.element_ids)
