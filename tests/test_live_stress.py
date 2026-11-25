"""Selection, ownership, evidence and recovery scope of live FE stress reads."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from anyfatigue.anyfem_adapter import from_solutions
from anyfatigue.errors import InputError
from anyfatigue.extract import ElementLocation, LineLocation, extract
from anyfatigue.model import StressSource, TENSOR_ORDER
from anyfatigue.plate import plate_stress_source

pytestmark = pytest.mark.anyfem


def record(level, director):
    values = {f"global_{component}_{side}": np.array([level + i, level + 2 + i]) * factor * 1e6
              for i, component in enumerate(TENSOR_ORDER) for side, factor in (("top", 1), ("bot", -2))}
    if director is not None:
        values["physical_director"] = director
    return values


class Shape:
    def __init__(self, mesh, label, level):
        self.built = SimpleNamespace(mesh=mesh)
        self.label = label
        self.level = level
        self.calls = []
        self.retained = {}
        self.normal = np.array((0., 1., -1.)) / np.sqrt(2)  # reversed physical side

    def retained_stress_recovery(self, kind):
        return self.retained.get(kind)

    def bundle(self, ids):
        return SimpleNamespace(element_stresses={eid: record(self.level + eid, self.normal) for eid in ids})

    def patch(self):
        field = {"nodal": {}, "node_diagnostics": {}}
        for nid in self.built.mesh.nodes:
            field["nodal"][nid] = {k: float(np.mean(v)) for k, v in record(777 + self.level, None).items()}
            field["node_diagnostics"][nid] = {"status": "unqualified" if nid == 2 else "qualified"}
        return SimpleNamespace(element_stresses={}, nodal_stresses=field)

    def stresses(self, **kwargs):
        if "patch_config" in kwargs:
            assert "recovery_config" not in kwargs
            self.calls.append("full patch")
            return self.patch()
        assert kwargs["return_global"]
        ids = kwargs["recovery_config"].element_ids
        self.calls.append(tuple(ids))
        return self.bundle(ids)


@pytest.fixture
def frames():
    from anymesher import Mesh
    mesh = Mesh(nodes={1: np.array((0., 0., 0.)), 2: np.array((1., 0., 0.)),
                       3: np.array((1., 1., 1.)), 4: np.array((0., 1., 1.)),
                       5: np.array((2., 0., 0.)), 6: np.array((2., 1., 1.))},
                quads={11: (1, 2, 3, 4), 21: (2, 5, 6, 3)})
    return [Shape(mesh, label, i * 100) for i, label in enumerate("ABC")]


def test_metadata_only_and_cold_preview_then_both_surfaces_reuse_selected_recovery(frames):
    source = from_solutions(frames, times=["0", "1", "2"], lazy=True)
    view = source.select_cases(["C", "A"])
    assert source.mesh.n_elements == 2 and view.case_times == (2, 0)
    assert all(not shape.calls for shape in frames)
    assert view.mesh.normals_for([1])[0] == pytest.approx(frames[2].normal)
    assert [s.calls for s in frames] == [[], [], [(21,)]]
    top, bottom = extract(view, ElementLocation((21,)))
    expected = np.array([[222 + i for i in range(6)], [22 + i for i in range(6)]])
    assert top.tensors == pytest.approx(expected)
    assert bottom.tensors == pytest.approx(-2 * expected)
    assert view.element_stress([1, 1], "bottom")[:, 1] == pytest.approx(-2 * expected)
    assert [s.calls for s in frames] == [[(21,)], [], [(21,)]]


def test_retained_signed_global_native_and_partial_fields(frames):
    first, second, third = frames
    first.retained["global"] = first.bundle([11])
    second.retained["native"] = second.bundle([11])
    third.retained["native"] = SimpleNamespace(element_stresses={11: {"physical_director": third.normal,
                                                                      "von_mises": 999}})
    source = from_solutions(frames, lazy=True)
    for shape in frames:
        selected = source.select_cases([shape.label])
        assert selected.mesh.normals_for([0])[0] == pytest.approx(shape.normal)
        selected.element_stress([0], "bottom")
        selected.element_stress([0], "top")
    assert [s.calls for s in frames] == [[], [], [(11,)]]
    source.select_cases(["A"]).element_stress([0, 1], "top")
    assert first.calls == [(21,)]


def test_owner_patch_qualified_and_incident_fallback_scope(frames):
    f = frames[1]
    f.retained["patch"] = f.patch()
    selected = from_solutions(frames, lazy=True).select_cases(["B"])
    rows = [selected.mesh.node_row(nid) for nid in (1, 2)]
    top = selected.node_stress(rows, "top")[0, :, 0]
    bottom = selected.node_stress(rows, "bottom")[0, :, 0]
    assert top == pytest.approx([878., 117.])
    assert bottom == pytest.approx(-2 * top)
    assert f.calls == [(11, 21)]
    assert selected.provenance["nodal_fallback_node_ids"] == [2]
    assert selected.provenance["patch_node_status_per_case"] == [{"qualified": 1, "other": 1}]
    plate = plate_stress_source(selected, [11])
    point, = extract(plate, LineLocation((2,), surfaces=("top",)))
    assert point.tensors[0, 0] == pytest.approx(112.)  # excludes adjacent shell 21
    assert frames[0].calls == frames[2].calls == []


def test_cold_owner_patch_is_full_only_for_asked_frame_and_interior_nodes(frames):
    view = from_solutions(frames, lazy=True).select_cases(["B"])
    for surface in ("top", "bottom"):
        view.node_stress([0], surface)
    assert [s.calls for s in frames] == [[], ["full patch"], []]
    element_only = from_solutions(frames, lazy=True, patch=False).select_cases(["A"])
    assert element_only.node_stress([0], "top")[0, 0, 0] == pytest.approx(12.)
    assert frames[0].calls == [(11,)]
    assert not element_only.provenance["nodal_fallback_node_ids"]


def test_corner_director_when_owner_global_has_none_and_native_can_override(frames):
    first, second, _ = frames
    reference = -first.normal
    for shape in (first, second):
        shape.normal = None
        shape.retained["global"] = shape.bundle([11])
    second.retained["native"] = SimpleNamespace(element_stresses={11: {"physical_director": -reference}})
    for shape, expected in ((first, reference), (second, -reference)):
        source = from_solutions(shape, lazy=True)
        assert source.mesh.normals_for([0])[0] == pytest.approx(expected)
        assert source.provenance["corner_order_director_element_ids"] == ([11] if shape is first else [])
        assert not shape.calls


def test_concurrent_repeated_request_and_cancellation_between_frames(frames):
    source = from_solutions(frames, lazy=True)
    selected = source.select_cases(["B"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = [pool.submit(selected.element_stress, [0], side) for side in ("top", "bottom")]
        top, bottom = [job.result() for job in work]
    assert bottom == pytest.approx(-2 * top)
    assert frames[1].calls == [(11,)]
    view = source.select_cases(["C", "A"], cancelled=lambda: bool(frames[2].calls))
    with pytest.raises(CancelledError):
        view.element_stress([1], "top")
    assert not frames[0].calls
    assert source.select_cases(["A"]).element_stress([1], "top")[0, 0, 0] == pytest.approx(22.)


def test_case_lookup_handles_old_mutable_labels_duplicate_and_shallow_copy():
    source = StressSource()
    source.case_labels = ("a", "b", "a")
    assert source.case_index("a") == 0
    view = copy(source)
    view.case_labels = ("b", "a")
    assert view.case_index("a") == 1
    view.case_labels = ["a", "b"]
    view.case_labels.reverse()
    assert view.case_index("a") == 1
    with pytest.raises(InputError):
        view.case_index("c")


def test_real_inclined_shell_selected_fields_match_full_owner_recovery():
    from anyfem import Project, steel
    from anyfem.post.results import LinearSolution
    from anyfem.solve.build import build_fe_model

    project = Project(name="inclined shell")
    project.add_material(steel("S355", 0.02))
    project.add_plate_section("plate", thickness=0.02, material="S355")
    geometry = project.geometry
    vertices = geometry.add_points([(0, 0, 0), (0, 1, .5), (1, 1, .5), (1, 0, 0)])
    face = geometry.add_face(geometry.add_polyline(vertices, close=True))
    project.assign_plate(face, "plate")
    mesh = project.generate_mesh(.4)
    built = build_fe_model(project, mesh, require_loads=False, require_supports=False)
    size = built.fe_model.mesh.dof_manager.total_dofs
    shape = LinearSolution(displacements=np.sin(np.arange(size)) * 1.e-5, built=built)
    reference = from_solutions(shape)
    source = from_solutions(shape, lazy=True)
    erows = [0, source.mesh.n_elements - 1]
    nrows = [0, source.mesh.n_nodes // 2]
    assert source.mesh.normals_for(erows) == pytest.approx(reference.mesh.normals()[erows])
    for side in ("top", "bottom"):
        assert source.element_stress(erows, side) == pytest.approx(reference.element_stress(erows, side))
        assert source.node_stress(nrows, side) == pytest.approx(reference.node_stress(nrows, side))
