"""ANYfem result artifacts (.anyres.h5) as a stress source.

One batch is solved once, written with ANYfem's own ``write_solution_artifact``
and read back.  The strongest check is equivalence with the live in-process
source built from the very same solve: every stress, normal, time and the
resulting fatigue damage must agree, so the file route cannot drift from the
in-process route.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.anyfem

from anyfatigue import sn_curves as sn
from anyfatigue.errors import ExtractionError, InputError, UnsupportedInputError
from anyfatigue.extract import ElementLocation, LineLocation, PointLocation, ReadOut
from anyfatigue.project import AnalysisDefinition
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings

ROOT = Path(__file__).resolve().parent.parent
SIDE, THICKNESS = 1.0, 0.010
FACTORS = [0.0, 1.0, -1.0, 0.5, 0.0, 1.0, -0.5]
TIMES = [0.5 * k for k in range(len(FACTORS))]


@pytest.fixture(scope="module")
def saved():
    """``(project_path, project, batch, work_dir)`` with mesh and result sidecars written."""

    from anyfem import Project, pinned, steel
    from anyfem.io.artifacts import ArtifactStore
    from anyfem.io.project_file import save_project
    from anyfem.io.result_artifact import write_solution_artifact
    from anyfem.solve.run import solve_linear_static_many
    from anysolver import PatchRecoveryConfig

    work = ROOT / f".pytest_tmp_{uuid.uuid4().hex}"
    work.mkdir()
    project = Project(name="deck")
    project.add_material(steel("S355", THICKNESS))
    project.add_plate_section("plate", thickness=THICKNESS, material="S355")
    geometry = project.geometry
    corners = geometry.add_points([(0, 0, 0), (SIDE, 0, 0), (SIDE, SIDE, 0), (0, SIDE, 0)])
    edges = geometry.add_polyline(corners, close=True)
    face = geometry.add_face(edges)
    project.assign_plate(face, "plate")
    for edge in edges:
        project.add_support(pinned(project.edge(edge)))
    names = []
    for k, factor in enumerate(FACTORS):
        names.append(f"c{k}")
        case = project.load_case(names[-1])
        case.add_pressure(project.face(face), 10_000.0 * factor)
        case.time = TIMES[k]
    mesh = project.generate_mesh(SIDE / 8)
    batch = solve_linear_static_many(project, mesh=mesh, load_cases=names)
    for shape in batch.shapes:
        shape.stresses()
        shape._requested_global_stress = shape.stresses(return_global=True)
        shape._requested_patch_stress = shape.stresses(patch_config=PatchRecoveryConfig())
    path = work / "deck.anyfem"
    save_project(project, path)
    store = ArtifactStore(path)
    store.write_mesh(mesh, mesh_id="mesh-1", document_id="doc", model_hash="m", mesh_hash="MESHHASH")
    write_solution_artifact(store, batch, job_id="job-1", document_id="doc", mesh_id="mesh-1",
                            model_hash="m", mesh_hash="MESHHASH", analysis_hash="a")
    try:
        yield path, project, batch, work
    finally:
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture(scope="module")
def live(saved):
    from anyfatigue.anyfem_adapter import from_solutions

    _path, project, batch, _work = saved
    return from_solutions(batch, project=project, patch=True)


@pytest.fixture(scope="module")
def artifact(saved):
    from anyfatigue.anyfem_artifacts import read_result

    return read_result(saved[0])


def test_list_results_describes_the_sidecar(saved):
    from anyfatigue.anyfem_artifacts import list_results

    found = list_results(saved[0])
    assert len(found) == 1
    info = found[0]
    assert info.job_id == "job-1" and info.frame_kind == "load_case" and info.n_frames == len(FACTORS)
    assert info.usable and info.has_global_stress and info.has_patch_stress
    assert info.path.name == "job-1.anyres.h5"


def test_labels_times_and_provenance(artifact):
    assert artifact.case_labels == tuple(f"c{k}" for k in range(len(FACTORS)))
    assert artifact.case_times == tuple(TIMES)                  # from the project's load cases
    p = artifact.provenance
    assert p["kind"] == "ANYfem result artifact" and p["job_id"] == "job-1"
    assert p["unit_conversion"].startswith("Pa -> MPa") and p["normals"] == "ANYfem physical_director"
    assert artifact.nodal_origin == "ANYfem patch recovery"


def test_mesh_and_normals_equal_the_live_source(artifact, live):
    a, b = artifact.mesh, live.mesh
    assert np.array_equal(a.node_ids, b.node_ids) and np.array_equal(a.element_ids, b.element_ids)
    assert np.allclose(a.node_xyz, b.node_xyz)
    assert np.array_equal(a.element_corner_ids, b.element_corner_ids)
    assert np.allclose(a.normals(), b.normals())


def test_element_and_nodal_stress_equal_the_live_source(artifact, live):
    rows = np.arange(live.mesh.n_elements)
    nodes = np.arange(live.mesh.n_nodes)
    for surface in ("top", "bottom"):
        assert artifact.element_stress(rows, surface) == pytest.approx(
            live.element_stress(rows, surface), rel=1e-12, abs=1e-12)
        assert artifact.node_stress(nodes, surface) == pytest.approx(
            live.node_stress(nodes, surface), rel=1e-12, abs=1e-12)


def test_partial_and_repeated_row_requests_are_ordered_like_the_request(artifact, live):
    rows = [5, 5, 2, 40, 2]
    assert artifact.element_stress(rows, "top") == pytest.approx(
        live.element_stress(rows, "top"), rel=1e-12, abs=1e-12)
    nodes = [30, 3, 30, 12]
    assert artifact.node_stress(nodes, "bottom") == pytest.approx(
        live.node_stress(nodes, "bottom"), rel=1e-12, abs=1e-12)


def test_every_route_into_the_result_finds_the_same_file(saved, artifact):
    from anyfatigue.anyfem_artifacts import read_result

    path, _project, _batch, work = saved
    by_file = read_result(work / "deck.anyfem-data" / "results" / "job-1.anyres.h5")
    by_folder = read_result(work / "deck.anyfem-data")
    by_id = read_result(path, job_id="job-1")
    rows = np.arange(artifact.mesh.n_elements)
    for source in (by_file, by_folder, by_id):
        assert source.case_labels == artifact.case_labels
        assert source.element_stress(rows, "top") == pytest.approx(artifact.element_stress(rows, "top"))
    with pytest.raises(InputError, match="no usable result"):
        read_result(path, job_id="nope")


def test_fatigue_damage_from_the_artifact_equals_the_live_route(artifact, live):
    ids = tuple(int(n) for n in live.mesh.node_ids if abs(live.mesh.node_xyz[live.mesh.node_row(n)][1]) < 1e-9)
    ids = tuple(sorted(ids, key=lambda n: live.mesh.node_xyz[live.mesh.node_row(n)][0]))
    locations = (
        LineLocation(ids, surfaces=("top", "bottom"), readout=ReadOut("hotspot_a", thickness_mm=10.0, side=1),
                     name="edge"),
        PointLocation(((0.5, 0.5, 0.0), (0.31, 0.62, 0.0)), surfaces=("top",), weld_direction=(0, 1, 0),
                      name="field"),
        ElementLocation((1, 9, 33), surfaces=("top",), weld_direction=(1, 0, 0), name="panel"),
    )
    settings = TimeSeriesSettings(
        curve=sn.get_curve("D", "air"), thickness_mm=10.0, design_life_years=20.0, dff=3.0,
        blocks=(LoadBlock("wave", tuple(live.case_labels)),))
    definition = AnalysisDefinition("time_series", settings, locations)
    a, b = definition.run(artifact), definition.run(live)
    assert [p.label for p in a.points] == [p.label for p in b.points]
    assert [p.damage for p in a.points] == pytest.approx([p.damage for p in b.points], rel=1e-10)
    assert a.critical.damage > 0.0


def test_without_patch_values_nodal_stress_is_the_element_average(saved, live):
    from anyfatigue.anyfem_adapter import from_solutions
    from anyfatigue.anyfem_artifacts import read_result

    path, project, batch, _work = saved
    plain = read_result(path, patch=False)
    expected = from_solutions(batch, project=project, patch=False)
    nodes = np.arange(plain.mesh.n_nodes)
    assert plain.nodal_origin == "element average"
    assert plain.node_stress(nodes, "top") == pytest.approx(expected.node_stress(nodes, "top"), rel=1e-12, abs=1e-12)


def test_an_unqualified_node_falls_back_to_the_element_average_and_is_reported(saved):
    from anyfatigue.anyfem_artifacts import read_result

    source = read_result(saved[0])
    row = 20
    patched = source.node_stress([row], "top")
    source._patch_index[row] = -1                                # as if recovery had not qualified it
    incident = source.mesh.incident_elements()[row]
    expected = source.element_stress(incident, "top").mean(axis=1)
    assert source.node_stress([row], "top")[:, 0, :] == pytest.approx(expected)
    assert not np.allclose(source.node_stress([row], "top"), patched, atol=1e-9)   # patch differs from the mean
    assert "1 unqualified" in source.nodal_origin


def test_times_can_be_given_and_are_checked(saved):
    from anyfatigue.anyfem_artifacts import read_result

    path = saved[0]
    custom = [2.0 * t for t in TIMES]
    assert read_result(path, times=custom).case_times == tuple(custom)
    with pytest.raises(InputError, match="one time per load case"):
        read_result(path, times=[0.0, 1.0])


def _copy_project(saved, tmp_path):
    path, _project, _batch, work = saved
    target = tmp_path / "copy"
    shutil.copytree(work, target)
    return target / "deck.anyfem"


def test_missing_mesh_sidecar_fails_closed(saved, tmp_path):
    from anyfatigue.anyfem_artifacts import read_result

    copy = _copy_project(saved, tmp_path)
    shutil.rmtree(copy.with_name("deck.anyfem-data") / "meshes")
    with pytest.raises(ExtractionError, match="mesh sidecar"):
        read_result(copy)


def test_unsupported_frame_kinds_are_refused(saved, tmp_path):
    import h5py
    from anyfatigue.anyfem_artifacts import list_results, read_result

    copy = _copy_project(saved, tmp_path)
    result = copy.with_name("deck.anyfem-data") / "results" / "job-1.anyres.h5"
    with h5py.File(result, "r+") as handle:
        handle["frames"].attrs["kind"] = "time"
    with pytest.raises(UnsupportedInputError, match="linear-static"):
        read_result(result)
    assert not list_results(copy)[0].usable
    with pytest.raises(InputError, match="no usable result"):
        read_result(copy)


def test_a_result_without_global_surface_stresses_is_refused(saved, tmp_path):
    from anyfem.io.artifacts import ArtifactStore
    from anyfem.io.result_artifact import write_solution_artifact
    from anyfatigue.anyfem_artifacts import list_results, read_result

    _path, _project, batch, _work = saved
    copy = _copy_project(saved, tmp_path)
    for shape in batch.shapes:
        shape._requested_global_stress_backup = shape._requested_global_stress
        shape._requested_patch_stress_backup = shape._requested_patch_stress
        shape._requested_global_stress = None
        shape._requested_patch_stress = None
    try:
        store = ArtifactStore(copy)
        write_solution_artifact(store, batch, job_id="job-2", document_id="doc", mesh_id="mesh-1",
                                model_hash="m", mesh_hash="MESHHASH", analysis_hash="a")
    finally:
        for shape in batch.shapes:                           # the module fixture is shared
            shape._requested_global_stress = shape._requested_global_stress_backup
            shape._requested_patch_stress = shape._requested_patch_stress_backup
    infos = {i.job_id: i for i in list_results(copy)}
    assert infos["job-1"].usable and not infos["job-2"].usable
    result = copy.with_name("deck.anyfem-data") / "results" / "job-2.anyres.h5"
    with pytest.raises(UnsupportedInputError, match="return_global"):
        read_result(result)
    assert read_result(copy).provenance["job_id"] == "job-1"      # the usable one is chosen


@pytest.mark.parametrize("bad", ["missing.anyres.h5", "notes.txt"])
def test_bad_paths_fail_closed(tmp_path, bad):
    from anyfatigue.anyfem_artifacts import read_result

    (tmp_path / "notes.txt").write_text("not a project", encoding="utf-8")
    with pytest.raises(InputError):
        read_result(tmp_path / bad)


def test_cli_runs_an_analysis_on_a_project(saved, artifact, tmp_path, capsys):
    from anyfatigue import cli

    path = saved[0]
    settings = TimeSeriesSettings(
        curve=sn.get_curve("E", "air"), thickness_mm=10.0, design_life_years=20.0, dff=3.0,
        blocks=(LoadBlock("wave", tuple(artifact.case_labels)),))
    loc = ElementLocation((1, 9), surfaces=("top",), weld_direction=(0, 1, 0), name="panel")
    analysis = AnalysisDefinition("time_series", settings, (loc,)).save(tmp_path / "deck.json")
    code = cli.main(["run", str(analysis), "--anyfem", str(path), "--out", str(tmp_path / "out")])
    assert code in (0, 2) and "governing" in capsys.readouterr().out
    assert (tmp_path / "out" / "deck-report.md").is_file()
    assert cli.main(["run", str(analysis), "--out", str(tmp_path / "o2")]) == 1      # no source given


@pytest.mark.qt
def test_gui_opens_an_anyfem_result(saved):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
        from anyfatigue.gui.window import MainWindow
    except ImportError as error:
        pytest.skip(f"Qt is not usable here: {error}")

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window._quiet = True
    try:
        window.open_anyfem_result(str(saved[0]))
        window.wait()
        assert window.source is not None and window.source.n_cases == len(FACTORS)
        assert "ANYfem result artifact" in window.source_label.text()
        window.kind_combo.setCurrentIndex(window.kind_combo.findData("element"))
        window.ids_edit.setText("1, 9")
        window.weld_edit.setText("0 1 0")
        window.add_location()
        window.run_analysis(wait=True)
        assert window.result is not None and len(window.result.points) == 4
    finally:
        window.wait()
        window.close()


def test_a_node_flagged_unqualified_in_the_file_uses_the_element_average(saved, tmp_path):
    """The production route to the fallback: the result file lists the node as unqualified."""

    import json

    import h5py
    from anyfatigue.anyfem_artifacts import read_result

    copy = _copy_project(saved, tmp_path)
    result = copy.with_name("deck.anyfem-data") / "results" / "job-1.anyres.h5"
    clean = read_result(result)
    row = 20
    node_id = int(clean.mesh.node_ids[row])
    with h5py.File(result, "r+") as handle:
        attrs = handle["fields/stress_patch_global_xx_top"].attrs
        provenance = json.loads(attrs["provenance"])
        provenance["unqualified_node_ids"] = [node_id]
        attrs["provenance"] = json.dumps(provenance)
    flagged = read_result(result)
    assert flagged.nodal_origin == "ANYfem patch recovery (element average at 1 unqualified nodes)"
    assert flagged.provenance["nodal_fallback_node_ids"] == [node_id]
    incident = flagged.mesh.incident_elements()[row]
    expected = flagged.element_stress(incident, "top").mean(axis=1)
    assert flagged.node_stress([row], "top")[:, 0, :] == pytest.approx(expected)
    assert not np.allclose(flagged.node_stress([row], "top"), clean.node_stress([row], "top"), atol=1e-9)
    other = row + 1                                                # neighbours keep their patch values
    assert flagged.node_stress([other], "top") == pytest.approx(clean.node_stress([other], "top"))
