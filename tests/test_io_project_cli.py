"""Exchange files, the analysis definition, reports and the command line."""

from __future__ import annotations

import json

import numpy as np
import pytest

from anyfatigue import cli, report, stress_io
from anyfatigue import sn_curves as sn
from anyfatigue.damage import SECONDS_PER_YEAR
from anyfatigue.errors import InputError
from anyfatigue.extract import ElementLocation, LineLocation, ReadOut
from anyfatigue.project import AnalysisDefinition
from anyfatigue.simplified import LoadCondition, SimplifiedSettings
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings
from helpers import plate_mesh, scaled, source_from_fields, tensor_field

FIELD = tensor_field(np.array([[20.0, 5, 0, 3, 0, 0], [100.0, 0, 0, 0, 0, 0], [30.0, 40, 0, 0, 0, 0]]))
LOC = ElementLocation((1, 2), surfaces=("top",), weld_direction=(0, 1, 0), name="plate")


def two_case_source():
    mesh = plate_mesh(4, 3)
    return source_from_fields(mesh, [FIELD, scaled(FIELD, -0.5)], labels=["up", "down"],
                              times=[0.0, 5.0])


def simple_settings():
    return SimplifiedSettings(
        curve=sn.get_curve("D", "seawater_cp"), thickness_mm=30.0, design_life_years=25.0,
        dff=3.0, n0=1.0e4, conditions=(LoadCondition("op", "up", 0.9, 8.0),))


# ---------------------------------------------------------------- npz ----
def test_stress_npz_round_trip_is_exact(tmp_path):
    source = two_case_source()
    path = stress_io.write_stress_npz(source, tmp_path / "s.npz")
    back = stress_io.read_stress_npz(path)
    assert back.case_labels == source.case_labels and back.case_times == source.case_times
    assert np.array_equal(back.mesh.node_xyz, source.mesh.node_xyz)
    assert np.array_equal(back.mesh.element_corner_ids, source.mesh.element_corner_ids)
    rows = np.arange(source.mesh.n_elements)
    for surface in ("top", "bottom"):
        assert np.array_equal(back.element_stress(rows, surface), source.element_stress(rows, surface))
    nodes = np.arange(source.mesh.n_nodes)
    assert np.array_equal(back.node_stress(nodes, "top"), source.node_stress(nodes, "top"))
    assert back.nodal_origin == source.nodal_origin


def test_npz_without_nodal_arrays_materialises_the_element_average(tmp_path):
    mesh = plate_mesh(2, 2)
    source = source_from_fields(mesh, [FIELD], with_nodal=False)
    back = stress_io.read_stress_npz(stress_io.write_stress_npz(source, tmp_path / "a.npz"))
    assert back.nodal_origin == "element average"
    nodes = np.arange(mesh.n_nodes)
    assert back.node_stress(nodes, "top") == pytest.approx(source.node_stress(nodes, "top"))


def test_npz_rejects_foreign_and_missing_files(tmp_path):
    np.savez(tmp_path / "other.npz", a=np.zeros(3))
    with pytest.raises(InputError):
        stress_io.read_stress_npz(tmp_path / "other.npz")
    with pytest.raises(InputError):
        stress_io.read_stress_npz(tmp_path / "missing.npz")


# ------------------------------------------------------------ history ----
def sine_csv(tmp_path, amplitude=40.0, n_cycles=10, dt=0.5, per_cycle=8, header="time,stress", sep=","):
    t = np.arange(n_cycles * per_cycle + 1) * dt
    s = amplitude * np.sin(2 * np.pi * (np.arange(len(t)) / per_cycle + 0.25))
    path = tmp_path / "history.csv"
    path.write_text(header.replace(",", sep) + "\n" + "\n".join(
        sep.join(f"{v:.12g}" for v in row) for row in zip(t, s)) + "\n", encoding="utf-8")
    return path, t, s


@pytest.mark.parametrize("sep", [",", ";", "\t"])
def test_history_csv_reproduces_the_miner_damage_of_a_regular_wave(tmp_path, sep):
    path, t, _s = sine_csv(tmp_path, sep=sep)
    source = stress_io.read_history_csv(path)
    n_cycles, amplitude = 10, 40.0
    curve = sn.get_curve("F", "air")
    settings = TimeSeriesSettings(
        curve=curve, thickness_mm=25.0, design_life_years=20.0, dff=3.0,
        blocks=(LoadBlock("wave", tuple(source.case_labels)),))
    loc = ElementLocation((stress_io.HISTORY_ELEMENT_ID,), surfaces=("top",),
                          weld_direction=(0, 1, 0), name="hot spot")
    result = AnalysisDefinition("time_series", settings, (loc,)).run(source)
    duration = t[-1] - t[0]
    expected = (20.0 * SECONDS_PER_YEAR / duration) * n_cycles / curve.cycles_to_failure(2 * amplitude)
    assert result.points[0].damage == pytest.approx(expected, rel=1e-9)


def test_history_csv_with_tensor_columns(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("time,xx,yy,xy\n0,10,0,0\n1,-10,5,2\n2,10,0,0\n", encoding="utf-8")
    source = stress_io.read_history_csv(path)
    assert source.element_stress([0], "top")[1, 0] == pytest.approx([-10.0, 5.0, 0.0, 2.0, 0.0, 0.0])


@pytest.mark.parametrize("content", [
    "t,stress\n0,1\n1,2\n",                    # no time column
    "time\n0\n1\n2\n",                          # neither stress nor components
    "time,stress,xx\n0,1,1\n1,2,2\n",           # both
    "time,stress\n0,1\n0,2\n1,3\n",             # not increasing
    "time,stress\n0,1\n1,abc\n2,3\n",           # not numeric
    "time,stress\n0,1\n",                       # too short
])
def test_history_csv_fails_closed(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(InputError):
        stress_io.read_history_csv(path)


# ---------------------------------------------------------- definition ----
def test_definition_round_trip_and_rerun(tmp_path):
    definition = AnalysisDefinition(
        "simplified", simple_settings(),
        # line along +x on the edge y = 0: n x t = z x x = +y points into the plate
        (LOC, LineLocation((100, 101, 102), readout=ReadOut("hotspot_a", thickness_mm=20.0, side=1))),
        source={"kind": "npz", "path": "s.npz"}, name="deck")
    path = definition.save(tmp_path / "a.json")
    again = AnalysisDefinition.load(path)
    assert again == definition
    source = two_case_source()
    first = definition.run(source)
    second = again.run(source)
    assert [p.damage for p in first.points] == [p.damage for p in second.points]


def test_definition_validation(tmp_path):
    with pytest.raises(InputError):
        AnalysisDefinition("simplified", simple_settings(), ())
    ts = TimeSeriesSettings(curve=sn.get_curve("D"), thickness_mm=20.0, design_life_years=20.0,
                            dff=2.0, blocks=(LoadBlock("a", ("x", "y")),))
    with pytest.raises(InputError):
        AnalysisDefinition("simplified", ts, (LOC,))
    with pytest.raises(InputError):
        AnalysisDefinition("harmonic", ts, (LOC,))
    (tmp_path / "x.json").write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(InputError):
        AnalysisDefinition.load(tmp_path / "x.json")
    (tmp_path / "y.json").write_text(json.dumps(
        {"schema": "anyfatigue.analysis", "version": 99}), encoding="utf-8")
    with pytest.raises(InputError, match="newer"):
        AnalysisDefinition.load(tmp_path / "y.json")
    with pytest.raises(InputError):
        AnalysisDefinition.load(tmp_path / "nope.json")


# ------------------------------------------------------------- reports ----
@pytest.fixture
def result():
    return AnalysisDefinition("simplified", simple_settings(), (LOC,)).run(two_case_source())


def test_points_csv_and_json(tmp_path, result):
    path = report.write_csv(result, tmp_path / "p.csv")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("label,kind,surface") and len(lines) == 1 + len(result.points)
    record = json.loads(report.write_json(result, tmp_path / "r.json").read_text(encoding="utf-8"))
    assert record["schema"] == "anyfatigue.result" and record["method"] == "simplified"
    assert record["critical"] == result.critical.label
    assert len(record["points"]) == len(result.points)
    assert record["points"][0]["parts"][0]["values"]["weibull_scale_q_mpa"] > 0.0


def test_markdown_report_states_standard_verdict_and_governing_point(result):
    text = report.report_markdown(result)
    assert "DNV-RP-C203" in text and "April 2010" in text
    assert "PASS" in text or "FAIL" in text
    assert result.critical.label in text
    assert "seawater_cp" in text


def test_time_series_report_names_the_rainflow_backend(tmp_path):
    path, _t, _s = sine_csv(tmp_path)
    source = stress_io.read_history_csv(path)
    settings = TimeSeriesSettings(curve=sn.get_curve("D"), thickness_mm=25.0, design_life_years=20.0,
                                  dff=2.0, blocks=(LoadBlock("w", tuple(source.case_labels)),))
    loc = ElementLocation((1,), surfaces=("top",), weld_direction=(0, 1, 0))
    result = AnalysisDefinition("time_series", settings, (loc,)).run(source)
    assert "anytimes" in report.report_markdown(result)
    full = report.result_to_dict(result, include_cycles=True)
    assert full["points"][0]["parts"][0]["cycles"]["count"]


# ----------------------------------------------------------------- cli ----
def test_cli_curves_lists_the_requested_environment(capsys):
    assert cli.main(["curves", "--environment", "seawater_cp"]) == 0
    out = capsys.readouterr().out
    assert "14.917" in out and "F3" in out


def test_cli_run_writes_reports_and_returns_the_verdict(tmp_path, capsys):
    source = two_case_source()
    npz = stress_io.write_stress_npz(source, tmp_path / "s.npz")
    analysis = AnalysisDefinition("simplified", simple_settings(), (LOC,)).save(tmp_path / "deck.json")
    code = cli.main(["run", str(analysis), "--stress", str(npz), "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code in (0, 2) and "governing" in out
    produced = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert produced == ["deck-points.csv", "deck-report.md", "deck-result.json"]


def test_cli_run_reports_errors_without_a_traceback(tmp_path, capsys):
    analysis = AnalysisDefinition("simplified", simple_settings(), (LOC,)).save(tmp_path / "d.json")
    assert cli.main(["run", str(analysis)]) == 1
    assert "exactly one" in capsys.readouterr().err
    assert cli.main(["run", str(tmp_path / "missing.json"), "--stress", "x.npz"]) == 1
