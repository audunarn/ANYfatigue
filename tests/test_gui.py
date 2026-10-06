"""GUI smoke and wiring tests, run offscreen.

The strongest check is equivalence: what the forms collect and what the window
shows must equal what the library computes for the same definition, so a wiring
slip (a swapped column, a unit, a stale combo) cannot hide behind a plausible
picture.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QApplication

from anyfatigue import examples, stress_io
from anyfatigue.errors import InputError
from anyfatigue.extract import ElementLocation
from anyfatigue.gui.plot import PlotSeries, PlotWidget, nice_ticks
from anyfatigue.gui.window import MainWindow, describe_location
from anyfatigue.project import AnalysisDefinition


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    w = MainWindow()
    w._quiet = True
    yield w
    w.wait()
    w.close()


@pytest.fixture(scope="module")
def synthetic():
    return examples.synthetic_plate(steps=60)


def load(window, synthetic):
    source, definition = synthetic
    window.load_source(source, definition)
    return source, definition


def test_window_starts_idle_and_refuses_to_run_without_a_source(window):
    assert window.verdict.property("state") == "idle"
    assert not window.export_report_action.isEnabled()
    window.run_analysis()
    assert "stress source" in window._last_error
    assert window.curve_label.text().startswith("D (air)") and "2024-10" in window.curve_label.text()


def test_forms_round_trip_the_definition(window, synthetic):
    _source, definition = load(window, synthetic)
    assert window.collect_definition() == definition
    assert window.method == "time_series"
    assert window.location_list.count() == 2
    assert "hotspot a" in window.location_list.item(0).text()


def test_time_series_run_matches_the_library(window, synthetic):
    source, definition = load(window, synthetic)
    window.run_analysis(wait=True)
    expected = definition.run(source)
    assert window.result is not None
    assert window.table.rowCount() == len(expected.points)
    assert window.verdict.property("state") == ("pass" if expected.passed else "fail")
    assert [p.damage for p in window.result.points] == pytest.approx([p.damage for p in expected.points])
    # the table is sorted by damage, so its first row is the governing point
    assert window.table.item(0, 0).text() == expected.critical.label
    assert f"{expected.critical.damage:.4g}" in window.table.item(0, 2).text()
    assert "Fatigue assessment" in window.report_view.toPlainText()
    assert window.history_plot.has_data() and window.spectrum_plot.has_data() and window.line_plot.has_data()
    assert window.export_report_action.isEnabled()


def test_changing_a_form_value_changes_the_result(window, synthetic):
    load(window, synthetic)
    window.run_analysis(wait=True)
    base = window.result.critical.damage
    window.scf_edit.setText("1.5")
    window.run_analysis(wait=True)
    # free-slope region scaling: bigger SCF must raise damage
    assert window.result.critical.damage > base * 1.5 ** 3 * 0.5
    window.scf_edit.setText("1.0")
    window.curve_combo.setCurrentText("F")
    window.run_analysis(wait=True)
    assert window.result.critical.damage > base            # F is weaker than D


def test_simplified_method_through_the_gui(window, synthetic):
    source, definition = load(window, synthetic)
    simplified = examples.simplified_for(source, definition)
    window.set_definition(simplified)
    assert window.method == "simplified"
    assert window.simplified_box.isVisibleTo(window) and not window.series_box.isVisibleTo(window)
    window.run_analysis(wait=True)
    expected = simplified.run(source)
    assert window.result.critical.damage == pytest.approx(expected.critical.damage)
    assert window.history_plot.series == []               # no stress history in this method
    assert window.spectrum_plot.has_data()
    assert window.collect_definition() == simplified


def test_dnv_effective_is_offered_only_for_the_simplified_method(window, synthetic):
    load(window, synthetic)
    model = window.measure_combo.model()
    index = window.measure_combo.findData("dnv_effective")
    assert not model.item(index).isEnabled()                # time series is the loaded method
    window.method_combo.setCurrentIndex(window.method_combo.findData("simplified"))
    assert model.item(index).isEnabled()
    window.measure_combo.setCurrentIndex(index)
    assert window.alpha_edit.isEnabled()
    window.method_combo.setCurrentIndex(window.method_combo.findData("time_series"))
    assert window.measure_combo.currentData() == "normal_to_weld"   # moved off the unsupported choice


def test_adding_and_replacing_and_removing_locations(window, synthetic):
    load(window, synthetic)
    before = window.location_list.count()
    window.kind_combo.setCurrentIndex(window.kind_combo.findData("element"))
    window.ids_edit.setText("1, 2, 5-7")
    window.weld_edit.setText("0, 1, 0")
    window.name_edit.setText("panel")
    # selecting the example's first location filled the form with its surfaces
    window.top_check.setChecked(True)
    window.bottom_check.setChecked(True)
    window.add_location()
    assert window.location_list.count() == before + 1
    added = window._locations[-1]
    assert isinstance(added, ElementLocation) and added.element_ids == (1, 2, 5, 6, 7)
    assert added.weld_direction == (0.0, 1.0, 0.0) and added.surfaces == ("top", "bottom")
    window.location_list.setCurrentRow(before)
    window.ids_edit.setText("9")
    window.replace_location()
    assert window._locations[-1].element_ids == (9,)
    window.remove_location()
    assert window.location_list.count() == before


def test_point_location_with_a_hot_spot_readout_from_the_form(window, synthetic):
    load(window, synthetic)
    window.kind_combo.setCurrentIndex(window.kind_combo.findData("point"))
    window.points_edit.setPlainText("0.6 0.2 0\n0.6 0.3 0")
    window.readout_combo.setCurrentIndex(window.readout_combo.findData("hotspot_a"))
    window.ro_thickness_edit.setText("")                    # blank: the plate thickness above
    window.thickness_edit.setText("20")
    window.weld_edit.setText("0 1 0")
    window.add_location()
    loc = window._locations[-1]
    assert len(loc.points) == 2 and loc.readout.method == "hotspot_a"
    assert loc.readout.thickness_mm == 20.0 and loc.readout.side == 1
    assert "hotspot a" in describe_location(loc)


@pytest.mark.parametrize("ids", ["", "abc", "9-3"])
def test_invalid_location_text_is_refused_with_a_message(window, synthetic, ids):
    load(window, synthetic)
    count = window.location_list.count()
    window.kind_combo.setCurrentIndex(window.kind_combo.findData("element"))
    window.ids_edit.setText(ids)
    window.add_location()
    assert window.location_list.count() == count and window._last_error


def test_a_bad_number_flags_the_field_and_blocks_the_run(window, synthetic):
    load(window, synthetic)
    window.thickness_edit.setText("thick")
    with pytest.raises(InputError):
        window.collect_definition()
    assert window.thickness_edit.property("invalid") is True
    window.run_analysis()
    assert "thickness" in window._last_error
    window.thickness_edit.setText("20")
    assert not window.thickness_edit.property("invalid")


def test_block_with_reversed_cases_is_rejected(window, synthetic):
    source, _ = load(window, synthetic)
    row = window.block_table
    row.cellWidget(0, 1).setCurrentIndex(row.cellWidget(0, 1).findData(source.case_labels[-1]))
    row.cellWidget(0, 2).setCurrentIndex(row.cellWidget(0, 2).findData(source.case_labels[0]))
    with pytest.raises(InputError, match="after the first"):
        window.collect_definition()


def test_analysis_file_round_trip_through_the_window(window, synthetic, tmp_path):
    _source, definition = load(window, synthetic)
    path = str(tmp_path / "plate.json")
    window.save_analysis(path)
    window.thickness_edit.setText("33")
    window.load_analysis(path)
    assert window.collect_definition() == definition
    assert AnalysisDefinition.load(path) == definition


def test_stress_file_and_history_routes(window, tmp_path):
    source, definition = examples.synthetic_plate(steps=30)
    npz = stress_io.write_stress_npz(source, tmp_path / "plate.npz")
    window.open_stress_file(str(npz))
    window.wait()
    assert window.source is not None and window.source.n_cases == 30
    assert "30 load cases" in window.source_label.text()
    assert window.export_stress_action.isEnabled()

    t = np.arange(41) * 0.5
    stress = 35.0 * np.sin(2 * np.pi * (np.arange(41) / 8 + 0.25))
    csv = tmp_path / "hotspot.csv"
    csv.write_text("time,stress\n" + "\n".join(f"{a},{b}" for a, b in zip(t, stress)), encoding="utf-8")
    window.open_history_csv(str(csv))
    window.wait()
    assert window.method == "time_series" and window.location_list.count() == 1
    window.run_analysis(wait=True)
    direct = window.collect_definition().run(stress_io.read_history_csv(csv))
    assert window.result.critical.damage == pytest.approx(direct.critical.damage)
    assert window.result.critical.parts[0].values["max_range_mpa"] == pytest.approx(70.0)


def test_reports_are_exported(window, synthetic, tmp_path):
    load(window, synthetic)
    window.run_analysis(wait=True)
    window.export_reports(str(tmp_path))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["synthetic_plate-points.csv", "synthetic_plate-report.md", "synthetic_plate-result.json"]


def test_a_failing_run_reports_instead_of_crashing(window, synthetic):
    source, definition = load(window, synthetic)
    window.block_table.item(0, 3).setText("1.5")           # exposure fraction above 1
    window.run_analysis(wait=True)
    assert "exposure" in window._last_error or "fraction" in window._last_error
    assert window.result is None


def test_plot_widget_renders_every_kind_without_error(app):
    plot = PlotWidget()
    plot.resize(520, 360)
    x = np.linspace(1.0, 100.0, 40)
    cases = [
        ([], {}),
        ([PlotSeries("line", x, x ** 2)], {}),
        ([PlotSeries("a", x, x), PlotSeries("b", x, 2 * x, kind="points")], {"x_log": True, "y_log": True}),
        ([PlotSeries("bars", x[:8], x[:8] ** 0.5, kind="bars")], {}),
        ([PlotSeries("gaps", x, np.where(x > 50, np.nan, x))], {}),
        ([PlotSeries("nonpositive", x - 200.0, x - 200.0)], {"x_log": True, "y_log": True}),
    ]
    for series, options in cases:
        plot.set_plot(series, x_label="x", y_label="y", title="t", **options)
        image = plot.grab().toImage()
        assert image.width() > 0
    assert not plot.has_data()                              # last case has no positive points on log axes


def test_nice_ticks_cover_the_range():
    ticks = nice_ticks(0.3, 97.0)
    assert ticks[0] >= 0.3 and ticks[-1] <= 97.0 and len(ticks) >= 4
    assert np.allclose(np.diff(ticks), np.diff(ticks)[0])


def test_run_gui_launcher_is_importable_and_smoke_starts(app):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parent.parent / "run_gui.py"
    spec = importlib.util.spec_from_file_location("anyfatigue_run_gui", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--example", "synthetic", "--smoke"]) == 0
