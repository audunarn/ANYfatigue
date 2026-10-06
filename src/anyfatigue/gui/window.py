"""The ANYfatigue main window."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QThreadPool, Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDockWidget, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QMainWindow, QMenu, QMessageBox,
    QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from .. import __version__
from .. import plotdata
from .. import sn_curves as sn
from ..errors import FatigueError, InputError
from ..extract import ElementLocation, LineLocation, PointLocation, ReadOut
from ..measures import MEASURES, StressMeasure
from ..model import TENSOR_ORDER, StressSource
from ..project import AnalysisDefinition
from ..rainflow import rainflow_backend_version
from ..report import report_markdown, write_csv, write_json, write_report
from ..results import AssessmentResult
from ..simplified import LoadCondition, SimplifiedSettings
from ..stress_io import read_history_csv, read_stress_npz, write_stress_npz, HISTORY_ELEMENT_ID
from ..timeseries import LoadBlock, TimeSeriesSettings
from . import parsing
from .plot import PlotSeries, PlotWidget
from .theme import apply_theme
from .worker import Task

__all__ = ["MainWindow", "main"]

METHODS = (("simplified", "Simplified (Weibull, one load case per condition)"),
           ("time_series", "Time series (rainflow counting)"))
READOUTS = (("direct", "Direct (stress at the point)"),
            ("hotspot_a", "Hot spot A: 0.5t / 1.5t extrapolation"),
            ("hotspot_b", "Hot spot B: stress at 0.5t"))
MEASURE_TEXT = {
    "normal_to_weld": "Normal to weld",
    "principal_abs_max": "Principal (largest magnitude)",
    "component": "Tensor component",
    "dnv_effective": "DNV effective hot-spot range (simplified only)",
}
LOCATION_KINDS = (("element", "Elements"), ("line", "Line (ordered nodes)"), ("point", "Points (x y z)"))


class NumItem(QTableWidgetItem):
    """A cell that shows text but sorts by a number."""

    def __init__(self, text: str, value: float) -> None:
        super().__init__(text)
        self._value = value
        self.setFlags(self.flags() & ~Qt.ItemIsEditable)

    def __lt__(self, other) -> bool:
        if isinstance(other, NumItem):
            return self._value < other._value
        return super().__lt__(other)


def describe_location(loc) -> str:
    surfaces = "+".join(loc.surfaces)
    if isinstance(loc, ElementLocation):
        return f"Elements '{loc.name}': {parsing.format_ids(loc.element_ids)} ({surfaces})"
    if isinstance(loc, LineLocation):
        n = len(loc.node_ids)
        shown = parsing.format_ids(loc.node_ids)
        text = f"Line '{loc.name}': nodes {shown if len(shown) < 40 else shown[:37] + '...'} ({n}; {surfaces})"
    else:
        text = f"Points '{loc.name}': {len(loc.points)} point(s) ({surfaces})"
    r = loc.readout
    if r.method != "direct":
        text += f", {r.method.replace('_', ' ')} t={r.thickness_mm:g} mm"
    return text


class MainWindow(QMainWindow):
    def __init__(self, source: Optional[StressSource] = None,
                 definition: Optional[AnalysisDefinition] = None) -> None:
        super().__init__()
        self.setWindowTitle(f"ANYfatigue {__version__}")
        self.resize(1480, 900)
        self.source: Optional[StressSource] = None
        self.result: Optional[AssessmentResult] = None
        self._locations: list = []
        self._tasks: set = set()
        self._building = False

        self._build_actions()
        self._build_inputs()
        self._build_results()
        self.statusBar().showMessage("Open a stress file, a stress history or an example to begin.")
        apply_theme(self)
        self._update_method_visibility()
        self._refresh_curve_label()
        if source is not None:
            self.load_source(source, definition)
        elif definition is not None:
            self.set_definition(definition)

    # ------------------------------------------------------------------
    # actions and menus
    # ------------------------------------------------------------------
    def _build_actions(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self._act(file_menu, "Open stress file (.npz)...", self.open_stress_file, "Ctrl+O")
        self._act(file_menu, "Open stress history (.csv)...", self.open_history_csv)
        self.anyfem_open_action = self._act(
            file_menu, "Open ANYfem result (.anyres.h5 or project)...", self.open_anyfem_result)
        examples = file_menu.addMenu("Examples")
        self._act(examples, "Synthetic plate (closed-form field, no FE)", lambda: self.load_example("synthetic"))
        self.anyfem_example_action = self._act(
            examples, "ANYfem plate (solves a small model, about a minute)", lambda: self.load_example("anyfem"))
        file_menu.addSeparator()
        self._act(file_menu, "Load analysis definition...", self.load_analysis)
        self._act(file_menu, "Save analysis definition...", self.save_analysis, "Ctrl+S")
        file_menu.addSeparator()
        self.export_stress_action = self._act(file_menu, "Export stress file (.npz)...", self.export_stress)
        self.export_report_action = self._act(file_menu, "Export reports...", self.export_reports)
        file_menu.addSeparator()
        self._act(file_menu, "Exit", self.close, "Ctrl+Q")
        run_menu = bar.addMenu("&Run")
        self.run_action = self._act(run_menu, "Run assessment", self.run_analysis, "Ctrl+R")
        help_menu = bar.addMenu("&Help")
        self._act(help_menu, "About ANYfatigue", self.about)
        try:
            import anyfem  # noqa: F401
        except ImportError:
            for action in (self.anyfem_example_action, self.anyfem_open_action):
                action.setEnabled(False)
                action.setToolTip("Install ANYfatigue[anyfem] to enable")
        self.export_stress_action.setEnabled(False)
        self.export_report_action.setEnabled(False)

    def _act(self, menu: QMenu, text: str, slot, shortcut: Optional[str] = None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False, s=slot: s())
        menu.addAction(action)
        return action

    # ------------------------------------------------------------------
    # input panel
    # ------------------------------------------------------------------
    def _build_inputs(self) -> None:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        # source -------------------------------------------------------
        box = QGroupBox("1. Stress source")
        form = QVBoxLayout(box)
        self.source_label = QLabel("No stress source loaded")
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("Hint")
        form.addWidget(self.source_label)
        row = QHBoxLayout()
        for text, slot in (("Stress file...", self.open_stress_file),
                           ("History CSV...", self.open_history_csv)):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, s=slot: s())
            row.addWidget(button)
        form.addLayout(row)
        layout.addWidget(box)

        # method and basis --------------------------------------------------
        box = QGroupBox("2. Method and fatigue basis")
        f = QFormLayout(box)
        self.method_combo = QComboBox()
        for key, text in METHODS:
            self.method_combo.addItem(text, key)
        self.method_combo.currentIndexChanged.connect(self._update_method_visibility)
        f.addRow("Method", self.method_combo)
        self.curve_combo = QComboBox()
        self.curve_combo.addItems(sn.curve_names())
        self.curve_combo.setCurrentText("D")
        self.env_combo = QComboBox()
        for key in sn.ENVIRONMENTS:
            self.env_combo.addItem(key.replace("_", " "), key)
        for combo in (self.curve_combo, self.env_combo):
            combo.currentIndexChanged.connect(self._refresh_curve_label)
        pair = QHBoxLayout()
        pair.addWidget(self.curve_combo)
        pair.addWidget(self.env_combo)
        f.addRow("S-N curve", pair)
        self.curve_label = QLabel()
        self.curve_label.setObjectName("Hint")
        self.curve_label.setWordWrap(True)
        f.addRow(self.curve_label)
        self.thickness_edit = self._edit("25", "Thickness through which the crack grows [mm]")
        self.scf_edit = self._edit("1.0", "Stress concentration factor applied to the stress range")
        self.life_edit = self._edit("20", "Design life [years]")
        self.dff_edit = self._edit("2", "Design fatigue factor; the criterion is damage x DFF <= 1")
        f.addRow("Thickness [mm]", self.thickness_edit)
        f.addRow("SCF", self.scf_edit)
        f.addRow("Design life [years]", self.life_edit)
        f.addRow("DFF", self.dff_edit)
        self.measure_combo = QComboBox()
        for key in MEASURES:
            self.measure_combo.addItem(MEASURE_TEXT[key], key)
        self.measure_combo.currentIndexChanged.connect(self._update_method_visibility)
        f.addRow("Stress measure", self.measure_combo)
        self.component_combo = QComboBox()
        self.component_combo.addItems(TENSOR_ORDER)
        f.addRow("Component", self.component_combo)
        self.alpha_edit = self._edit("1.0", "alpha of DNV-RP-C203 (4.3.1): 0.72 / 0.80 / 0.90 for C / C1 / C2 parallel to the weld, else 1")
        f.addRow("alpha", self.alpha_edit)
        layout.addWidget(box)

        # simplified ------------------------------------------------------------
        self.simplified_box = QGroupBox("3. Load conditions (simplified method)")
        v = QVBoxLayout(self.simplified_box)
        form = QFormLayout()
        self.n0_edit = self._edit("1e4", "dS0 is the largest stress range out of n0 cycles")
        form.addRow("n0 (cycles at dS0)", self.n0_edit)
        v.addLayout(form)
        hint = QLabel("Each condition takes its stress range from one load case: "
                      "range = factor x stress(case), or |stress(case) - stress(reference)|.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.condition_table = self._table(
            ["Name", "Load case", "Reference", "Weibull h", "Period [s]", "Fraction", "Factor"])
        v.addWidget(self.condition_table)
        v.addLayout(self._row_buttons(self.add_condition, self.remove_condition))
        layout.addWidget(self.simplified_box)

        # time series ---------------------------------------------------------------
        self.series_box = QGroupBox("3. Load blocks (time-series method)")
        v = QVBoxLayout(self.series_box)
        hint = QLabel("A block is an ordered run of load cases (first to last, in source order) "
                      "and the share of the design life it represents (life share, 0 to 1). "
                      "Leave the duration blank to take it from the case times.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.block_table = self._table(["Name", "First case", "Last case", "Life share", "Duration [s]"])
        v.addWidget(self.block_table)
        v.addLayout(self._row_buttons(self.add_block, self.remove_block))
        form = QFormLayout()
        self.mean_combo = QComboBox()
        self.mean_combo.addItem("None (welded details)", "none")
        self.mean_combo.addItem("Non-welded base material, DNV 2.5.1", "non_welded")
        form.addRow("Mean stress", self.mean_combo)
        v.addLayout(form)
        layout.addWidget(self.series_box)

        # locations --------------------------------------------------------------------
        box = QGroupBox("4. Stress locations")
        v = QVBoxLayout(box)
        self.location_list = QListWidget()
        self.location_list.setMinimumHeight(90)
        self.location_list.currentRowChanged.connect(self._location_selected)
        v.addWidget(self.location_list)
        f = QFormLayout()
        self.kind_combo = QComboBox()
        for key, text in LOCATION_KINDS:
            self.kind_combo.addItem(text, key)
        self.kind_combo.currentIndexChanged.connect(self._update_location_form)
        f.addRow("Type", self.kind_combo)
        self.ids_edit = self._edit("", "Ids such as 3, 5, 8-11; for a line, nodes in order along the weld")
        self.points_edit = QPlainTextEdit()
        self.points_edit.setPlaceholderText("0.30 0.20 0.0\n0.30 0.40 0.0")
        self.points_edit.setFixedHeight(60)
        self.ids_label = QLabel("Element ids")
        stack = QStackedWidget()
        stack.addWidget(self.ids_edit)
        stack.addWidget(self.points_edit)
        self._ids_stack = stack
        f.addRow(self.ids_label, stack)
        self.top_check, self.bottom_check = QCheckBox("top"), QCheckBox("bottom")
        self.top_check.setChecked(True)
        self.bottom_check.setChecked(True)
        pair = QHBoxLayout()
        pair.addWidget(self.top_check)
        pair.addWidget(self.bottom_check)
        pair.addStretch()
        f.addRow("Surfaces", pair)
        self.readout_combo = QComboBox()
        for key, text in READOUTS:
            self.readout_combo.addItem(text, key)
        self.readout_combo.currentIndexChanged.connect(self._update_location_form)
        f.addRow("Read-out", self.readout_combo)
        self.ro_thickness_edit = self._edit("", "Plate thickness t for the 0.5t / 1.5t points [mm]; blank = thickness above")
        self.side_combo = QComboBox()
        self.side_combo.addItem("+1  (n x weld direction)", 1)
        self.side_combo.addItem("-1  (opposite)", -1)
        self.ro_dir_edit = self._edit("", "Optional in-plane read-out direction x, y, z (away from the weld toe)")
        self.weld_edit = self._edit("", "Weld direction x, y, z. Lines default to the line tangent")
        self.name_edit = self._edit("", "A name for reports")
        f.addRow("Read-out t [mm]", self.ro_thickness_edit)
        f.addRow("Side", self.side_combo)
        f.addRow("Read-out direction", self.ro_dir_edit)
        f.addRow("Weld direction", self.weld_edit)
        f.addRow("Name", self.name_edit)
        v.addLayout(f)
        v.addLayout(self._row_buttons(self.add_location, self.remove_location, extra=("Replace", self.replace_location)))
        layout.addWidget(box)

        self.run_button = QPushButton("Run assessment")
        self.run_button.setProperty("primary", True)
        self.run_button.clicked.connect(lambda _c=False: self.run_analysis())
        layout.addWidget(self.run_button)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        dock = QDockWidget("Analysis", self)
        dock.setWidget(scroll)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        dock.setMinimumWidth(540)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [580], Qt.Horizontal)
        self._update_location_form()

    def _edit(self, text: str, tip: str = "") -> QLineEdit:
        e = QLineEdit(text)
        e.setToolTip(tip)
        e.textChanged.connect(lambda _t, w=e: self._clear_invalid(w))
        return e

    @staticmethod
    def _clear_invalid(widget) -> None:
        if widget.property("invalid"):
            widget.setProperty("invalid", False)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    @staticmethod
    def _flag_invalid(widget) -> None:
        widget.setProperty("invalid", True)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _table(self, headers) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        header = t.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setStretchLastSection(True)
        for column, width in enumerate((80, 76, 76, 64, 64, 64)):
            t.setColumnWidth(column, width)
        t.verticalHeader().setVisible(False)
        t.setMinimumHeight(110)
        t.setAlternatingRowColors(True)
        return t

    def _row_buttons(self, add, remove, extra=None) -> QHBoxLayout:
        row = QHBoxLayout()
        buttons = [("Add", add)] + ([extra] if extra else []) + [("Remove", remove)]
        for text, slot in buttons:
            b = QPushButton(text)
            b.clicked.connect(lambda _c=False, s=slot: s())
            row.addWidget(b)
        row.addStretch()
        return row

    # ------------------------------------------------------------------
    # results panel
    # ------------------------------------------------------------------
    def _build_results(self) -> None:
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(10, 10, 10, 10)
        self.verdict = QLabel("No assessment yet")
        self.verdict.setObjectName("Verdict")
        self.verdict.setProperty("state", "idle")
        self.verdict.setWordWrap(True)
        v.addWidget(self.verdict)
        self.notice = QLabel()
        self.notice.setObjectName("Notice")
        self.notice.setWordWrap(True)
        self.notice.setVisible(False)
        v.addWidget(self.notice)

        splitter = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Point", "Surface", "Damage", "Life [years]", "D x DFF", "OK"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._point_selected)
        splitter.addWidget(self.table)

        self.tabs = QTabWidget()
        self.history_plot = PlotWidget()
        self.spectrum_plot = PlotWidget()
        self.line_plot = PlotWidget()
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.tabs.addTab(self.history_plot, "Stress history")
        self.tabs.addTab(self.spectrum_plot, "Spectrum vs S-N")
        self.tabs.addTab(self.line_plot, "Damage along line")
        self.tabs.addTab(self.report_view, "Report")
        splitter.addWidget(self.tabs)
        splitter.setSizes([260, 460])
        v.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self._clear_plots()

    def _clear_plots(self) -> None:
        self.history_plot.clear("Run an assessment, then select a point.")
        self.spectrum_plot.clear("Run an assessment, then select a point.")
        self.line_plot.clear("Run an assessment with a line location.")

    # ------------------------------------------------------------------
    # form <-> definition
    # ------------------------------------------------------------------
    @property
    def method(self) -> str:
        return self.method_combo.currentData()

    def _update_method_visibility(self) -> None:
        simple = self.method == "simplified"
        self.simplified_box.setVisible(simple)
        self.series_box.setVisible(not simple)
        kind = self.measure_combo.currentData()
        self.component_combo.setEnabled(kind == "component")
        self.alpha_edit.setEnabled(kind == "dnv_effective")
        model = self.measure_combo.model()
        item = model.item(MEASURES.index("dnv_effective"))
        item.setEnabled(simple)
        if not simple and kind == "dnv_effective":
            self.measure_combo.setCurrentIndex(MEASURES.index("normal_to_weld"))

    def _refresh_curve_label(self) -> None:
        curve = sn.get_curve(self.curve_combo.currentText(), self.env_combo.currentData())
        note = "\n".join(sn.edition_warnings(curve)[:1])
        self.curve_label.setText(curve.describe() + f"\n{curve.source}" + (f"\nNote: {note}" if note else ""))

    def _update_location_form(self) -> None:
        kind = self.kind_combo.currentData()
        self._ids_stack.setCurrentIndex(1 if kind == "point" else 0)
        self.ids_label.setText({"element": "Element ids", "line": "Node ids (ordered)",
                                "point": "Points x y z"}[kind])
        hot = self.readout_combo.currentData() != "direct" and kind != "element"
        self.readout_combo.setEnabled(kind != "element")
        for w in (self.ro_thickness_edit, self.side_combo, self.ro_dir_edit):
            w.setEnabled(hot)

    def _case_labels(self) -> List[str]:
        return list(self.source.case_labels) if self.source is not None else []

    def _combo(self, items, current=None, allow_none=False) -> QComboBox:
        c = QComboBox()
        if allow_none:
            c.addItem("(none)", None)
        for item in items:
            c.addItem(str(item), str(item))
        if current is not None:
            if c.findData(current) < 0:       # e.g. a definition opened before any source
                c.addItem(str(current), str(current))
            c.setCurrentIndex(c.findData(current))
        return c

    def add_condition(self, values: Optional[dict] = None) -> None:
        v = {"name": f"condition {self.condition_table.rowCount() + 1}", "case": None,
             "reference_case": None, "weibull_h": 0.8, "period_s": 9.0, "fraction": 1.0,
             "range_factor": 2.0}
        v.update(values or {})
        t = self.condition_table
        row = t.rowCount()
        t.insertRow(row)
        labels = self._case_labels()
        t.setItem(row, 0, QTableWidgetItem(str(v["name"])))
        t.setCellWidget(row, 1, self._combo(labels, v["case"]))
        t.setCellWidget(row, 2, self._combo(labels, v["reference_case"], allow_none=True))
        for col, key in ((3, "weibull_h"), (4, "period_s"), (5, "fraction"), (6, "range_factor")):
            t.setItem(row, col, QTableWidgetItem(f"{v[key]:g}"))

    def remove_condition(self) -> None:
        row = self.condition_table.currentRow()
        if row >= 0:
            self.condition_table.removeRow(row)

    def add_block(self, values: Optional[dict] = None) -> None:
        labels = self._case_labels()
        v = {"name": f"block {self.block_table.rowCount() + 1}",
             "first": labels[0] if labels else None, "last": labels[-1] if labels else None,
             "exposure_fraction": 1.0, "duration_s": None}
        v.update(values or {})
        t = self.block_table
        row = t.rowCount()
        t.insertRow(row)
        t.setItem(row, 0, QTableWidgetItem(str(v["name"])))
        t.setCellWidget(row, 1, self._combo(labels, v["first"]))
        t.setCellWidget(row, 2, self._combo(labels, v["last"]))
        t.setItem(row, 3, QTableWidgetItem(f"{v['exposure_fraction']:g}"))
        t.setItem(row, 4, QTableWidgetItem("" if v["duration_s"] is None else f"{v['duration_s']:g}"))

    def remove_block(self) -> None:
        row = self.block_table.currentRow()
        if row >= 0:
            self.block_table.removeRow(row)

    # -- locations ---------------------------------------------------------
    def _location_from_form(self):
        kind = self.kind_combo.currentData()
        surfaces = tuple(s for s, c in (("top", self.top_check), ("bottom", self.bottom_check)) if c.isChecked())
        if not surfaces:
            raise InputError("choose at least one surface")
        name = self.name_edit.text().strip() or {"element": "elements", "line": "line", "point": "points"}[kind]
        weld = parsing.parse_vector(self.weld_edit.text(), "weld direction")
        readout = ReadOut()
        if kind != "element" and self.readout_combo.currentData() != "direct":
            t_text = self.ro_thickness_edit.text().strip() or self.thickness_edit.text()
            readout = ReadOut(
                self.readout_combo.currentData(),
                thickness_mm=parsing.parse_float(t_text, "read-out thickness", positive=True),
                side=int(self.side_combo.currentData()),
                direction=parsing.parse_vector(self.ro_dir_edit.text(), "read-out direction"),
            )
        if kind == "element":
            return ElementLocation(parsing.parse_ids(self.ids_edit.text(), "element ids"),
                                   surfaces=surfaces, weld_direction=weld, name=name)
        if kind == "line":
            return LineLocation(parsing.parse_ids(self.ids_edit.text(), "node ids"), surfaces=surfaces,
                                readout=readout, weld_direction=weld, name=name)
        return PointLocation(parsing.parse_points(self.points_edit.toPlainText()), surfaces=surfaces,
                             readout=readout, weld_direction=weld, name=name)

    def _location_to_form(self, loc) -> None:
        kind = "element" if isinstance(loc, ElementLocation) else "line" if isinstance(loc, LineLocation) else "point"
        self.kind_combo.setCurrentIndex(self.kind_combo.findData(kind))
        if kind == "point":
            self.points_edit.setPlainText("\n".join(" ".join(f"{v:g}" for v in p) for p in loc.points))
        else:
            ids = loc.element_ids if kind == "element" else loc.node_ids
            self.ids_edit.setText(parsing.format_ids(ids))
        self.top_check.setChecked("top" in loc.surfaces)
        self.bottom_check.setChecked("bottom" in loc.surfaces)
        self.name_edit.setText(loc.name)
        self.weld_edit.setText("" if loc.weld_direction is None else ", ".join(f"{v:g}" for v in loc.weld_direction))
        if kind != "element":
            r = loc.readout
            self.readout_combo.setCurrentIndex(self.readout_combo.findData(r.method))
            self.ro_thickness_edit.setText("" if r.thickness_mm is None else f"{r.thickness_mm:g}")
            self.side_combo.setCurrentIndex(self.side_combo.findData(r.side))
            self.ro_dir_edit.setText("" if r.direction is None else ", ".join(f"{v:g}" for v in r.direction))
        self._update_location_form()

    def add_location(self) -> None:
        try:
            self._locations.append(self._location_from_form())
        except FatigueError as error:
            self._error(str(error))
            return
        self._refresh_location_list(len(self._locations) - 1)

    def replace_location(self) -> None:
        row = self.location_list.currentRow()
        if row < 0:
            self._error("Select a location to replace.")
            return
        try:
            self._locations[row] = self._location_from_form()
        except FatigueError as error:
            self._error(str(error))
            return
        self._refresh_location_list(row)

    def remove_location(self) -> None:
        row = self.location_list.currentRow()
        if row >= 0:
            del self._locations[row]
            self._refresh_location_list(min(row, len(self._locations) - 1))

    def _refresh_location_list(self, select: int = -1) -> None:
        self.location_list.blockSignals(True)
        self.location_list.clear()
        for loc in self._locations:
            self.location_list.addItem(describe_location(loc))
        self.location_list.blockSignals(False)
        if select >= 0:
            self.location_list.setCurrentRow(select)

    def _location_selected(self, row: int) -> None:
        if 0 <= row < len(self._locations):
            self._location_to_form(self._locations[row])

    # -- definition ----------------------------------------------------------
    def _float(self, edit: QLineEdit, name: str, **kw) -> float:
        try:
            return parsing.parse_float(edit.text(), name, **kw)
        except InputError:
            self._flag_invalid(edit)
            raise

    def _cell_float(self, table, row, col, name, **kw) -> float:
        return parsing.parse_float(table.item(row, col).text(), f"{name} (row {row + 1})", **kw)

    def collect_definition(self) -> AnalysisDefinition:
        """Read every form into an :class:`AnalysisDefinition` (raises InputError)."""

        curve = sn.get_curve(self.curve_combo.currentText(), self.env_combo.currentData())
        common = dict(
            curve=curve,
            thickness_mm=self._float(self.thickness_edit, "thickness", positive=True),
            design_life_years=self._float(self.life_edit, "design life", positive=True),
            dff=self._float(self.dff_edit, "DFF", positive=True),
            scf=self._float(self.scf_edit, "SCF", positive=True),
        )
        kind = self.measure_combo.currentData()
        measure = StressMeasure(
            kind, self.component_combo.currentText() if kind == "component" else None,
            self._float(self.alpha_edit, "alpha", positive=True) if kind == "dnv_effective" else 1.0)
        if not self._locations:
            raise InputError("add at least one stress location")
        if self.method == "simplified":
            t = self.condition_table
            if t.rowCount() == 0:
                raise InputError("add at least one load condition")
            conditions = []
            for row in range(t.rowCount()):
                case = t.cellWidget(row, 1).currentData()
                if case is None:
                    raise InputError(f"load condition {row + 1}: no load case (open a stress source first)")
                conditions.append(LoadCondition(
                    name=t.item(row, 0).text().strip() or f"condition {row + 1}", case=case,
                    reference_case=t.cellWidget(row, 2).currentData(),
                    weibull_h=self._cell_float(t, row, 3, "Weibull h", positive=True),
                    period_s=self._cell_float(t, row, 4, "period", positive=True),
                    fraction=self._cell_float(t, row, 5, "fraction", minimum=0.0),
                    range_factor=self._cell_float(t, row, 6, "range factor", positive=True)))
            settings = SimplifiedSettings(
                n0=self._float(self.n0_edit, "n0", positive=True), conditions=tuple(conditions),
                measure=measure, **common)
        else:
            t = self.block_table
            if t.rowCount() == 0:
                raise InputError("add at least one load block")
            labels = self._case_labels()
            blocks = []
            for row in range(t.rowCount()):
                first, last = t.cellWidget(row, 1).currentData(), t.cellWidget(row, 2).currentData()
                if first is None or last is None:
                    raise InputError(f"block {row + 1}: no load cases (open a stress source first)")
                i, j = labels.index(first), labels.index(last)
                if j <= i:
                    raise InputError(f"block {row + 1}: the last case must come after the first")
                duration_text = t.item(row, 4).text().strip()
                blocks.append(LoadBlock(
                    name=t.item(row, 0).text().strip() or f"block {row + 1}", cases=tuple(labels[i:j + 1]),
                    exposure_fraction=self._cell_float(t, row, 3, "exposure fraction", minimum=0.0),
                    duration_s=None if not duration_text else
                    self._cell_float(t, row, 4, "duration", positive=True)))
            settings = TimeSeriesSettings(
                blocks=tuple(blocks), measure=measure, mean_stress=self.mean_combo.currentData(), **common)
        return AnalysisDefinition(self.method, settings, tuple(self._locations),
                                  source=dict(getattr(self, "_source_ref", {})), name=self._analysis_name())

    def _analysis_name(self) -> str:
        return getattr(self, "_name", "analysis")

    def set_definition(self, definition: AnalysisDefinition) -> None:
        """Fill every form from a definition."""

        self._building = True
        try:
            self._name = definition.name
            s = definition.settings
            self.method_combo.setCurrentIndex(self.method_combo.findData(definition.method))
            self.curve_combo.setCurrentText(s.curve.name)
            env = s.curve.environment if s.curve.environment in sn.ENVIRONMENTS else "air"
            self.env_combo.setCurrentIndex(self.env_combo.findData(env))
            self.thickness_edit.setText(f"{s.thickness_mm:g}")
            self.scf_edit.setText(f"{s.scf:g}")
            self.life_edit.setText(f"{s.design_life_years:g}")
            self.dff_edit.setText(f"{s.dff:g}")
            self.measure_combo.setCurrentIndex(self.measure_combo.findData(s.measure.kind))
            if s.measure.component:
                self.component_combo.setCurrentText(s.measure.component)
            self.alpha_edit.setText(f"{s.measure.alpha:g}")
            self._locations = list(definition.locations)
            self._refresh_location_list(0 if self._locations else -1)
            self.condition_table.setRowCount(0)
            self.block_table.setRowCount(0)
            if definition.method == "simplified":
                self.n0_edit.setText(f"{s.n0:g}")
                for c in s.conditions:
                    self.add_condition(c.to_dict())
            else:
                self.mean_combo.setCurrentIndex(self.mean_combo.findData(s.mean_stress))
                labels = self._case_labels()
                for b in s.blocks:
                    self.add_block({"name": b.name, "first": b.cases[0], "last": b.cases[-1],
                                    "exposure_fraction": b.exposure_fraction, "duration_s": b.duration_s})
                    if labels and (b.cases[0] not in labels or b.cases[-1] not in labels):
                        self._warn(f"Block '{b.name}' refers to load cases the loaded source does not have.")
            self._source_ref = dict(definition.source)
        finally:
            self._building = False
        self._update_method_visibility()
        self._refresh_curve_label()

    # ------------------------------------------------------------------
    # sources
    # ------------------------------------------------------------------
    def _start(self, fn, done, busy_text: str) -> Task:
        task = Task(fn, done)
        if not self._tasks:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        self._tasks.add(task)
        # Connect to bound slots of this QObject so Qt queues delivery onto the GUI thread.
        task.signals.finished.connect(self._on_task_finished)
        task.signals.failed.connect(self._on_task_failed)
        self.statusBar().showMessage(busy_text)
        self.run_button.setEnabled(False)
        QThreadPool.globalInstance().start(task)
        return task

    def _end_busy(self, task) -> None:
        self._tasks.discard(task)
        if not self._tasks:
            QApplication.restoreOverrideCursor()
            self.run_button.setEnabled(True)

    @Slot(object, object)
    def _on_task_finished(self, task, result) -> None:
        self._end_busy(task)
        try:
            task.done(result)
        except FatigueError as error:
            self._error(str(error))

    @Slot(object, str)
    def _on_task_failed(self, task, message: str) -> None:
        self._end_busy(task)
        self.statusBar().showMessage("Failed")
        self._error(message)

    def wait(self) -> None:
        """Block until background work has finished (used by tests and scripts)."""

        QThreadPool.globalInstance().waitForDone()
        QApplication.processEvents()
        QApplication.processEvents()

    def open_stress_file(self, path: Optional[str] = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open stress file", "", "ANYfatigue stress (*.npz)")
        if path:
            self._start(lambda: read_stress_npz(path),
                        lambda s: self.load_source(s, ref={"kind": "npz", "path": str(path)}),
                        f"Reading {Path(path).name}...")

    def open_anyfem_result(self, path: Optional[str] = None) -> None:
        """Open a saved ANYfem result: a ``.anyres.h5`` file, or a project (its newest usable result)."""

        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open ANYfem result", "", "ANYfem result or project (*.anyres.h5 *.anyfem)")
        if path:
            from ..anyfem_artifacts import read_result

            self._start(lambda: read_result(path),
                        lambda s: self.load_source(s, ref={"kind": "anyfem_result", "path": str(path)}),
                        f"Reading {Path(path).name}...")

    def open_history_csv(self, path: Optional[str] = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open stress history", "", "Stress history (*.csv *.txt)")
        if path:
            def done(source):
                self.load_source(source, ref={"kind": "history", "path": str(path)})
                if not self._locations:
                    self._locations = [ElementLocation((HISTORY_ELEMENT_ID,), surfaces=("top",),
                                                       weld_direction=(0.0, 1.0, 0.0), name="hot spot")]
                    self._refresh_location_list(0)
                    self.method_combo.setCurrentIndex(self.method_combo.findData("time_series"))
                    self.statusBar().showMessage(
                        "History loaded; the stress is taken as the component normal to a weld along y.")
            self._start(lambda: read_history_csv(path), done, f"Reading {Path(path).name}...")

    def load_example(self, kind: str) -> None:
        from .. import examples

        build = examples.synthetic_plate if kind == "synthetic" else examples.anyfem_plate
        text = "Building the synthetic plate..." if kind == "synthetic" else "Solving the ANYfem plate and recovering stresses (about a minute)..."
        self._start(build, lambda pair: self.load_source(pair[0], pair[1]), text)

    def load_source(self, source: StressSource, definition: Optional[AnalysisDefinition] = None,
                    ref: Optional[dict] = None) -> None:
        self.source = source
        self._source_ref = dict(ref or (definition.source if definition else {}))
        mesh = source.mesh
        kind = source.provenance.get("kind", "stress source")
        if source.provenance.get("file"):
            kind = f"{kind}: {source.provenance['file']}"
        times = [t for t in source.case_times if t is not None]
        span = f", {times[0]:g}-{times[-1]:g} s" if len(times) == source.n_cases and times else ""
        self.source_label.setText(
            f"{kind}\n{source.n_cases} load cases{span}; {mesh.n_elements} elements, {mesh.n_nodes} nodes\n"
            f"Nodal stress: {source.nodal_origin}. Stresses in MPa.")
        self.export_stress_action.setEnabled(True)
        self.result = None
        self._clear_plots()
        self.table.setRowCount(0)
        if definition is not None:
            self.set_definition(definition)
        else:
            # keep what the user set up, but re-point the case selectors at this source
            self._refresh_case_selectors()
            if self.condition_table.rowCount() == 0 and self.block_table.rowCount() == 0:
                labels = list(source.case_labels)
                self.add_condition({"case": labels[0]})
                self.add_block()
        self.statusBar().showMessage(f"Loaded {source.n_cases} load cases.")

    def _refresh_case_selectors(self) -> None:
        """Point the case selectors of existing rows at the loaded source's cases."""

        labels = self._case_labels()
        if not labels:
            return
        # (table, per column: allow "(none)", fallback label when the old one is gone)
        plan = (
            (self.condition_table, ((False, labels[0]), (True, None))),
            (self.block_table, ((False, labels[0]), (False, labels[-1]))),
        )
        for table, columns in plan:
            for row in range(table.rowCount()):
                for col, (none_ok, fallback) in zip((1, 2), columns):
                    old = table.cellWidget(row, col).currentData()
                    keep = old if old in labels else fallback
                    table.setCellWidget(row, col, self._combo(labels, keep, allow_none=none_ok))

    # ------------------------------------------------------------------
    # running and showing results
    # ------------------------------------------------------------------
    def run_analysis(self, wait: bool = False) -> None:
        if self.source is None:
            self._error("Open a stress source first.")
            return
        try:
            definition = self.collect_definition()
        except FatigueError as error:
            self._error(str(error))
            return
        source = self.source
        self._definition = definition
        self._start(lambda: definition.run(source), self._show_result,
                    f"Running the {definition.method.replace('_', ' ')} assessment...")
        if wait:
            self.wait()

    def _show_result(self, result: AssessmentResult) -> None:
        self.result = result
        critical = result.critical
        state = "pass" if result.passed else "fail"
        self.verdict.setText(
            f"{'PASS' if result.passed else 'FAIL'}  |  governing {critical.label}: damage {critical.damage:.4g}, "
            f"D x DFF = {critical.usage:.4g}, life {critical.life_text} years")
        self.verdict.setProperty("state", state)
        self.verdict.style().unpolish(self.verdict)
        self.verdict.style().polish(self.verdict)
        warnings = list(result.warnings)
        self.notice.setVisible(bool(warnings))
        self.notice.setText("\n".join(warnings))
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(result.points))
        for row, p in enumerate(result.points):
            cells = [QTableWidgetItem(p.label), QTableWidgetItem(p.surface),
                     NumItem(f"{p.damage:.4g}", p.damage), NumItem(p.life_text, p.life_years),
                     NumItem(f"{p.usage:.4g}", p.usage), QTableWidgetItem("yes" if p.passed else "no")]
            cells[0].setData(Qt.UserRole, row)
            for col, item in enumerate(cells):
                if col in (0, 1, 5):
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        self.table.sortItems(2, Qt.DescendingOrder)
        self.report_view.setPlainText(report_markdown(result))
        self.export_report_action.setEnabled(True)
        self._update_line_plot()
        self.table.selectRow(0)
        self._point_selected()
        self.statusBar().showMessage(f"Assessed {len(result.points)} points.")

    def _selected_point(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.result is None:
            return None
        item = self.table.item(rows[0].row(), 0)
        return self.result.points[int(item.data(Qt.UserRole))]

    def _point_selected(self) -> None:
        point = self._selected_point()
        if point is None or self.result is None:
            return
        result = self.result
        curve = sn.SNCurve.from_dict(result.settings["curve"])
        # stress history (time series only)
        series = [PlotSeries(part.name, part.series_times if part.series_times is not None
                             else np.arange(len(part.series)), part.series)
                  for part in point.parts if part.series is not None]
        self.history_plot.set_plot(
            series, x_label="time [s]" if series and point.parts[0].series_times is not None else "load case index",
            y_label="stress [MPa]", title=point.label,
            empty_text="The simplified method has no stress history: it uses one load case per condition.")
        # spectrum against the S-N curve
        curves = []
        for part in point.parts:
            spectrum = plotdata.design_life_spectrum(result, part)
            if spectrum is not None:
                n, s = spectrum
                curves.append(PlotSeries(f"spectrum: {part.name}", n, s, kind="points" if result.method == "time_series" else "line"))
        if curves:
            n_all = np.concatenate([c.x for c in curves])
            n_low = max(10.0 ** np.floor(np.log10(max(n_all.min() / 10.0, 1.0))), 1.0)
            n_high = max(10.0 ** np.ceil(np.log10(n_all.max() * 10.0)), 1.0e8)
            n, s = plotdata.sn_curve_between_cycles(curve, max(n_low, 1.0e2), n_high)
            curves.insert(0, PlotSeries(f"S-N {curve.name}", n, s, color="#111827", width=2.2))
        self.spectrum_plot.set_plot(
            curves, x_label="cycles over the design life", y_label="effective stress range [MPa]",
            title="Design-life spectrum against the S-N curve", x_log=True, y_log=True,
            empty_text="No stress cycles at this point.")

    def _update_line_plot(self) -> None:
        grouped = plotdata.damage_along_line(self.result) if self.result else {}
        self.line_plot.set_plot(
            [PlotSeries(name, c, d) for name, (c, d) in grouped.items()],
            x_label="distance along the line [m]", y_label="damage", title="Fatigue damage along the line",
            y_log=True, empty_text="No line locations in this assessment.")

    # ------------------------------------------------------------------
    # files and dialogs
    # ------------------------------------------------------------------
    def load_analysis(self, path: Optional[str] = None) -> None:
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Load analysis", "", "ANYfatigue analysis (*.json)")
        if not path:
            return
        try:
            self.set_definition(AnalysisDefinition.load(path))
        except FatigueError as error:
            self._error(str(error))
            return
        self.statusBar().showMessage(f"Loaded analysis {Path(path).name}.")

    def save_analysis(self, path: Optional[str] = None) -> None:
        try:
            definition = self.collect_definition()
        except FatigueError as error:
            self._error(str(error))
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save analysis", "analysis.json", "ANYfatigue analysis (*.json)")
        if path:
            definition.save(path)
            self.statusBar().showMessage(f"Saved {Path(path).name}.")

    def export_stress(self, path: Optional[str] = None) -> None:
        if self.source is None:
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Export stress file", "stress.npz", "ANYfatigue stress (*.npz)")
        if path:
            source = self.source
            self._start(lambda: write_stress_npz(source, path),
                        lambda _p: self.statusBar().showMessage(f"Exported {Path(path).name}."),
                        "Writing the stress file...")

    def export_reports(self, directory: Optional[str] = None) -> None:
        if self.result is None:
            return
        if not directory:
            directory = QFileDialog.getExistingDirectory(self, "Export reports to")
        if directory:
            out = Path(directory)
            stem = getattr(self, "_name", "analysis").replace(" ", "_")
            write_csv(self.result, out / f"{stem}-points.csv")
            write_json(self.result, out / f"{stem}-result.json")
            write_report(self.result, out / f"{stem}-report.md")
            self.statusBar().showMessage(f"Reports written to {out}.")

    def about(self) -> None:
        try:
            backend = rainflow_backend_version()
        except FatigueError as error:
            backend = f"not available ({error})"
        QMessageBox.about(
            self, "About ANYfatigue",
            f"<b>ANYfatigue {__version__}</b><br>Fatigue assessment to DNV rules on ANYfem stresses.<br><br>"
            f"S-N data: {sn.STANDARD_EDITION}<br>Rainflow counting: ANYtimeseries (anytimes {backend})<br><br>"
            "Early development: results have not been independently qualified for design use.")

    def _error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self._last_error = message
        if not getattr(self, "_quiet", False):
            QMessageBox.warning(self, "ANYfatigue", message)

    def _warn(self, message: str) -> None:
        self.notice.setText(message)
        self.notice.setVisible(True)

    def closeEvent(self, event) -> None:  # noqa: N802
        QThreadPool.globalInstance().waitForDone(5000)
        super().closeEvent(event)


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    smoke = "--smoke" in args
    if smoke:
        args.remove("--smoke")
    app = QApplication.instance() or QApplication([sys.argv[0]])
    window = MainWindow()
    window.show()
    i = 0
    while i < len(args):
        flag = args[i]
        if flag == "--stress" and i + 1 < len(args):
            window.open_stress_file(args[i + 1])
            i += 1
        elif flag == "--history" and i + 1 < len(args):
            window.open_history_csv(args[i + 1])
            i += 1
        elif flag == "--anyfem" and i + 1 < len(args):
            window.open_anyfem_result(args[i + 1])
            i += 1
        elif flag == "--example" and i + 1 < len(args):
            window.load_example(args[i + 1])
            i += 1
        i += 1
    if smoke:
        window.wait()
        window.close()
        return 0
    return app.exec()
