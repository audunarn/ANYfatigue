"""Stress-source files and a direct history route.

``write_stress_npz`` / ``read_stress_npz``
    A self-contained exchange file for a whole stress source (mesh, load-case
    labels and times, element and nodal tensors, provenance).  It lets the GUI
    open stresses exported from ANYfem on a machine that has no ANYfem.  Only
    plain arrays and one JSON string are stored (``allow_pickle`` is never used).

``source_from_history``
    Wraps a stress history that already exists at one hot spot (from any
    finite-element tool or a measurement) as a one-element source, so the same
    rainflow pipeline, reports and GUI apply.  Stress is uniform over a unit
    triangle in the xy plane (normal ``+z``); the weld direction is carried on
    the location, so give the history in components resolved in those axes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .errors import InputError
from .model import TENSOR_ORDER, ArrayStressSource, StressSource, SurfaceMesh

__all__ = [
    "HISTORY_ELEMENT_ID",
    "materialise",
    "read_history_csv",
    "read_stress_npz",
    "source_from_history",
    "write_stress_npz",
]

SCHEMA = "anyfatigue.stress"
SCHEMA_VERSION = 1
HISTORY_ELEMENT_ID = 1


def materialise(source: StressSource) -> ArrayStressSource:
    """Copy any source into memory (all elements and nodes, both surfaces)."""

    mesh = source.mesh
    e_rows = np.arange(mesh.n_elements)
    n_rows = np.arange(mesh.n_nodes)
    top_e = source.element_stress(e_rows, "top")
    bot_e = source.element_stress(e_rows, "bottom")
    top_n = source.node_stress(n_rows, "top")
    bot_n = source.node_stress(n_rows, "bottom")
    return ArrayStressSource(
        mesh, source.case_labels, top_e, bot_e, times=source.case_times,
        node_top=top_n, node_bottom=bot_n, nodal_origin=source.nodal_origin,
        provenance=dict(source.provenance),
    )


def write_stress_npz(source: StressSource, path) -> Path:
    data = materialise(source)
    mesh = data.mesh
    target = Path(path)
    meta = {
        "schema": SCHEMA, "version": SCHEMA_VERSION, "unit": "MPa",
        "tensor_order": list(TENSOR_ORDER), "nodal_origin": data.nodal_origin,
        "provenance": _json_safe(data.provenance),
    }
    normals = mesh.element_normals if mesh.element_normals is not None else np.zeros((0, 3))
    np.savez_compressed(
        target,
        meta=np.array(json.dumps(meta, sort_keys=True)),
        node_ids=mesh.node_ids, node_xyz=mesh.node_xyz,
        element_ids=mesh.element_ids, element_corner_ids=mesh.element_corner_ids,
        element_normals=normals,
        labels=np.array(data.case_labels, dtype=str),
        times=np.array([np.nan if t is None else t for t in data.case_times]),
        element_top=data._el["top"], element_bottom=data._el["bottom"],
        node_top=data._nd["top"], node_bottom=data._nd["bottom"],
    )
    return target


def read_stress_npz(path) -> ArrayStressSource:
    target = Path(path)
    try:
        with np.load(target, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            if meta.get("schema") != SCHEMA:
                raise InputError(f"{target.name} is not an ANYfatigue stress file")
            if int(meta.get("version", 0)) > SCHEMA_VERSION:
                raise InputError(f"{target.name} was written by a newer ANYfatigue")
            normals = archive["element_normals"]
            mesh = SurfaceMesh(
                archive["node_ids"], archive["node_xyz"], archive["element_ids"],
                archive["element_corner_ids"],
                element_normals=normals if normals.size else None,
            )
            times = [None if np.isnan(t) else float(t) for t in archive["times"]]
            return ArrayStressSource(
                mesh, [str(x) for x in archive["labels"]],
                archive["element_top"], archive["element_bottom"], times=times,
                node_top=archive["node_top"], node_bottom=archive["node_bottom"],
                nodal_origin=meta.get("nodal_origin"),
                provenance=dict(meta.get("provenance", {}), file=target.name),
            )
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, InputError):
            raise
        raise InputError(f"cannot read stress file {target}: {error}") from None


def _json_safe(value):
    return json.loads(json.dumps(value, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


# ----------------------------------------------------------------------
# direct history
# ----------------------------------------------------------------------
def source_from_history(
    times: Sequence[float],
    tensors: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    bottom: Optional[np.ndarray] = None,
    provenance: Optional[dict] = None,
) -> ArrayStressSource:
    """A one-element source from a stress history ``(n, 6)`` in MPa.

    ``bottom`` defaults to the same history as the top surface (a membrane
    state); pass the mirrored history for bending.  Times must be strictly
    increasing.
    """

    t = np.asarray(times, dtype=float)
    s = np.asarray(tensors, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise InputError("a stress history needs at least two samples")
    if s.shape != (t.size, 6):
        raise InputError(f"stress history must have shape ({t.size}, 6)")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(s))):
        raise InputError("stress history contains non-finite values")
    if np.any(np.diff(t) <= 0.0):
        raise InputError("history times must be strictly increasing")
    b = s if bottom is None else np.asarray(bottom, dtype=float)
    if b.shape != s.shape:
        raise InputError("bottom history must match the top history")
    mesh = SurfaceMesh(
        np.array([1, 2, 3]), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([HISTORY_ELEMENT_ID]), np.array([[1, 2, 3, -1]]),
    )
    labels = list(labels) if labels is not None else [f"t{k}" for k in range(t.size)]
    nodal = np.repeat(s[:, None, :], 3, axis=1)
    nodal_b = np.repeat(b[:, None, :], 3, axis=1)
    return ArrayStressSource(
        mesh, labels, s[:, None, :], b[:, None, :], times=t,
        node_top=nodal, node_bottom=nodal_b, nodal_origin="uniform history",
        provenance={"kind": "direct stress history", **(provenance or {})},
    )


def read_history_csv(path, *, scalar_component: str = "xx") -> ArrayStressSource:
    """Read ``time`` plus either one ``stress`` column or tensor components.

    A header row is required.  Columns: ``time`` (s) and ``stress`` (MPa,
    stored as component ``scalar_component``), or any of ``xx yy zz xy yz xz``
    (MPa; missing ones are zero).  Delimiters ``,`` ``;`` and tab are detected.
    """

    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError as error:
        raise InputError(f"cannot read {target}: {error}") from None
    try:
        dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t")
    except (csv.Error, IndexError):
        dialect = csv.excel
    rows = list(csv.reader(text.splitlines(), dialect))
    if len(rows) < 3:
        raise InputError("the history file needs a header and at least two samples")
    header = [h.strip().lower() for h in rows[0]]
    if "time" not in header:
        raise InputError("the history file needs a 'time' column")
    if scalar_component not in TENSOR_ORDER:
        raise InputError(f"scalar_component must be one of {TENSOR_ORDER}")
    has_scalar = "stress" in header
    components = [c for c in TENSOR_ORDER if c in header]
    if has_scalar == bool(components):
        raise InputError(
            "give either a 'stress' column or tensor component columns "
            f"({', '.join(TENSOR_ORDER)}), not both and not neither"
        )
    try:
        data = np.array([[float(x) for x in row] for row in rows[1:] if any(c.strip() for c in row)])
    except ValueError as error:
        raise InputError(f"non-numeric value in {target.name}: {error}") from None
    if data.shape[1] != len(header):
        raise InputError("every row must have as many values as the header has columns")
    times = data[:, header.index("time")]
    tensors = np.zeros((len(times), 6))
    if has_scalar:
        tensors[:, TENSOR_ORDER.index(scalar_component)] = data[:, header.index("stress")]
    else:
        for c in components:
            tensors[:, TENSOR_ORDER.index(c)] = data[:, header.index(c)]
    return source_from_history(times, tensors, provenance={"file": target.name})
