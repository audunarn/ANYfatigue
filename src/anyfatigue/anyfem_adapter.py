"""Stress sources from solved ANYfem results (in-process).

What is read from ANYfem, and nothing else:

* the solved shapes' ``stresses(return_global=True)`` recovery: per shell
  element, the top and bottom surface tensors in **global axes**
  (``global_xx_top`` ... ``global_xz_bot``), one value per integration point,
  in Pa;
* optionally ``stresses(patch_config=PatchRecoveryConfig())``: owner-continuous
  nodal values, used wherever the recovery qualified the node;
* the shell mesh (node coordinates, corner nodes) and each element's
  ``physical_director`` as its normal.

Conversion and reductions, all recorded in the source's ``provenance``:

* Pa to MPa (x 1e-6);
* element value = mean over the integration points (ANYfem's default ``mean``
  reduction);
* a node that patch recovery did not qualify takes the mean of its incident
  element values, and the node ids are listed in the provenance so the fallback
  is visible, never silent.

Only linear-static recoveries are handled (a ``LinearBatchSolution`` or
``LinearSolution``); a result without global surface stresses is refused.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .errors import ExtractionError, InputError, MissingDependencyError, UnsupportedInputError
from .extract import ElementLocation, LineLocation, ReadOut
from .model import SURFACES, TENSOR_ORDER, ArrayStressSource, SurfaceMesh

__all__ = [
    "element_location", "fill_unqualified_nodes", "from_solutions", "line_location",
    "surface_mesh_from_anyfem",
]

PA_TO_MPA = 1.0e-6
_SIDE = {"top": "top", "bottom": "bot"}


def _require_anyfem() -> None:
    try:
        import anyfem  # noqa: F401
        import anysolver  # noqa: F401
    except ImportError as error:
        raise MissingDependencyError(
            "reading ANYfem results needs ANYfem and ANYsolver: "
            "`pip install ANYfatigue[anyfem]`"
        ) from error


def _shapes(solutions) -> List[Any]:
    """Normalise a batch, a single shape or a sequence of shapes to a list."""

    batch = getattr(solutions, "shapes", None)
    if batch is not None and getattr(solutions, "case_names", None) is not None:
        return list(batch)
    if hasattr(solutions, "stresses"):
        return [solutions]
    try:
        items = list(solutions)
    except TypeError:
        raise InputError("expected a LinearBatchSolution, a LinearSolution or a list of them") from None
    if not items or not all(hasattr(s, "stresses") for s in items):
        raise InputError("every solution needs a stresses() recovery (linear static results)")
    return items


def from_solutions(
    solutions,
    *,
    labels: Optional[Sequence[str]] = None,
    times: Optional[Sequence[Optional[float]]] = None,
    project=None,
    patch: bool = True,
) -> ArrayStressSource:
    """Build a stress source from solved ANYfem linear load cases.

    ``solutions`` is a ``LinearBatchSolution`` (the natural result of a
    load-case series), one ``LinearSolution`` or a list of them.  ``labels``
    default to the batch's case names.  ``times`` default to the load-case
    times of ``project`` (set by ``add_load_case_series``) when given, else
    ``None``; a time-series analysis then needs ``duration_s`` on its blocks.
    """

    _require_anyfem()
    shapes = _shapes(solutions)
    if labels is None:
        names = getattr(solutions, "case_names", None)
        labels = list(names) if names is not None and len(names) == len(shapes) else [
            getattr(s, "label", f"case{i}") for i, s in enumerate(shapes)
        ]
    labels = [str(x) for x in labels]
    if len(labels) != len(shapes):
        raise InputError("one label per solved load case is required")
    if times is None:
        times = [_case_time(project, name) for name in labels]
    if len(times) != len(shapes):
        raise InputError("one time per solved load case is required")

    # One global-surface recovery per case; the first also supplies the element normals.
    recoveries = [shape.stresses(return_global=True) for shape in shapes]
    mesh, element_ids, node_ids = surface_mesh_from_anyfem(
        shapes[0].built.mesh, _directors(recoveries[0])
    )

    element = {surface: np.zeros((len(shapes), len(element_ids), 6)) for surface in SURFACES}
    nodal = {surface: None for surface in SURFACES}
    fallback_nodes: set = set()
    patch_status: List[Dict[str, int]] = []
    for k, recovery in enumerate(recoveries):
        for surface in SURFACES:
            element[surface][k] = _element_tensors(recovery, element_ids, surface, labels[k])
    if patch:
        from anysolver import PatchRecoveryConfig

        for surface in SURFACES:
            nodal[surface] = np.zeros((len(shapes), len(node_ids), 6))
        for k, shape in enumerate(shapes):
            recovered = shape.stresses(patch_config=PatchRecoveryConfig())
            bundle = getattr(recovered, "nodal_stresses", None)
            if not isinstance(bundle, dict) or "nodal" not in bundle:
                raise UnsupportedInputError(
                    f"load case {labels[k]!r} has no patch-recovered nodal stresses"
                )
            diagnostics = bundle.get("node_diagnostics", {})
            counts = {"qualified": 0, "other": 0}
            for row, node_id in enumerate(node_ids):
                values = bundle["nodal"].get(int(node_id))
                status = diagnostics.get(int(node_id), {}).get("status", "unclassified")
                if values is None or status != "qualified":
                    fallback_nodes.add(int(node_id))
                    counts["other"] += 1
                    for surface in SURFACES:
                        nodal[surface][k, row, :] = np.nan
                    continue
                counts["qualified"] += 1
                for surface in SURFACES:
                    nodal[surface][k, row, :] = [
                        values[f"global_{c}_{_SIDE[surface]}"] * PA_TO_MPA for c in TENSOR_ORDER
                    ]
            patch_status.append(counts)
        fallback_nodes |= fill_unqualified_nodes(mesh, labels, element, nodal)
        origin = "ANYfem patch recovery"
        if fallback_nodes:
            origin += f" (element average at {len(fallback_nodes)} unqualified nodes)"
    else:
        origin = None

    provenance = {
        "kind": "ANYfem linear static results",
        "producer": _versions(),
        "unit_conversion": "Pa -> MPa (x 1e-6)",
        "element_reduction": "mean over integration points",
        "surfaces": {"top": "ANYfem '*_top' (+physical_director side)", "bottom": "ANYfem '*_bot'"},
        "cases": labels,
        "nodal_fallback_node_ids": sorted(fallback_nodes),
        "patch_node_status_per_case": patch_status,
    }
    return ArrayStressSource(
        mesh, labels, element["top"], element["bottom"], times=times,
        node_top=nodal["top"], node_bottom=nodal["bottom"],
        nodal_origin=origin, provenance=provenance,
    )


def _case_time(project, name: str) -> Optional[float]:
    if project is None:
        return None
    case = getattr(project, "load_cases", {}).get(name)
    t = getattr(case, "time", None)
    return None if t is None else float(t)


def _directors(recovery) -> Dict[int, np.ndarray]:
    """``physical_director`` of every element in a recovery, as unit-length normals."""

    out = {}
    for eid, record in recovery.element_stresses.items():
        director = record.get("physical_director") if isinstance(record, dict) else None
        if director is not None:
            out[int(eid)] = np.asarray(director, dtype=float).reshape(3)
    return out


def surface_mesh_from_anyfem(mesh, normals: Optional[Mapping[int, np.ndarray]] = None):
    """Shell corner mesh from an ANYfem/ANYmesher ``Mesh``.

    ``normals`` maps element id to its normal (ANYfem's ``physical_director``).  If
    it covers every shell it is used; otherwise normals follow the corner order.
    Returns ``(surface_mesh, element_ids, node_ids)``, both id arrays sorted.
    """

    shells = sorted(mesh.shells)
    if not shells:
        raise UnsupportedInputError("the ANYfem model has no shell elements to assess")
    used: set = set()
    corners = np.full((len(shells), 4), -1, dtype=np.int64)
    for row, eid in enumerate(shells):
        ids = tuple(int(n) for n in mesh.corners_of(eid))
        if len(ids) not in (3, 4):
            raise UnsupportedInputError(f"element {eid} has {len(ids)} corners; only 3 or 4 are supported")
        corners[row, : len(ids)] = ids
        used.update(ids)
    node_ids = np.array(sorted(used), dtype=np.int64)
    xyz = np.array([mesh.nodes[int(n)] for n in node_ids], dtype=float)
    element_normals = None
    if normals is not None and all(int(e) in normals for e in shells):
        element_normals = np.array([normals[int(e)] for e in shells])
    surface_mesh = SurfaceMesh(
        node_ids, xyz, np.array(shells, dtype=np.int64), corners, element_normals=element_normals,
    )
    return surface_mesh, np.array(shells, dtype=np.int64), node_ids


def fill_unqualified_nodes(mesh: SurfaceMesh, labels, element, nodal) -> set:
    """Replace NaN nodal stresses by the mean of the incident element stresses.

    ``element`` and ``nodal`` map ``surface`` to arrays ``(n_cases, n, 6)``; NaN marks
    a node that patch recovery did not qualify.  Returns the ids of the nodes that
    were filled, so the caller can record the fallback.
    """

    average = ArrayStressSource(mesh, labels, element["top"], element["bottom"])
    rows = np.arange(mesh.n_nodes)
    filled = np.zeros(mesh.n_nodes, dtype=bool)
    for surface in SURFACES:
        bad = np.isnan(nodal[surface]).any(axis=2)               # (n_cases, n_nodes)
        filled |= bad.any(axis=0)
        fallback = average.node_stress(rows, surface)
        nodal[surface][bad] = fallback[bad]
    return {int(n) for n in mesh.node_ids[filled]}


def _element_tensors(recovery, element_ids, surface: str, label: str) -> np.ndarray:
    out = np.empty((len(element_ids), 6))
    side = _SIDE[surface]
    for row, eid in enumerate(element_ids):
        record = recovery.element_stresses.get(int(eid))
        if record is None:
            raise ExtractionError(f"load case {label!r}: no recovered stress for element {int(eid)}")
        for col, component in enumerate(TENSOR_ORDER):
            key = f"global_{component}_{side}"
            if key not in record:
                raise UnsupportedInputError(
                    f"load case {label!r}: the recovery has no {key!r}; recover stresses "
                    "with return_global=True (global surface stresses are required so "
                    "that the stress can be resolved against the weld)"
                )
            out[row, col] = float(np.mean(np.asarray(record[key], dtype=float))) * PA_TO_MPA
    return out


def _versions() -> Dict[str, str]:
    from importlib import metadata

    out = {}
    for name in ("ANYfem", "ANYsolver"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "source checkout"
    return out


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def line_location(
    solutions,
    edge,
    *,
    surfaces: Sequence[str] = SURFACES,
    readout: Optional[ReadOut] = None,
    weld_direction=None,
    name: Optional[str] = None,
) -> LineLocation:
    """A :class:`LineLocation` along an ANYfem geometry edge.

    ``edge`` is an ``EntityRef`` such as ``project.edge(edge_id)``.  Corner
    nodes only, in ANYfem's order along the edge.
    """

    _require_anyfem()
    shape = _shapes(solutions)[0]
    mesh = shape.built.mesh
    corner_nodes = {int(n) for eid in mesh.shells for n in mesh.corners_of(eid)}
    ids = tuple(int(n) for n in mesh.nodes_on(edge) if int(n) in corner_nodes)
    if not ids:
        raise ExtractionError(f"{edge} has no corner nodes in the mesh")
    return LineLocation(
        ids, surfaces=tuple(surfaces), readout=readout or ReadOut(),
        weld_direction=weld_direction, name=name or str(edge),
    )


def element_location(
    solutions,
    face,
    *,
    surfaces: Sequence[str] = SURFACES,
    weld_direction=None,
    name: Optional[str] = None,
) -> ElementLocation:
    """An :class:`ElementLocation` for every shell element of an ANYfem face."""

    _require_anyfem()
    shape = _shapes(solutions)[0]
    ids = tuple(int(e) for e in shape.built.mesh.elements_on(face) if int(e) in shape.built.mesh.shells)
    if not ids:
        raise ExtractionError(f"{face} has no shell elements in the mesh")
    return ElementLocation(ids, surfaces=tuple(surfaces), weld_direction=weld_direction,
                           name=name or str(face))
