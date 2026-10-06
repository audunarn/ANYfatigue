"""Neutral stress-source contract.

ANYfatigue does not depend on how a stress field was produced.  A *stress
source* is a surface mesh plus, for each load case or time step, the stress
tensor on the top and bottom surface of every shell element (and optionally at
nodes), in **MPa and global axes**.  ANYfem adapters (``anyfatigue.anyfem``)
build one; so do the tests, from closed-form fields, and ``stress_io`` from
files.

Conventions
-----------
* Length in metres, stress in MPa (the unit DNV S-N curves use).  Unit
  conversion happens once, in the adapter, and is recorded in the source's
  ``provenance``.
* Tensor components are ordered ``xx, yy, zz, xy, yz, xz`` (ANYfem's order).
* ``surface`` is ``"top"`` or ``"bottom"``; the adapter defines which side is
  which and records it.  Both are evaluated by default because fatigue cracks
  start where the surface stress range is largest.
* Nodal values are *continuous* fields (patch recovery or node averaging) and
  are what interpolation uses.  Element values are piecewise constant and are
  only ever read per element.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import ExtractionError, InputError

__all__ = [
    "SURFACES",
    "TENSOR_ORDER",
    "ArrayStressSource",
    "StressSource",
    "SurfaceMesh",
]

TENSOR_ORDER = ("xx", "yy", "zz", "xy", "yz", "xz")
SURFACES = ("top", "bottom")


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norms > 0.0, vectors / np.where(norms > 0.0, norms, 1.0), 0.0)


@dataclass(frozen=True, eq=False)
class SurfaceMesh:
    """Triangle and quadrilateral shell mesh, corner nodes only.

    Mid-side nodes of quadratic elements are ignored: stress interpolation in
    this package is linear between corner nodes.  Corner ids are listed in
    cyclic order; a triangle uses ``-1`` for its fourth corner.
    """

    node_ids: np.ndarray
    node_xyz: np.ndarray
    element_ids: np.ndarray
    element_corner_ids: np.ndarray
    element_normals: Optional[np.ndarray] = None
    corner_rows: np.ndarray = field(init=False, repr=False)
    _cache: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        ids = np.asarray(self.node_ids, dtype=np.int64)
        xyz = np.asarray(self.node_xyz, dtype=float)
        eids = np.asarray(self.element_ids, dtype=np.int64)
        corners = np.asarray(self.element_corner_ids, dtype=np.int64)
        if ids.ndim != 1 or xyz.shape != (ids.size, 3):
            raise InputError("node_xyz must have shape (n_nodes, 3)")
        if not np.all(np.isfinite(xyz)):
            raise InputError("node coordinates must be finite")
        if np.unique(ids).size != ids.size:
            raise InputError("node ids must be unique")
        if corners.ndim != 2 or corners.shape[1] != 4 or corners.shape[0] != eids.size:
            raise InputError("element_corner_ids must have shape (n_elements, 4)")
        if np.unique(eids).size != eids.size:
            raise InputError("element ids must be unique")
        if np.any(corners[:, :3] < 0):
            raise InputError("only the fourth corner may be absent (triangle)")
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        flat = corners.ravel()
        absent = flat < 0
        if ids.size:
            pos = np.minimum(np.searchsorted(sorted_ids, flat), ids.size - 1)
            found = sorted_ids[pos] == flat
            mapped = order[pos]
        else:
            found = np.zeros(flat.shape, dtype=bool)
            mapped = np.zeros(flat.shape, dtype=np.int64)
        bad = ~found & ~absent
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            raise InputError(
                f"element {int(eids[first // 4])} references unknown node {int(flat[first])}"
            )
        rows = np.where(absent, -1, mapped).reshape(corners.shape).astype(np.int64)
        element_order = np.argsort(eids, kind="stable")
        normals = None
        if self.element_normals is not None:
            normals = _unit(np.asarray(self.element_normals, dtype=float))
            if normals.shape != (eids.size, 3) or not np.all(np.linalg.norm(normals, axis=1) > 0):
                raise InputError("element_normals must be non-zero with shape (n_elements, 3)")
        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "node_xyz", xyz)
        object.__setattr__(self, "element_ids", eids)
        object.__setattr__(self, "element_corner_ids", corners)
        object.__setattr__(self, "corner_rows", rows)
        object.__setattr__(self, "element_normals", normals)
        object.__setattr__(
            self,
            "_cache",
            {
                "node_sorted": (sorted_ids, order),
                "element_sorted": (eids[element_order], element_order),
            },
        )

    # ------------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def n_elements(self) -> int:
        return int(self.element_ids.size)

    def _lookup(self, key: str, value: int, what: str) -> int:
        ids, order = self._cache[key]
        pos = int(np.searchsorted(ids, int(value)))
        if pos >= ids.size or ids[pos] != int(value):
            raise ExtractionError(f"{what} {int(value)} is not in the mesh")
        return int(order[pos])

    def node_row(self, node_id: int) -> int:
        return self._lookup("node_sorted", node_id, "node")

    def element_row(self, element_id: int) -> int:
        return self._lookup("element_sorted", element_id, "element")

    def n_corners(self, element_row: int) -> int:
        return 3 if self.corner_rows[element_row, 3] < 0 else 4

    def corner_xyz(self, element_row: int) -> np.ndarray:
        n = self.n_corners(element_row)
        return self.node_xyz[self.corner_rows[element_row, :n]]

    # ------------------------------------------------------------------
    def geometric_normals(self) -> np.ndarray:
        """Unit normals from corner order (right-hand rule), cached."""

        cached = self._cache.get("geometric_normals")
        if cached is None:
            rows = self.corner_rows
            p0 = self.node_xyz[rows[:, 0]]
            p1 = self.node_xyz[rows[:, 1]]
            p2 = self.node_xyz[rows[:, 2]]
            quad = rows[:, 3] >= 0
            p3 = np.where(quad[:, None], self.node_xyz[np.where(quad, rows[:, 3], 0)], p2)
            # Diagonal cross product is exact for planar quads and a good
            # average for warped ones; for a triangle it is twice the area vector.
            cached = _unit(np.cross(p2 - p0, p3 - p1))
            tri = ~quad
            if tri.any():
                cached[tri] = _unit(np.cross(p1[tri] - p0[tri], p2[tri] - p0[tri]))
            self._cache["geometric_normals"] = cached
        return cached

    def normals(self) -> np.ndarray:
        """Element normals: those supplied by the producer, else geometric."""

        return self.element_normals if self.element_normals is not None else self.geometric_normals()

    def centroids(self) -> np.ndarray:
        cached = self._cache.get("centroids")
        if cached is None:
            rows = self.corner_rows
            xyz = self.node_xyz[np.where(rows >= 0, rows, 0)]
            mask = (rows >= 0)[..., None]
            cached = (xyz * mask).sum(axis=1) / mask.sum(axis=1)
            self._cache["centroids"] = cached
        return cached

    def _corner_positions(self) -> np.ndarray:
        """(E, 4, 3) corner positions; a triangle's fourth corner repeats its third."""

        rows = self.corner_rows
        fourth = np.where(rows[:, 3] >= 0, rows[:, 3], rows[:, 2])
        return self.node_xyz[np.column_stack([rows[:, :3], fourth])]

    def characteristic_length(self) -> float:
        """Median element edge length, the scale for search tolerances."""

        cached = self._cache.get("length")
        if cached is None:
            corners = self._corner_positions()
            edges = np.linalg.norm(corners - np.roll(corners, -1, axis=1), axis=2)
            quad = self.corner_rows[:, 3] >= 0
            # A triangle repeats its third corner: edges 0-1, 1-2 and 3-0
            # (the closing edge); the repeated 2-3 edge has zero length.
            lengths = np.concatenate([edges[quad].ravel(), edges[~quad][:, [0, 1, 3]].ravel()])
            cached = float(np.median(lengths)) if lengths.size else 0.0
            self._cache["length"] = cached
        return cached

    def incident_elements(self) -> Sequence[np.ndarray]:
        """Element rows touching each node row."""

        cached = self._cache.get("incident")
        if cached is None:
            buckets = [[] for _ in range(self.n_nodes)]
            for e in range(self.n_elements):
                for row in self.corner_rows[e]:
                    if row >= 0:
                        buckets[row].append(e)
            cached = [np.asarray(b, dtype=np.int64) for b in buckets]
            self._cache["incident"] = cached
        return cached

    def node_normals(self, node_rows: Sequence[int]) -> np.ndarray:
        """Average normal of the elements at each node (sign-aligned)."""

        incident = self.incident_elements()
        normals = self.normals()
        out = np.zeros((len(node_rows), 3))
        for i, row in enumerate(node_rows):
            elements = incident[int(row)]
            if elements.size == 0:
                raise ExtractionError(
                    f"node {int(self.node_ids[int(row)])} belongs to no element"
                )
            reference = normals[elements[0]]
            total = np.zeros(3)
            for e in elements:
                n = normals[e]
                total += n if float(n @ reference) >= 0.0 else -n
            out[i] = total
        return _unit(out)

    # ------------------------------------------------------------------
    def locate(
        self, points: np.ndarray, tolerance: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the element under each point.

        Returns ``(element_rows, weights, distances)``: ``weights`` has shape
        ``(n, 4)`` (linear shape-function values at the corners, 0 for a
        missing fourth corner) and ``distances`` is each point's distance to
        the element plane.  A point further than ``tolerance`` from every
        element raises :class:`ExtractionError`; nothing is snapped silently.
        """

        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if pts.shape[1] != 3 or not np.all(np.isfinite(pts)):
            raise InputError("points must be finite with shape (n, 3)")
        tol = self.characteristic_length() * 0.1 if tolerance is None else float(tolerance)
        if not (math.isfinite(tol) and tol >= 0.0):
            raise InputError("search tolerance must be non-negative")
        bounds = self._cache.get("bounds")
        if bounds is None:
            corners = self._corner_positions()
            bounds = (corners.min(axis=1), corners.max(axis=1))
            self._cache["bounds"] = bounds
        lo, hi = bounds
        out_rows = np.empty(len(pts), dtype=np.int64)
        out_weights = np.zeros((len(pts), 4))
        out_dist = np.empty(len(pts))
        slack = 1.0e-9
        for i, p in enumerate(pts):
            candidates = np.flatnonzero(np.all((p >= lo - tol) & (p <= hi + tol), axis=1))
            best = None
            for e in candidates:
                found = self._local(int(e), p)
                if found is None:
                    continue
                weights, distance, inside = found
                if distance > tol or inside > slack:
                    continue
                key = (round(distance, 12), int(self.element_ids[e]))
                if best is None or key < best[0]:
                    best = (key, int(e), weights, distance)
            if best is None:
                raise ExtractionError(
                    f"point {tuple(float(v) for v in p)} lies on no element "
                    f"within {tol:.3g} m of the mesh surface"
                )
            _key, out_rows[i], out_weights[i], out_dist[i] = best
        return out_rows, out_weights, out_dist

    def _local(self, e: int, p: np.ndarray):
        """Natural coordinates of ``p`` in element ``e``.

        Returns ``(weights, plane_distance, outside)`` where ``outside`` is how
        far the natural coordinates stray beyond the element (0 inside).
        """

        corners = self.corner_xyz(e)
        origin = corners[0]
        normal = self.geometric_normals()[e]
        if not np.any(normal):
            return None
        distance = float(abs((p - origin) @ normal))
        e1 = _unit((corners[1] - corners[0])[None, :])[0]
        e2 = np.cross(normal, e1)
        local = np.column_stack([(corners - origin) @ e1, (corners - origin) @ e2])
        q = np.array([(p - origin) @ e1, (p - origin) @ e2])
        if len(corners) == 3:
            a = np.array([[local[1, 0] - local[0, 0], local[2, 0] - local[0, 0]],
                          [local[1, 1] - local[0, 1], local[2, 1] - local[0, 1]]])
            det = np.linalg.det(a)
            if abs(det) < 1e-300:
                return None
            b1, b2 = np.linalg.solve(a, q - local[0])
            w = np.array([1.0 - b1 - b2, b1, b2, 0.0])
            outside = float(max(0.0, -w[:3].min()))
            return w, distance, outside
        xi = np.zeros(2)
        signs = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
        for _ in range(30):
            shape = 0.25 * (1.0 + signs[:, 0] * xi[0]) * (1.0 + signs[:, 1] * xi[1])
            position = shape @ local
            residual = position - q
            if np.linalg.norm(residual) < 1e-14 * max(1.0, np.abs(local).max()):
                break
            d_xi = 0.25 * signs[:, 0] * (1.0 + signs[:, 1] * xi[1])
            d_eta = 0.25 * signs[:, 1] * (1.0 + signs[:, 0] * xi[0])
            jac = np.column_stack([d_xi @ local, d_eta @ local])
            if abs(np.linalg.det(jac)) < 1e-300:
                return None
            xi = xi - np.linalg.solve(jac, residual)
            if np.abs(xi).max() > 50.0:
                return None
        shape = 0.25 * (1.0 + signs[:, 0] * xi[0]) * (1.0 + signs[:, 1] * xi[1])
        outside = float(max(0.0, np.abs(xi).max() - 1.0))
        return shape, distance, outside


class StressSource:
    """What the extraction code needs from any producer of stresses.

    Subclasses provide ``mesh``, ``case_labels``, ``case_times``,
    :meth:`element_stress` and optionally :meth:`node_stress`.  All stresses
    are MPa in global axes, shaped ``(n_cases, n_locations, 6)``.
    """

    mesh: SurfaceMesh
    case_labels: Tuple[str, ...]
    case_times: Tuple[Optional[float], ...]
    provenance: Mapping[str, object]

    @property
    def n_cases(self) -> int:
        return len(self.case_labels)

    def case_index(self, label: str) -> int:
        try:
            return self.case_labels.index(str(label))
        except ValueError:
            raise InputError(
                f"no load case {label!r}; available: {', '.join(self.case_labels)}"
            ) from None

    def element_stress(self, element_rows: Sequence[int], surface: str) -> np.ndarray:
        raise NotImplementedError

    def node_stress(self, node_rows: Sequence[int], surface: str) -> np.ndarray:
        raise NotImplementedError

    @property
    def nodal_origin(self) -> str:
        """How nodal stresses were obtained (recorded in every result)."""

        return "unknown"

    @staticmethod
    def check_surface(surface: str) -> str:
        if surface not in SURFACES:
            raise InputError(f"surface must be one of {SURFACES}, got {surface!r}")
        return surface


class ArrayStressSource(StressSource):
    """A stress source held in memory.

    ``element_top``/``element_bottom`` have shape ``(n_cases, n_elements, 6)``;
    ``node_top``/``node_bottom`` ``(n_cases, n_nodes, 6)``.  If nodal arrays are
    absent, nodal stresses are the unweighted mean of the incident element
    stresses (stated in ``nodal_origin``), the convention ANYfem's own
    along-line postprocessing uses.
    """

    def __init__(
        self,
        mesh: SurfaceMesh,
        labels: Sequence[str],
        element_top: np.ndarray,
        element_bottom: np.ndarray,
        *,
        times: Optional[Sequence[Optional[float]]] = None,
        node_top: Optional[np.ndarray] = None,
        node_bottom: Optional[np.ndarray] = None,
        nodal_origin: Optional[str] = None,
        provenance: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.mesh = mesh
        self.case_labels = tuple(str(label) for label in labels)
        if len(set(self.case_labels)) != len(self.case_labels):
            raise InputError("load case labels must be unique")
        n = len(self.case_labels)
        if times is None:
            times = (None,) * n
        if len(times) != n:
            raise InputError("one time per load case is required")
        self.case_times = tuple(None if t is None else float(t) for t in times)
        self._el = {
            "top": _checked(element_top, (n, mesh.n_elements, 6), "element_top"),
            "bottom": _checked(element_bottom, (n, mesh.n_elements, 6), "element_bottom"),
        }
        if (node_top is None) != (node_bottom is None):
            raise InputError("give nodal stresses for both surfaces or neither")
        self._nd = None
        if node_top is not None:
            self._nd = {
                "top": _checked(node_top, (n, mesh.n_nodes, 6), "node_top"),
                "bottom": _checked(node_bottom, (n, mesh.n_nodes, 6), "node_bottom"),
            }
        self._nodal_origin = nodal_origin or (
            "supplied nodal field" if self._nd is not None else "element average"
        )
        self._averaged: Dict[str, np.ndarray] = {}
        self.provenance = dict(provenance or {})

    @property
    def nodal_origin(self) -> str:
        return self._nodal_origin

    def element_stress(self, element_rows, surface):
        return self._el[self.check_surface(surface)][:, np.asarray(element_rows, dtype=int), :]

    def node_stress(self, node_rows, surface):
        self.check_surface(surface)
        rows = np.asarray(node_rows, dtype=int)
        if self._nd is not None:
            return self._nd[surface][:, rows, :]
        if surface not in self._averaged:
            self._averaged[surface] = self._average_to_nodes(self._el[surface])
        return self._averaged[surface][:, rows, :]

    def _average_to_nodes(self, element_values: np.ndarray) -> np.ndarray:
        out = np.zeros((element_values.shape[0], self.mesh.n_nodes, 6))
        count = np.zeros(self.mesh.n_nodes)
        for e in range(self.mesh.n_elements):
            for row in self.mesh.corner_rows[e]:
                if row >= 0:
                    out[:, row, :] += element_values[:, e, :]
                    count[row] += 1.0
        with np.errstate(invalid="ignore", divide="ignore"):
            out /= np.where(count > 0.0, count, 1.0)[None, :, None]
        return out


def _checked(array, shape, name) -> np.ndarray:
    a = np.asarray(array, dtype=float)
    if a.shape != shape:
        raise InputError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise InputError(f"{name} contains non-finite stresses")
    return a
