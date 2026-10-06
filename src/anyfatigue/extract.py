"""Stress locations: element, line and interpolated points.

A *location* says where to read stress; :func:`extract` resolves it against a
:class:`~anyfatigue.model.StressSource` into :class:`SamplePoint` objects, each
carrying the full tensor history ``(n_cases, 6)`` of one point on one surface.

``ElementLocation``
    The stress of whole elements, read at the element (piecewise constant).
``LineLocation``
    An ordered chain of nodes, typically a weld line.  Every node is a sample
    point; the line tangent is the default weld direction.
``PointLocation``
    Arbitrary points, e.g. weld toes, interpolated linearly between the corner
    nodes of the element underneath from the continuous nodal stress field.

Read-out (:class:`ReadOut`)
    ``direct``
        the stress at the point itself.
    ``hotspot_a``
        DNV-RP-C203 4.3.4 method A: surface stress read at ``0.5 t`` and
        ``1.5 t`` from the weld toe, in the plate, and extrapolated linearly to
        the toe: ``s_hs = 1.5 s(0.5t) - 0.5 s(1.5t)``, applied per tensor
        component and load case.
    ``hotspot_b``
        method B: the stress at ``0.5 t``.

The read-out direction is the in-plane direction away from the weld toe,
``side * (n x t)`` for weld direction ``t``, or an explicit vector.  Read-out
points that fall off the mesh raise :class:`ExtractionError`; they are never
snapped to a neighbour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import ExtractionError, InputError
from .model import SURFACES, StressSource

__all__ = [
    "ElementLocation",
    "LineLocation",
    "LOCATION_TYPES",
    "PointLocation",
    "ReadOut",
    "SamplePoint",
    "extract",
    "location_from_dict",
]

_READOUT_METHODS = ("direct", "hotspot_a", "hotspot_b")


@dataclass(frozen=True)
class ReadOut:
    method: str = "direct"
    thickness_mm: Optional[float] = None
    side: int = 1
    direction: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        if self.method not in _READOUT_METHODS:
            raise InputError(f"read-out method must be one of {_READOUT_METHODS}")
        if self.method != "direct":
            if self.thickness_mm is None or not (
                math.isfinite(self.thickness_mm) and self.thickness_mm > 0.0
            ):
                raise InputError("a hot-spot read-out needs the plate thickness t (mm) > 0")
        if self.side not in (-1, 1):
            raise InputError("side must be +1 or -1")
        if self.direction is not None:
            d = np.asarray(self.direction, dtype=float)
            if d.shape != (3,) or not np.all(np.isfinite(d)) or np.linalg.norm(d) == 0.0:
                raise InputError("read-out direction must be a non-zero 3-vector")
            object.__setattr__(self, "direction", tuple(float(v) for v in d))

    @property
    def offsets_m(self) -> Tuple[float, ...]:
        if self.method == "direct":
            return ()
        t = self.thickness_mm / 1000.0
        return (0.5 * t, 1.5 * t) if self.method == "hotspot_a" else (0.5 * t,)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "thickness_mm": self.thickness_mm,
            "side": self.side,
            "direction": None if self.direction is None else list(self.direction),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReadOut":
        return cls(**{**data, "direction": _opt_vector(data.get("direction"))})


@dataclass(frozen=True)
class SamplePoint:
    """One point on one surface with its stress-tensor history."""

    label: str
    kind: str
    surface: str
    xyz: np.ndarray
    normal: np.ndarray
    weld_direction: Optional[np.ndarray]
    tensors: np.ndarray
    chainage: Optional[float] = None
    readout: str = "direct"
    notes: str = ""


def _opt_vector(value) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    v = np.asarray(value, dtype=float)
    if v.shape != (3,):
        raise InputError("a direction must be a 3-vector")
    return tuple(float(x) for x in v)


def _surfaces(value: Sequence[str]) -> Tuple[str, ...]:
    out = tuple(value)
    if not out:
        raise InputError("at least one surface is required")
    for s in out:
        if s not in SURFACES:
            raise InputError(f"surface must be one of {SURFACES}, got {s!r}")
    if len(set(out)) != len(out):
        raise InputError("surfaces must not repeat")
    return out


def _project(vector, normal) -> Optional[np.ndarray]:
    """Unit vector in the plane normal to ``normal``; None when degenerate."""

    v = np.asarray(vector, dtype=float)
    v = v - (v @ normal) * normal
    norm = np.linalg.norm(v)
    return None if norm < 1e-12 * max(np.linalg.norm(vector), 1e-300) else v / norm


# ----------------------------------------------------------------------
# element locations
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ElementLocation:
    element_ids: Tuple[int, ...]
    surfaces: Tuple[str, ...] = SURFACES
    weld_direction: Optional[Tuple[float, float, float]] = None
    name: str = "elements"

    def __post_init__(self) -> None:
        ids = tuple(int(i) for i in self.element_ids)
        if not ids:
            raise InputError("an element location needs at least one element id")
        object.__setattr__(self, "element_ids", ids)
        object.__setattr__(self, "surfaces", _surfaces(self.surfaces))
        object.__setattr__(self, "weld_direction", _opt_vector(self.weld_direction))

    def resolve(self, source: StressSource) -> List[SamplePoint]:
        mesh = source.mesh
        rows = [mesh.element_row(i) for i in self.element_ids]
        centroids = mesh.centroids()[rows]
        normals = mesh.normals()[rows]
        points: List[SamplePoint] = []
        for surface in self.surfaces:
            stress = source.element_stress(rows, surface)
            for k, element_id in enumerate(self.element_ids):
                weld = (
                    None if self.weld_direction is None
                    else _project(self.weld_direction, normals[k])
                )
                points.append(SamplePoint(
                    label=f"{self.name}: element {element_id} {surface}",
                    kind="element",
                    surface=surface,
                    xyz=centroids[k],
                    normal=normals[k],
                    weld_direction=weld,
                    tensors=stress[:, k, :],
                    readout="direct",
                    notes="element stress (piecewise constant)",
                ))
        return points

    def to_dict(self) -> dict:
        return {
            "type": "element", "name": self.name,
            "element_ids": list(self.element_ids), "surfaces": list(self.surfaces),
            "weld_direction": None if self.weld_direction is None else list(self.weld_direction),
        }


# ----------------------------------------------------------------------
# interpolation
# ----------------------------------------------------------------------
def _interpolate(
    source: StressSource, points: np.ndarray, surface: str, tolerance: Optional[float]
):
    """Nodal stress interpolated at ``points``: ``(n_cases, M, 6)``, normals, element rows."""

    mesh = source.mesh
    rows, weights, _distance = mesh.locate(points, tolerance)
    corner = mesh.corner_rows[rows]
    needed = np.unique(corner[corner >= 0])
    nodal = source.node_stress(needed, surface)
    index = np.searchsorted(needed, np.where(corner >= 0, corner, needed[0]))
    out = np.zeros((nodal.shape[0], len(points), 6))
    for j in range(4):
        out += weights[:, j][None, :, None] * nodal[:, index[:, j], :]
    return out, mesh.normals()[rows], rows


def _check_probes(
    source: StressSource,
    origin: np.ndarray,
    direction: np.ndarray,
    readout: ReadOut,
    surface: str,
    tolerance: Optional[float],
    what: str,
) -> None:
    """Raise a contextual :class:`ExtractionError` if a read-out point is off the mesh.

    Used only to name the offending location after a batched read-out failed; the
    values themselves come from :func:`_hot_spot_batch`.
    """

    offsets = readout.offsets_m
    probes = np.array([origin + s * direction for s in offsets])
    try:
        _interpolate(source, probes, surface, tolerance)
    except ExtractionError as error:
        raise ExtractionError(
            f"{what}: read-out points at {', '.join(f'{s * 1000:g} mm' for s in offsets)} "
            f"from the weld toe are not on the mesh ({error})"
        ) from None


def _hot_spot_batch(
    source: StressSource,
    origins: np.ndarray,
    directions: np.ndarray,
    readout: ReadOut,
    surface: str,
    tolerance: Optional[float],
    names: Sequence[str],
) -> np.ndarray:
    """Hot-spot tensors for many origins with one interpolation call.

    One call matters for sources that read from disk: the nodal field is fetched
    once for all read-out points instead of once per point.  On failure the
    points are retried one by one so the error names the offending location.
    """

    offsets = readout.offsets_m
    n = len(origins)
    probes = np.concatenate([origins + s * directions for s in offsets])
    try:
        values, _normals, _rows = _interpolate(source, probes, surface, tolerance)
    except ExtractionError:
        for i in range(n):
            _check_probes(source, origins[i], directions[i], readout, surface, tolerance, names[i])
        raise
    values = values.reshape(values.shape[0], len(offsets), n, 6)
    if readout.method == "hotspot_b":
        return values[:, 0]
    return 1.5 * values[:, 0] - 0.5 * values[:, 1]


# ----------------------------------------------------------------------
# line locations
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LineLocation:
    """An ordered chain of node ids; every node becomes a sample point."""

    node_ids: Tuple[int, ...]
    surfaces: Tuple[str, ...] = SURFACES
    readout: ReadOut = field(default_factory=ReadOut)
    weld_direction: Optional[Tuple[float, float, float]] = None
    tolerance: Optional[float] = None
    name: str = "line"

    def __post_init__(self) -> None:
        ids = tuple(int(i) for i in self.node_ids)
        if not ids:
            raise InputError("a line location needs at least one node id")
        if len(set(ids)) != len(ids):
            raise InputError("a line location must not repeat nodes")
        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "surfaces", _surfaces(self.surfaces))
        object.__setattr__(self, "weld_direction", _opt_vector(self.weld_direction))

    def resolve(self, source: StressSource) -> List[SamplePoint]:
        mesh = source.mesh
        rows = [mesh.node_row(i) for i in self.node_ids]
        xyz = mesh.node_xyz[rows]
        normals = mesh.node_normals(rows)
        tangents = self._tangents(xyz)
        chainage = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))))
        weld = []
        for i in range(len(rows)):
            explicit = self.weld_direction if self.weld_direction is not None else tangents[i]
            if explicit is None and self.readout.direction is not None:
                explicit = np.cross(normals[i], self.readout.direction)
            weld.append(None if explicit is None else _project(explicit, normals[i]))

        points: List[SamplePoint] = []
        for surface in self.surfaces:
            if self.readout.method == "direct":
                stress = source.node_stress(rows, surface)
            else:
                directions = np.array([
                    self._readout_direction(normals[i], weld[i], i) for i in range(len(rows))
                ])
                stress = _hot_spot_batch(
                    source, xyz, directions, self.readout, surface, self.tolerance,
                    [f"{self.name} node {nid}" for nid in self.node_ids],
                )
            for i, node_id in enumerate(self.node_ids):
                points.append(SamplePoint(
                    label=f"{self.name}: node {node_id} {surface}",
                    kind="line",
                    surface=surface,
                    xyz=xyz[i],
                    normal=normals[i],
                    weld_direction=weld[i],
                    tensors=stress[:, i, :],
                    chainage=float(chainage[i]),
                    readout=self.readout.method,
                    notes=_notes(source, self.readout),
                ))
        return points

    def _tangents(self, xyz: np.ndarray):
        n = len(xyz)
        if n < 2:
            return [None]
        out = []
        for i in range(n):
            lo, hi = max(i - 1, 0), min(i + 1, n - 1)
            v = xyz[hi] - xyz[lo]
            norm = np.linalg.norm(v)
            out.append(v / norm if norm > 0.0 else None)
        return out

    def _readout_direction(self, normal, weld, i) -> np.ndarray:
        if self.readout.direction is not None:
            d = _project(self.readout.direction, normal)
        elif weld is not None:
            d = self.readout.side * np.cross(normal, weld)
        else:
            d = None
        if d is None:
            raise InputError(
                f"{self.name}: a hot-spot read-out needs a weld direction or an "
                f"explicit read-out direction (node {self.node_ids[i]})"
            )
        return d / np.linalg.norm(d)

    def to_dict(self) -> dict:
        return {
            "type": "line", "name": self.name, "node_ids": list(self.node_ids),
            "surfaces": list(self.surfaces), "readout": self.readout.to_dict(),
            "weld_direction": None if self.weld_direction is None else list(self.weld_direction),
            "tolerance": self.tolerance,
        }


# ----------------------------------------------------------------------
# point locations
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class PointLocation:
    """Points (e.g. weld toes) read by interpolating the nodal stress field."""

    points: Tuple[Tuple[float, float, float], ...]
    surfaces: Tuple[str, ...] = SURFACES
    readout: ReadOut = field(default_factory=ReadOut)
    weld_direction: Optional[Tuple[float, float, float]] = None
    tolerance: Optional[float] = None
    name: str = "points"

    def __post_init__(self) -> None:
        pts = tuple(tuple(float(v) for v in p) for p in self.points)
        if not pts or any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in pts):
            raise InputError("a point location needs finite (x, y, z) points")
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "surfaces", _surfaces(self.surfaces))
        object.__setattr__(self, "weld_direction", _opt_vector(self.weld_direction))

    def resolve(self, source: StressSource) -> List[SamplePoint]:
        pts = np.asarray(self.points, dtype=float)
        out: List[SamplePoint] = []
        for surface in self.surfaces:
            if self.readout.method == "direct":
                stress, normals, _rows = _interpolate(source, pts, surface, self.tolerance)
            else:
                _unused, normals, _rows = _interpolate(source, pts, surface, self.tolerance)
                directions = np.array([
                    self._readout_direction(normals[i], self._weld(normals[i])) for i in range(len(pts))
                ])
                stress = _hot_spot_batch(
                    source, pts, directions, self.readout, surface, self.tolerance,
                    [f"{self.name} point {i + 1}" for i in range(len(pts))],
                )
            for i in range(len(pts)):
                out.append(SamplePoint(
                    label=f"{self.name}: point {i + 1} {surface}",
                    kind="point",
                    surface=surface,
                    xyz=pts[i],
                    normal=normals[i],
                    weld_direction=self._weld(normals[i]),
                    tensors=stress[:, i, :],
                    readout=self.readout.method,
                    notes=_notes(source, self.readout),
                ))
        return out

    def _weld(self, normal) -> Optional[np.ndarray]:
        if self.weld_direction is not None:
            return _project(self.weld_direction, normal)
        if self.readout.direction is not None:
            return _project(np.cross(normal, self.readout.direction), normal)
        return None

    def _readout_direction(self, normal, weld) -> np.ndarray:
        if self.readout.direction is not None:
            d = _project(self.readout.direction, normal)
        elif weld is not None:
            d = self.readout.side * np.cross(normal, weld)
        else:
            d = None
        if d is None:
            raise InputError(
                f"{self.name}: a hot-spot read-out needs a weld direction or an "
                "explicit read-out direction"
            )
        return d / np.linalg.norm(d)

    def to_dict(self) -> dict:
        return {
            "type": "point", "name": self.name, "points": [list(p) for p in self.points],
            "surfaces": list(self.surfaces), "readout": self.readout.to_dict(),
            "weld_direction": None if self.weld_direction is None else list(self.weld_direction),
            "tolerance": self.tolerance,
        }


def _notes(source: StressSource, readout: ReadOut) -> str:
    base = f"nodal stress: {source.nodal_origin}"
    return base if readout.method == "direct" else f"{base}; {readout.method}"


LOCATION_TYPES = (ElementLocation, LineLocation, PointLocation)


def extract(source: StressSource, location) -> List[SamplePoint]:
    """Resolve a location against a stress source."""

    if not isinstance(location, LOCATION_TYPES):
        raise InputError(f"unsupported location {type(location).__name__}")
    return location.resolve(source)


def location_from_dict(data: Dict) -> "ElementLocation | LineLocation | PointLocation":
    kind = data.get("type")
    body = {k: v for k, v in data.items() if k != "type"}
    if kind == "element":
        return ElementLocation(**body)
    if kind == "line":
        body["readout"] = ReadOut.from_dict(body.get("readout", {}))
        return LineLocation(**body)
    if kind == "point":
        body["readout"] = ReadOut.from_dict(body.get("readout", {}))
        return PointLocation(**body)
    raise InputError(f"unknown location type {kind!r}")
