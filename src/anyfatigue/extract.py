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
    "ReadOutStation",
    "SamplePoint",
    "extract",
    "location_from_dict",
    "readout_preview",
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
        normals = mesh.normals_for(rows)
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
def _interpolate(source, rows, weights, surface):
    """Interpolate one batch using previously resolved element rows and weights."""

    corner = source.mesh.corner_rows[rows]
    needed = np.unique(corner[corner >= 0])
    nodal = source.node_stress(needed, surface)
    index = np.searchsorted(needed, np.where(corner >= 0, corner, needed[0]))
    out = np.zeros((nodal.shape[0], len(rows), 6))
    for j in range(4):
        out += weights[:, j][None, :, None] * nodal[:, index[:, j], :]
    return out


@dataclass(frozen=True)
class ReadOutStation:
    """One station on the shell mesh, with physical director and sample positions (m)."""

    xyz: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    direction: Optional[Tuple[float, float, float]]
    probes: Tuple[Tuple[float, float, float], ...]


@dataclass
class _SamplingGeometry:
    xyz: np.ndarray
    normals: np.ndarray
    welds: Sequence[Optional[np.ndarray]]
    directions: Optional[np.ndarray]
    probes: np.ndarray  # (number of probes per station, number of stations, 3)
    rows: np.ndarray
    weights: Optional[np.ndarray]  # absent for direct reads at line nodes

    def tensors(self, source, surface):
        if self.weights is None:
            return source.node_stress(self.rows, surface)
        values = _interpolate(source, self.rows, self.weights, surface)
        if len(self.probes) == 2:
            near, far = np.split(values, 2, axis=1)
            return 1.5 * near - 0.5 * far
        return values


def _surface_positions(mesh, rows, weights):
    # The fourth weight of a triangle is zero. A valid index there lets quads
    # and triangles share the same shape-function interpolation.
    corners = np.maximum(mesh.corner_rows[rows], 0)
    return np.einsum("ij,ijk->ik", weights, mesh.node_xyz[corners])


def _sampling_geometry(location, mesh, xyz, normals, welds, names, rows, weights):
    readout = location.readout
    station_xyz = xyz if weights is None else _surface_positions(mesh, rows, weights)
    if readout.method == "direct":
        return _SamplingGeometry(station_xyz, normals, welds, None, station_xyz[None, :, :],
                                 rows, weights)

    directions = []
    for normal, weld, name in zip(normals, welds, names):
        if readout.direction is not None:
            direction = _project(readout.direction, normal)
        elif weld is not None:
            direction = readout.side * np.cross(normal, weld)
        else:
            direction = None
        if direction is None:
            raise InputError(
                f"{name}: a hot-spot read-out needs a weld direction or an "
                "explicit read-out direction in the plate"
            )
        directions.append(direction / np.linalg.norm(direction))
    directions = np.asarray(directions)
    probes = np.array([xyz + offset * directions for offset in readout.offsets_m])
    try:
        rows, weights, _ = mesh.locate(probes.reshape(-1, 3), location.tolerance)
    except ExtractionError:
        # Identify the offending station after a failed batch without reading stresses.
        for i, name in enumerate(names):
            try:
                mesh.locate(probes[:, i], location.tolerance)
            except ExtractionError as error:
                distances = ", ".join(f"{s * 1000:g} mm" for s in readout.offsets_m)
                raise ExtractionError(
                    f"{name}: read-out points at {distances} from the weld toe "
                    f"are not on the mesh ({error})"
                ) from None
        raise
    sample_xyz = _surface_positions(mesh, rows, weights).reshape(probes.shape)
    return _SamplingGeometry(station_xyz, normals, welds, directions, sample_xyz, rows, weights)


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
    closed: bool = False

    def __post_init__(self) -> None:
        ids = tuple(int(i) for i in self.node_ids)
        if not ids:
            raise InputError("a line location needs at least one node id")
        if len(set(ids)) != len(ids):
            raise InputError("a line location must not repeat nodes")
        if self.closed and len(ids) < 3:
            raise InputError("a closed weld line needs at least three nodes")
        object.__setattr__(self, "node_ids", ids)
        object.__setattr__(self, "surfaces", _surfaces(self.surfaces))
        object.__setattr__(self, "weld_direction", _opt_vector(self.weld_direction))

    def _geometry(self, mesh):
        rows = np.array([mesh.node_row(i) for i in self.node_ids])
        xyz = mesh.node_xyz[rows]
        normals = mesh.node_normals(rows)
        tangents = self._tangents(xyz)
        weld = []
        for i in range(len(rows)):
            explicit = self.weld_direction if self.weld_direction is not None else tangents[i]
            if explicit is None and self.readout.direction is not None:
                explicit = np.cross(normals[i], self.readout.direction)
            weld.append(None if explicit is None else _project(explicit, normals[i]))

        names = [f"{self.name} node {nid}" for nid in self.node_ids]
        return _sampling_geometry(self, mesh, xyz, normals, weld, names, rows, None)

    def resolve(self, source: StressSource) -> List[SamplePoint]:
        geometry = self._geometry(source.mesh)
        lengths = np.linalg.norm(np.diff(geometry.xyz, axis=0), axis=1)
        chainage = np.concatenate(([0.0], np.cumsum(lengths)))
        points: List[SamplePoint] = []
        for surface in self.surfaces:
            stress = geometry.tensors(source, surface)
            for i, node_id in enumerate(self.node_ids):
                points.append(SamplePoint(
                    label=f"{self.name}: node {node_id} {surface}",
                    kind="line",
                    surface=surface,
                    xyz=geometry.xyz[i],
                    normal=geometry.normals[i],
                    weld_direction=geometry.welds[i],
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
            lo, hi = ((i - 1) % n, (i + 1) % n) if self.closed else (max(i - 1, 0), min(i + 1, n - 1))
            v = xyz[hi] - xyz[lo]
            norm = np.linalg.norm(v)
            out.append(v / norm if norm > 0.0 else None)
        return out

    def to_dict(self) -> dict:
        return {
            "type": "line", "name": self.name, "node_ids": list(self.node_ids),
            **({"closed": True} if self.closed else {}),
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

    def _geometry(self, mesh):
        pts = np.asarray(self.points, dtype=float)
        rows, weights, _ = mesh.locate(pts, self.tolerance)
        normals = mesh.normals_for(rows)
        welds = [self._weld(normal) for normal in normals]
        names = [f"{self.name} point {i + 1}" for i in range(len(pts))]
        return _sampling_geometry(self, mesh, pts, normals, welds, names, rows, weights)

    def resolve(self, source: StressSource) -> List[SamplePoint]:
        geometry = self._geometry(source.mesh)
        out: List[SamplePoint] = []
        for surface in self.surfaces:
            stress = geometry.tensors(source, surface)
            for i in range(len(geometry.xyz)):
                out.append(SamplePoint(
                    label=f"{self.name}: point {i + 1} {surface}",
                    kind="point",
                    surface=surface,
                    xyz=np.asarray(self.points[i]),
                    normal=geometry.normals[i],
                    weld_direction=geometry.welds[i],
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


def readout_preview(source: StressSource, location) -> Tuple[ReadOutStation, ...]:
    """Resolve the same toe positions, normals and probes as extraction, without reading stress.

    Each station is returned once across both surfaces. A direct read samples
    the station itself and has no direction. Hot-spot probes are ordered by
    increasing distance from the toe. Coordinates and offsets are in metres.
    """

    if not isinstance(location, (PointLocation, LineLocation)):
        raise InputError("read-out preview requires a point or line location")
    geometry = location._geometry(source.mesh)
    return tuple(
        ReadOutStation(
            tuple(xyz), tuple(geometry.normals[i]),
            None if geometry.directions is None else tuple(geometry.directions[i]),
            tuple(tuple(probe) for probe in geometry.probes[:, i]),
        )
        for i, xyz in enumerate(geometry.xyz)
    )


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
