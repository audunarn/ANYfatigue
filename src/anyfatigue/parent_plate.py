"""Explicit parent-plate patches; no geometric plate recognition or new recovery.

The first supported recovery is a deliberately requested, one-sided element
average on a connected, consistently oriented planar patch. Supplied global
nodal stresses are NEVER reused: their support may include the attached member.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Optional, Tuple

import numpy as np

from .errors import ExtractionError, InputError
from .model import StressSource, SurfaceMesh
from .tensor import completeness_of


def _ids(values):
    try:
        result = tuple(values)
    except TypeError:
        raise InputError("parent-plate element_ids must be an integer sequence") from None
    if not result or any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in result):
        raise InputError("parent-plate element_ids must be a non-empty integer sequence")
    if len(set(result)) != len(result):
        raise InputError("parent-plate element_ids must not repeat")
    return tuple(sorted(int(x) for x in result))


@dataclass(frozen=True)
class ParentPlate:
    """Model-relative element IDs defining the one-sided parent recovery patch.

    ``nodal_recovery`` is intentionally required: 'element_average' is an
    explicit approximation, not a substitute for a qualified nodal recovery.
    ``mesh_id`` optionally binds the upstream ``provenance['mesh_hash']``; IDs
    are not inferred from coordinates or silently transferred across meshes.
    """
    element_ids: Tuple[int, ...]
    nodal_recovery: str
    name: str = "parent plate"
    mesh_id: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "element_ids", _ids(self.element_ids))
        if self.nodal_recovery != "element_average":
            raise InputError("parent nodal_recovery must explicitly be 'element_average'; "
                             "unscoped supplied nodal recovery is not supported")
        if not isinstance(self.name, str) or not self.name.strip():
            raise InputError("parent plate needs a name")
        if self.mesh_id is not None and (not isinstance(self.mesh_id, str) or not self.mesh_id.strip()):
            raise InputError("mesh_id must be a non-empty upstream mesh hash")

    def to_dict(self):
        return {"element_ids": list(self.element_ids), "nodal_recovery": self.nodal_recovery,
                "name": self.name, "mesh_id": self.mesh_id}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise InputError("parent_plate must be an object")
        try:
            return cls(**data)
        except TypeError as error:
            raise InputError(f"invalid parent_plate: {error}") from None

    def bind(self, source):
        return _ParentSource(source, self)


def _connected(mesh):
    """Check supplied element topology; do not discover or label plates."""
    neighbours = [set() for _ in range(mesh.n_elements)]
    edges = {}
    cells = set()
    for e in range(mesh.n_elements):
        corners = tuple(int(n) for n in mesh.element_corner_ids[e] if n >= 0)
        cell = tuple(sorted(corners))
        if len(set(corners)) != len(corners) or cell in cells:
            raise ExtractionError("parent patch has repeated corners or duplicate elements")
        cells.add(cell)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            edge = tuple(sorted((a, b)))
            owners = edges.setdefault(edge, [])
            for other in owners:
                neighbours[e].add(other)
                neighbours[other].add(e)
            owners.append(e)
            if len(owners) > 2:
                raise ExtractionError("parent patch has a non-manifold edge")
    visited, pending = set(), [0]
    while pending:
        e = pending.pop()
        if e not in visited:
            visited.add(e)
            pending.extend(neighbours[e] - visited)
    if len(visited) != mesh.n_elements:
        raise ExtractionError("parent patch must be connected through element edges; split the selection")


class _ParentSource(StressSource):
    def __init__(self, source, parent):
        self._source, self.parent = source, parent
        if parent.mesh_id is not None and source.provenance.get("mesh_hash") != parent.mesh_id:
            raise ExtractionError("parent-plate mesh_id does not match the source's upstream mesh_hash")
        original = source.mesh
        self._rows = np.array([original.element_row(i) for i in parent.element_ids])
        corners = original.element_corner_ids[self._rows]
        ids = np.unique(corners[corners >= 0])
        nodes = np.array([original.node_row(int(i)) for i in ids])
        self.mesh = SurfaceMesh(ids, original.node_xyz[nodes], original.element_ids[self._rows],
                                corners, element_normals=original.normals()[self._rows])
        mesh = self.mesh
        _connected(mesh)
        length = mesh.characteristic_length()
        # A floating-point geometric guard, not a mesh-angle acceptance setting.
        scale = max(float(np.max(np.abs(mesh.node_xyz))), length, np.finfo(float).tiny)
        self._distance_bound = 256 * np.finfo(float).eps * scale
        if length <= self._distance_bound:
            raise ExtractionError("parent patch coordinates have insufficient numerical resolution")
        angular_bound = self._distance_bound / length
        normal = mesh.normals()[0]
        geometry = mesh.geometric_normals()
        if np.any(np.linalg.norm(geometry, axis=1) == 0):
            raise ExtractionError("parent patch contains a degenerate shell")
        if np.any(np.linalg.norm(np.cross(mesh.normals(), normal), axis=1) > angular_bound):
            raise ExtractionError("parent patch must be planar; split attached or curved plates")
        if np.any(mesh.normals() @ normal <= 0):
            raise ExtractionError("parent patch mixes top/bottom orientations; supply a consistent physical face")
        if np.any(np.linalg.norm(np.cross(geometry, normal), axis=1) > angular_bound):
            raise ExtractionError("parent patch directors do not describe the geometric surface")
        if np.max(np.abs((mesh.node_xyz - mesh.node_xyz[0]) @ normal)) > self._distance_bound:
            raise ExtractionError("parent patch is not coplanar; split the selection")
        # Producers' physical directors define faces; geometric winding alone
        # may be opposite, and is not used to swap physical top/bottom values.
        self.case_labels, self.case_times = source.case_labels, source.case_times
        self.tensor_completeness = completeness_of(source)
        self.provenance = {**source.provenance, "parent_plate": parent.to_dict()}
        self._cache = {}

    @property
    def nodal_origin(self):
        return f"parent-only element average ({self.parent.name}); unscoped nodal field not used"

    def element_stress(self, rows, surface):
        return self._source.element_stress(self._rows[np.asarray(rows, dtype=int)], surface)

    def node_stress(self, rows, surface):
        self.check_surface(surface)
        rows = np.asarray(rows, dtype=int)
        if surface not in self._cache:
            # Fetch selected elements once per surface, including lazy producers.
            values = self.element_stress(np.arange(self.mesh.n_elements), surface)
            out = np.zeros((self.n_cases, self.mesh.n_nodes, 6))
            count = np.zeros(self.mesh.n_nodes)
            for e, corners in enumerate(self.mesh.corner_rows):
                corners = corners[corners >= 0]
                out[:, corners, :] += values[:, e, None, :]
                count[corners] += 1
            self._cache[surface] = out / count[None, :, None]
        return self._cache[surface][:, rows, :]

    def check_readout_side(self, origins, directions):
        """All recovery support must stay on the chosen side of each local toe."""
        for origin, direction in zip(origins, directions):
            if np.min((self.mesh.node_xyz - origin) @ direction) < -self._distance_bound:
                raise ExtractionError(
                    f"{self.parent.name}: recovery patch crosses the read-out side; "
                    "select only parent elements on that side, or split the weld into patches"
                )


def bind_parent(source, parent):
    if parent is None:
        return source
    if not isinstance(parent, ParentPlate):
        raise InputError("parent_plate must be ParentPlate")
    return parent.bind(source)


def extraction_provenance(provenance, locations):
    """Retain model-relative selections and distinguish source fields from recovery."""
    out = dict(provenance)
    out["locations"] = [loc.to_dict() for loc in locations]
    if any(getattr(loc, "parent_plate", None) is not None for loc in locations):
        out["nodal_stress"] = (
            "parent-only element average on explicitly selected patches; "
            "other locations use " + str(out.get("nodal_stress", "unknown source recovery"))
        )
    return out
