"""Restrict shell stress sampling to the selected physical plate."""

from __future__ import annotations

import operator
from typing import Sequence

import numpy as np

from .errors import ExtractionError, InputError
from .model import StressSource, SurfaceMesh

__all__ = ["plate_stress_source"]


class _PlateStressSource(StressSource):
    def __init__(self, source: StressSource, element_rows: np.ndarray) -> None:
        parent = source.mesh
        corners = parent.element_corner_ids[element_rows]
        corner_rows = parent.corner_rows[element_rows]
        self._node_rows = np.unique(corner_rows[corner_rows >= 0])
        self.mesh = SurfaceMesh(
            parent.node_ids[self._node_rows], parent.node_xyz[self._node_rows],
            parent.element_ids[element_rows], corners, parent.normals_for(element_rows),
        )
        self._source = source
        self._element_rows = element_rows
        self.case_labels = source.case_labels
        self.case_times = source.case_times

        # At a junction the parent has only one nodal tensor for all incident
        # plates. Its value cannot identify the tensor on the chosen plate.
        parent_incident = parent.incident_elements()
        incident = self.mesh.incident_elements()
        self._shared = np.array([
            len(parent_incident[parent_row]) != len(group)
            for parent_row, group in zip(self._node_rows, incident)
        ])
        shared_count = int(self._shared.sum())
        self._use_element_stress = (
            type(source).node_stress is StressSource.node_stress
            or source.nodal_origin == "element average"
        )
        if self._use_element_stress:
            self._nodal_origin = "selected-plate element average"
        elif shared_count:
            self._nodal_origin = (
                f"{source.nodal_origin}; selected-plate element average at {shared_count} shared nodes"
            )
        else:
            self._nodal_origin = source.nodal_origin
        normals = self.mesh.normals()
        for node_id, group in zip(self.mesh.node_ids, incident):
            local = normals[group]
            if len(group) > 1 and np.any(local @ local.T <= 0.0):
                raise ExtractionError(
                    f"selected shells have inconsistent surface directors at node {node_id}; "
                    "top and bottom must denote consistent physical sides of one plate"
                )

        self._selection = {
            "element_ids": self.mesh.element_ids.tolist(),
            "shared_node_count": shared_count,
            "nodal_stress": self.nodal_origin,
        }

    @property
    def provenance(self):
        return {**self._source.provenance, "plate_selection": self._selection}

    @property
    def nodal_origin(self) -> str:
        return self._nodal_origin

    def element_stress(self, element_rows, surface):
        return self._source.element_stress(
            self._element_rows[np.asarray(element_rows, dtype=int)], self.check_surface(surface)
        )

    def node_stress(self, node_rows, surface):
        self.check_surface(surface)
        rows = np.asarray(node_rows, dtype=int)
        average = np.ones(len(rows), dtype=bool) if self._use_element_stress else self._shared[rows]
        out = np.empty((self.n_cases, len(rows), 6))
        if not average.all():
            out[:, ~average, :] = self._source.node_stress(self._node_rows[rows[~average]], surface)
        if average.any():
            incident = self.mesh.incident_elements()
            groups = [incident[row] for row in rows[average]]
            needed = np.unique(np.concatenate(groups))
            tensors = self.element_stress(needed, surface)
            for slot, group in zip(np.flatnonzero(average), groups):
                out[:, slot, :] = tensors[:, np.searchsorted(needed, group), :].mean(axis=1)
        return out


def plate_stress_source(source: StressSource, element_ids: Sequence[int]) -> StressSource:
    """Build a mesh view from producer shell element IDs, reading stresses on demand.

    Element stresses retain the source's global axes, units and physical sides.
    Nodes incident only to the chosen elements retain the producer's nodal field.
    At a node shared with excluded elements, the view averages the selected
    incident element stresses separately on each surface. This approximation is
    recorded in the view's provenance and nodal origin.
    """

    try:
        values = tuple(element_ids)
        if any(isinstance(value, (bool, np.bool_)) for value in values):
            raise TypeError
        ids = tuple(operator.index(value) for value in values)
    except TypeError as error:
        raise InputError("plate selection requires integer shell element IDs") from error
    if not ids or len(set(ids)) != len(ids):
        raise InputError("plate selection requires a nonempty list of unique shell element IDs")
    rows = np.array([source.mesh.element_row(value) for value in ids], dtype=int)
    return _PlateStressSource(source, rows)
