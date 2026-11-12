"""On-demand shell tensors from one immutable live ANYfem result."""

from collections import OrderedDict
from concurrent.futures import CancelledError
from copy import copy
from threading import RLock

import numpy as np

from .anyfem_adapter import PA_TO_MPA, _SIDE, _element_tensors, _versions, surface_mesh_from_anyfem
from .errors import ExtractionError, InputError, UnsupportedInputError
from .model import SURFACES, TENSOR_ORDER, StressSource


_KEYS = tuple(f"global_{component}_{_SIDE[side]}" for side in SURFACES for component in TENSOR_ORDER)


def _retained(shape, kind):
    accessor = getattr(shape, "retained_stress_recovery", None)
    return accessor(kind) if accessor is not None else None


class _LiveRecovery:
    """Share compact, two-sided rows across views of exactly these solved shapes."""

    def __init__(self, shapes, labels, times, recover, patch, transient):
        self.shapes = tuple(shapes)
        self.labels = tuple(labels)
        self.times = tuple(None if value is None else float(value) for value in times)
        self.recover = recover
        self.patch = patch
        self.lock = RLock()
        # 16,384 element/node rows carry 1.5 MiB of raw tensor data. Each
        # row holds both surfaces; the cache never retains full solver output.
        self.cache = OrderedDict()
        self.directors = {}
        self.corner_directors = set()
        self.mesh, _, _ = surface_mesh_from_anyfem(shapes[0].built.mesh)
        self.metadata = {
            "kind": "ANYfem linear transient history" if transient else "ANYfem linear static results",
            "producer": _versions(), "unit_conversion": "Pa -> MPa (x 1e-6)",
            "element_reduction": "mean over integration points",
            "surfaces": {"top": "ANYfem '*_top' (+physical_director side)", "bottom": "ANYfem '*_bot'"},
        }

    def _cache_rows(self, kind, frame, ids, fetch):
        found = {}
        missing = []
        for eid in dict.fromkeys(map(int, ids)):
            key = kind, frame, eid
            if key in self.cache:
                found[eid] = self.cache[key]
                self.cache.move_to_end(key)
            else:
                missing.append(eid)
        if missing:
            new = fetch(frame, missing)
            found.update(new)
            self._store(kind, frame, new)
        return found

    def _store(self, kind, frame, values):
        for eid, value in values.items():
            self.cache[kind, frame, eid] = value
        while len(self.cache) > 16384:
            self.cache.popitem(last=False)

    def _read_elements(self, frame, ids, existing=None, *, need_directors=False):
        from anyfesolver import RecoveryConfig

        shape = self.shapes[frame]
        retained = _retained(shape, "global")
        # A patch bundle can already contain the requested global surface tensors.
        candidates = [(retained, True)]
        if not need_directors:
            candidates.append((existing if existing is not None else _retained(shape, "patch"), False))
        candidates.append((_retained(shape, "native"), False))
        missing = set(ids)
        result = {}

        def take(bundle, wanted, global_result):
            for surface in SURFACES:
                block = _element_tensors(bundle, wanted, surface, self.labels[frame])
                if not np.all(np.isfinite(block)):
                    raise ExtractionError(f"non-finite shell stresses in case {self.labels[frame]!r}")
                for eid, tensor in zip(wanted, block):
                    if eid not in result:
                        result[eid] = np.empty((2, 6))
                    result[eid][SURFACES.index(surface)] = tensor
            for eid in wanted:
                director = bundle.element_stresses[eid].get("physical_director")
                if director is not None:
                    self.directors[eid] = np.asarray(director, dtype=float).reshape(3).copy()
                    self.corner_directors.discard(eid)
                elif global_result and eid not in self.directors:
                    self.corner_directors.add(eid)
            missing.difference_update(wanted)

        for bundle, global_result in candidates:
            if bundle is None or not missing:
                continue
            records = bundle.element_stresses
            wanted = [eid for eid in missing if eid in records and all(k in records[eid] for k in _KEYS)
                      and (not need_directors or global_result or "physical_director" in records[eid])]
            if wanted:
                take(bundle, wanted, global_result)
        if missing:
            wanted = sorted(missing)
            config = RecoveryConfig(element_ids=wanted, include_displacements=False, include_reactions=False)
            bundle = self.recover(shape, return_global=True, recovery_config=config)
            take(bundle, wanted, True)
        return result

    def element_rows(self, frame, ids, existing=None):
        return self._cache_rows("element", frame, ids,
                                lambda f, missing: self._read_elements(f, missing, existing))

    def _read_nodes(self, frame, ids):
        from anyfesolver import PatchRecoveryConfig

        shape = self.shapes[frame]
        bundle = None
        qualified = {}
        if self.patch:
            bundle = _retained(shape, "patch")
            if bundle is None:
                # Owner qualification needs the complete surrounding shell patch.
                bundle = self.recover(shape, patch_config=PatchRecoveryConfig())
            field = getattr(bundle, "nodal_stresses", None)
            if field is None or "nodal" not in field:
                raise UnsupportedInputError(f"load case {self.labels[frame]!r} has no patch-recovered nodal stresses")
            diagnostics = field.get("node_diagnostics", {})
            for nid in ids:
                values = field["nodal"].get(nid)
                if values is not None and diagnostics.get(nid, {}).get("status") == "qualified":
                    qualified[nid] = np.array([values[k] for k in _KEYS], dtype=float).reshape(2, 6) * PA_TO_MPA
        fallback = [nid for nid in ids if nid not in qualified or np.isnan(qualified[nid]).any()]
        result = {nid: (tensor, False) for nid, tensor in qualified.items()}
        if fallback:
            incidence = self.mesh.incident_elements()
            groups = {nid: self.mesh.element_ids[incidence[self.mesh.node_row(nid)]] for nid in fallback}
            tensors = self.element_rows(frame, np.unique(np.concatenate(list(groups.values()))), bundle)
            for nid, group in groups.items():
                average = np.mean([tensors[eid] for eid in group], axis=0)
                value = qualified.get(nid, average).copy()
                bad = np.isnan(value).any(axis=1)
                value[bad] = average[bad]
                result[nid] = value, True
        if any(not np.isfinite(tensor).all() for tensor, _ in result.values()):
            raise ExtractionError(f"non-finite nodal stresses in case {self.labels[frame]!r}")
        return result

    def node_rows(self, frame, ids):
        return self._cache_rows("node", frame, ids, self._read_nodes)

    def normals(self, frame, rows):
        with self.lock:
            ids = self.mesh.element_ids[rows]
            shape = self.shapes[frame]
            retained = _retained(shape, "global")
            records = {} if retained is None else retained.element_stresses
            alternate = [_retained(shape, kind) for kind in ("patch", "native")]
            model = getattr(shape.built, "fe_model", None)
            solver_mesh = getattr(model, "mesh", None)
            unknown = []
            for eid in dict.fromkeys(map(int, ids)):
                if eid in self.directors:
                    continue
                director = records.get(eid, {}).get("physical_director")
                for bundle in alternate:
                    if director is None and bundle is not None:
                        director = bundle.element_stresses.get(eid, {}).get("physical_director")
                if director is None and solver_mesh is not None:
                    element = solver_mesh.elements[eid]
                    native = getattr(element, "native_reference_directors", None)
                    if native is not None:
                        # The S3 owner returns the same physical facet director at each corner.
                        director = native(solver_mesh)[0]
                if director is None:
                    if eid in records:
                        self.corner_directors.add(eid)
                    if eid not in self.corner_directors:
                        unknown.append(eid)
                else:
                    self.directors[eid] = np.asarray(director, dtype=float).reshape(3).copy()
                    self.corner_directors.discard(eid)
            if unknown:
                values = self._read_elements(frame, unknown, need_directors=True)
                self._store("element", frame, values)
            for eid, row in zip(ids, rows):
                if eid not in self.directors:
                    self.directors[eid] = self.mesh.geometric_normals()[row]
            return np.array([self.directors[eid] for eid in ids], dtype=float).reshape(-1, 3)


class LiveStressSource(StressSource):
    """Frame-selected, row-lazy global shell stresses with full producer topology."""

    @classmethod
    def from_shapes(cls, shapes, labels, times, recover, patch, transient):
        if not shapes or len(set(labels)) != len(labels):
            raise InputError("a live stress source needs frames with unique labels")
        if any(shape.built.mesh is not shapes[0].built.mesh for shape in shapes):
            raise InputError("live fatigue frames must use the same submitted mesh")
        storage = _LiveRecovery(shapes, labels, times, recover, patch, transient)
        return cls(storage, tuple(range(len(shapes))))

    def __init__(self, storage, frames, cancelled=None):
        self._storage = storage
        self._frames = frames
        self._cancelled = cancelled
        self.case_labels = tuple(storage.labels[i] for i in frames)
        self.case_times = tuple(storage.times[i] for i in frames)
        # The immutable full topology and its geometric lookup tables are shared.
        # Any recovery needed for a director must come from a selected frame.
        self.mesh = copy(storage.mesh)
        object.__setattr__(self.mesh, "director_provider", self._normals)
        self._node_status = {}

    def _check_cancel(self):
        if self._cancelled is not None and self._cancelled():
            raise CancelledError("fatigue result request cancelled")

    def _normals(self, rows):
        with self._storage.lock:
            self._check_cancel()
            return self._storage.normals(self._frames[0], rows)

    def select_cases(self, labels, *, cancelled=None):
        """Share recovery storage with an ordered view of the requested case labels."""
        frames = tuple(self._frames[self.case_index(label)] for label in labels)
        if not frames or len(frames) != len(set(frames)):
            raise InputError("select one or more distinct fatigue cases")
        return LiveStressSource(self._storage, frames, cancelled)

    @property
    def nodal_origin(self):
        return ("ANYfem patch recovery (incident element average at unqualified nodes)"
                if self._storage.patch else "element average")

    @property
    def provenance(self):
        with self._storage.lock:
            statuses = [self._node_status.get(frame, {}) for frame in self._frames]
            return {**self._storage.metadata, "cases": list(self.case_labels),
                    "corner_order_director_element_ids": sorted(self._storage.corner_directors),
                    "nodal_fallback_node_ids": sorted({nid for state in statuses for nid, bad in state.items() if bad}),
                    "patch_status_scope": "sampled nodes of selected cases",
                    "patch_node_status_per_case": [
                        {"qualified": len(state) - sum(state.values()), "other": sum(state.values())}
                        for state in statuses] if self._storage.patch else []}

    def _read(self, rows, surface, *, nodal):
        side = SURFACES.index(self.check_surface(surface))
        ids = (self.mesh.node_ids if nodal else self.mesh.element_ids)[np.asarray(rows, dtype=int)]
        out = np.empty((self.n_cases, len(ids), 6))
        if not len(ids):
            return out
        for i, frame in enumerate(self._frames):
            with self._storage.lock:
                self._check_cancel()
                if nodal:
                    values = self._storage.node_rows(frame, ids)
                    status = self._node_status.setdefault(frame, {})
                    for col, nid in enumerate(ids):
                        tensor, fallback = values[nid]
                        if self._storage.patch:
                            status[int(nid)] = fallback
                        out[i, col] = tensor[side]
                else:
                    values = self._storage.element_rows(frame, ids)
                    out[i] = [values[eid][side] for eid in ids]
        return out

    def node_stress(self, node_rows, surface):
        return self._read(node_rows, surface, nodal=True)

    def element_stress(self, element_rows, surface):
        return self._read(element_rows, surface, nodal=False)
