"""ANYfem result artifacts (``.anyres.h5``) as a stress source.

A saved ANYfem project ``deck.anyfem`` keeps its meshes and results as HDF5
sidecars in ``deck.anyfem-data/``.  This module reads those files directly, so a
fatigue assessment needs neither a re-solve nor the ANYfem application.

What is read, and from where
----------------------------
* **Stresses**: ``stress_global_{xx,yy,zz,xy,yz,xz}_{top,bot}`` (Pa, global axes,
  one value per integration point; the element value is their mean) with the
  matching ``*_element_ids`` tables.  A result saved without them is refused:
  global surface stresses are needed to resolve the stress against a weld.
* **Nodal stresses**: ``stress_patch_global_*`` (patch recovery) where present.
  Nodes that patch recovery did not qualify, and nodes the patch table does not
  contain, take the mean of their incident element stresses; the count is in the
  provenance and in :attr:`ArtifactStressSource.nodal_origin`.
* **Normals**: ``stress_physical_director`` when present, else the corner order.
* **Mesh**: the ``meshes/*.anymesh.h5`` sidecar whose ``mesh_hash`` equals the
  result's.  Load-case results do not name their mesh sidecar, so the hash is
  the link; none or no match is an error.
* **Load-case names**: the ``load_case_names`` table; **times**: the project's
  ``load_cases`` (as set by ``add_load_case_series``) unless ``times`` is given.

Only linear-static results (a load-case batch or one static case) are supported;
other frame kinds fail with a typed error.  The source is lazy: only the requested
elements and nodes are read from disk, so a long series on a large model is not
held in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anyfem_adapter import surface_mesh_from_anyfem
from .errors import ExtractionError, InputError, MissingDependencyError, UnsupportedInputError
from .model import SURFACES, TENSOR_ORDER, StressSource

__all__ = ["ArtifactStressSource", "ResultInfo", "list_results", "read_result"]

PA_TO_MPA = 1.0e-6
RESULT_SUFFIX = ".anyres.h5"
STORE_SUFFIX = "-data"
SUPPORTED_FRAMES = ("load_case", "static")
_SIDE = {"top": "top", "bottom": "bot"}


def _global_key(component: str, surface: str) -> str:
    return f"stress_global_{component}_{_SIDE[surface]}"


def _patch_key(component: str, surface: str) -> str:
    return f"stress_patch_global_{component}_{_SIDE[surface]}"


def _modules():
    try:
        import h5py
        from anyfem.io.artifacts import ArtifactError, ArtifactStore
    except ImportError as error:
        raise MissingDependencyError(
            "reading ANYfem result files needs ANYfem and h5py: `pip install ANYfatigue[anyfem]`"
        ) from error
    return h5py, ArtifactStore, ArtifactError


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class ResultInfo:
    """One result sidecar of an ANYfem project."""

    path: Path
    job_id: str
    created_utc: str
    frame_kind: str
    n_frames: int
    solution_type: str
    has_global_stress: bool
    has_patch_stress: bool

    @property
    def usable(self) -> bool:
        return self.has_global_stress and self.frame_kind in SUPPORTED_FRAMES


def _store_for(path) -> Tuple[object, Path, Optional[Path]]:
    """``(ArtifactStore, data root, result file or None)`` for a project, data folder or result."""

    _h5py, ArtifactStore, _error = _modules()
    p = Path(path)
    if not p.exists():
        raise InputError(f"{p} does not exist")
    result_file = None
    if p.is_file() and p.name.endswith(RESULT_SUFFIX):
        result_file = p
        root = p.parent.parent
    elif p.is_dir() and p.name.endswith(STORE_SUFFIX):
        root = p
    elif p.is_file():
        project = p
        return ArtifactStore(project), ArtifactStore(project).root, None
    else:
        raise InputError(f"{p} is not an ANYfem project, data folder or {RESULT_SUFFIX} file")
    if not root.name.endswith(STORE_SUFFIX) or result_file is not None and p.parent.name != "results":
        raise InputError(f"{p} is not inside an ANYfem project data folder (…{STORE_SUFFIX}/results/)")
    project = root.with_name(root.name[: -len(STORE_SUFFIX)])
    return ArtifactStore(project), root, result_file


def list_results(path) -> List[ResultInfo]:
    """The results of a project, oldest first; works on a project, data folder or result file."""

    _h5py, _store, ArtifactError = _modules()
    from anyfem.io.artifacts import LazyResultDataset

    store, root, _file = _store_for(path)
    out: List[ResultInfo] = []
    for file in sorted((root / "results").glob(f"*{RESULT_SUFFIX}")):
        try:
            dataset = LazyResultDataset(file)
            dataset.validate()
            identity = dataset.identity
            keys = set(dataset.field_keys)
            try:
                solution = str(dataset.metadata("provenance").get("solution_type", ""))
            except ArtifactError:
                solution = ""
            out.append(ResultInfo(
                path=file, job_id=identity["artifact_id"], created_utc=identity["created_utc"],
                frame_kind=identity["frame_kind"], n_frames=int(dataset.frames.size),
                solution_type=solution,
                has_global_stress=all(_global_key(c, s) in keys for c in TENSOR_ORDER for s in SURFACES),
                has_patch_stress=all(_patch_key(c, s) in keys for c in TENSOR_ORDER for s in SURFACES),
            ))
        except (ArtifactError, OSError, KeyError, ValueError):
            continue                                    # an unreadable sidecar is not a result
    return sorted(out, key=lambda r: (r.created_utc, r.job_id))


def read_result(
    path,
    *,
    job_id: Optional[str] = None,
    times: Optional[Sequence[Optional[float]]] = None,
    patch: bool = True,
) -> "ArtifactStressSource":
    """Open an ANYfem result as a lazy :class:`~anyfatigue.model.StressSource`.

    ``path`` is a ``.anyres.h5`` file, or a project / data folder, in which case
    ``job_id`` picks the result and the newest usable one is taken by default.
    """

    h5py, _ArtifactStore, ArtifactError = _modules()
    store, root, file = _store_for(path)
    if file is None:
        usable = [r for r in list_results(path) if r.usable and (job_id is None or r.job_id == job_id)]
        if not usable:
            raise InputError(
                "no usable result in the project"
                + (f" with job id {job_id!r}" if job_id else "")
                + ": a result needs global surface stresses (recover with return_global=True) "
                  "from a linear-static load-case batch"
            )
        file = usable[-1].path
    try:
        return ArtifactStressSource(store, root, file, times=times, use_patch=patch)
    except (ArtifactError, OSError, KeyError) as error:
        raise InputError(f"cannot read result {file.name}: {error}") from None


class ArtifactStressSource(StressSource):
    """A stress source backed by an ANYfem result sidecar (reads on demand)."""

    def __init__(self, store, root: Path, file: Path, *, times=None, use_patch: bool = True) -> None:
        h5py, _ArtifactStore, ArtifactError = _modules()
        from anyfem.io.artifacts import LazyResultDataset

        self._file = file
        dataset = LazyResultDataset(file)
        dataset.validate()
        self._dataset = dataset
        identity = dataset.identity
        kind = identity["frame_kind"]
        if kind not in SUPPORTED_FRAMES:
            raise UnsupportedInputError(
                f"{file.name}: result frames are of kind {kind!r}; only linear-static results "
                f"({', '.join(SUPPORTED_FRAMES)}) are supported"
            )
        keys = set(dataset.field_keys)
        missing = [_global_key(c, s) for c in TENSOR_ORDER for s in SURFACES if _global_key(c, s) not in keys]
        if missing:
            raise UnsupportedInputError(
                f"{file.name} has no global surface stresses ({missing[0]} ...); recover stresses "
                "with return_global=True (the 'global stress' output) so the stress can be "
                "resolved against the weld"
            )
        n_frames = int(dataset.frames.size)
        if kind == "load_case":
            names = [_text(x) for x in dataset.table("load_case_names")]
            if len(names) != n_frames:
                raise InputError(f"{file.name}: {len(names)} load-case names for {n_frames} frames")
        else:
            names = ["static"]
        self.case_labels = tuple(names)
        self.n_frames = n_frames

        ids = np.asarray(dataset.table(f"{_global_key('xx', 'top')}_element_ids"), dtype=np.int64)
        for c in TENSOR_ORDER:
            for s in SURFACES:
                if not np.array_equal(np.asarray(dataset.table(f"{_global_key(c, s)}_element_ids")), ids):
                    raise InputError(f"{file.name}: surface stress fields cover different elements")
        self._element_ids = ids

        mesh = self._find_mesh(store, root, identity["mesh_hash"], ArtifactError)
        directors = self._read_directors(dataset, keys, ids)
        surface_mesh, shells, node_ids = surface_mesh_from_anyfem(mesh, directors)
        if not np.array_equal(shells, ids):
            raise UnsupportedInputError(
                f"{file.name} covers {ids.size} of the model's {shells.size} shell elements; "
                "a partial recovery cannot be assessed"
            )
        self.mesh = surface_mesh

        self._init_patch(dataset, keys, node_ids, use_patch)
        self.case_times = self._times(store, names, times)
        self.provenance = self._provenance(dataset, identity, file, directors is not None)

    # -- construction helpers -----------------------------------------------------
    @staticmethod
    def _find_mesh(store, root: Path, mesh_hash: str, _error=None):
        h5py, _s, _e = _modules()
        folder = root / "meshes"
        for sidecar in sorted(folder.glob("*.anymesh.h5")):
            try:
                with h5py.File(sidecar, "r") as handle:
                    found = _text(handle.attrs.get("mesh_hash", ""))
            except OSError:
                continue
            if found == mesh_hash:
                return store.read_mesh(f"meshes/{sidecar.name}")
        raise ExtractionError(
            f"the mesh sidecar (mesh_hash {mesh_hash[:12]}...) is not in {folder}; restore the "
            "project's meshes/ folder next to its results/"
        )

    @staticmethod
    def _read_directors(dataset, keys, ids) -> Optional[Dict[int, np.ndarray]]:
        if "stress_physical_director" not in keys:
            return None
        table = np.asarray(dataset.table("stress_physical_director_element_ids"), dtype=np.int64)
        values = np.asarray(dataset.field("stress_physical_director").read(0), dtype=float)
        if values.ndim != 2 or values.shape != (table.size, 3):
            return None
        return {int(e): v for e, v in zip(table, values)}

    def _init_patch(self, dataset, keys, node_ids, use_patch: bool) -> None:
        n = self.mesh.n_nodes
        self._patch_index = np.full(n, -1, dtype=np.int64)
        self._patch_ok = False
        self._unqualified = set()
        has_patch = all(_patch_key(c, s) in keys for c in TENSOR_ORDER for s in SURFACES)
        if not (use_patch and has_patch):
            return
        ids = np.asarray(dataset.table(f"{_patch_key('xx', 'top')}_node_ids"), dtype=np.int64)
        for c in TENSOR_ORDER:
            for s in SURFACES:
                if not np.array_equal(np.asarray(dataset.table(f"{_patch_key(c, s)}_node_ids")), ids):
                    return                               # inconsistent: do not use patch values
        provenance = dataset.field(_patch_key("xx", "top")).descriptor.provenance
        self._unqualified = {int(x) for x in provenance.get("unqualified_node_ids", [])}
        position = {int(nid): i for i, nid in enumerate(ids)}
        for row, nid in enumerate(self.mesh.node_ids):
            index = position.get(int(nid), -1)
            if index >= 0 and int(nid) not in self._unqualified:
                self._patch_index[row] = index
        self._patch_ok = bool((self._patch_index >= 0).any())

    def _times(self, store, names, times) -> Tuple[Optional[float], ...]:
        if times is not None:
            if len(times) != len(names):
                raise InputError("one time per load case is required")
            return tuple(None if t is None else float(t) for t in times)
        project = getattr(store, "project_path", None)
        found: Dict[str, Optional[float]] = {}
        if project is not None and Path(project).is_file():
            try:
                data = json.loads(Path(project).read_text(encoding="utf-8"))
                for case in data.get("load_cases", []):
                    t = case.get("time")
                    found[str(case.get("name"))] = None if t is None else float(t)
            except (OSError, ValueError, AttributeError):
                found = {}
        return tuple(found.get(name) for name in names)

    def _provenance(self, dataset, identity, file: Path, directors: bool) -> dict:
        try:
            meta = dataset.metadata("provenance")
        except Exception:                                  # metadata is informative only
            meta = {}
        fallback = sorted(
            int(self.mesh.node_ids[r]) for r in np.flatnonzero(self._patch_index < 0)
        ) if self._patch_ok else []
        return {
            "kind": "ANYfem result artifact",
            "file": file.name,
            "job_id": identity["artifact_id"],
            "created_utc": identity["created_utc"],
            "mesh_hash": identity["mesh_hash"],
            "analysis_hash": identity["analysis_hash"],
            "producer": dict(meta.get("producer_versions", {})) if isinstance(meta, dict) else {},
            "unit_conversion": "Pa -> MPa (x 1e-6)",
            "element_reduction": "mean over integration points",
            "normals": "ANYfem physical_director" if directors else "element corner order",
            "surfaces": {"top": "ANYfem '*_top'", "bottom": "ANYfem '*_bot'"},
            "cases": list(self.case_labels),
            "nodal_fallback_node_ids": fallback,
        }

    # -- StressSource ----------------------------------------------------------------
    @property
    def nodal_origin(self) -> str:
        if not self._patch_ok:
            return "element average"
        count = int((self._patch_index < 0).sum())
        text = "ANYfem patch recovery"
        return text + (f" (element average at {count} unqualified nodes)" if count else "")

    def _read(self, key: str, index: np.ndarray, mean_last: bool) -> np.ndarray:
        """Rows ``index`` (ascending, unique) of ``fields/key/values`` for every frame.

        Few rows are read as an HDF5 selection; many rows (where per-row selection is
        slower than a full read) are read whole and indexed in memory.
        """

        h5py, _s, _e = _modules()
        with h5py.File(self._file, "r") as handle:
            values = handle[f"fields/{key}/values"]
            if len(index) > min(1000, values.shape[1] // 2):
                block = np.asarray(values[...], dtype=float)[:, index]
            else:
                block = np.asarray(values[:, index], dtype=float)
        if mean_last and block.ndim == 3:
            block = block.mean(axis=2)
        return block[..., 0] if block.ndim == 3 else block

    def element_stress(self, element_rows, surface):
        self.check_surface(surface)
        rows = np.asarray(element_rows, dtype=np.int64)
        unique, inverse = np.unique(rows, return_inverse=True)
        out = np.empty((self.n_frames, unique.size, 6))
        for k, component in enumerate(TENSOR_ORDER):
            out[:, :, k] = self._read(_global_key(component, surface), unique, mean_last=True)
        return out[:, inverse, :] * PA_TO_MPA

    def node_stress(self, node_rows, surface):
        self.check_surface(surface)
        rows = np.asarray(node_rows, dtype=np.int64)
        result = np.empty((self.n_frames, rows.size, 6))
        patch_mask = self._patch_index[rows] >= 0 if self._patch_ok else np.zeros(rows.size, dtype=bool)
        if patch_mask.any():
            wanted = self._patch_index[rows[patch_mask]]
            unique, inverse = np.unique(wanted, return_inverse=True)
            block = np.empty((self.n_frames, unique.size, 6))
            for k, component in enumerate(TENSOR_ORDER):
                block[:, :, k] = self._read(_patch_key(component, surface), unique, mean_last=False)
            result[:, patch_mask, :] = block[:, inverse, :] * PA_TO_MPA
        average = np.flatnonzero(~patch_mask)
        if average.size:
            incident = self.mesh.incident_elements()
            groups = [incident[int(r)] for r in rows[average]]
            if any(g.size == 0 for g in groups):
                raise ExtractionError("a requested node belongs to no shell element")
            needed = np.unique(np.concatenate(groups))
            element = self.element_stress(needed, surface)
            position = {int(e): i for i, e in enumerate(needed)}
            for slot, group in zip(average, groups):
                result[:, slot, :] = element[:, [position[int(e)] for e in group], :].mean(axis=1)
        return result
