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
* **Normals**: saved physical shell directors when available, else the corner order.
* **Mesh**: the ``meshes/*.anymesh.h5`` sidecar whose ``mesh_hash`` equals the
  result's.  Load-case results do not name their mesh sidecar, so the hash is
  the link; none or no match is an error.
* **Times**: stored load-case times or the original transient frame times. Older
  static batches can use the project's named cases or explicit block duration.

Static shell results and completed linear transient histories with signed global
surface tensors are supported. Samples are read from the artifact on demand.
"""

from __future__ import annotations

import json
from concurrent.futures import CancelledError
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .anyfem_adapter import surface_mesh_from_anyfem
from .errors import ExtractionError, InputError, MissingDependencyError, UnsupportedInputError
from .model import SURFACES, TENSOR_ORDER, StressSource

__all__ = ["ArtifactStressSource", "ResultInfo", "list_results", "read_result", "read_result_dataset"]

PA_TO_MPA = 1.0e-6
RESULT_SUFFIX = ".anyres.h5"
STORE_SUFFIX = "-data"
SUPPORTED_FRAMES = ("load_case", "static")
_SIDE = {"top": "top", "bottom": "bot"}


def _global_key(component: str, surface: str) -> str:
    return f"stress_global_{component}_{_SIDE[surface]}"


def _artifact_failures() -> tuple:
    """Exceptions that mean a saved artifact is malformed (never the caller's own times)."""

    failures = [TypeError, ValueError, IndexError, KeyError, AttributeError, OSError, InputError]
    try:
        failures.append(_modules()[2])
    except MissingDependencyError:
        pass
    return tuple(failures)


def _check_field(key: str, shape, dtype, ndims: Tuple[int, ...], n_frames: int, n_entities: int) -> None:
    """Validate rank, then shape, then dtype of a stored field before anything indexes it."""

    shape = tuple(int(x) for x in shape)
    if len(shape) not in ndims:
        raise UnsupportedInputError(f"{key}: stored rank {len(shape)} is not one of {ndims}")
    if shape[0] != n_frames or shape[1] != n_entities:
        raise UnsupportedInputError(
            f"{key}: stored shape {shape} does not cover {n_frames} frames x {n_entities} entities"
        )
    if dtype is not None and np.dtype(dtype).kind != "f":
        raise UnsupportedInputError(f"{key}: stored values are {np.dtype(dtype)}, not real floating point")


def _stored_dtype(dataset, key: str):
    """dtype of a stored field from the file header, or None when it cannot be known."""

    path = getattr(dataset, "path", None)
    if path is None:
        return None
    h5py, _s, _e = _modules()
    with h5py.File(path, "r") as handle:
        return handle[f"fields/{key}/values"].dtype


class _CallerTimesError(InputError):
    """The caller's explicit ``times`` do not fit the result; never falls back to another job."""


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
        return self.has_global_stress and (
            self.frame_kind in SUPPORTED_FRAMES or
            (self.frame_kind == "time" and self.solution_type == "TransientSolution")
        )


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
        dataset = LazyResultDataset(file)
        try:
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
        finally:
            if hasattr(dataset, "close"):
                dataset.close()
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
        candidates = [r.path for r in reversed(usable)]
        if job_id is not None:
            candidates = candidates[:1]
    else:
        candidates = [file]
    first_error = last_error = None
    for candidate in candidates:
        try:
            return ArtifactStressSource(store, root, candidate, times=times, use_patch=patch)
        except _CallerTimesError:
            raise                                  # the caller's own times never switch result
        except (UnsupportedInputError, ExtractionError, ArtifactError, OSError, KeyError) as error:
            # Newest first: an unusable newer result must not hide an older usable one.
            if len(candidates) == 1:
                if isinstance(error, (UnsupportedInputError, ExtractionError)):
                    raise
                raise UnsupportedInputError(f"cannot read result {candidate.name}: {error}") from error
            first_error = first_error or error
            last_error = error
    raise UnsupportedInputError(
        f"no readable result in the project; newest failed with {type(first_error).__name__}: {first_error}; "
        f"oldest failed with {type(last_error).__name__}: {last_error}"
    ) from last_error


def read_result_dataset(
    dataset, *, times: Optional[Sequence[Optional[float]]] = None, patch: bool = True
) -> "ArtifactStressSource":
    """Bind a caller-owned open ANYfem result dataset to its submitted mesh.

    The source borrows the dataset only while validating and reading metadata.
    Later stress requests open the result sidecar independently, including after
    the caller closes the dataset. The mesh is verified as for :func:`read_result`.
    """
    _h5py, _ArtifactStore, ArtifactError = _modules()
    try:
        store, root, file = _store_for(dataset.path)
        if file is None:
            raise InputError("the open dataset must refer to an ANYfem result sidecar")
        return ArtifactStressSource(store, root, file, times=times, use_patch=patch, _dataset=dataset)
    except (ArtifactError, OSError, KeyError) as error:
        raise InputError(f"cannot read open result dataset: {error}") from None


class ArtifactStressSource(StressSource):
    """A stress source backed by an ANYfem result sidecar (reads on demand)."""

    def __init__(self, store, root: Path, file: Path, *, times=None, use_patch: bool = True,
                 _dataset=None) -> None:
        from anyfem.io.artifacts import LazyResultDataset

        self._file = file
        if _dataset is not None:
            self._load_metadata(_dataset, store, root, file, times, use_patch)
            return

        dataset = LazyResultDataset(file)
        try:
            self._load_metadata(dataset, store, root, file, times, use_patch)
        finally:
            if hasattr(dataset, "close"):
                dataset.close()

    def _load_metadata(self, dataset, store, root, file, times, use_patch):
        """Read and validate a saved artifact; any malformation is one typed refusal."""

        try:
            self._load_metadata_unchecked(dataset, store, root, file, times, use_patch)
        except (_CallerTimesError, UnsupportedInputError, ExtractionError):
            raise
        except _artifact_failures() as error:
            raise UnsupportedInputError(
                f"{file.name}: unusable ANYfem result ({type(error).__name__}: {error})"
            ) from error

    def _load_metadata_unchecked(self, dataset, store, root, file, times, use_patch):
        dataset.validate()
        identity = dataset.identity
        kind = identity["frame_kind"]
        transient = kind == "time" and dataset.metadata("provenance").get("solution_type") == "TransientSolution"
        if kind not in SUPPORTED_FRAMES and not transient:
            raise UnsupportedInputError(
                f"{file.name}: unsupported frames {kind!r}. Select a linear-static result "
                "or a linear transient with signed shell stresses."
            )
        if identity.get("partial"):
            raise UnsupportedInputError("fatigue requires a completed result history")
        keys = set(dataset.field_keys)
        missing = [_global_key(c, s) for c in TENSOR_ORDER for s in SURFACES if _global_key(c, s) not in keys]
        if missing:
            raise UnsupportedInputError(
                f"{file.name} has no global surface stresses ({missing[0]} ...); recover stresses "
                "with return_global=True (the 'global stress' output) so the stress can be "
                "resolved against the weld"
            )
        frames = dataset.frames
        n_frames = int(frames.size)
        if kind == "load_case":
            names = [_text(x) for x in dataset.table("load_case_names")]
            if len(names) != n_frames:
                raise InputError(f"{file.name}: {len(names)} load-case names for {n_frames} frames")
        elif transient:
            if n_frames == 0 or not np.all(np.isfinite(frames)) or np.any(np.diff(frames) <= 0):
                raise InputError("transient stress frames need finite, strictly increasing times")
            if times is not None and not np.array_equal(times, frames):
                raise _CallerTimesError("use the original stored times of the transient result")
            times = frames
            names = [f"Step {i + 1} ({t:g} s)" for i, t in enumerate(frames)]
        else:
            names = ["static"]
        self.case_labels = tuple(names)
        self.n_frames = n_frames
        self._frame_ids = tuple(range(n_frames))
        self._frame_runs = (slice(0, n_frames),)
        self._cancelled = None

        ids = np.asarray(dataset.table(f"{_global_key('xx', 'top')}_element_ids"), dtype=np.int64)
        self._sample_shapes = {}
        for c in TENSOR_ORDER:
            for s in SURFACES:
                key = _global_key(c, s)
                field = dataset.field(key)
                desc = field.descriptor
                _check_field(key, field.shape, _stored_dtype(dataset, key), (2, 3), n_frames, ids.size)
                if (desc.unit != "Pa" or desc.basis != "global" or desc.recovery == "envelope"
                        or desc.components != (f"global_{c}_{_SIDE[s]}",)
                        or desc.location not in ("element", "integration_point")):
                    raise UnsupportedInputError(f"{key}: expected a signed global shell stress in Pa for every frame")
                if transient:
                    if not np.array_equal(desc.frames, frames):
                        raise UnsupportedInputError(f"{key}: shell stress times differ from the stored result times")
                spatial = desc.provenance.get("element_sample_shapes")
                by_frame = desc.provenance.get("frame_element_sample_shapes")
                if spatial is not None or by_frame is not None:
                    self._sample_shapes[key] = (spatial, by_frame)
                if not np.array_equal(np.asarray(dataset.table(f"{_global_key(c, s)}_element_ids")), ids):
                    raise InputError(f"{file.name}: surface stress fields cover different elements")
        self._element_ids = ids

        mesh = self._find_mesh(store, root, identity["mesh_hash"])
        directors = self._read_directors(dataset, keys, ids)
        surface_mesh, shells, node_ids = surface_mesh_from_anyfem(mesh, directors)
        if not np.array_equal(shells, ids):
            raise UnsupportedInputError(
                f"{file.name} covers {ids.size} of the model's {shells.size} shell elements; "
                "a partial recovery cannot be assessed"
            )
        self.mesh = surface_mesh

        self._init_patch(dataset, keys, node_ids, use_patch, frames if transient else None, n_frames)
        if kind == "load_case" and times is None and "load_case_times_s" in dataset.table_keys:
            stored_times = np.asarray(dataset.table("load_case_times_s"), dtype=float)
            if stored_times.shape != (n_frames,) or np.isinf(stored_times).any():
                raise InputError("invalid saved load-case times")
            times = [None if np.isnan(value) else value for value in stored_times]
        self.case_times = self._times(store, names, times)
        self.provenance = self._provenance(dataset, identity, file, directors is not None)

    # -- construction helpers -----------------------------------------------------
    @staticmethod
    def _find_mesh(store, root: Path, mesh_hash: str):
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
        if "shell_surface_directors" in dataset.table_keys:
            directors = dataset.table("shell_surface_directors")
            if not isinstance(directors, dict) or any(str(e) not in directors for e in ids):
                raise UnsupportedInputError("the saved shell surface directors are incomplete")
            return {int(e): np.asarray(directors[str(e)], dtype=float) for e in ids}
        if "stress_physical_director" not in keys:
            return None
        table = np.asarray(dataset.table("stress_physical_director_element_ids"), dtype=np.int64)
        values = np.asarray(dataset.field("stress_physical_director").read(0), dtype=float)
        if values.ndim != 2 or values.shape != (table.size, 3):
            return None
        return {int(e): v for e, v in zip(table, values)}

    def _init_patch(self, dataset, keys, node_ids, use_patch: bool, frames=None, n_frames=None) -> None:
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
        for c in TENSOR_ORDER:
            for s in SURFACES:
                key = _patch_key(c, s)
                patch_field = dataset.field(key)
                desc = patch_field.descriptor
                if desc.unit != "Pa" or desc.basis != "global":
                    raise UnsupportedInputError(f"{key}: expected a patch stress in Pa on the global basis")
                _check_field(key, patch_field.shape, _stored_dtype(dataset, key), (2,), n_frames, ids.size)
                if frames is not None and not np.array_equal(desc.frames, frames):
                    raise UnsupportedInputError(f"{key}: patch stress times differ from the stored result times")
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
                raise _CallerTimesError("one time per load case is required")
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
    def select_cases(self, labels: Sequence[str], *, cancelled=None) -> "ArtifactStressSource":
        """Share the mesh and metadata, reading only the listed frames in their given order."""
        labels = tuple(labels)
        if labels == self.case_labels:
            if cancelled is self._cancelled:
                return self
            view = copy(self)
            view._cancelled = cancelled
            return view
        if not labels or len(set(labels)) != len(labels):
            raise InputError("select one or more distinct load cases")
        rows = tuple(self.case_index(label) for label in labels)
        view = copy(self)
        view._cancelled = cancelled
        view.case_labels = labels
        view.case_times = tuple(self.case_times[i] for i in rows)
        view.provenance = {**self.provenance, "cases": list(labels)}
        view.n_frames = len(rows)
        view._frame_ids = tuple(self._frame_ids[i] for i in rows)
        runs = []
        start = previous = view._frame_ids[0]
        for frame in view._frame_ids[1:]:
            if frame != previous + 1:
                runs.append(slice(start, previous + 1))
                start = frame
            previous = frame
        view._frame_runs = (*runs, slice(start, previous + 1))
        return view

    @property
    def nodal_origin(self) -> str:
        if not self._patch_ok:
            return "element average"
        count = int((self._patch_index < 0).sum())
        text = "ANYfem patch recovery"
        return text + (f" (element average at {count} unqualified nodes)" if count else "")

    def _read(self, handle, key: str, index: np.ndarray, mean_last: bool) -> np.ndarray:
        """Read only the selected frames and requested entity rows from an open file."""
        values = handle[f"fields/{key}/values"]
        # HDF5 slices are efficient for contiguous IDs, including one picked shell.
        width = index[-1] - index[0] + 1 if len(index) else 0
        dense = bool(len(index) and width <= 2 * len(index))
        rows = slice(index[0], index[-1] + 1) if dense else index
        blocks = []
        for run in self._frame_runs:
            if self._cancelled is not None and self._cancelled():
                raise CancelledError()
            blocks.append(np.asarray(values[run, rows], dtype=float))
        block = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)
        if dense and width != len(index):
            block = block[:, index - index[0]]
        if mean_last and block.ndim >= 3:
            shapes = self._sample_shapes.get(key)
            if shapes is None:
                block = block.mean(axis=tuple(range(2, block.ndim)))
            else:
                # Mixed shell formulations need not have the same number of
                # Gauss points. Average exactly the samples the producer kept.
                spatial, by_frame = shapes
                means = np.empty(block.shape[:2])
                for frame, stored_frame in enumerate(self._frame_ids):
                    extents = spatial if by_frame is None else by_frame[str(stored_frame)]
                    for slot, eid in enumerate(self.mesh.element_ids[index]):
                        native = tuple(extents[str(eid)])
                        sample_shape = block.shape[2:]
                        # The artifact writer appends singleton axes to lower-rank
                        # samples, including a scalar stored alongside a vector.
                        native += (1,) * (len(sample_shape) - len(native))
                        if len(native) != len(sample_shape) or any(
                            type(size) is not int or not 1 <= size <= limit
                            for size, limit in zip(native, sample_shape)
                        ):
                            raise ExtractionError(f"{key}: invalid sample shape for shell {eid}")
                        region = tuple(slice(extent) for extent in native)
                        means[frame, slot] = block[(frame, slot, *region)].mean()
                block = means
        if not np.isfinite(block).all():
            raise ExtractionError(f"{key}: a requested shell stress sample is missing or non-finite")
        return block[..., 0] if block.ndim == 3 else block

    def element_stress(self, element_rows, surface):
        self.check_surface(surface)
        rows = np.asarray(element_rows, dtype=np.int64)
        unique, inverse = np.unique(rows, return_inverse=True)
        out = np.empty((self.n_frames, unique.size, 6))
        h5py, _, _ = _modules()
        with h5py.File(self._file, "r") as handle:
            for k, component in enumerate(TENSOR_ORDER):
                out[:, :, k] = self._read(handle, _global_key(component, surface), unique, mean_last=True)
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
            h5py, _, _ = _modules()
            with h5py.File(self._file, "r") as handle:
                for k, component in enumerate(TENSOR_ORDER):
                    block[:, :, k] = self._read(handle, _patch_key(component, surface), unique, mean_last=False)
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
