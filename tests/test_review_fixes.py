"""Regression tests for plate-scoped lazy assessment review findings (no ANYfem stack needed)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from anyfatigue import anyfem_artifacts as art
from anyfatigue.anyfem_adapter import _directors
from anyfatigue.anyfem_artifacts import ArtifactStressSource, SURFACES, TENSOR_ORDER, _patch_key
from anyfatigue.errors import InputError, UnsupportedInputError


def test_eager_directors_fall_back_to_the_native_reference_director():
    native = np.array([0.0, 0.0, -1.0])
    element = SimpleNamespace(native_reference_directors=lambda mesh: [native, native])
    shape = SimpleNamespace(built=SimpleNamespace(fe_model=SimpleNamespace(
        mesh=SimpleNamespace(elements={1: element, 2: SimpleNamespace()}))))
    recovery = SimpleNamespace(element_stresses={1: {}, 2: {"physical_director": [0, 1, 0]}})
    out = _directors(recovery, shape)
    assert np.allclose(out[1], native) and np.allclose(out[2], [0, 1, 0])


class _Dataset:
    def __init__(self, frames, n_frames=None, n_nodes=3, unit="Pa", basis="global"):
        self.frames = np.asarray(frames, float)
        self.shape = (len(frames) if n_frames is None else n_frames, n_nodes)
        self.unit, self.basis = unit, basis

    def table(self, key):
        return np.array([1, 2, 3])

    def field(self, key):
        return SimpleNamespace(
            shape=self.shape,
            descriptor=SimpleNamespace(frames=self.frames, provenance={}, unit=self.unit, basis=self.basis),
        )


KEYS = {_patch_key(c, s) for c in TENSOR_ORDER for s in SURFACES}
FAKE = SimpleNamespace(mesh=SimpleNamespace(n_nodes=3, node_ids=np.array([1, 2, 3])), _field_shapes={})


def init_patch(dataset, frames, n_frames):
    ArtifactStressSource._init_patch(FAKE, dataset, KEYS, None, True, frames, n_frames)


def test_patch_fields_with_different_frame_times_are_refused():
    t = np.array([0.0, 1.0])
    with pytest.raises(UnsupportedInputError):
        init_patch(_Dataset([1.0, 0.0]), t, 2)
    init_patch(_Dataset([0.0, 1.0]), t, 2)


def test_patch_field_with_missing_stored_frames_is_refused_also_for_static_batches():
    with pytest.raises(UnsupportedInputError):
        init_patch(_Dataset([0.0, 1.0], n_frames=1), np.array([0.0, 1.0]), 2)
    with pytest.raises(UnsupportedInputError):
        init_patch(_Dataset([0.0, 1.0], n_frames=1), None, 2)          # static batch


def test_patch_field_with_missing_node_rows_is_refused():
    with pytest.raises(UnsupportedInputError):
        init_patch(_Dataset([0.0, 1.0], n_nodes=1), None, 2)


@pytest.mark.parametrize("unit,basis", [("MPa", "global"), ("Pa", "local")])
def test_patch_field_with_wrong_unit_or_basis_is_refused(unit, basis):
    with pytest.raises(UnsupportedInputError):
        init_patch(_Dataset([0.0, 1.0], unit=unit, basis=basis), None, 2)


def _project(monkeypatch, tmp_path, behaviours):
    """Synthetic project whose results open (or fail) per ``behaviours`` (oldest first)."""

    files = [tmp_path / f"r{i}.anyres.h5" for i in range(len(behaviours))]
    monkeypatch.setattr(art, "_modules", lambda: (None, None, OSError))
    monkeypatch.setattr(art, "_store_for", lambda path: (None, tmp_path, None))
    monkeypatch.setattr(
        art, "list_results",
        lambda path: [SimpleNamespace(usable=True, job_id=f.name, path=f) for f in files],
    )
    opened = []

    def fake(store, root, file, **kwargs):
        opened.append(file.name)
        outcome = behaviours[files.index(file)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(art, "ArtifactStressSource", fake)
    return files, opened


@pytest.mark.parametrize("error", [
    UnsupportedInputError("partial result"),
    KeyError("missing table"),
    OSError("truncated file"),
])
def test_read_result_falls_back_to_an_older_result_on_artifact_errors(monkeypatch, tmp_path, error):
    _files, opened = _project(monkeypatch, tmp_path, ["older", error])
    assert art.read_result(tmp_path) == "older"
    assert opened == ["r1.anyres.h5", "r0.anyres.h5"]


def test_read_result_does_not_switch_result_on_the_callers_bad_times(monkeypatch, tmp_path):
    _files, opened = _project(monkeypatch, tmp_path, ["older", art._CallerTimesError("one time per load case")])
    with pytest.raises(InputError):
        art.read_result(tmp_path, times=[1.0])
    assert opened == ["r1.anyres.h5"]


# -- malformed artifacts: refused with a typed error, never a crash or silent value ----------

class _Global:
    """A fake result dataset that reaches the per-field checks of ArtifactStressSource."""

    field_keys = None
    table_keys = ()
    frames = np.array([0.0, 1.0])
    identity = {"frame_kind": "load_case", "partial": False, "mesh_hash": "h", "artifact_id": "j",
                "created_utc": "t", "analysis_hash": "a"}

    def __init__(self, names=("a", "b"), shape=(2, 3), dtype=None, unit="Pa", times=None, drop_unit=False):
        self.names, self.shape, self.dtype, self.times = names, shape, dtype, times
        self.unit, self.drop_unit = unit, drop_unit
        self.field_keys = [art._global_key(c, s) for c in TENSOR_ORDER for s in SURFACES]
        self.table_keys = () if times is None else ("load_case_times_s",)

    def validate(self):
        pass

    def metadata(self, key):
        return {}

    def table(self, key):
        if key == "load_case_names":
            return self.names
        if key == "load_case_times_s":
            return self.times
        return np.array([1, 2, 3])

    def field(self, key):
        c = key.split("global_")[1].rsplit("_", 1)[0]
        side = key.rsplit("_", 1)[1]
        desc = SimpleNamespace(
            unit=self.unit, basis="global", recovery="native", components=(f"global_{c}_{side}",),
            location="element", frames=self.frames, provenance={},
        )
        if self.drop_unit:
            del desc.unit
        return SimpleNamespace(shape=self.shape, descriptor=desc)


def open_global(monkeypatch, dataset, dtype=None):
    monkeypatch.setattr(art, "_modules", lambda: (None, None, OSError))
    monkeypatch.setattr(art, "_stored_dtype", lambda ds, key: dtype, raising=False)
    monkeypatch.setattr(ArtifactStressSource, "_find_mesh", staticmethod(lambda *a: SimpleNamespace()))
    monkeypatch.setattr(art, "surface_mesh_from_anyfem",
                        lambda mesh, directors: (SimpleNamespace(n_nodes=1, node_ids=np.array([1])), np.array([1, 2, 3]), np.array([1])))
    source = ArtifactStressSource.__new__(ArtifactStressSource)
    source._load_metadata(dataset, None, Path("."), Path("r.anyres.h5"), None, False)


def test_a_well_formed_fake_reaches_the_end(monkeypatch):
    open_global(monkeypatch, _Global())                 # guards the fixture itself


@pytest.mark.parametrize("make", [
    lambda: _Global(names=np.array("scalar")),                  # scalar load_case_names -> TypeError
    lambda: _Global(times=np.array(["x", "y"])),                # non-numeric times -> ValueError
    lambda: _Global(shape=(2,)),                                # rank 1
    lambda: _Global(shape=(2, 3, 4, 5)),                        # rank 4 (global allows 2 or 3)
    lambda: _Global(shape=(1, 3)),                              # missing frames
    lambda: _Global(shape=(2, 1)),                              # missing entities
    lambda: _Global(drop_unit=True),                            # missing attribute
])
def test_malformed_global_artifacts_are_refused_with_a_typed_error(monkeypatch, make):
    with pytest.raises(UnsupportedInputError):
        open_global(monkeypatch, make())


def test_non_real_stored_dtype_is_refused(monkeypatch):
    with pytest.raises(UnsupportedInputError):
        open_global(monkeypatch, _Global(), dtype=np.complex128)


def test_rank_one_and_rank_three_patch_fields_are_refused():
    for shape in ((2,), (2, 3, 6)):
        ds = _Dataset([0.0, 1.0])
        ds.shape = shape
        with pytest.raises(UnsupportedInputError):
            init_patch(ds, None, 2)


def test_non_finite_stored_samples_are_refused_when_read():
    source = ArtifactStressSource.__new__(ArtifactStressSource)
    source._frame_runs = (slice(0, 1),)
    source._field_shapes = {}
    source._file = Path("x.anyres.h5")
    source._cancelled = None
    source._sample_shapes = {}
    handle = {"fields/k/values": np.array([[np.nan, 1.0]])}
    with pytest.raises(art.ExtractionError):
        source._read(handle, "k", np.array([0, 1]), mean_last=True)


def test_read_result_names_the_last_cause_when_every_result_fails(monkeypatch, tmp_path):
    _files, _opened = _project(monkeypatch, tmp_path, [UnsupportedInputError("old bad"), UnsupportedInputError("new bad")])
    with pytest.raises(UnsupportedInputError, match="newest failed with UnsupportedInputError: new bad; oldest failed with UnsupportedInputError: old bad"):
        art.read_result(tmp_path)


# -- deferred read guard --------------------------------------------------------------------

def reader(tmp_path, stored, validated=(3, 4), dtype=float, name="r.anyres.h5"):
    import h5py

    file = tmp_path / name
    with h5py.File(file, "w") as handle:
        handle.create_dataset("fields/k/values", data=np.ones(stored, dtype=dtype))
    source = ArtifactStressSource.__new__(ArtifactStressSource)
    source._file = file
    source._frame_runs = (slice(0, validated[0]),)
    source._cancelled = None
    source._sample_shapes = {}
    source._field_shapes = {"k": tuple(validated)}
    return source, file


@pytest.mark.parametrize("stored,dtype", [
    ((1, 4), float),            # truncated to one frame: used to broadcast
    ((3, 2), float),            # fewer entities than validated
    ((3, 4, 2, 2), float),      # rank changed
    ((3, 4), complex),          # not real floating point
    ((3, 4), np.int32),
])
def test_deferred_read_refuses_a_field_that_changed_since_open(tmp_path, stored, dtype):
    import h5py

    source, file = reader(tmp_path, stored, dtype=dtype)
    with h5py.File(file, "r") as handle:
        with pytest.raises(UnsupportedInputError):
            source._read(handle, "k", np.arange(4), mean_last=True)


def test_deferred_read_of_a_well_formed_field_is_unchanged(tmp_path):
    import h5py

    source, file = reader(tmp_path, (3, 4))
    with h5py.File(file, "r") as handle:
        assert source._read(handle, "k", np.array([0, 2]), mean_last=True).shape == (3, 2)
        assert source._read(handle, "k", np.array([1, 2, 3]), mean_last=True).shape == (3, 3)


def test_deferred_read_wraps_missing_dataset_and_unreadable_files(tmp_path):
    import h5py

    source, file = reader(tmp_path, (3, 4))
    with h5py.File(file, "r") as handle:
        with pytest.raises(UnsupportedInputError):
            source._read(handle, "absent", np.arange(2), mean_last=True)
    source._file = tmp_path / "gone.anyres.h5"
    source.n_frames = 3
    with pytest.raises(UnsupportedInputError):
        source.element_stress(np.array([0]), "top")


def test_deferred_read_wraps_missing_per_frame_sample_metadata(tmp_path):
    import h5py

    source, file = reader(tmp_path, (3, 4, 2), validated=(3, 4, 2))
    source._sample_shapes = {"k": ({}, {})}               # no entry for any frame
    source._frame_ids = (0, 1, 2)
    source.mesh = SimpleNamespace(element_ids=np.arange(4))
    with h5py.File(file, "r") as handle:
        with pytest.raises(UnsupportedInputError):
            source._read(handle, "k", np.arange(4), mean_last=True)
