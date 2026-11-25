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
        self.shape = (len(frames) if n_frames is None else n_frames, n_nodes, 3)
        self.unit, self.basis = unit, basis

    def table(self, key):
        return np.array([1, 2, 3])

    def field(self, key):
        return SimpleNamespace(
            shape=self.shape,
            descriptor=SimpleNamespace(frames=self.frames, provenance={}, unit=self.unit, basis=self.basis),
        )


KEYS = {_patch_key(c, s) for c in TENSOR_ORDER for s in SURFACES}
FAKE = SimpleNamespace(mesh=SimpleNamespace(n_nodes=3, node_ids=np.array([1, 2, 3])))


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
    InputError("1 load-case names for 2 frames"),
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
