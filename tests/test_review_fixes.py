"""Regression tests for plate-scoped lazy assessment review findings (no ANYfem stack needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from anyfatigue.anyfem_adapter import _directors
from anyfatigue.anyfem_artifacts import ArtifactStressSource, SURFACES, TENSOR_ORDER, _patch_key
from anyfatigue.errors import UnsupportedInputError


def test_eager_directors_fall_back_to_the_native_reference_director():
    native = np.array([0.0, 0.0, -1.0])
    element = SimpleNamespace(native_reference_directors=lambda mesh: [native, native])
    shape = SimpleNamespace(built=SimpleNamespace(fe_model=SimpleNamespace(
        mesh=SimpleNamespace(elements={1: element, 2: SimpleNamespace()}))))
    recovery = SimpleNamespace(element_stresses={1: {}, 2: {"physical_director": [0, 1, 0]}})
    out = _directors(recovery, shape)
    assert np.allclose(out[1], native) and np.allclose(out[2], [0, 1, 0])


class _Field:
    def __init__(self, frames):
        self.descriptor = SimpleNamespace(frames=np.asarray(frames, float), provenance={})


class _Dataset:
    def __init__(self, frames):
        self._frames = frames

    def table(self, key):
        return np.array([1, 2, 3])

    def field(self, key):
        return _Field(self._frames)


def test_patch_fields_with_different_frame_times_are_refused():
    keys = {_patch_key(c, s) for c in TENSOR_ORDER for s in SURFACES}
    fake = SimpleNamespace(mesh=SimpleNamespace(n_nodes=3, node_ids=np.array([1, 2, 3])))
    with pytest.raises(UnsupportedInputError):
        ArtifactStressSource._init_patch(fake, _Dataset([1.0, 0.0]), keys, None, True, np.array([0.0, 1.0]))
    ArtifactStressSource._init_patch(fake, _Dataset([0.0, 1.0]), keys, None, True, np.array([0.0, 1.0]))
