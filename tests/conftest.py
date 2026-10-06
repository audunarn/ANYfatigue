"""Test environment.

ANYfatigue consumes ANYtimeseries (rainflow) and, optionally, ANYfem.  In the
ecosystem's editable-sibling layout those are checkouts next to this one, so
when they are not installed they are put on ``sys.path`` for the test run only.
This never replaces an installed copy.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import uuid
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_WORKSPACE = _ROOT.parent

_SIBLINGS = (
    ("anyqats", _WORKSPACE / "ANYtimeseries"),
    ("anyfem", _WORKSPACE / "ANYfem" / "src"),
    ("anyloads", _WORKSPACE / "ANYloads" / "src"),       # a dependency of ANYfem since 0.4.x
)
for _module, _path in _SIBLINGS:
    if importlib.util.find_spec(_module) is None and _path.is_dir():
        sys.path.append(str(_path))


@pytest.fixture
def tmp_path():
    """A unique scratch directory inside the checkout.

    The default ``%TEMP%/pytest-of-<user>`` is not writable in every ANY
    workstation sandbox, so (as in the sibling repositories) scratch lives in a
    git-ignored ``.pytest_tmp_*`` folder that is removed after each test.
    """

    root = _ROOT / f".pytest_tmp_{uuid.uuid4().hex}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _importable(name: str) -> bool:
    """Whether ``import name`` really works (a package can be found yet fail to import
    because one of *its* dependencies is missing, e.g. a newer ANYfem on an older env)."""

    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


HAVE_ANYFEM = _importable("anyfem") and _importable("anysolver") and _importable("h5py")
HAVE_QT = _importable("PySide6.QtWidgets")


def pytest_collection_modifyitems(config, items):
    skip_fem = pytest.mark.skip(
        reason="ANYfem producer packages are not importable "
        "(owner: ANYfatigue; remove when CI installs the 'anyfem' extra)"
    )
    skip_qt = pytest.mark.skip(reason="PySide6 is not importable")
    for item in items:
        if "anyfem" in item.keywords and not HAVE_ANYFEM:
            item.add_marker(skip_fem)
        if "qt" in item.keywords and not HAVE_QT:
            item.add_marker(skip_qt)
