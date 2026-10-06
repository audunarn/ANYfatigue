#!/usr/bin/env python
"""Run the ANYfatigue desktop application straight from a checkout.

Point an IDE's Run button at this file, or::

    python run_gui.py
    python run_gui.py --stress stresses.npz
    python run_gui.py --history hotspot.csv
    python run_gui.py --example synthetic

``src`` is put on ``sys.path`` first, so this works in a fresh clone with nothing
of ANYfatigue installed.  The GUI needs PySide6 (``pip install PySide6``, or
``pip install -e ".[gui]"``) and rainflow counting from ANYtimeseries
(``pip install anytimes``).

Sibling checkouts are used only when the package is *not* installed: the
ecosystem's editable layout keeps ANYtimeseries and the ANYfem stack beside this
repository.  ANYfem is optional; without it the stress file, history and
synthetic example routes still work and only the ANYfem example is disabled.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    # Prepended so a checkout is what runs, not a stale installed copy: running
    # this file is a statement about *this* working tree.
    sys.path.insert(0, str(_SRC))

_WORKSPACE = _ROOT.parent
# (importable module, sibling source directory).  Appended, never prepended.
_SIBLINGS = (
    ("anyqats", _WORKSPACE / "ANYtimeseries"),          # rainflow counting (required)
    ("anyfem", _WORKSPACE / "ANYfem" / "src"),          # optional producer of stresses
    ("anysolver", _WORKSPACE / "ANYsolver" / "src"),
    ("anyloads", _WORKSPACE / "ANYloads" / "src"),
    ("anymaterial", _WORKSPACE / "ANYmaterial" / "src"),
    ("anygeometry", _WORKSPACE / "ANYgeometry" / "src"),
    ("anyfileio", _WORKSPACE / "ANYfileIO" / "src"),
    ("anymesher", _WORKSPACE / "ANYmesh" / "src"),
)
for _module, _path in _SIBLINGS:
    try:
        _missing = importlib.util.find_spec(_module) is None
    except (ImportError, ValueError):
        _missing = True
    if _missing and _path.is_dir() and str(_path) not in sys.path:
        sys.path.append(str(_path))

def main(argv=None) -> int:
    """Launch the GUI (Qt is imported here, not at package import)."""

    try:
        from anyfatigue.gui.window import main as launch  # noqa: E402 - after the path setup
    except ImportError as exc:  # PySide6 is an optional extra of ANYfatigue
        if "PySide6" not in str(exc):
            raise
        raise SystemExit(
            f"The ANYfatigue GUI needs PySide6 ({exc}).\n"
            'Install it with:  python -m pip install PySide6   (or  pip install -e ".[gui]")'
        )
    return launch(argv)


if __name__ == "__main__":
    raise SystemExit(main())
