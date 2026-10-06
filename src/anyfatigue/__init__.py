"""ANYfatigue: fatigue assessment to DNV rules on ANYfem stresses.

Two methods share one S-N and damage core:

* :mod:`anyfatigue.simplified` - DNV-RP-C203 section 5, a Weibull long-term
  stress-range distribution scaled from the stress of one load case.
* :mod:`anyfatigue.timeseries` - rainflow counting of a quasi-static stress
  history built from a series of load cases.

Importing the package never loads Qt, ANYfem or h5py.
"""

from __future__ import annotations

from .errors import (
    ExtractionError,
    FatigueError,
    InputError,
    MissingDependencyError,
    UnsupportedInputError,
)
from .sn_curves import SNCurve, get_curve

__version__ = "0.1.0"

__all__ = [
    "ExtractionError",
    "FatigueError",
    "InputError",
    "MissingDependencyError",
    "SNCurve",
    "UnsupportedInputError",
    "__version__",
    "get_curve",
]
