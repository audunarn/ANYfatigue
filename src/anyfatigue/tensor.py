"""Declared tensor-component availability, independent of numerical zero values.

All declarations refer to physical stresses in the source's fixed GLOBAL axes.
An absent component is not an assumed zero. Array constructors take six explicit
components; importers must retain what their input actually supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .errors import InputError

TENSOR_ORDER = ("xx", "yy", "zz", "xy", "yz", "xz")
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def _components(values, name):
    if isinstance(values, str):
        raise InputError(f"{name} must be a sequence of component names, not text")
    try:
        result = tuple(values)
    except TypeError:
        raise InputError(f"{name} must be a sequence of component names") from None
    if any(not isinstance(x, str) or x not in TENSOR_ORDER for x in result):
        raise InputError(f"{name} must contain only {TENSOR_ORDER}")
    if len(set(result)) != len(result):
        raise InputError(f"{name} must not repeat components")
    return tuple(c for c in TENSOR_ORDER if c in result)


@dataclass(frozen=True)
class TensorCompleteness:
    """Availability in global axes, shared by all locations/cases/surfaces.

    Use separate sources when availability differs. An empty declaration is
    unknown, as for a legacy file without component metadata. ``basis`` explains
    any assumed zeros; declarations are not inferred from the values themselves.
    """
    provided: Tuple[str, ...] = ()
    assumed_zero: Tuple[str, ...] = ()
    basis: str = ""

    def __post_init__(self):
        provided = _components(self.provided, "provided")
        zero = _components(self.assumed_zero, "assumed_zero")
        if set(provided) & set(zero):
            raise InputError("provided and assumed-zero components must be disjoint")
        if not isinstance(self.basis, str) or (zero and not self.basis.strip()):
            raise InputError("assumed-zero components require a non-empty engineering basis")
        object.__setattr__(self, "provided", provided)
        object.__setattr__(self, "assumed_zero", zero)

    @property
    def missing(self):
        return tuple(c for c in TENSOR_ORDER if c not in self.provided + self.assumed_zero)

    @property
    def complete(self):
        return not self.missing

    def to_dict(self):
        return {"frame": "global", "provided": list(self.provided),
                "assumed_zero": list(self.assumed_zero), "basis": self.basis}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or data.get("frame") != "global":
            raise InputError("tensor completeness requires an explicit global frame")
        if set(data) - {"frame", "provided", "assumed_zero", "basis"}:
            raise InputError("unknown tensor-completeness fields")
        if "provided" not in data:
            raise InputError("tensor completeness must declare provided components")
        return cls(data["provided"], data.get("assumed_zero", ()), data.get("basis", ""))

    def validate_values(self, values):
        for component in self.assumed_zero:
            if np.any(np.asarray(values)[..., TENSOR_ORDER.index(component)] != 0.0):
                raise InputError(f"component {component} is declared zero but contains non-zero stresses")

    def require(self, components, *, context="fatigue assessment"):
        missing = [c for c in components if c in self.missing]
        if missing:
            raise InputError(
                f"{context}: missing tensor components {', '.join(missing)} in global axes; "
                "export them or explicitly declare and justify assumed-zero components"
            )


def completeness_of(source):
    """Unknown legacy/third-party declarations are never promoted to complete."""
    value = getattr(source, "tensor_completeness", None)
    if value is None:
        return TensorCompleteness()
    if not isinstance(value, TensorCompleteness):
        raise InputError("source.tensor_completeness must be TensorCompleteness")
    return value


def require_in_plane_tensor(source, normal, *, context="multidirectional fatigue"):
    """Require the components used by projection onto this physical surface.

    A declared xx/yy/xy tensor suffices in the global xy plane. An inclined
    surface generally needs all six GLOBAL components. Unknown transverse
    components remain unknown in the provenance; this does not assume them zero.
    """
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.all(np.isfinite(n)) or np.linalg.norm(n) == 0:
        raise InputError("tensor completeness needs a finite non-zero surface normal")
    n = n / np.linalg.norm(n)
    projection = np.eye(3) - np.outer(n, n)
    needed = []
    for c, (i, j) in zip(TENSOR_ORDER, _PAIRS):
        unit = np.zeros((3, 3))
        unit[i, j] = unit[j, i] = 1.0
        # Only exact zeros remove a dependency. A tiny inclination must not
        # silently discard an unknown component of unbounded magnitude.
        if np.any(projection @ unit @ projection != 0.0):
            needed.append(c)
    completeness_of(source).require(needed, context=context)
