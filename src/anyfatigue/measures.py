"""From a surface stress tensor to the scalar a fatigue curve needs.

A shell surface stress is a 3-D tensor in global axes.  Fatigue cracks grow in
the plate surface, so only the in-plane part matters: the tensor is projected
onto the plane normal to the element normal ``n`` and then resolved against the
weld.  With ``t`` the weld direction in the surface and ``e = n x t`` the
in-plane direction normal to the weld (DNV-RP-C203 figures 2-3 and 2-4)::

    sigma_perp = e.S.e      sigma_par = t.S.t      tau_par = e.S.t

Measures (``StressMeasure``):

``normal_to_weld``
    signed ``sigma_perp(t)``; needs a weld direction.
``principal_abs_max``
    the in-plane principal stress of larger magnitude, signed; for base
    material or an unknown weld direction.  It can jump between the two
    principal values in non-proportional loading, so prefer ``normal_to_weld``
    when the weld direction is known.
``component:<xx|yy|zz|xy|yz|xz>``
    one global tensor component, as given.
``dnv_effective``
    the effective hot-spot stress *range* of DNV-RP-C203 section 4.3.4,
    meaningful for ranges only, hence for the simplified method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .errors import InputError, UnsupportedInputError
from .model import TENSOR_ORDER

__all__ = [
    "MEASURES",
    "StressMeasure",
    "effective_hot_spot_range",
    "in_plane_components",
    "in_plane_principal",
    "plane_basis",
]

MEASURES = ("normal_to_weld", "principal_abs_max", "component", "dnv_effective")


def _vector(value, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise InputError(f"{name} must be a finite 3-vector")
    return v


def plane_basis(
    normal, weld_direction=None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Unit ``(n, t, e)``: normal, weld direction in the surface, in-plane normal to weld.

    ``weld_direction`` need not be exactly in the surface: it is projected onto
    it.  It must not be parallel to the normal.
    """

    n = _vector(normal, "normal")
    length = np.linalg.norm(n)
    if length == 0.0:
        raise InputError("surface normal has zero length")
    n = n / length
    if weld_direction is None:
        return n, None, None
    w = _vector(weld_direction, "weld direction")
    t = w - (w @ n) * n
    norm = np.linalg.norm(t)
    if norm < 1e-9 * max(np.linalg.norm(w), 1e-300):
        raise InputError("weld direction is parallel to the surface normal")
    t = t / norm
    return n, t, np.cross(n, t)


def _matrix(tensors: np.ndarray) -> np.ndarray:
    a = np.asarray(tensors, dtype=float)
    if a.shape[-1] != 6:
        raise InputError("a stress tensor needs 6 components (xx yy zz xy yz xz)")
    m = np.empty(a.shape[:-1] + (3, 3))
    m[..., 0, 0], m[..., 1, 1], m[..., 2, 2] = a[..., 0], a[..., 1], a[..., 2]
    m[..., 0, 1] = m[..., 1, 0] = a[..., 3]
    m[..., 1, 2] = m[..., 2, 1] = a[..., 4]
    m[..., 0, 2] = m[..., 2, 0] = a[..., 5]
    return m


def in_plane_components(tensors, normal, weld_direction):
    """``(sigma_perp, sigma_par, tau_par)`` for tensors shaped ``(..., 6)``."""

    _n, t, e = plane_basis(normal, weld_direction)
    if t is None:
        raise InputError("a weld direction is required to resolve stress against the weld")
    m = _matrix(tensors)
    s_perp = np.einsum("i,...ij,j->...", e, m, e)
    s_par = np.einsum("i,...ij,j->...", t, m, t)
    tau = np.einsum("i,...ij,j->...", e, m, t)
    return s_perp, s_par, tau


def in_plane_principal(tensors, normal):
    """In-plane principal stresses ``(s1 >= s2)`` for tensors shaped ``(..., 6)``."""

    n, _t, _e = plane_basis(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    a = np.cross(n, helper)
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    m = _matrix(tensors)
    saa = np.einsum("i,...ij,j->...", a, m, a)
    sbb = np.einsum("i,...ij,j->...", b, m, b)
    sab = np.einsum("i,...ij,j->...", a, m, b)
    centre = 0.5 * (saa + sbb)
    radius = np.sqrt((0.5 * (saa - sbb)) ** 2 + sab ** 2)
    return centre + radius, centre - radius


def effective_hot_spot_range(
    d_perp, d_par, d_tau, *, alpha: float = 1.0, method: str = "A"
):
    """Effective hot-spot stress range, DNV-RP-C203 (4.3.1)-(4.3.4).

    ``d_perp``, ``d_par`` and ``d_tau`` are the changes (a range, or a case
    minus a reference) of the stress normal to the weld, parallel to the weld
    and the in-plane shear.  ``method`` ``"A"`` (extrapolated hot-spot stress)
    uses equation (4.3.1); ``"B"`` (stress at 0.5 t) multiplies all three terms
    by 1.12, equation (4.3.4).  ``alpha`` is 0.72, 0.80 or 0.90 for C, C1 or C2
    details with stress parallel to the weld and 1.0 otherwise.
    """

    if method not in ("A", "B"):
        raise InputError("hot-spot method must be 'A' or 'B'")
    if not (math.isfinite(alpha) and alpha > 0.0):
        raise InputError("alpha must be positive")
    p = np.asarray(d_perp, dtype=float)
    q = np.asarray(d_par, dtype=float)
    s = np.asarray(d_tau, dtype=float)
    mean = 0.5 * (p + q)
    radius = 0.5 * np.sqrt((p - q) ** 2 + 4.0 * s ** 2)
    first, second = mean + radius, mean - radius
    effective = np.maximum.reduce([
        np.sqrt(p ** 2 + 0.81 * s ** 2),
        alpha * first,
        alpha * np.abs(second),
    ])
    factor = 1.12 if method == "B" else 1.0
    result = factor * effective
    return result if result.ndim else float(result)


@dataclass(frozen=True)
class StressMeasure:
    """Which scalar to take from the surface stress tensor."""

    kind: str = "normal_to_weld"
    component: Optional[str] = None
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in MEASURES:
            raise InputError(f"unknown stress measure {self.kind!r}; use one of {MEASURES}")
        if self.kind == "component":
            if self.component not in TENSOR_ORDER:
                raise InputError(f"component must be one of {TENSOR_ORDER}")
        elif self.component is not None:
            raise InputError("only the 'component' measure takes a component name")

    # -- text form used by files and the GUI ------------------------------
    @classmethod
    def parse(cls, text: str) -> "StressMeasure":
        if text.startswith("component:"):
            return cls("component", text.split(":", 1)[1])
        return cls(text)

    def to_text(self) -> str:
        return f"component:{self.component}" if self.kind == "component" else self.kind

    @property
    def needs_weld_direction(self) -> bool:
        return self.kind in ("normal_to_weld", "dnv_effective")

    # ------------------------------------------------------------------
    def signed_series(self, tensors, normal, weld_direction=None) -> np.ndarray:
        """Signed scalar history for rainflow counting; shape ``(n_cases,)``."""

        if self.kind == "dnv_effective":
            raise UnsupportedInputError(
                "the DNV effective hot-spot stress (4.3.4) is defined for stress "
                "ranges, not for a signed history; use it with the simplified "
                "method, or choose 'normal_to_weld' or 'principal_abs_max' for "
                "a time series"
            )
        t = np.asarray(tensors, dtype=float)
        if self.kind == "component":
            return t[..., TENSOR_ORDER.index(self.component)]
        if self.kind == "normal_to_weld":
            return in_plane_components(t, normal, weld_direction)[0]
        s1, s2 = in_plane_principal(t, normal)
        return np.where(np.abs(s1) >= np.abs(s2), s1, s2)

    def range_of(self, delta_tensors, normal, weld_direction=None, *, method: str = "A") -> np.ndarray:
        """Unsigned stress range from the *change* of the tensor between two states."""

        d = np.asarray(delta_tensors, dtype=float)
        if self.kind == "dnv_effective":
            p, q, s = in_plane_components(d, normal, weld_direction)
            return effective_hot_spot_range(p, q, s, alpha=self.alpha, method=method)
        if self.kind == "principal_abs_max":
            s1, s2 = in_plane_principal(d, normal)
            return np.maximum(np.abs(s1), np.abs(s2))
        return np.abs(self.signed_series(d, normal, weld_direction))
