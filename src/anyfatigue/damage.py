"""Palmgren-Miner damage: from cycle counts and from a Weibull distribution.

Everything works in MPa.  Stress concentration and thickness effects enter
through :meth:`anyfatigue.sn_curves.SNCurve.effective_range`; the damage
functions never apply them a second time.

References (DNV-RP-C203, April 2010):

* (5.1.1) ``Q(dS) = exp(-(dS / q)**h)``, exceedance probability.
* (5.1.2) ``q = dS0 / (ln n0)**(1/h)``, ``dS0`` the largest range in ``n0`` cycles.
* (5.1.3) ``D = nu0 Td / a * q**m * Gamma(1 + m/h)`` for a one-slope curve.
* (D.13-1) the two-slope form with incomplete gamma functions.
* (D.13-2/3) the same damage by direct integration of the Weibull density.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import integrate
from scipy.special import gammainc, gammaincc, gammaln

from .errors import InputError
from .sn_curves import SNCurve

__all__ = [
    "SECONDS_PER_YEAR",
    "MinerResult",
    "design_check",
    "miner_damage",
    "weibull_damage",
    "weibull_damage_by_integration",
    "weibull_scale",
    "non_welded_mean_stress_factor",
]

# 365 days, as ANYstructure and ANYtimeseries use.
SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class MinerResult:
    """Cycle-by-cycle damage."""

    total: float
    per_cycle: np.ndarray
    cycles_to_failure: np.ndarray
    effective_ranges: np.ndarray


def miner_damage(
    ranges,
    counts,
    curve: SNCurve,
    *,
    scf: float = 1.0,
    thickness_mm: Optional[float] = None,
    scale: float = 1.0,
) -> MinerResult:
    """``D = scale * sum(n_i / N_i)`` over a cycle table.

    ``scale`` extrapolates a short record to its exposure (``T_exposure /
    T_record``).  Raw, unbinned cycles are preferred: rebinning shifts cycles
    to bin mid-points, which changes the damage.
    """

    r = np.asarray(ranges, dtype=float)
    c = np.asarray(counts, dtype=float)
    if r.shape != c.shape or r.ndim != 1:
        raise InputError("ranges and counts must be 1-D arrays of the same length")
    if not (math.isfinite(scale) and scale >= 0.0):
        raise InputError("damage scale must be finite and non-negative")
    if np.any(c < 0.0) or np.any(~np.isfinite(c)):
        raise InputError("cycle counts must be finite and non-negative")
    effective = curve.effective_range(r, scf=scf, thickness_mm=thickness_mm)
    if r.size == 0:
        empty = np.zeros(0)
        return MinerResult(0.0, empty, empty, empty)
    n_fail = np.asarray(curve.cycles_to_failure(effective), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cycle = np.where(np.isfinite(n_fail), scale * c / n_fail, 0.0)
    return MinerResult(float(per_cycle.sum()), per_cycle, n_fail, effective)


def weibull_scale(range_at_n0: float, shape_h: float, n0: float) -> float:
    """Weibull scale parameter ``q = dS0 / (ln n0)**(1/h)`` (5.1.2)."""

    _check_weibull(shape_h)
    if not (math.isfinite(range_at_n0) and range_at_n0 >= 0.0):
        raise InputError("stress range dS0 must be finite and non-negative")
    if not (math.isfinite(n0) and n0 > 1.0):
        raise InputError("n0 must exceed 1 so that ln(n0) is positive")
    return range_at_n0 / math.log(n0) ** (1.0 / shape_h)


def _check_weibull(shape_h: float) -> None:
    if not (math.isfinite(shape_h) and shape_h > 0.0):
        raise InputError(f"Weibull shape parameter must be positive, got {shape_h!r}")


def weibull_damage(
    curve: SNCurve,
    *,
    scale_q: float,
    shape_h: float,
    n_cycles: float,
    scf: float = 1.0,
    thickness_mm: Optional[float] = None,
) -> float:
    """Closed-form damage for a Weibull long-term stress range distribution.

    ``n_cycles`` is ``nu0 * Td``, the number of stress cycles in the period.
    ``scale_q`` is the Weibull scale of the *applied* stress range; the SCF and
    thickness factor are applied to it here (a scale factor scales ``q``).
    """

    _check_weibull(shape_h)
    if not (math.isfinite(scale_q) and scale_q >= 0.0):
        raise InputError("Weibull scale must be finite and non-negative")
    if not (math.isfinite(n_cycles) and n_cycles >= 0.0):
        raise InputError("number of cycles must be finite and non-negative")
    q = float(curve.effective_range(scale_q, scf=scf, thickness_mm=thickness_mm))
    if q == 0.0 or n_cycles == 0.0:
        return 0.0
    if not curve.bilinear:
        return n_cycles * _gamma_term(curve.m1, curve.a1, q, shape_h, "full")
    # x = (S1 / q)**h, evaluated through the logarithm and capped so that a
    # very large shape parameter saturates the regularised gamma functions
    # (to 0 or 1) instead of overflowing.
    x = math.exp(min(shape_h * math.log(curve.stress_switch / q), 700.0))
    # Ranges above S1 follow slope 1 (upper incomplete gamma), those below
    # follow slope 2 (lower incomplete gamma): (D.13-1).
    upper = _gamma_term(curve.m1, curve.a1, q, shape_h, "upper", x)
    lower = _gamma_term(curve.m2, curve.a2, q, shape_h, "lower", x)
    return n_cycles * (upper + lower)


def _gamma_term(m, a, q, h, kind, x=0.0) -> float:
    """``q^m / a * Gamma(1+m/h)`` times a regularised incomplete-gamma ratio.

    ``kind`` is ``"full"`` (no ratio), ``"upper"`` (``Q(1+m/h, x)``) or
    ``"lower"`` (``P(1+m/h, x)``).
    """

    shape = 1.0 + m / h
    if kind == "full":
        regularised = 1.0
    elif kind == "lower":
        regularised = gammainc(shape, x)
    elif kind == "upper":
        regularised = gammaincc(shape, x)
    else:  # pragma: no cover - internal misuse
        raise ValueError(kind)
    if regularised == 0.0:
        return 0.0
    log_value = m * math.log(q) - math.log(a) + gammaln(shape) + math.log(regularised)
    return math.exp(log_value)


def weibull_damage_by_integration(
    curve: SNCurve,
    *,
    scale_q: float,
    shape_h: float,
    n_cycles: float,
    scf: float = 1.0,
    thickness_mm: Optional[float] = None,
    upper_range: Optional[float] = None,
) -> float:
    """Damage by direct integration of the Weibull density (D.13-2, D.13-3).

    Independent of the incomplete-gamma form above, so the two cross-check each
    other.  ``upper_range`` truncates the distribution at a largest range (the
    printed (D.13-2) integrates to ``dS0``); the default integrates to
    effectively infinity, which is what (D.13-1) represents.
    """

    _check_weibull(shape_h)
    q = float(curve.effective_range(scale_q, scf=scf, thickness_mm=thickness_mm))
    if q == 0.0 or n_cycles == 0.0:
        return 0.0
    h = shape_h

    def density(s: float) -> float:
        if s <= 0.0:
            return 0.0
        z = s / q
        return (h / q) * z ** (h - 1.0) * math.exp(-(z ** h))

    def integrand(s: float) -> float:
        if s <= 0.0:
            return 0.0
        return density(s) / float(curve.cycles_to_failure(s))

    s_max = q * 80.0 ** (1.0 / h) if upper_range is None else float(upper_range)
    if s_max <= 0.0:
        return 0.0
    knots = [0.0]
    if curve.bilinear and 0.0 < curve.stress_switch < s_max:
        knots.append(curve.stress_switch)
    knots.append(s_max)
    total = 0.0
    for low, high in zip(knots[:-1], knots[1:]):
        # Break each piece at the geometric centre so the quadrature sees both
        # the small-range tail (m ~ 3-5 weights it little) and the bulk.
        edges = np.unique(
            np.concatenate(([low], np.geomspace(max(low, high * 1e-6), high, 24), [high]))
        )
        for a, b in zip(edges[:-1], edges[1:]):
            value, _error = integrate.quad(integrand, a, b, limit=200, epsabs=0.0, epsrel=1e-10)
            total += value
    return n_cycles * total


def non_welded_mean_stress_factor(maximum, minimum):
    """Stress-range reduction ``f_m`` for non-welded regions, equation (2.5.1).

    ``f_m = (s_t + 0.6 s_c) / (s_t + s_c)`` with ``s_t`` the largest tension
    and ``s_c`` the largest compression magnitude in the cycle.  Applies only
    to base material not affected by residual stresses from welding, never to
    welded details.  A cycle without compression has ``f_m = 1``.
    """

    smax = np.asarray(maximum, dtype=float)
    smin = np.asarray(minimum, dtype=float)
    tension = np.maximum(smax, 0.0)
    compression = np.maximum(-smin, 0.0)
    total = tension + compression
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(total > 0.0, (tension + 0.6 * compression) / total, 1.0)
    return factor if factor.ndim else float(factor)


def design_check(damage: float, dff: float) -> dict:
    """DNV criterion ``D * DFF <= 1``: usage factor, pass flag and life margin."""

    if not (math.isfinite(dff) and dff > 0.0):
        raise InputError("design fatigue factor must be positive")
    if not (math.isfinite(damage) and damage >= 0.0):
        raise InputError("damage must be finite and non-negative")
    usage = damage * dff
    return {"damage": damage, "dff": dff, "usage": usage, "passed": usage <= 1.0}
