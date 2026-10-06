"""Numbers behind the GUI plots, computed without Qt.

The "design-life spectrum against the S-N curve" diagram puts both on the same
axes: stress range *entering the curve* (after SCF and thickness) against
number of cycles.  The S-N curve gives the cycles to failure at a range; the
spectrum gives the cycles at or above that range over the design life.  Damage
is the area where the two meet, so a spectrum lying left of the curve is safe.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .errors import InputError
from .results import AssessmentResult, PartResult
from .sn_curves import SNCurve

__all__ = ["damage_along_line", "design_life_spectrum", "sn_curve_line", "sn_curve_between_cycles"]


def sn_curve_line(curve: SNCurve, low: float, high: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """``(cycles, effective range)`` along the curve between two ranges (MPa)."""

    if not (low > 0.0 and high > low):
        raise InputError("S-N plot limits must satisfy 0 < low < high")
    s = np.geomspace(low, high, n)
    return np.asarray(curve.cycles_to_failure(s), dtype=float), s


def sn_curve_between_cycles(
    curve: SNCurve, n_low: float, n_high: float, n: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """The curve between two cycle numbers, ``n_low < n_high`` (effective ranges)."""

    if not (0.0 < n_low < n_high):
        raise InputError("S-N plot limits must satisfy 0 < n_low < n_high")
    return sn_curve_line(curve, curve.allowable_range(n_high), curve.allowable_range(n_low), n)


def design_life_spectrum(
    result: AssessmentResult, part: PartResult
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(cycles at or above, effective range)`` over the design life, or None."""

    s = result.settings
    curve = SNCurve.from_dict(s["curve"])
    scf, thickness = s["scf"], s["thickness_mm"]
    if result.method == "time_series":
        if part.cycles is None or len(part.cycles) == 0:
            return None
        eff = np.asarray(curve.effective_range(part.cycles.ranges, scf=scf, thickness_mm=thickness))
        order = np.argsort(-eff)
        counts = part.cycles.counts[order] * part.values["exposure_scale"]
        return np.cumsum(counts), eff[order]
    q = part.values["weibull_scale_q_mpa"]
    h = part.values["weibull_shape_h"]
    n_total = part.values["cycles_in_design_life"]
    if q <= 0.0 or n_total <= 0.0:
        return None
    q_eff = float(curve.effective_range(q, scf=scf, thickness_mm=thickness))
    # From the range that 80 % of the cycles exceed (below it the curve is flat at the
    # total cycle count and says nothing) up to where one cycle is exceeded.
    s_max = q_eff * max(math.log(max(n_total, 2.0)), 1.0) ** (1.0 / h)
    s_min = q_eff * (-math.log(0.8)) ** (1.0 / h)
    ranges = np.geomspace(min(s_min, 0.5 * s_max), s_max, 200)
    return n_total * np.exp(-((ranges / q_eff) ** h)), ranges


def damage_along_line(result: AssessmentResult) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per surface: ``(chainage, damage)`` of the line sample points, in order."""

    out: Dict[str, list] = {}
    for p in result.points:
        if p.chainage is not None:
            out.setdefault(f"{p.label.split(':')[0]} {p.surface}", []).append((p.chainage, p.damage))
    return {k: (np.array([c for c, _ in v]), np.array([d for _, d in v])) for k, v in out.items()}
