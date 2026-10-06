"""Simplified fatigue analysis, DNV-RP-C203 section 5.

The long-term stress-range distribution is a two-parameter Weibull distribution
(5.1.1) whose scale ``q`` is fixed by one stress range ``dS0``, the largest range
in ``n0`` cycles (5.1.2).  Here ``dS0`` is taken from the stress of **one load
case** per load condition, read from ANYfem at every sample point:

``range = range_factor * measure(stress(case))``       (default factor 2: the
    load case is the stress *amplitude* of a fully reversed load), or
``range = measure(stress(case) - stress(reference_case))``  when two states bound
    the cycle (e.g. a loaded and an unloaded condition).

Damage per condition follows (5.1.3) or, for the two-slope curves, (D.13-1), and
is weighted by the fraction of the design life spent in that condition, as in
ANYstructure's simplified fatigue.  ``D * DFF <= 1`` is the criterion.

The measure is applied to the *range* (the change of tensor between two states),
which is why ``dnv_effective`` (4.3.4) is available here and not for a signed
history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from . import __version__
from .damage import SECONDS_PER_YEAR, weibull_damage, weibull_scale
from .errors import InputError
from .extract import SamplePoint, extract
from .measures import StressMeasure
from .model import StressSource
from .results import AssessmentResult, PartResult, PointResult
from .sn_curves import STANDARD_EDITION, SNCurve, edition_warnings

__all__ = ["LoadCondition", "SimplifiedSettings", "assess_simplified"]


@dataclass(frozen=True)
class LoadCondition:
    """One loading condition of the long-term distribution.

    ``fraction`` is the share of the design life spent in it, ``period_s`` the
    average zero-crossing period (``nu0 = 1 / period``) and ``weibull_h`` the
    Weibull shape parameter.
    """

    name: str
    case: str
    weibull_h: float
    period_s: float
    fraction: float = 1.0
    reference_case: Optional[str] = None
    range_factor: float = 2.0

    def __post_init__(self) -> None:
        if not self.name:
            raise InputError("a load condition needs a name")
        if not (math.isfinite(self.weibull_h) and self.weibull_h > 0.0):
            raise InputError(f"{self.name}: Weibull shape must be positive")
        if not (math.isfinite(self.period_s) and self.period_s > 0.0):
            raise InputError(f"{self.name}: period must be positive")
        if not (math.isfinite(self.fraction) and 0.0 <= self.fraction <= 1.0):
            raise InputError(f"{self.name}: fraction of design life must be in [0, 1]")
        if not (math.isfinite(self.range_factor) and self.range_factor > 0.0):
            raise InputError(f"{self.name}: range factor must be positive")

    def to_dict(self) -> dict:
        return {
            "name": self.name, "case": self.case, "weibull_h": self.weibull_h,
            "period_s": self.period_s, "fraction": self.fraction,
            "reference_case": self.reference_case, "range_factor": self.range_factor,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoadCondition":
        return cls(**data)


@dataclass(frozen=True)
class SimplifiedSettings:
    curve: SNCurve
    thickness_mm: float
    design_life_years: float
    dff: float
    n0: float
    conditions: Tuple[LoadCondition, ...]
    scf: float = 1.0
    measure: StressMeasure = StressMeasure("normal_to_weld")

    def __post_init__(self) -> None:
        if not (math.isfinite(self.thickness_mm) and self.thickness_mm > 0.0):
            raise InputError("thickness must be positive (mm)")
        if not (math.isfinite(self.design_life_years) and self.design_life_years > 0.0):
            raise InputError("design life must be positive (years)")
        if not (math.isfinite(self.dff) and self.dff > 0.0):
            raise InputError("design fatigue factor must be positive")
        if not (math.isfinite(self.n0) and self.n0 > 1.0):
            raise InputError("n0 must exceed 1")
        if not (math.isfinite(self.scf) and self.scf > 0.0):
            raise InputError("SCF must be positive")
        conditions = tuple(self.conditions)
        if not conditions:
            raise InputError("at least one load condition is required")
        names = [c.name for c in conditions]
        if len(set(names)) != len(names):
            raise InputError("load condition names must be unique")
        if sum(c.fraction for c in conditions) > 1.0 + 1e-9:
            raise InputError("the load condition fractions add up to more than 1")
        object.__setattr__(self, "conditions", conditions)

    def to_dict(self) -> dict:
        return {
            "curve": self.curve.to_dict(), "thickness_mm": self.thickness_mm,
            "design_life_years": self.design_life_years, "dff": self.dff, "n0": self.n0,
            "scf": self.scf, "measure": self.measure.to_text(),
            "alpha": self.measure.alpha,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimplifiedSettings":
        body = dict(data)
        measure = StressMeasure.parse(body.pop("measure"))
        alpha = body.pop("alpha", 1.0)
        measure = StressMeasure(measure.kind, measure.component, alpha)
        return cls(
            curve=SNCurve.from_dict(body.pop("curve")),
            conditions=tuple(LoadCondition.from_dict(c) for c in body.pop("conditions")),
            measure=measure,
            **body,
        )


def assess_simplified(
    source: StressSource,
    locations: Sequence,
    settings: SimplifiedSettings,
) -> AssessmentResult:
    """Simplified fatigue damage at every sample point of ``locations``."""

    if not locations:
        raise InputError("at least one stress location is required")
    case_rows = {}
    for condition in settings.conditions:
        case_rows[condition.name] = (
            source.case_index(condition.case),
            None if condition.reference_case is None
            else source.case_index(condition.reference_case),
        )
    points = [p for location in locations for p in extract(source, location)]
    warnings = list(edition_warnings(settings.curve))
    results = [_assess_point(p, settings, case_rows, warnings) for p in points]
    return AssessmentResult(
        method="simplified",
        points=tuple(results),
        settings=settings.to_dict(),
        provenance=_provenance(source),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _assess_point(
    point: SamplePoint, settings: SimplifiedSettings, case_rows, warnings: list
) -> PointResult:
    measure = settings.measure
    if measure.needs_weld_direction and point.weld_direction is None:
        raise InputError(
            f"{point.label}: the stress measure {measure.to_text()!r} needs a weld "
            "direction; give one on the location (or use 'principal_abs_max')"
        )
    if measure.kind == "dnv_effective" and point.readout == "direct":
        warnings.append(
            "dnv_effective (DNV-RP-C203 4.3.4) is defined for hot-spot stress but was "
            "applied to directly read-out stress"
        )
    method = "B" if point.readout == "hotspot_b" else "A"
    design_seconds = settings.design_life_years * SECONDS_PER_YEAR
    parts = []
    total = 0.0
    for condition in settings.conditions:
        row, ref_row = case_rows[condition.name]
        if ref_row is None:
            delta = condition.range_factor * point.tensors[row]
        else:
            delta = point.tensors[row] - point.tensors[ref_row]
        dS0 = float(measure.range_of(delta[None, :], point.normal, point.weld_direction,
                                     method=method)[0])
        q = weibull_scale(dS0, condition.weibull_h, settings.n0)
        n_cycles = condition.fraction * design_seconds / condition.period_s
        damage = weibull_damage(
            settings.curve, scale_q=q, shape_h=condition.weibull_h, n_cycles=n_cycles,
            scf=settings.scf, thickness_mm=settings.thickness_mm,
        )
        total += damage
        parts.append(PartResult(
            name=condition.name,
            damage=damage,
            values={
                "stress_range_n0_mpa": dS0,
                "effective_range_n0_mpa": float(settings.curve.effective_range(
                    dS0, scf=settings.scf, thickness_mm=settings.thickness_mm)),
                "weibull_scale_q_mpa": q,
                "weibull_shape_h": condition.weibull_h,
                "cycles_in_design_life": n_cycles,
            },
        ))
    life = math.inf if total <= 0.0 else settings.design_life_years / total
    usage = total * settings.dff
    return PointResult(
        label=point.label, kind=point.kind, surface=point.surface,
        xyz=tuple(float(v) for v in point.xyz), chainage=point.chainage,
        damage=total, life_years=life, usage=usage, passed=usage <= 1.0,
        parts=tuple(parts), notes=point.notes,
    )


def _provenance(source: StressSource) -> dict:
    return {
        "anyfatigue_version": __version__,
        "standard": STANDARD_EDITION,
        "method": "DNV-RP-C203 section 5 (simplified fatigue analysis), (5.1.2), (D.13-1)",
        "stress_unit": "MPa",
        "nodal_stress": source.nodal_origin,
        "source": dict(source.provenance),
        "cases": list(source.case_labels),
    }
