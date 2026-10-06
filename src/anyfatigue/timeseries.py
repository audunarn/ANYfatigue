"""Quasi-static time-series fatigue by rainflow counting.

ANYfem solves a *series* of load cases (a wave, a sea state, a stepped
operation) that together form a quasi-static load history.  Each step has a
stress tensor at every sample point; the chosen :class:`StressMeasure` reduces it
to one signed stress history per point and surface.  That history is rainflow
counted (ANYtimeseries), each cycle is entered into the S-N curve and the
damage of the record is extrapolated to the share of the design life that the
record represents::

    D_block = (T_exposure / T_record) * sum(n_i / N_i)
    T_exposure = exposure_fraction * design_life

Blocks are independent records (for example sea states of a scatter diagram);
their damages add (Palmgren-Miner).  ``T_record`` is ``duration_s`` when given,
otherwise the span of the case times, last minus first.

The record is treated as non-repeating: its first and last samples start and
close half cycles.  For a record that genuinely repeats, close it (append the
first sample) so that its half cycles pair up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from . import __version__
from .damage import SECONDS_PER_YEAR, miner_damage, non_welded_mean_stress_factor
from .errors import InputError
from .extract import SamplePoint, extract
from .measures import StressMeasure
from .model import StressSource
from .rainflow import CycleTable, count_rainflow_cycles, rainflow_backend_version
from .results import AssessmentResult, PartResult, PointResult
from .sn_curves import STANDARD_EDITION, SNCurve, edition_warnings

__all__ = ["LoadBlock", "TimeSeriesSettings", "assess_time_series"]

MEAN_STRESS = ("none", "non_welded")


@dataclass(frozen=True)
class LoadBlock:
    """An ordered series of load cases and the share of life it represents."""

    name: str
    cases: Tuple[str, ...]
    exposure_fraction: float = 1.0
    duration_s: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise InputError("a load block needs a name")
        cases = tuple(str(c) for c in self.cases)
        if len(cases) < 2:
            raise InputError(f"block {self.name!r}: a time series needs at least two load cases")
        object.__setattr__(self, "cases", cases)
        if not (math.isfinite(self.exposure_fraction) and 0.0 <= self.exposure_fraction <= 1.0):
            raise InputError(f"block {self.name!r}: exposure fraction must be in [0, 1]")
        if self.duration_s is not None and not (
            math.isfinite(self.duration_s) and self.duration_s > 0.0
        ):
            raise InputError(f"block {self.name!r}: duration must be positive")

    def to_dict(self) -> dict:
        return {
            "name": self.name, "cases": list(self.cases),
            "exposure_fraction": self.exposure_fraction, "duration_s": self.duration_s,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoadBlock":
        return cls(**data)


@dataclass(frozen=True)
class TimeSeriesSettings:
    curve: SNCurve
    thickness_mm: float
    design_life_years: float
    dff: float
    blocks: Tuple[LoadBlock, ...]
    scf: float = 1.0
    measure: StressMeasure = StressMeasure("normal_to_weld")
    mean_stress: str = "none"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.thickness_mm) and self.thickness_mm > 0.0):
            raise InputError("thickness must be positive (mm)")
        if not (math.isfinite(self.design_life_years) and self.design_life_years > 0.0):
            raise InputError("design life must be positive (years)")
        if not (math.isfinite(self.dff) and self.dff > 0.0):
            raise InputError("design fatigue factor must be positive")
        if not (math.isfinite(self.scf) and self.scf > 0.0):
            raise InputError("SCF must be positive")
        if self.mean_stress not in MEAN_STRESS:
            raise InputError(f"mean_stress must be one of {MEAN_STRESS}")
        blocks = tuple(self.blocks)
        if not blocks:
            raise InputError("at least one load block is required")
        names = [b.name for b in blocks]
        if len(set(names)) != len(names):
            raise InputError("load block names must be unique")
        if sum(b.exposure_fraction for b in blocks) > 1.0 + 1e-9:
            raise InputError("the block exposure fractions add up to more than 1")
        if self.measure.kind == "dnv_effective":
            raise InputError(
                "the DNV effective hot-spot stress is defined for stress ranges; "
                "it cannot be used for a signed time history"
            )
        object.__setattr__(self, "blocks", blocks)

    def to_dict(self) -> dict:
        return {
            "curve": self.curve.to_dict(), "thickness_mm": self.thickness_mm,
            "design_life_years": self.design_life_years, "dff": self.dff,
            "scf": self.scf, "measure": self.measure.to_text(),
            "mean_stress": self.mean_stress,
            "blocks": [b.to_dict() for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TimeSeriesSettings":
        body = dict(data)
        return cls(
            curve=SNCurve.from_dict(body.pop("curve")),
            blocks=tuple(LoadBlock.from_dict(b) for b in body.pop("blocks")),
            measure=StressMeasure.parse(body.pop("measure")),
            **body,
        )


def _block_geometry(source: StressSource, block: LoadBlock):
    rows = [source.case_index(c) for c in block.cases]
    if len(set(rows)) != len(rows):
        raise InputError(f"block {block.name!r}: a load case appears twice")
    times = [source.case_times[r] for r in rows]
    have_times = all(t is not None for t in times)
    if have_times:
        t = np.asarray(times, dtype=float)
        if np.any(np.diff(t) <= 0.0):
            raise InputError(
                f"block {block.name!r}: load cases must be in strictly increasing time order"
            )
    if block.duration_s is not None:
        duration = float(block.duration_s)
    elif have_times:
        duration = float(times[-1] - times[0])
    else:
        raise InputError(
            f"block {block.name!r}: give duration_s, or provide case times so the "
            "duration of the record is known"
        )
    return rows, (np.asarray(times, dtype=float) if have_times else None), duration


def assess_time_series(
    source: StressSource,
    locations: Sequence,
    settings: TimeSeriesSettings,
    *,
    keep_series: bool = True,
) -> AssessmentResult:
    """Rainflow fatigue damage at every sample point of ``locations``."""

    if not locations:
        raise InputError("at least one stress location is required")
    geometry = {b.name: _block_geometry(source, b) for b in settings.blocks}
    points = [p for location in locations for p in extract(source, location)]
    results = [_assess_point(p, settings, geometry, keep_series) for p in points]
    warnings = list(edition_warnings(settings.curve))
    covered = sum(b.exposure_fraction for b in settings.blocks)
    if covered < 1.0 - 1e-9:
        warnings.append(
            f"the load blocks represent {covered:.1%} of the design life; the rest is "
            "assumed to cause no damage"
        )
    return AssessmentResult(
        method="time_series",
        points=tuple(results),
        settings=settings.to_dict(),
        provenance=_provenance(source),
        warnings=tuple(warnings),
    )


def _assess_point(point: SamplePoint, settings: TimeSeriesSettings, geometry, keep_series) -> PointResult:
    measure = settings.measure
    if measure.needs_weld_direction and point.weld_direction is None:
        raise InputError(
            f"{point.label}: the stress measure {measure.to_text()!r} needs a weld "
            "direction; give one on the location (or use 'principal_abs_max')"
        )
    design_seconds = settings.design_life_years * SECONDS_PER_YEAR
    parts = []
    total = 0.0
    for block in settings.blocks:
        rows, times, duration = geometry[block.name]
        series = measure.signed_series(point.tensors[rows], point.normal, point.weld_direction)
        cycles = count_rainflow_cycles(series, endpoints=True)
        ranges = cycles.ranges
        mean_factor = 1.0
        if settings.mean_stress == "non_welded" and len(cycles):
            factors = non_welded_mean_stress_factor(cycles.maxima, cycles.minima)
            ranges = ranges * factors
            mean_factor = float(np.average(factors, weights=cycles.counts))
            # keep the table that the damage was computed from
            cycles = CycleTable(ranges=ranges, means=cycles.means, counts=cycles.counts)
        scale = block.exposure_fraction * design_seconds / duration
        damage = miner_damage(
            ranges, cycles.counts, settings.curve,
            scf=settings.scf, thickness_mm=settings.thickness_mm, scale=scale,
        ).total
        total += damage
        parts.append(PartResult(
            name=block.name,
            damage=damage,
            values={
                "cycles_in_record": cycles.total_cycles,
                "max_range_mpa": float(ranges.max()) if ranges.size else 0.0,
                "record_duration_s": duration,
                "exposure_scale": scale,
                "mean_stress_factor": mean_factor,
                "stress_max_mpa": float(series.max()),
                "stress_min_mpa": float(series.min()),
            },
            cycles=cycles,
            series=series if keep_series else None,
            series_times=times if keep_series else None,
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
        "method": "rainflow counting (ASTM E1049-85 5.4.4 via ANYtimeseries) and Palmgren-Miner",
        "rainflow_backend": f"anytimes {rainflow_backend_version()}",
        "stress_unit": "MPa",
        "nodal_stress": source.nodal_origin,
        "source": dict(source.provenance),
        "cases": list(source.case_labels),
    }
