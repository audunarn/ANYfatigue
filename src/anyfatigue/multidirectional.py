"""Fixed-material-plane normal-master fatigue with explicit shear association.

The interaction is gamma * hypot(normal_range, 0.9 * shear_range), as in
DNVGL-RP-C203 (2019, amended 2020), F.12.4; see docs/MULTIDIRECTIONAL.md.
The global-range alternative is preserved. The opt-in cycle_associated mode uses
ANYfatigue chronological first-return policy v1, not a claim of complete F.12
master/slave qualification or F.12.1 crack-sector accumulation.
Rainflow counting remains owned by ANYtimeseries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple
from numbers import Integral

import numpy as np

from .damage import SECONDS_PER_YEAR, miner_damage
from .errors import InputError
from .measures import in_plane_components
from .rainflow import CycleTable, count_rainflow_cycles, count_indexed_rainflow_cycles
from .cycle_association import associate_shear_cycles
from .results import DirectionResult, PartResult, PointResult
from .sn_curves import SNCurve

if TYPE_CHECKING:
    from .extract import SamplePoint
    from .timeseries import TimeSeriesSettings

__all__ = [
    "FatiguePlane", "MultidirectionalSettings", "directional_stress_history",
    "multidirectional_equivalent_range",
]

METHOD_ID = "normal_master_global_shear_range_v1"
ASSOCIATED_METHOD_ID = "normal_master_cycle_associated_shear_v1"
METHOD_SOURCE = "DNVGL-RP-C203 September 2019, amended January 2020, F.12"
METHOD_WARNINGS = (
    "Multidirectional assessment uses normal-master rainflow with the maximum "
    "minus minimum shear stress of each whole load block on each fixed plane. "
    "This is not time-matched shear-cycle counting; its per-cycle shear envelope "
    "does not establish a conservative damage bound for arbitrary histories.",
    "The governing damage is max_over_planes(sum_over_blocks(damage)); "
    "F.12.1 crack-sector accumulation and automatic directional S-N "
    "classification are not implemented. Check angular-grid convergence.",
    "The multidirectional procedure is an implementation of the stated 2019/2020 "
    "method subset, not an independently qualified or current-edition DNV assessment. "
    "The supplied gamma and directional curve assignments require engineering review.",
)


ASSOCIATED_METHOD_WARNINGS = (
    "Cycle-associated shear uses chronological first-return windows, with exactly matching "
    "adjacent half cycles paired and unmatched halves assessed over their own intervals. "
    "Nested excursions remain in parent windows. This is an explicit envelope convention, "
    "not a unique multiaxial decomposition or independent fatigue-life qualification.",
    METHOD_WARNINGS[1],
    "The interaction equation retains its stated 2019/2020 source; temporal support is "
    "ANYfatigue policy v1, not a verified current-edition DNV requirement. Explicit gamma "
    "and directional S-N assignments still require engineering review.",
)


def _number(value, name: str) -> float:
    try:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise InputError(f"{name} must be a finite number") from None
    if not math.isfinite(result):
        raise InputError(f"{name} must be a finite number")
    return result


def _array(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise InputError(f"{name} must be real time-domain values, not complex amplitudes")
    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise InputError(f"{name} must contain finite numbers") from None
    if not np.all(np.isfinite(out)):
        raise InputError(f"{name} must contain finite numbers")
    return out


def _angle(value) -> float:
    angle = _number(value, "plane angle")
    if not 0.0 <= angle < 180.0:
        raise InputError("plane angle must be in [0, 180) degrees")
    return angle


@dataclass(frozen=True)
class FatiguePlane:
    """A trial stress-normal direction and its explicitly assigned S-N curve.

    Zero is perpendicular to the weld. Positive angles rotate from
    ``cross(surface_normal, weld_tangent)`` towards the weld tangent. This is
    the stress/plane-normal direction, not the crack-trace direction.
    """

    angle_deg: float
    curve: SNCurve

    def __post_init__(self) -> None:
        object.__setattr__(self, "angle_deg", _angle(self.angle_deg))
        if not isinstance(self.curve, SNCurve):
            raise InputError("every fatigue plane needs an explicit SNCurve")

    def to_dict(self) -> dict:
        return {"angle_deg": self.angle_deg, "curve": self.curve.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "FatiguePlane":
        return cls(data["angle_deg"], SNCurve.from_dict(data["curve"]))


@dataclass(frozen=True)
class MultidirectionalSettings:
    """Opt-in settings; no universal gamma or directional curve is assumed.

    At least three distinct directions (including zero) are required so that
    normal projections span the three independent in-plane tensor components.
    A fine, converged angular grid is still the caller's responsibility.
    The zero-degree curve must match TimeSeriesSettings.curve (the baseline).
    """

    planes: Tuple[FatiguePlane, ...]
    gamma: float
    gamma_basis: str
    curve_basis: str
    shear_association: str = "global_range"
    cycle_chunk_size: Optional[int] = None

    def __post_init__(self) -> None:
        try:
            planes = tuple(self.planes)
        except TypeError:
            raise InputError("planes must be a sequence of FatiguePlane objects") from None
        if not all(isinstance(p, FatiguePlane) for p in planes):
            raise InputError("planes must contain FatiguePlane objects")
        angles = [p.angle_deg for p in planes]
        if len(angles) < 3 or len(set(angles)) != len(angles) or 0.0 not in angles:
            raise InputError("use at least three unique plane angles, including zero degrees")
        a = np.deg2rad(2.0 * np.asarray(angles))
        if np.linalg.matrix_rank(np.column_stack((np.ones(len(a)), np.cos(a), np.sin(a)))) < 3:
            raise InputError("plane directions cannot resolve all in-plane tensor components")
        gamma = _number(self.gamma, "gamma")
        if gamma < 1.0:
            raise InputError("gamma must be >= 1; a fatigue-strength benefit is not supported")
        for name in ("gamma_basis", "curve_basis"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InputError(f"{name} must describe the engineering basis")
        if self.shear_association not in ("global_range", "cycle_associated"):
            raise InputError("shear_association must be 'global_range' or 'cycle_associated'")
        if self.cycle_chunk_size is not None:
            if (isinstance(self.cycle_chunk_size, (bool, np.bool_))
                or not isinstance(self.cycle_chunk_size, Integral) or self.cycle_chunk_size <= 0):
                raise InputError("cycle_chunk_size must be a positive integer or None")
            if self.shear_association != "cycle_associated":
                raise InputError("cycle_chunk_size requires cycle_associated mode")
            object.__setattr__(self, "cycle_chunk_size", int(self.cycle_chunk_size))
        object.__setattr__(self, "planes", tuple(sorted(planes, key=lambda p: p.angle_deg)))
        object.__setattr__(self, "gamma", gamma)

    @classmethod
    def uniform(
        cls, curve: SNCurve, *, gamma: float, gamma_basis: str, curve_basis: str,
        step_deg: float = 5.0, shear_association: str = "global_range",
        cycle_chunk_size: Optional[int] = None,
    ) -> "MultidirectionalSettings":
        """Explicitly assign one curve to all grid directions; this is not a DNV classification.

        The actual uniform step is 180 / ceil(180 / step_deg), no larger than
        the requested step. Supply FatiguePlane objects directly for mixed curves.
        """
        step = _number(step_deg, "angular step")
        if not 0.0 < step <= 60.0:
            raise InputError("angular step must be in (0, 60] degrees")
        if step < 180.0 / 10000:
            raise InputError("angular grid exceeds 10000 planes; use a bounded convergence study")
        count = math.ceil(180.0 / step)
        return cls(
            tuple(FatiguePlane(float(a), curve) for a in np.linspace(0.0, 180.0, count, endpoint=False)),
            gamma, gamma_basis, curve_basis, shear_association, cycle_chunk_size,
        )

    def to_dict(self) -> dict:
        return {
            "planes": [p.to_dict() for p in self.planes], "gamma": self.gamma,
            "gamma_basis": self.gamma_basis, "curve_basis": self.curve_basis,
            "shear_association": self.shear_association,
            **({"cycle_chunk_size": self.cycle_chunk_size} if self.cycle_chunk_size is not None else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MultidirectionalSettings":
        body = dict(data)
        body["planes"] = tuple(FatiguePlane.from_dict(p) for p in body["planes"])
        return cls(**body)


def _project(perp, parallel, shear, angle_deg):
    # Exact quadrantal double angles avoid artificial cycles on degenerate planes.
    special = {0.0: (1.0, 0.0), 45.0: (0.0, 1.0),
               90.0: (-1.0, 0.0), 135.0: (0.0, -1.0)}
    if angle_deg in special:
        c, s = special[angle_deg]
    else:
        a = math.radians(2.0 * angle_deg)
        c, s = math.cos(a), math.sin(a)
    mean = 0.5 * perp + 0.5 * parallel
    difference = 0.5 * perp - 0.5 * parallel
    return mean + difference * c + shear * s, -difference * s + shear * c


def directional_stress_history(tensors, normal, weld_direction, angle_deg):
    """Resolve signed global surface tensors (N,6) into one fixed plane, in MPa.

    Tensor order is xx, yy, zz, xy, yz, xz, with physical shear stresses.
    The surface frame and weld tangent must remain material-fixed through time.
    """
    values = _array(tensors, "surface tensors")
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < 2:
        raise InputError("surface tensors must have shape (N, 6), N >= 2")
    if weld_direction is None:
        raise InputError("multidirectional fatigue requires a fixed weld direction")
    angle = _angle(angle_deg)
    sn, st = _project(*in_plane_components(values, normal, weld_direction), angle)
    if not np.all(np.isfinite(sn)) or not np.all(np.isfinite(st)):
        raise InputError("stress projection overflowed; check stress units and magnitudes")
    return sn, st


def multidirectional_equivalent_range(normal_range, shear_range, *, gamma):
    """gamma * sqrt(normal_range**2 + 0.81*shear_range**2), applied AFTER counting."""
    g = _number(gamma, "gamma")
    if g < 1.0:
        raise InputError("gamma must be >= 1")
    nr, sr = _array(normal_range, "normal range"), _array(shear_range, "shear range")
    if np.any(nr < 0.0) or np.any(sr < 0.0):
        raise InputError("stress ranges must be non-negative")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = g * np.hypot(nr, 0.9 * sr)
    except ValueError:
        raise InputError("normal and shear range shapes are incompatible") from None
    if not np.all(np.isfinite(result)):
        raise InputError("equivalent stress range overflowed")
    return float(result) if result.ndim == 0 else result


def _assess_multidirectional_point(
    point: "SamplePoint", settings: "TimeSeriesSettings", geometry, keep_series: bool,
    baseline_damage: float,
) -> PointResult:
    md = settings.multidirectional
    assert md is not None  # dispatched only by the validated time-series path
    prepared = []
    for block in settings.blocks:
        rows, times, duration = geometry[block.name]
        tensors = _array(point.tensors[rows], "surface tensors")
        components = in_plane_components(tensors, point.normal, point.weld_direction)
        # An explicit floating-point resolution guard, NOT a material fatigue limit.
        # 128 eps bounds the short basis/contraction/transformation/range operation
        # chain. Without it, cancellation noise can receive the FULL shear envelope.
        resolution = 128.0 * np.finfo(float).eps * float(np.max(np.abs(tensors)))
        prepared.append((block, times, duration, components, resolution))

    best_parts = ()
    best_damage = -1.0
    directions = []
    ignored_cycles = 0.0
    inactive_shear_planes = 0
    for plane in md.planes:
        parts = []
        for block, times, duration, components, resolution in prepared:
            normal, shear = _project(*components, plane.angle_deg)
            if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(shear)):
                raise InputError(f"{point.label}: stress projection overflowed")
            indexed = None
            if md.shear_association == "cycle_associated":
                indexed = count_indexed_rainflow_cycles(normal, endpoints=True, chunk_size=md.cycle_chunk_size)
                cycles = indexed.cycles
            else:
                cycles = count_rainflow_cycles(normal, endpoints=True)
            keep = cycles.ranges > resolution
            discarded = float(cycles.counts[~keep].sum())
            ignored_cycles += discarded
            cycles = CycleTable(cycles.ranges[keep], cycles.means[keep], cycles.counts[keep])
            shear_range = float(np.max(shear) - np.min(shear))
            if shear_range <= resolution:
                shear_range = 0.0
            if not len(cycles) and shear_range > 0.0:
                inactive_shear_planes += 1
            association = None
            applied_shear = shear_range
            if indexed is not None:
                association = associate_shear_cycles(normal, shear, indexed.select(keep), times=times, resolution=resolution)
                applied_shear = association.shear_ranges
            equivalent = multidirectional_equivalent_range(cycles.ranges, applied_shear, gamma=md.gamma)
            scale = block.exposure_fraction * settings.design_life_years * SECONDS_PER_YEAR / duration
            miner = miner_damage(
                equivalent, cycles.counts, plane.curve, scf=settings.scf,
                thickness_mm=settings.thickness_mm, scale=scale,
            )
            damage = miner.total
            if not math.isfinite(damage):
                raise InputError(f"{point.label}: multidirectional damage overflowed")
            parts.append(PartResult(
                name=block.name, damage=damage, cycles=cycles,
                equivalent_ranges=equivalent, curve=plane.curve,
                cycle_association=association,
                cycle_damage=miner.per_cycle if association is not None else None,
                series=normal if keep_series else None,
                series_times=times if keep_series else None,
                values={
                    "plane_angle_deg": plane.angle_deg,
                    "cycles_in_record": cycles.total_cycles,
                    "max_range_mpa": float(equivalent.max()) if equivalent.size else 0.0,
                    "max_normal_range_mpa": cycles.max_range,
                    "global_shear_range_mpa": shear_range,
                    **({"max_associated_shear_range_mpa": float(applied_shear.max()) if len(applied_shear) else 0.0,
                        "paired_half_cycle_rows": float(np.count_nonzero(association.paired_cycle_indices >= 0)),
                        "unmatched_half_cycle_rows": float(sum(k == "half_interval" for k in association.support_kinds))}
                       if association is not None else {}),
                    "gamma": md.gamma,
                    "roundoff_range_bound_mpa": resolution,
                    "discarded_roundoff_cycles": discarded,
                    "record_duration_s": duration, "exposure_scale": scale,
                    "mean_stress_factor": 1.0,
                    "stress_max_mpa": float(normal.max()),
                    "stress_min_mpa": float(normal.min()),
                },
            ))
        try:
            total = math.fsum(p.damage for p in parts)
        except OverflowError:
            raise InputError(f"{point.label}: accumulated fatigue damage overflowed") from None
        directions.append(DirectionResult(plane.angle_deg, plane.curve, total,
                                          tuple(p.damage for p in parts)))
        if total > best_damage:  # deterministic: the first (smallest) angle wins exact ties
            best_damage, best_parts = total, tuple(parts)

    notes = [point.notes] if point.notes else []
    if ignored_cycles:
        notes.append(f"Discarded {ignored_cycles:g} roundoff-scale normal cycles across all trial planes/blocks.")
    if inactive_shear_planes:
        notes.append(
            f"{inactive_shear_planes} plane/block combinations have cyclic shear but no resolved "
            "normal-master cycle; their zero is not a stand-alone pure-shear assessment."
        )
    usage = best_damage * settings.dff
    if not math.isfinite(usage):
        raise InputError(f"{point.label}: fatigue usage overflowed")
    return PointResult(
        label=point.label, kind=point.kind, surface=point.surface,
        xyz=tuple(float(v) for v in point.xyz), chainage=point.chainage,
        damage=best_damage,
        life_years=math.inf if best_damage == 0.0 else settings.design_life_years / best_damage,
        usage=usage, passed=usage <= 1.0, parts=best_parts, notes=" ".join(notes),
        directions=tuple(directions), normal_to_weld_damage=baseline_damage,
    )
