"""Result containers shared by both methods."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .rainflow import CycleTable
from .cycle_association import CycleAssociation
from .sn_curves import SNCurve

__all__ = ["AssessmentResult", "DirectionResult", "PartResult", "PointResult"]


@dataclass(frozen=True)
class PartResult:
    """Damage of one load condition (simplified) or one load block (time series)."""

    name: str
    damage: float
    values: Mapping[str, float] = field(default_factory=dict)
    cycles: Optional[CycleTable] = None
    series: Optional[np.ndarray] = None
    series_times: Optional[np.ndarray] = None
    # Raw cycles retain NORMAL ranges/means; these are the ranges used for damage.
    equivalent_ranges: Optional[np.ndarray] = None
    curve: Optional[SNCurve] = None
    cycle_association: Optional[CycleAssociation] = None
    cycle_damage: Optional[np.ndarray] = None


@dataclass(frozen=True)
class DirectionResult:
    """Summary on one fixed plane; block damages follow the settings' block order."""

    angle_deg: float
    curve: SNCurve
    damage: float
    block_damages: Tuple[float, ...]


@dataclass(frozen=True)
class PointResult:
    label: str
    kind: str
    surface: str
    xyz: Tuple[float, float, float]
    chainage: Optional[float]
    damage: float
    life_years: float
    usage: float
    passed: bool
    parts: Tuple[PartResult, ...]
    notes: str = ""
    directions: Tuple[DirectionResult, ...] = ()
    normal_to_weld_damage: Optional[float] = None

    @property
    def critical_angle_deg(self) -> Optional[float]:
        return max(self.directions, key=lambda d: d.damage).angle_deg if self.directions else None

    @property
    def life_text(self) -> str:
        return "infinite" if math.isinf(self.life_years) else f"{self.life_years:.4g}"


@dataclass(frozen=True)
class AssessmentResult:
    method: str
    points: Tuple[PointResult, ...]
    settings: Mapping[str, object]
    provenance: Mapping[str, object]
    warnings: Tuple[str, ...] = ()

    @property
    def critical(self) -> PointResult:
        if not self.points:
            raise ValueError("assessment has no points")
        return max(self.points, key=lambda p: p.damage)

    @property
    def passed(self) -> bool:
        return all(p.passed for p in self.points)

    def by_label(self) -> Dict[str, PointResult]:
        return {p.label: p for p in self.points}
