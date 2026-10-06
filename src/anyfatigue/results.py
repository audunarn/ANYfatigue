"""Result containers shared by both methods."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from .rainflow import CycleTable

__all__ = ["AssessmentResult", "PartResult", "PointResult"]


@dataclass(frozen=True)
class PartResult:
    """Damage of one load condition (simplified) or one load block (time series)."""

    name: str
    damage: float
    values: Mapping[str, float] = field(default_factory=dict)
    cycles: Optional[CycleTable] = None
    series: Optional[np.ndarray] = None
    series_times: Optional[np.ndarray] = None


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
