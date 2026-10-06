"""A saved analysis: method, settings and locations (not the stresses).

The file holds the *definition* of an assessment, so it can be re-run on a new
stress source (a new mesh or load series) without re-entering anything, and so
a report can name exactly what was assessed.  Stress data stay in their own file
(``.npz`` from :mod:`anyfatigue.stress_io`) or come from ANYfem; the definition
only records where the stresses came from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Union

from .errors import InputError
from .extract import location_from_dict
from .model import StressSource
from .results import AssessmentResult
from .simplified import SimplifiedSettings, assess_simplified
from .timeseries import TimeSeriesSettings, assess_time_series

__all__ = ["AnalysisDefinition", "SCHEMA", "SCHEMA_VERSION"]

SCHEMA = "anyfatigue.analysis"
SCHEMA_VERSION = 1
METHODS = ("simplified", "time_series")


@dataclass(frozen=True)
class AnalysisDefinition:
    method: str
    settings: Union[SimplifiedSettings, TimeSeriesSettings]
    locations: Tuple
    source: Dict[str, object] = field(default_factory=dict)
    name: str = "analysis"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise InputError(f"method must be one of {METHODS}")
        expected = SimplifiedSettings if self.method == "simplified" else TimeSeriesSettings
        if not isinstance(self.settings, expected):
            raise InputError(f"method {self.method!r} needs {expected.__name__}")
        if not self.locations:
            raise InputError("an analysis needs at least one stress location")
        object.__setattr__(self, "locations", tuple(self.locations))

    def run(self, source: StressSource, *, keep_series: bool = True) -> AssessmentResult:
        if self.method == "simplified":
            return assess_simplified(source, self.locations, self.settings)
        return assess_time_series(source, self.locations, self.settings, keep_series=keep_series)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA, "version": SCHEMA_VERSION, "name": self.name,
            "method": self.method, "source": dict(self.source),
            "settings": self.settings.to_dict(),
            "locations": [loc.to_dict() for loc in self.locations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisDefinition":
        if data.get("schema") != SCHEMA:
            raise InputError("not an ANYfatigue analysis file")
        if int(data.get("version", 0)) > SCHEMA_VERSION:
            raise InputError("this analysis file was written by a newer ANYfatigue")
        method = data["method"]
        settings_cls = SimplifiedSettings if method == "simplified" else TimeSeriesSettings
        if method not in METHODS:
            raise InputError(f"unknown method {method!r}")
        return cls(
            method=method,
            settings=settings_cls.from_dict(data["settings"]),
            locations=tuple(location_from_dict(x) for x in data["locations"]),
            source=dict(data.get("source", {})),
            name=data.get("name", "analysis"),
        )

    def save(self, path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path) -> "AnalysisDefinition":
        target = Path(path)
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise InputError(f"cannot read analysis file {target}: {error}") from None
        return cls.from_dict(data)
