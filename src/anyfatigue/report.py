"""Reports: a points table (CSV), a JSON record and a Markdown summary."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import List

from .results import AssessmentResult, PointResult

__all__ = ["points_table", "report_markdown", "result_to_dict", "write_csv", "write_json", "write_report"]

_HEADER = (
    "label", "kind", "surface", "x_m", "y_m", "z_m", "chainage_m",
    "damage", "life_years", "usage_damage_x_dff", "passed",
)


def points_table(result: AssessmentResult) -> List[list]:
    rows = []
    for p in result.points:
        rows.append([
            p.label, p.kind, p.surface, *p.xyz,
            "" if p.chainage is None else p.chainage,
            p.damage, "inf" if math.isinf(p.life_years) else p.life_years,
            p.usage, "yes" if p.passed else "no",
        ])
    return rows


def write_csv(result: AssessmentResult, path) -> Path:
    target = Path(path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER)
        writer.writerows(points_table(result))
    return target


def result_to_dict(result: AssessmentResult, *, include_cycles: bool = False) -> dict:
    """A JSON-serialisable record: settings, provenance, points and parts."""

    def part(p):
        out = {"name": p.name, "damage": p.damage, "values": dict(p.values)}
        if include_cycles and p.cycles is not None:
            out["cycles"] = {
                "range_mpa": p.cycles.ranges.tolist(),
                "mean_mpa": p.cycles.means.tolist(),
                "count": p.cycles.counts.tolist(),
            }
        return out

    def point(p: PointResult):
        return {
            "label": p.label, "kind": p.kind, "surface": p.surface, "xyz_m": list(p.xyz),
            "chainage_m": p.chainage, "damage": p.damage,
            "life_years": None if math.isinf(p.life_years) else p.life_years,
            "usage": p.usage, "passed": p.passed, "notes": p.notes,
            "parts": [part(x) for x in p.parts],
        }

    return {
        "schema": "anyfatigue.result", "version": 1, "method": result.method,
        "settings": dict(result.settings), "provenance": dict(result.provenance),
        "warnings": list(result.warnings), "passed": result.passed,
        "critical": result.critical.label if result.points else None,
        "points": [point(p) for p in result.points],
    }


def write_json(result: AssessmentResult, path, *, include_cycles: bool = False) -> Path:
    target = Path(path)
    target.write_text(
        json.dumps(result_to_dict(result, include_cycles=include_cycles), indent=2, default=str),
        encoding="utf-8",
    )
    return target


def report_markdown(result: AssessmentResult, *, top: int = 15) -> str:
    s = result.settings
    prov = result.provenance
    critical = result.critical
    method = {"simplified": "Simplified (Weibull) fatigue analysis",
              "time_series": "Time-series (rainflow) fatigue analysis"}[result.method]
    curve = s["curve"]
    lines = [
        f"# Fatigue assessment: {method}",
        "",
        f"* Standard: {prov['standard']}",
        f"* S-N curve: {curve['name']} ({curve['environment']}), "
        f"m1 = {curve['m1']:g}, log a1 = {curve['log_a1']:g}"
        + (f", m2 = {curve['m2']:g}, log a2 = {curve['log_a2']:g}" if curve.get("m2") else ""),
        f"* Thickness {s['thickness_mm']:g} mm, SCF {s['scf']:g}, "
        f"design life {s['design_life_years']:g} years, DFF {s['dff']:g}",
        f"* Stress measure: {s['measure']}",
        f"* Stresses in {prov['stress_unit']}; nodal stress: {prov['nodal_stress']}",
        f"* ANYfatigue {prov['anyfatigue_version']}; {prov['method']}",
    ]
    if "rainflow_backend" in prov:
        lines.append(f"* Rainflow backend: {prov['rainflow_backend']}")
    lines += ["", "## Verdict", ""]
    verdict = "PASS" if result.passed else "FAIL"
    lines += [
        f"**{verdict}** at {len(result.points)} points: governing point "
        f"`{critical.label}` has damage {critical.damage:.4g}, "
        f"usage D x DFF = {critical.usage:.4g} "
        f"(limit 1.0), fatigue life {critical.life_text} years.",
        "",
    ]
    for w in result.warnings:
        lines.append(f"> Warning: {w}")
    if result.warnings:
        lines.append("")
    lines += ["## Highest damage", "",
              "| point | surface | damage | life [years] | D x DFF | ok |",
              "|---|---|---:|---:|---:|:---:|"]
    for p in sorted(result.points, key=lambda p: -p.damage)[:top]:
        lines.append(
            f"| {p.label} | {p.surface} | {p.damage:.4g} | {p.life_text} | {p.usage:.4g} | "
            f"{'yes' if p.passed else 'no'} |"
        )
    lines += ["", f"## Governing point: contributions", ""]
    for part in critical.parts:
        detail = ", ".join(f"{k} = {v:.5g}" for k, v in part.values.items())
        lines.append(f"* **{part.name}**: damage {part.damage:.4g} ({detail})")
    lines.append("")
    return "\n".join(lines)


def write_report(result: AssessmentResult, path) -> Path:
    target = Path(path)
    target.write_text(report_markdown(result), encoding="utf-8")
    return target
