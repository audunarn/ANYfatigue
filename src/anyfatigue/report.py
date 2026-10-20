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
        if p.curve is not None:
            out["curve"] = p.curve.to_dict()
        if include_cycles and p.cycles is not None:
            out["cycles"] = {
                "range_mpa": p.cycles.ranges.tolist(),
                "mean_mpa": p.cycles.means.tolist(),
                "count": p.cycles.counts.tolist(),
            }
            if p.equivalent_ranges is not None:
                out["cycles"]["range_kind"] = "normal_master"
                out["cycles"]["equivalent_range_mpa"] = p.equivalent_ranges.tolist()
            if p.cycle_association is not None:
                out["cycles"]["association"] = p.cycle_association.to_dict()
                out["cycles"]["design_life_damage"] = p.cycle_damage.tolist()
        return out

    def point(p: PointResult):
        out = {
            "label": p.label, "kind": p.kind, "surface": p.surface, "xyz_m": list(p.xyz),
            "chainage_m": p.chainage, "damage": p.damage,
            "life_years": None if math.isinf(p.life_years) else p.life_years,
            "usage": p.usage, "passed": p.passed, "notes": p.notes,
            "parts": [part(x) for x in p.parts],
        }
        if p.directions:
            out.update(
                critical_angle_deg=p.critical_angle_deg,
                normal_to_weld_damage=p.normal_to_weld_damage,
                directions=[{"angle_deg": d.angle_deg, "damage": d.damage,
                             "curve": d.curve.to_dict(), "block_damages": list(d.block_damages)}
                            for d in p.directions],
            )
        return out

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
        ("* S-N curve (weld-normal baseline): " if s.get("multidirectional") else "* S-N curve: ")
        + f"{curve['name']} ({curve['environment']}), "
        f"m1 = {curve['m1']:g}, log a1 = {curve['log_a1']:g}"
        + (f", m2 = {curve['m2']:g}, log a2 = {curve['log_a2']:g}" if curve.get("m2") else ""),
        f"* Thickness {s['thickness_mm']:g} mm, SCF {s['scf']:g}, "
        f"design life {s['design_life_years']:g} years, DFF {s['dff']:g}",
        f"* Stress measure: {s['measure']}",
        f"* Stresses in {prov['stress_unit']}; nodal stress: {prov['nodal_stress']}",
        f"* ANYfatigue {prov['anyfatigue_version']}; {prov['method']}",
    ]
    declaration = prov.get("tensor_completeness", prov.get("source", {}).get("tensor_completeness"))
    if declaration is not None:
        known = declaration["provided"] + declaration["assumed_zero"]
        missing = [c for c in ("xx", "yy", "zz", "xy", "yz", "xz") if c not in known]
        lines.append("* Tensor components (global): provided " + (", ".join(declaration["provided"]) or "none")
                     + "; assumed zero " + (", ".join(declaration["assumed_zero"]) or "none")
                     + "; missing " + (", ".join(missing) or "none"))
        if declaration["basis"]:
            lines.append("* Tensor assumption basis: " + declaration["basis"])
    for location in prov.get("locations", ()):
        parent = location.get("parent_plate")
        if parent:
            lines.append(f"* Parent patch for {location['name']}: {parent['name']}; "
                         f"element IDs {parent['element_ids']}; recovery {parent['nodal_recovery']}; "
                         f"mesh binding {parent['mesh_id'] or 'not supplied (mesh-relative IDs)'}")
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
    if critical.directions:
        md = s["multidirectional"]
        lines += ["", "## Multidirectional fixed-plane assessment", "",
                  f"Critical stress-normal angle: {critical.critical_angle_deg:g} deg "
                  "from perpendicular-to-weld (not the crack-trace angle).",
                  f"Weld-normal baseline damage: {critical.normal_to_weld_damage:.6g}.",
                  f"gamma = {md['gamma']:g}; basis: {md['gamma_basis']}",
                  f"Directional curve basis: {md['curve_basis']}",
                  f"Shear association: {md['shear_association']}; "
                  f"policy: {prov.get('cycle_association_policy', 'whole-block range')}",
                  "Block contributions below use the same governing plane throughout.", "",
                  "| angle [deg] | S-N curve | environment | accumulated damage |",
                  "|---:|---|---|---:|"]
        for direction in critical.directions:
            lines.append(f"| {direction.angle_deg:g} | {direction.curve.name} | "
                         f"{direction.curve.environment} | {direction.damage:.6g} |")
        if critical.notes:
            lines += ["", critical.notes]
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
