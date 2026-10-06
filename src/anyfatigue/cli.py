"""Command line: list curves, run a saved analysis on a stress file.

    anyfatigue curves [--environment air]
    anyfatigue run analysis.json --stress stresses.npz --out report-dir
    anyfatigue run analysis.json --history hotspot.csv --out report-dir
    anyfatigue run analysis.json --anyfem deck.anyfem --out report-dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .errors import FatigueError
from .project import AnalysisDefinition
from .report import write_csv, write_json, write_report
from .sn_curves import ENVIRONMENTS, iter_curves
from .stress_io import read_history_csv, read_stress_npz


def _curves(args) -> int:
    print(f"{'curve':6s} {'m1':>4s} {'log a1':>8s} {'m2':>4s} {'log a2':>8s} {'k':>5s} {'t_ref':>6s}")
    for c in iter_curves(args.environment):
        m2 = f"{c.m2:g}" if c.bilinear else "-"
        a2 = f"{c.log_a2:.3f}" if c.bilinear else "-"
        print(f"{c.name:6s} {c.m1:4g} {c.log_a1:8.3f} {m2:>4s} {a2:>8s} {c.k:5g} {c.t_ref_mm:6g}")
    return 0


def _run(args) -> int:
    definition = AnalysisDefinition.load(args.analysis)
    if sum(bool(x) for x in (args.stress, args.history, args.anyfem)) != 1:
        raise FatigueError("give exactly one of --stress, --history and --anyfem")
    if args.stress:
        source = read_stress_npz(args.stress)
    elif args.history:
        source = read_history_csv(args.history)
    else:
        from .anyfem_artifacts import read_result

        source = read_result(args.anyfem, job_id=args.job_id)
    result = definition.run(source, keep_series=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.analysis).stem
    write_csv(result, out / f"{stem}-points.csv")
    write_json(result, out / f"{stem}-result.json")
    write_report(result, out / f"{stem}-report.md")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    critical = result.critical
    print(f"{'PASS' if result.passed else 'FAIL'}: {len(result.points)} points; governing "
          f"{critical.label}: damage {critical.damage:.4g}, D x DFF {critical.usage:.4g}, "
          f"life {critical.life_text} years")
    print(f"reports written to {out}")
    return 0 if result.passed else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="anyfatigue", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    curves = sub.add_parser("curves", help="list the DNV S-N curves")
    curves.add_argument("--environment", choices=ENVIRONMENTS, default="air")
    curves.set_defaults(func=_curves)
    run = sub.add_parser("run", help="run a saved analysis")
    run.add_argument("analysis", help="analysis definition (.json)")
    run.add_argument("--stress", help="stress file (.npz) exported from ANYfem")
    run.add_argument("--history", help="stress history at one hot spot (.csv)")
    run.add_argument("--anyfem", help="ANYfem project (.anyfem), its -data folder or a .anyres.h5 result")
    run.add_argument("--job-id", help="result to use when --anyfem names a project with several")
    run.add_argument("--out", default=".", help="directory for the reports")
    run.set_defaults(func=_run)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FatigueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
