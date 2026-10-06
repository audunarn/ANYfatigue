# ANYfatigue: design record and verification evidence

Living task note for 0.1.0 (2026-10-06). Policy: `ANY_ECOSYSTEM_RISK_PROPORTIONATE_ENGINEERING_V1`,
revision `2026-09-24.2`. Level A (integrated development) with focused checks; no release,
qualification or publication claim is made.

## 1. Scope and ownership

ANYfatigue is the canonical owner of fatigue assessment (S-N curves, damage, hot-spot read-out,
reports). It consumes, through public contracts:

| Need | Owner | How consumed |
|---|---|---|
| Rainflow counting | ANYtimeseries | `anyqats.fatigue.rainflow.count_cycles`, one adapter (`rainflow.py`); missing package raises `MissingDependencyError`, no fallback counter |
| Stresses, meshes | ANYfem / ANYsolver | `anyfem_adapter.py` reads `stresses(return_global=True)` and patch recovery; the core depends only on `model.StressSource` |

The fatigue core never imports ANYfem, Qt or h5py. Units at the boundary: metres, **MPa**, global axes;
conversion (Pa to MPa) happens once in the adapter and is recorded in `provenance`.

## 2. What is implemented

* `sn_curves.py`: Tables 2-1/2-2/2-3 (B1..W3, T) with thickness exponent, `t_ref`, knee at 1e7 (air)
  or 1e6 (cathodic protection), free corrosion single slope; custom curves; ANYstructure key mapping.
* `damage.py`: Miner sum over cycles; Weibull scale (5.1.2); one-slope (5.1.3) and two-slope (D.13-1)
  closed forms; direct integration (D.13-2/3) as an independent check; non-welded mean-stress factor
  (2.5.1); `D x DFF <= 1`.
* `extract.py`, `measures.py`, `model.py`: element / line / interpolated-point locations; direct,
  hot-spot A (`1.5 s(0.5t) - 0.5 s(1.5t)`) and B (`s(0.5t)`) read-out (4.3.3/4.3.4); in-plane projection
  and weld resolution; DNV effective hot-spot range (4.3.1 to 4.3.4, 1.12 factor for method B).
* `simplified.py`, `timeseries.py`: the two methods; `project.py` saved analysis; `report.py`,
  `cli.py`, `stress_io.py` (stress `.npz`, history CSV), `plotdata.py`, `examples.py`.
* `gui/`: PySide6 workbench in ANYfem's theme, `run_gui.py` at the root.

## 3. Decisions

1. **One neutral stress contract** (mesh + tensors per case, MPa, global axes) instead of calling ANYfem
   inside the core. Extraction logic is therefore tested against closed-form fields, and a stress file
   exported once runs anywhere.
2. **Global-axes surface tensors are required.** Element-local stresses cannot be resolved against a weld
   without the local frame, so a recovery lacking `global_*_top/bot` is refused (typed error naming
   `return_global=True`).
3. **Only in-plane components enter** the fatigue stress (projection onto the surface plane). ANYfem
   reports the same transverse shear on both faces; it is a through-thickness average, not a surface
   stress, and is never read.
4. **Interpolation uses the continuous nodal field**, never the piecewise-constant element field.
   Nodes that patch recovery did not qualify use the mean of their incident elements and are listed
   in the provenance.
5. **No silent fallback.** Read-out points off the mesh, unknown cases, missing weld directions,
   unordered times, a signed history with the range-only effective stress: all raise typed errors.
6. **Rainflow end points are kept** (first and last sample start/close half cycles). ANYtimeseries'
   default drops them, which can discard the largest cycle of a quasi-static record that starts at
   its peak (tested).
7. **Simplified method input** is one load case per condition (range = `factor x |stress|`, default
   factor 2 for a fully reversed amplitude, or the difference to a reference case), with the fraction of
   life, Weibull shape, zero-crossing period and `n0` per ANYstructure's conventions.
8. **Licence: MPL-2.0** (chosen by the owner, 2026-10-06), as ANYfem.
9. **Edition status is part of every result.** DNV's current edition is 2024-10 (amended 2025-10); only
   April 2010 is verified, so built-in curves add a warning to every result, report, CLI run and the GUI
   until the current edition has been transcribed and checked (open item 1).
10. **Artifact reader is lazy** and links result to mesh by `mesh_hash` (load-case results do not name
   their mesh sidecar); `physical_director` is used as the normal when the file has it.

## 4. Package evaluation (decision: no new runtime dependency)

| Package | Licence | Finding |
|---|---|---|
| [py_fatigue](https://github.com/owi-lab/py_fatigue) | GPL-3.0 | The ecosystem licence gate (`ANYfem/tools/check_licenses.py`) forbids GPL dependencies. Also numba/pandas/plotly/pydantic; its v1 line needs Python <3.11 and v2 supports 3.10 to 3.13 (not 3.14). No Weibull simplified method or hot-spot read-out. |
| [pyLife](https://pylife.readthedocs.io) | Apache-2.0 | Permitted and maintained; numpy/scipy/pandas>=2.2/h5py. Generic Woehler curve (`k_1`, `ND`, `SD`, optional `k_2`); no DNV tables, thickness exponent, Weibull simplified method or 0.5t/1.5t read-out found; pandas-MultiIndex data model. An open issue reports Miner-Haibach ignoring `k_2`. Not adopted; possible test-only oracle. Pages for `strength`/`mesh` returned 404 during review, so the module list rests on the reference index. |
| fatpack | ISC | Rainflow + trilinear S-N + Miner; duplicates ANYtimeseries, no DNV bilinear/thickness. |
| rainflow (PyPI) | MIT | Rainflow only; duplicates ANYtimeseries. |

## 5. Verification evidence

Environment: Python 3.13.9 and 3.14.2 on Windows 11, numpy 2.4.3 / scipy 1.16.3 (3.13 venv),
PySide6 6.10, ANYfem 0.4.1, ANYsolver 0.4.7, anytimes sources from the sibling checkout.
Command: `python -m pytest tests` with `QT_QPA_PLATFORM=offscreen`. **351 passed on both interpreters**
(including the ANYfem integration and artifact tests). The core suite (no ANYfem) takes about 5 s.
ANYfem is under concurrent development: when it gained a dependency (`ANYloads`) that the 3.14 environment
lacked, the ANYfem tests errored instead of skipping; the conftest now tests that ANYfem really imports.

References that are independent of the code under test:

* **Standard, transcribed from the PDF** (rules.dnv.com, DNV-RP-C203 April 2010; the tables from the extracted text, the
  equations from rendered pages 17, 32, 33, 36 and 140): Tables 2-1 to 2-3; equations (2.4.3), (4.3.1) to (4.3.4), (5.1.1) to (5.1.3), (D.13-1) to
  (D.13-5), (2.5.1); Table 5-1 gamma values; the printed *fatigue limit at 10^7 cycles* column, used to
  check the transcribed `log a` values within the tables' own rounding (tolerances derived from the
  rounding bound, not tuned).
* **ANYstructure parity** (commit `dc977b7c`, `CalcFatigue`, curve `Ec`, h 0.8, T 9 s, n0 1e4, 20 y):
  four thicknesses including a case with both slopes carrying damage (D = 1.378). Relative difference
  <= 1e-14. ANYstructure rounds a few `log a2` values (B1 17.15 vs 17.146); ANYfatigue uses the printed
  values.
* **Closed forms and invariants**: incomplete-gamma damage against direct numerical integration (80
  parameter combinations, 1e-6); h to infinity limit equals constant-amplitude Miner; damage scales with
  `SCF^m` and range^m; time-reversal invariance; area between the plotted spectrum and the S-N curve
  equals the damage; linear/cubic nodal fields interpolate and extrapolate exactly on quad, triangle and
  skewed meshes.
* **ANYfem as producer**: stresses equal ANYfem's own `membrane +/- bending` field; response is linear in
  load factor; centre stress of a simply supported square plate matches Timoshenko (6 x 0.0479 q a^2 / t^2)
  within 3 %; a regular pressure wave gives the hand-counted rainflow damage.
* **Artifact reader**: every stress, nodal value, normal, time and the resulting fatigue damage equal the
  in-process route built from the same solve (1e-10 or better); 9 injected faults (unit factor, first
  integration point instead of the mean, shifted patch index, fallback statistic, component order, top/bottom
  swapped, row order, ignored times, ignored unqualified nodes) are all caught after closing one gap (the
  unqualified-node route is now exercised through the file, as in production).
* **Mutation check**: 17 injected faults (hot-spot factors, shear weight, swapped gamma functions, knee
  slope, dropped end points, ignored range factor/thickness/SCF/fraction/exposure, side sign for line and
  point read-out, weld resolution, bilinear weights, half-cycle counts, mean-stress factor). All are caught.
  Two gaps found and closed while doing this (a hot-spot side test that could not see the sign because a
  linear field extrapolates exactly; a quadratic field is even in the offset, so a cubic is needed).
* **Installed wheel** built offline, installed outside the source tree, imported and run (analysis + CLI).
* **GUI**: offscreen tests check that what the forms collect equals what the library computes and that the
  window shows the same damage; plots were inspected visually (real fonts) and two layout defects fixed.

Scale: 9 600 elements x 3 000 steps, 162 hot-spot points in 0.9 s; 4 000 element points in 8.5 s.
ANYfem adapter: extraction is cheap; patch recovery (ANYsolver) is about 2 s per load case at 576 elements.

## 6. Findings in sibling repositories (not changed here)

* **ANYtimeseries `SNCurve.n` (anyqats/fatigue/sn.py)** chooses the slope branch from the uncorrected
  range while `sswitch` is in effective-range terms. With thickness correction it can pick the wrong
  branch: for a 100 mm plate it underestimates life by up to ~40 % (conservative) and `N` rises with stress
  between 52 and 60 MPa. A follow-up task with a reproduction was raised. ANYfatigue is unaffected (it
  applies the factor first).
* ANYtimeseries also carries its own S-N/damage module (`anyqats.fatigue.sn`) that overlaps this
  repository; consolidating on ANYfatigue is an ecosystem decision, not made here.
* ANYstructure still carries its own `CalcFatigue` and a rounded S-N table; moving it onto ANYfatigue
  is a compatibility decision for its owner.

## 7. Open items and limits

1. **Blocked on a source document.** The owner asked for the latest edition: DNV-RP-C203 **2024-10, amended
   2025-10**. It is behind DNV's subscription portal (Rules and Standards Explorer+) and could not be read
   here; the 2010 PDF was public on rules.dnv.com, the newer ones are not reachable. A vendor page (SDC Verifier)
   states a tubular reference thickness of 16 mm for 2024, against 32 mm in 2010, and cites hot-spot
   "Method B" under chapter 4.8 (2010: section 4.3.4); unverified hearsay, but enough to show that the 2010
   data must not be assumed valid for 2024. To close this the owner
   supplies the PDF (not committed: it is DNV's copyright); then Tables 2-1..2-3, the thickness
   equation, the hot-spot method (4.3.x), section 5 and the mean-stress clause are re-verified and
   `STANDARD_EDITION` updated, with the 2010 values kept selectable if they differ.
2. Not implemented: beam elements, nonlinear or transient results, the second-order stress fit for
   coarse 4-node shell meshes (4.3.3), welds' mean-stress/residual-stress treatment, tubular-joint SCF
   methods, crack-growth assessment, non-proportional multiaxial counting (the principal measure can
   switch branch).
3. A time-series record is treated as non-repeating; a repeating record should be closed by the user.
4. ANYfem results are read in-process, from saved projects (`.anyres.h5`, lazily) or via an exported `.npz`.
   The reader needs the project's `meshes/` folder and global surface stresses in the result; a result
   that covers only part of the shell elements is refused. Normals follow `physical_director`; without it
   (older files) they follow the corner order and the sign of the hot-spot `side` can differ per element,
   so give an explicit read-out direction.
5. CI (`.github/workflows/tests.yml`) is written but has not been run; it covers the core and Qt tests. The
   ANYfem integration tests need the ANYfem stack, which ANYfem's own CI pins by commit, so they are not
   in this workflow yet.
6. No release, tag or publication.
7. GUI: no cancel for a running job, no mesh view of the damage, patch recovery cost can be minutes on large models.
8. Unverified here: behaviour on Linux and macOS (only Windows was run).
