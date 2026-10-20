# ANYfatigue

Fatigue assessment to DNV rules for the ANY ecosystem. Stresses come from
ANYfem; the S-N and damage mathematics live here.

> **Status: 0.1.0, early development.** The S-N data are transcribed from
> DNV-RP-C203 **April 2010** and checked against that edition only. DNV now
> publishes edition **2024-10 (amended 2025-10)**; it is behind DNV's subscription
> portal, has **not** been checked here, and is known to differ in places (for
> example the tubular reference thickness). Every result, report and the GUI say
> so. The results have not been independently qualified; do not use them for a
> design decision without your own verification against the edition your project
> requires. See [docs/DESIGN.md](docs/DESIGN.md) for exactly what was verified.

## Two methods

| | Simplified (DNV-RP-C203 section 5) | Time series (rainflow) |
|---|---|---|
| Load | one load case per load condition | a *series* of quasi-static load cases (a wave, a sea state) |
| Stress | `dS0 = factor x measure(stress(case))`, or `|stress(case) - stress(reference)|` | signed stress history `measure(stress(t))` |
| Damage | Weibull long-term distribution, closed form `(5.1.3)` and `(D.13-1)` | rainflow counting (ANYtimeseries), Palmgren-Miner, scaled to the share of the design life |
| Conditions | fraction of life, Weibull shape `h`, period, `n0` | blocks with a life share and a duration |

Both use the same S-N core (air, seawater with cathodic protection, free
corrosion; thickness correction `(t/t_ref)^k`; SCF; DFF) and report
`D x DFF <= 1`.

The simplified method reproduces ANYstructure's `CalcFatigue` to better than 1e-13
(relative) on the same inputs, so the two can be compared directly.

## Where stress is read

* **Elements**: the surface stress of whole shell elements (piecewise constant).
* **Line**: an ordered chain of nodes, typically a weld; every node is a point
  and the line tangent is the weld direction.
* **Points (interpolated)**: arbitrary points, linearly interpolated from the
  continuous nodal stress field.

For line and point locations the read-out is **direct**, or the DNV-RP-C203
4.3.4 **hot spot**: surface stress at `0.5 t` and `1.5 t` from the weld toe
extrapolated linearly (method A) or the stress at `0.5 t` (method B). Stress is
resolved against the weld in the surface plane (normal to the weld, principal,
a tensor component, or for the simplified method the DNV effective hot-spot
range, equations 4.3.1 to 4.3.4). Read-out points that fall off the mesh are an
error, never snapped.

## Install and run

```bash
python -m pip install -e ".[gui,anyfem]"      # anytimes (rainflow) is a core dependency
python run_gui.py                              # the desktop application
python run_gui.py --example synthetic          # open with a closed-form example
python run_gui.py --anyfem deck.anyfem         # a saved ANYfem project or result (.anyres.h5)
python run_gui.py --stress stresses.npz        # or a stress file / --history hotspot.csv
```

`run_gui.py` works in a fresh clone: it puts `src` first and falls back to
sibling checkouts of ANYtimeseries and the ANYfem stack when they are not
installed. The Qt look follows ANYfem's workbench theme.

Without ANYfem installed everything works except the ANYfem example; a stress
file exported once from ANYfem is enough to run the GUI elsewhere.

## Python API

```python
from anyfatigue import get_curve
from anyfatigue.anyfem_adapter import from_solutions, line_location
from anyfatigue.extract import ReadOut
from anyfatigue.project import AnalysisDefinition
from anyfatigue.timeseries import LoadBlock, TimeSeriesSettings

# `batch` is ANYfem's solve_linear_static_many(...) result for a load-case series
source = from_solutions(batch, project=project)           # MPa, global axes, provenance recorded

weld = line_location(batch, project.edge(edge_id),
                     readout=ReadOut("hotspot_a", thickness_mm=20.0, side=1), name="weld toe")
settings = TimeSeriesSettings(
    curve=get_curve("D", "seawater_cp"), thickness_mm=20.0, design_life_years=25.0, dff=3.0,
    blocks=(LoadBlock("sea state", tuple(source.case_labels), exposure_fraction=1.0),))

result = AnalysisDefinition("time_series", settings, (weld,)).run(source)
print(result.critical.label, result.critical.damage, result.critical.usage)
```

### Rotating stress directions at shell hot spots

Time-series assessments now accept an explicit `MultidirectionalSettings` option.
It resolves the full surface tensor onto fixed material planes, counts the signed
normal history through ANYtimeseries, and combines each counted range with that
plane's **whole-block shear range**, or optionally a cycle-associated shear window. Damage is accumulated across blocks on each
plane before selecting the governing plane. The weld-normal baseline, directional
curves, all directional damage totals and governing angle are retained in results.

Both existing read-outs work unchanged: `hotspot_b` at 0.5t with a
**detail-appropriate E curve**, or component-wise `hotspot_a` extrapolation with D.
There is no automatic curve reassignment or additional 1.12 multiplier.

This is an opt-in **2019/2020 F.12 method subset**, not current-edition DNV
qualification. Gamma and the basis for directional curve assignments must be
supplied explicitly. The new `cycle_associated` option uses an explicitly documented first-return
window convention, not a claim of complete DNV master/slave qualification.
F.12.1 crack-sector accumulation is not implemented. The Python API and saved-definition CLI support
it; the GUI rejects these definitions rather than silently losing the settings.
See [the algorithm, runnable example and verification record](docs/MULTIDIRECTIONAL.md).

Simplified method: `SimplifiedSettings(..., n0=1e4, conditions=(LoadCondition("loaded", "case_name", weibull_h=0.8, period_s=9.0),))`
with `assess_simplified`. An analysis is saved and re-run with
`AnalysisDefinition.save/load`; reports with `anyfatigue.report`.

Command line:

```bash
anyfatigue curves --environment seawater_cp
anyfatigue run analysis.json --stress stresses.npz --out reports/
anyfatigue run analysis.json --history hotspot.csv --out reports/
anyfatigue run analysis.json --anyfem deck.anyfem --out reports/   # saved ANYfem project
```

## What is read from ANYfem

Either an in-process solve (`anyfem_adapter.from_solutions`) or a **saved project**
(`anyfem_artifacts.read_result`, the `.anyres.h5` sidecars next to `deck.anyfem`; no
re-solve, read lazily so a long series on a large model is not held in memory). The
result must have been saved with global surface stresses (`return_global=True`);
its mesh sidecar is found by `mesh_hash`, load-case names and times come from the
result and the project's load cases.

Linear-static recoveries only: per shell element the top and bottom surface
tensors in **global axes** (`stresses(return_global=True)`), optionally the
patch-recovered nodal stresses, the corner nodes and each element's
`physical_director` as its normal. Pa becomes MPa (x 1e-6); the element value is
the mean over integration points; a node that patch recovery did not qualify
takes the mean of its elements and is listed in the provenance. Patch recovery
costs about 2 s per load case at 576 elements (ANYsolver's cost); pass
`patch=False` to skip it.

## Ecosystem ownership

* S-N curves, damage, hot-spot read-out, reports: **ANYfatigue** (this repository).
* Rainflow counting: **ANYtimeseries** (`anyqats.fatigue.rainflow`), consumed, not copied.
* Stresses and meshes: **ANYfem**, through its public postprocessing; the fatigue
  core depends only on a small neutral contract (`anyfatigue.model.StressSource`).

## Licence

MPL-2.0, as ANYfem ([LICENSE](LICENSE), [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).

## Known limits

The S-N data are April 2010 only (see above). Beam elements, nonlinear/transient
results, the second-order stress fit for coarse 4-node meshes (DNV 4.3.3),
mean-stress effects in welds, tubular-joint SCF methods and fracture-mechanics
assessment are not implemented. See
[docs/DESIGN.md](docs/DESIGN.md).


### Cycle-associated shear (development addition)

Set `shear_association="cycle_associated"` on `MultidirectionalSettings` (or its
`uniform` constructor). This requires the ANYtimeseries **indexed contract v1**;
the old published `anytimes 1.1.0` does not provide it. The missing capability
raises an actionable error; no global-envelope fallback is made.

Full cycles include the returning branch through the first return to the
starting normal level, including fractional-sample closure. Exactly matching
adjacent half cycles share a complete loop window while retaining their counts;
unmatched halves use their own intervals. `cycle_chunk_size` is optional and
changes computation only. Different LoadBlocks are still independent records.
Use `anyfatigue run ... --include-cycles` for JSON cycle-support and damage audit.
See [CYCLE_ASSOCIATION.md](docs/CYCLE_ASSOCIATION.md) for installation, examples,
conventions, verification and engineering limits. Existing definitions continue
to use `global_range`.
