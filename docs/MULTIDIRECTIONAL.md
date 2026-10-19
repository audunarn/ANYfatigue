# Multidirectional shell fatigue — implementation and verification record

Living task note, 2026-10-19. Development addition only; no release or qualification.

## Scope, acceptance and ownership

The request is fatigue at shell-model bracket toes when stress directions rotate
within a time series. Preserve direct 0.5t read-out with a detail-appropriate E
curve and component-wise hot-spot extrapolation with D. Existing scalar analyses
must remain unchanged. ANYfatigue owns stress resolution, S-N resistance, damage
and reporting; ANYtimeseries continues to own rainflow counting. No second
counter or new runtime dependency is introduced.

The prospective acceptance checks were: transformation and uniaxial closed
forms, rotating constant principal stresses, non-proportional shear, frame
invariance, block accumulation, both extraction paths, persistence/reporting,
invalid inputs, existing tests and an installed-wheel smoke check. Current
results and limitations are recorded below, not treated as fatigue qualification.

## Implemented algorithm

`TimeSeriesSettings.multidirectional` is an optional `MultidirectionalSettings`.
When absent, serialization and the scalar path are unchanged. When present:

1. Extract the existing synchronized **surface tensor** histories. Stress is in
   MPa, geometry in metres, thickness in millimetres. The global tensor order is
   `xx, yy, zz, xy, yz, xz`; shear entries are physical stresses, not engineering
   shear strains. The selected top/bottom faces are assessed separately.
2. Use a fixed material frame: surface normal `n`, projected weld tangent `t`,
   perpendicular-to-weld axis `e = cross(n, t)`. An angle is the **stress/plane
   normal direction**, measured from `e` towards `t`, not the crack-trace angle.
3. For each fixed angle, transform the in-plane tensor to signed normal and
   shear histories. With `p = e.S.e`, `q = t.S.t`, `v = e.S.t`:

   ```text
   normal(theta) = p*cos(theta)^2 + q*sin(theta)^2 + 2*v*sin(theta)*cos(theta)
   shear(theta)  = (q-p)*sin(theta)*cos(theta) + v*(cos(theta)^2-sin(theta)^2)
   ```

4. Rainflow-count the **normal history** using the existing ANYtimeseries
   adapter with endpoints retained. Do not count an instantaneous principal,
   von Mises or unsigned effective-stress history.
5. For each block and plane, use `shear_range = max(shear) - min(shear)` over the
   **entire block**. For each counted normal range:

   ```text
   equivalent_range = gamma * sqrt(normal_range^2 + 0.81*shear_range^2)
   ```

   Gamma multiplies the whole expression. It is a required input, not inferred
   from a principal-direction tolerance. Values below one are rejected. The
   user must record `gamma_basis` and `curve_basis`. Gamma is common to all
   blocks in this assessment and must be justified accordingly.
6. Enter the equivalent range into the explicitly assigned plane's S-N curve.
   The existing SCF, curve-specific thickness correction, block duration/life
   exposure and DFF logic is reused. Extrapolation is already component-wise.
   There is **no hidden 1.12 multiplier or automatic E/D curve conversion**.
7. Accumulate on each fixed plane before taking the maximum:

   ```text
   D(theta) = sum(block_exposure_scale * sum(count / N(equivalent_range)))
   D_governing = max_theta D(theta)
   ```

   Neither all trial directions nor block-by-block direction maxima are added.
   The ordinary weld-normal result is retained as a baseline. Exactly tied
   direction totals select the smallest supplied angle deterministically.

`FatiguePlane(angle_deg, curve)` allows explicit direction-dependent resistance.
Angles must be unique in `[0, 180)`, include zero, and span all three independent
in-plane tensor components. The zero-degree curve must equal the baseline
`TimeSeriesSettings.curve`. `MultidirectionalSettings.uniform(...)` explicitly
assigns one chosen curve to all angles; it does **not** classify a welded detail.
The actual uniform step is `180 / ceil(180 / step_deg)`, no larger than the
requested step. The uniform constructor limits the grid to 10000 planes as an
input-size guard, not a convergence criterion. Refine the grid and verify damage
convergence for the application; the result is a sampled maximum, not a certified
continuous-angle maximum.

### Numerical resolution and degenerate planes

The implementation uses exact quadrantal double angles. It also records a
normal-range resolution guard of `128 * float64_epsilon * max(abs(tensor))` for
each block/point. This is a floating-point operation-chain allowance, not a
material fatigue limit or a configurable noise filter. Normal cycles at or
below that scale are discarded **before** adding the full shear envelope;
otherwise cancellation noise can become significant artificial fatigue damage.
Discarded counts and the bound are reported. Shear ranges below the same
resolution are zeroed. Data requiring finer precision must be recovered with
adequate numerical resolution; this guard cannot recover lost input precision.

A plane with cyclic shear but no resolved normal-master cycles does not produce
cycles on that plane. Such combinations are noted explicitly, not labelled a
stand-alone pure-shear assessment. Other plane orientations detect the pure-shear
regression. This is a normal-master model, not a general shear-initiation model.

## Public API example

This executable example uses the existing synthetic shell model. **Gamma=1 is a
software demonstration input here, not a recommendation for a non-proportional
bracket-toe assessment.** Project gamma and curve assignments need review.

```python
from dataclasses import replace
from anyfatigue import MultidirectionalSettings
from anyfatigue.examples import synthetic_plate
from anyfatigue.report import write_json, write_report

source, base = synthetic_plate(steps=60)
md = MultidirectionalSettings.uniform(
    base.settings.curve,
    gamma=1.0,
    gamma_basis="Demonstration only; no design value inferred.",
    curve_basis="Uniform curve explicitly assigned for this software demonstration.",
    step_deg=5.0,
)
definition = replace(base, settings=replace(base.settings, multidirectional=md))
result = definition.run(source, keep_series=False)
point = result.critical
print(point.damage, point.critical_angle_deg, point.normal_to_weld_damage)
for direction in point.directions:
    print(direction.angle_deg, direction.curve.name, direction.damage)
definition.save("multidirectional-analysis.json")
write_json(result, "multidirectional-result.json", include_cycles=True)
write_report(result, "multidirectional-report.md")
```

For a real analysis, attach the option to the existing settings and keep the
chosen locations/read-outs. An E curve at 0.5t is not automatically valid for
all trial directions. Use explicit `FatiguePlane` assignments when classifications
differ; no unverified angular interpolation is embedded.

The existing CLI consumes the saved definition:

```bash
anyfatigue run multidirectional-analysis.json --stress stresses.npz --out reports/
```

`directional_stress_history(...)` and `multidirectional_equivalent_range(...)`
are also public numerical helpers in `anyfatigue.multidirectional`.

## Results, persistence and GUI boundary

`PointResult.directions` records every trial angle, full curve definition, total
damage and per-block damages in settings block order. `critical_angle_deg` and
`normal_to_weld_damage` are exposed directly. Only the governing plane's detailed
`parts` are retained, so memory does not grow with all angles' complete histories.

`PartResult.cycles` retains **normal** ranges, means and counts. Its new
`equivalent_ranges` holds the ranges actually used for damage, and `curve` names
the actual directional resistance. Equivalent means are not invented. A retained
`series` is the signed governing-plane normal history, not an effective history.
JSON with `include_cycles=True` labels this distinction; all JSON reports include
directional summaries. Markdown identifies the baseline curve and lists the
critical point's directional damages. The existing CSV remains the points-level
summary; use JSON for the full directional audit. The spectrum helper uses the
actual equivalent ranges and that part's curve/thickness exponent.

Mean-stress correction and incompatible scalar measures are rejected rather than
silently ignored. The old scalar GUI cannot represent these controls and therefore
rejects a multidirectional definition **before changing its forms**, with a Python
API/CLI instruction. No GUI editor for this option is claimed.

## Source basis and boundaries

* DNVGL-RP-C203, September 2019 amended January 2020, Appendix F.12, describes
  the normal-master/global-shear-range alternative. The available text was read
  from the [reproduced DNV document](https://www.academia.edu/144077767/DNV_RP_C203_2020_fatigue_design_of_offshore_steel_structure),
  printed pp. 265–267. No standard file is redistributed in this repository.
* The interaction and transformation equations were visually checked against
  Winther et al. (2024), *Effect of non-proportional stress states caused by
  varying phase shifts on the fatigue life of welded joints*, International
  Journal of Fatigue 185, 108351, equations (3)–(4), printed p. 3;
  [DOI](https://doi.org/10.1016/j.ijfatigue.2024.108351),
  [publisher PDF in the university repository](https://vbn.aau.dk/ws/portalfiles/portal/773039071/1-s2.0-S0142112324002093-main.pdf).
  Its experimental factors are **not** adopted as universal design values.
* Built-in S-N data remain **April 2010**, exactly as documented in DESIGN.md.
  DNV's [catalogue](https://www.dnv.com/energy/standards-guidelines/dnv-rp-c203-fatigue-design-of-offshore-steel-structures/)
  lists edition 2024-10 amended 2025-10; those full provisions were not verified.

This is **not a complete implementation of every F.12 procedure**. In particular,
F.12.1 crack-sector accumulation, automatic classification, experimentally
calibrated non-proportional factors and true indexed master/slave association
are outside this addition. Whole-block shear is an envelope for each assigned
cycle amplitude; it is **not a proof of conservative total fatigue damage for
arbitrary independently varying histories**. It may strongly overestimate damage
when a rare shear peak is reused on numerous normal cycles. Longer blocks and
changes in shear/normal cycle-rate relationships require particular care. Do not
subdivide records just to improve the answer: doing so also changes counting and
the shear envelope.

The installed owner API (`anytimes 1.1.0`) supplies no cycle interval indices.
No indices are guessed from aggregated ranges. Exact matching needs an explicit
indexed-cycle contract from ANYtimeseries and separate validation. The input
remains a fixed-frame, in-plane shell surface history, not a rotating-frame or
3D solid-fatigue implementation.

## Verification evidence and handoff

Base: main `98fcbaee9079b64871420b61bae84c04059b3d88`. Work isolated on
`feat/multidirectional-shell-fatigue`. The connected workstation was unavailable;
its files were not modified. Repository AGENTS/ECOSYSTEM_GUIDE and the accessible
ANYopenSoft philosophy were read; unavailable workstation-specific instructions
were not assumed. The temporary development workflow is removed from the final
source tree. No version bump, release, tag or package publication is requested.

Local environment: Linux, Python 3.13.5, numpy 2.3.5, scipy 1.17.0, pytest 9.0.2,
setuptools 82.0.1, published anytimes 1.1.0. Commands/evidence:

* Baseline `python -m pytest -q -rs`: **299 passed, 33 skipped**.
* After implementation, full suite with JUnit output: **363 passed, 33 skipped**.
  The 64 new headless cases include analytical transformation/interaction,
  uniaxial limit, gamma/SCF/thickness applied once, rotating fixed eigenvalues,
  shear extrema between normal extrema, pure shear, half cycles, constant
  tensors, max-after-block-sum, mixed directional curves/exponents, 3D rigid-frame
  rotation, angle refinement, roundoff protection, A+D and B+E extraction on both
  surfaces, save/load/CLI/reporting and invalid input/overflow checks.
* The skips are the existing optional ANYfem producer tests (32) and the entire
  Qt module (one module skip). A new offscreen GUI guard test is included for the
  existing Ubuntu/Windows, Python 3.11–3.14 CI matrix. Exact CI outcome belongs to
  the pull request's checks; workstation/ANYfem integration is not claimed here.
* `pip wheel . --no-deps --no-build-isolation --no-index`: successful. Installed
  outside the checkout with existing runtime dependencies explicitly exposed in
  the test venv; confirmed import from the installed wheel, no Qt/ANYfem/h5py
  import, saved-definition round trip, API/report and CLI run on 10 synthetic
  hot-spot surface points and 36 directions. CLI exited zero. An initial smoke
  harness run lacked scipy because the nested venv did not inherit the outer
  venv; after correcting dependency visibility, the same wheel passed.

These checks establish the stated implementation behaviour, not independent
fatigue-life qualification. Real bracket-toe model validation, current-edition
review, classification/gamma justification and grid/record convergence remain
engineering acceptance activities.
