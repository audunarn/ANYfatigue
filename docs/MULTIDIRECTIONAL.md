# Multidirectional shell fatigue — living implementation note

## Task and prospective acceptance (2026-10-19)

Add an opt-in, headless multidirectional shell fatigue path for bracket toes, preserving existing scalar analyses and both extraction paths: direct stress at 0.5t with a detail-appropriate E curve, and component-wise extrapolation with a D curve. ANYfatigue owns tensor resolution and damage; ANYtimeseries remains the rainflow owner. Do not turn instantaneous unsigned equivalent/principal stress into a rainflow history. Assess complete synchronized signed tensors in a fixed material frame.

Before merge: analytical transformation and uniaxial limits; rotating constant principal stresses; non-proportional shear; angle invariance; block accumulation; extraction-path integration; persistence/reporting; invalid inputs; existing CI. Explicitly record the shear association policy, direction-dependent curve selection, non-proportional factor and source editions. No claim of current-edition DNV qualification or a release.

Principal uncertainty: whether the owner counting API retains cycle intervals. Inspect before selecting an association algorithm; do not copy the rainflow implementation into this repository or infer intervals from aggregated ranges. The documented global-shear-range alternative may provide an explicit supported first mode if cycle indices are unavailable. No invented non-proportional factor.

Environment: connected workstation offline; its global AGENTS and locally adopted policy revision are unavailable. Repository instructions, ECOSYSTEM_GUIDE and the remotely available ANYopenSoft philosophy were read. Work is isolated on feat/multidirectional-shell-fatigue; do not alter the workstation. A temporary branch-only workflow exports the exact public source snapshot and published rainflow dependency for local checks; remove it before final delivery.

Status: implementation and evidence pending.
