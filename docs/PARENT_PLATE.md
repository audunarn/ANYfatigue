# Parent-plate extraction and tensor completeness

Living task record, 2026-10-19. Base: main `0c7f146aee8418e24e4036be9e403a231cad3eb5`.
Development implementation, not a release or a new fatigue-method qualification.

## Scope and prospective acceptance

Add an explicit one-sided parent recovery patch to line and point locations.
Restrict point search, normals and stress recovery to that patch. Never reuse an
unscoped nodal field that may already include the attached member. Retain both
0.5t (method B) and component-wise extrapolated (method A) read-outs, without
changing S-N curves, the rainflow counter or the multidirectional interaction.

Primary uncertainty: can attached-plate stress or opposite-side recovery support
contaminate the selected surface? Check this with independently specified tensors
on a T-junction, element renumbering, rigid-frame rotation and both physical faces.
Also check side, topology, ambiguity, off-patch sampling, persistence and reporting.
For completeness check scalar, partial, complete and explicitly completed CSVs,
NPZ round trips/legacy migration, and the public multidirectional entry point.

## ParentPlate contract

```python
from anyfatigue import ParentPlate
from anyfatigue.extract import LineLocation, ReadOut

# These IDs must be taken from this model's selected supporting-plate patch.
# Select the elements on ONE side of the toe, not the attached bracket and
# not the complete plate on both sides of the weld.
parent = ParentPlate(
    element_ids=(20, 30),
    nodal_recovery="element_average",  # deliberately required, not a hidden fallback
    name="supporting plate, positive-x side",
    mesh_id="junction-v1",             # optional upstream provenance['mesh_hash']
)
toe = LineLocation(
    node_ids=(1, 4),
    surfaces=("top", "bottom"),
    parent_plate=parent,
    weld_direction=(0, 1, 0),
    readout=ReadOut("hotspot_a", thickness_mm=20, direction=(1, 0, 0)),
    name="bracket toe",
)
```

`PointLocation` accepts the same `parent_plate` field. `ReadOut("hotspot_b", ...)`
selects the 0.5t route. Existing analysis settings supply the appropriate S-N curve;
this addition does not automatically classify B as E or A as D. Old definitions
without `parent_plate` retain their JSON structure and normal single-plate route.

The patch retains the producer's element/node IDs. When `mesh_id` is supplied it
must match the source's upstream `mesh_hash`; missing or different identity is an
error. Without that binding, selection is explicitly relative to the loaded mesh.
No coordinate-based identity or automatic remeshing migration is introduced.

### Actual recovery and physical faces

The initial supported method is **explicit, unweighted incident-element averaging
inside the selected patch**. Element stresses are fetched once per face, including
for lazy producers; the original source's global nodal stress function is never
called. Patch-local values are then interpolated using the existing shape functions.

This prevents contamination but is NOT equivalent to a qualified extrapolated or
patch-recovered nodal stress field. Sampling-point values and resulting fatigue
still need a mesh/recovery convergence check. A region-qualified producer nodal
contract is a future extension; requesting `source_nodal` is rejected, not silently
replaced by averaging.

The patch must be edge-connected, planar, free of duplicate cells/non-manifold
edges, and have consistent producer physical directors. Top and bottom values are
never silently swapped to align opposite directors. Geometric corner winding may
be opposite when the producer explicitly supplies a consistent physical director.
Curved/folded regions must be split; this first slice does not qualify curved-plate
parent recovery. The geometric checks use a floating-point guard of
`256 * epsilon * max(abs(coordinates), characteristic_length)` in metres, with
that guard divided by characteristic length for direction checks. It is not an
engineering allowable plate angle. Poorly resolved coordinates fail closed.

For hot-spot sampling, **every patch vertex must lie on the selected read-out side
of each toe origin**. Patches spanning both sides are rejected, even if a probe
happens to lie on the intended side. Split a changing weld direction into suitable
patches. This is an explicit region contract, not automatic geometry segmentation.
The sampled points must lie on the selected mesh patch within the existing search
tolerance; there is no fallback to the bracket or another region. Probe membership
is checked; a full geometric certification of every connecting ray segment through
holes is not claimed.

Unselected hot-spot line nodes with inconsistent physical normals now fail with an
instruction to supply a parent patch. Point interpolation also rejects coincident
ambiguous surfaces. A genuine common edge/node of the same oriented sheet remains
valid and deterministic. These guards intentionally reject previously ambiguous
inputs rather than preserving an arbitrary element-ID-based result.

## TensorCompleteness contract

```python
from anyfatigue import TensorCompleteness
from anyfatigue.stress_io import read_history_csv

# Legacy scalar input still reads, but only xx is declared supplied.
scalar_source = read_history_csv("scalar-history.csv")

# Explicit engineering assumptions, never inferred from zero-filled slots.
completed_source = read_history_csv(
    "scalar-history.csv",
    assumed_zero_components=("yy", "zz", "xy", "yz", "xz"),
    assumption_basis="Prescribed uniaxial global-x loading for this verification case",
)
```

The record distinguishes `provided`, `assumed_zero`, and the remaining `missing`
components, all in **fixed global axes** with order `xx, yy, zz, xy, yz, xz` and
physical shear stresses. Assumptions require a non-empty basis, must not overlap
provided components, and must agree with the actual data. Repeated column names,
unknown components, conflicting metadata and complex frequency-domain amplitudes
are rejected. Numerical zeros in a supplied column remain *provided*, not assumed.

Programmatic `ArrayStressSource` and `source_from_history` constructors take six
explicit values and therefore default to six provided components. Importers with
padded/unknown slots must pass their actual `tensor_completeness` declaration.
A third-party StressSource without a declaration is unknown. The ANYfem artifact
adapter declares all six only after its existing complete-global-field validation.

Before multidirectional damage is calculated, the declaration must cover every
component that affects the selected in-plane tensor. Consequently `xx, yy, xy`
are sufficient for an exactly global-xy surface; absent `zz, yz, xz` stay unknown
and are reported as such, not reclassified as zero. An inclined surface generally
requires all six global components. There is no angular threshold that drops a
small but nonzero dependence on an unknown component. Scalar histories remain
available through the legacy scalar analysis; no claim of tensor completeness is
made for them.

### Files, reports and GUI boundary

Stress NPZ writes use **schema version 2** and retain the declaration explicitly.
Version 1 files remain readable for scalar use, but files without a component
declaration are marked unknown. Re-saving or materialising them cannot promote
padded arrays or old provenance into a complete tensor. To enable multidirectional
assessment, re-import an explicit tensor or deliberately supply and document the
reviewed component declaration. New v2 files missing this metadata are invalid.
Both the stress unit and component order are checked on import.

Analysis definitions, CLI runs and result JSON retain the parent selection,
read-out side and upstream mesh binding. Report text names the actual parent
recovery rather than claiming that the source's unscoped nodal field was used.
Tensor availability and zero-assumption basis are included in report provenance
and Markdown. The existing GUI rejects parent-plate definitions before changing
forms, because it cannot edit/preserve this option; use the Python API or saved
analysis CLI. No GUI editor is included in this slice.

## Verification and execution record

The repository instructions, ecosystem guide and accessible canonical philosophy
were read. Workstation-specific global instructions were unavailable; workstation
files were not modified. Work is isolated on `feat/parent-plate-extraction`.
A development snapshot workflow creation was blocked and was not retried. Source
was instead reconstructed from the retained public source snapshot and connector
reads; the merged multidirectional core, result types and report baseline were
checked against their Git blob hashes before integration.

Local Linux/Python 3.13.5: **64 new headless tests passed**, covering the analytical
T-junction, contamination independence, both read-outs/faces, renumbering, 3-D
rotation/translation, side/topology/orientation errors, lazy reads, NPZ/CLI/analysis
round trips, actual recovery reporting, scalar vs multiaxial completeness, complete
in-plane and full-global tensors, explicit zeros, legacy files and invalid inputs.
The available local regression set gives **363 passed, 34 optional-dependency
skips**: 299 original headless regressions plus the 64 new tests. The 64 previously
merged multidirectional test cases were not part of the retained local test-file
snapshot; the PR's full repository CI is the authoritative combined-suite result.
Qt and optional ANYfem dependencies are unavailable locally. A new Qt form-safety
regression is included for the repository's existing CI matrix.

Built the development wheel with `pip wheel . --no-deps --no-build-isolation
--no-index`, installed it to a separate directory, and ran from outside the source
checkout with only that installed package on PYTHONPATH. The smoke test passed:
parent-only T-junction assessment, analytical uniaxial damage, both faces, saved
analysis and stress-file round trips, result/Markdown reports, CLI exit zero,
and persistence/rejection of missing tensor components. Confirmed the package
imported from the installation and did not load Qt/ANYfem/h5py on import.
Development wheel SHA-256:
`b0d2c415b4edac716ce4ee6490ffff1c8a8bcfa45cdffc24accd9d829317baa5`.
Runtime source was unchanged after this build; documentation records may be
updated to record subsequent CI evidence.

No version bump, tag, package publication, current-edition DNV verification or
fatigue-life qualification is claimed. The PR verification records the exact
commit and complete CI outcome. Existing multidirectional/gamma/edition limits
remain unchanged.
