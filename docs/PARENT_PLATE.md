# Parent-plate extraction and tensor completeness

Living task record, 2026-10-19. Base: main 0c7f146aee8418e24e4036be9e403a231cad3eb5.

## Prospective scope

Add an explicit parent-plate/one-sided recovery patch to line and point locations.
Restrict point search, local normals and stress recovery to the selected element IDs.
Never reuse globally averaged or patch-recovered nodal stresses without an explicit
region-scoped producer contract. The initial supported recovery is deliberately
selected element averaging on a consistently oriented planar patch; it is not a
new stress-recovery qualification. Reject ambiguous multi-surface locations,
mixed physical-face orientations, off-patch probes and patches crossing the
specified read-out side. Preserve conventional single-plate definitions and both
0.5t (B) and component-wise extrapolated (A) read-outs.

Track provided, explicitly assumed-zero and missing tensor components separately.
Retain legacy scalar CSV use, but require complete declared tensors for
multidirectional analysis. Preserve that declaration through history import,
materialisation, NPZ and reporting; old files without declarations remain unknown
rather than being silently promoted to complete.

## Acceptance and decision-oriented checks

Primary uncertainty: whether either attached-plate stress or opposite-side recovery
support can contaminate the selected fatigue surface. Use analytical connected-plate
fixtures with distinct tensors, renumbering and rigid-frame transforms. Test both
surfaces, both read-outs, invalid selection/frame/side, serialization and reports.
For completeness test scalar, partial, full and explicitly completed CSV histories;
NPZ round trips and old-file migration; missing metadata at public analysis entry.
Run focused checks and the existing CI matrix, with no changed counting/S-N model.

No release, standards-edition update, automatic plate recognition, GUI editor or
workstation modification is included. GUI must fail closed for definitions it
cannot preserve. Global workstation instructions were unavailable; repository
AGENTS, ECOSYSTEM_GUIDE and the accessible canonical philosophy were read.

Status: implementation and evidence pending.
