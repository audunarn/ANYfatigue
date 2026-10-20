# Cycle-associated shear — living task record

## Scope and prospective acceptance, 2026-10-20

Stacked on PR #2 (84fcfa25882236069a91bd3b6c2d34d39f283d9f), not merged.
Keep global_range and old saved definitions unchanged. Opt-in cycle_associated
uses the indexed owner contract, rejecting absent/incompatible capabilities.
No new S-N data, gamma policy, crack-sector rule or standards qualification.

Prospective policy: chronological first-return envelope v1. A full normal cycle
uses the original history from its start reversal through its first return to
that normal level after its turning reversal, linearly interpolated between raw
samples. Half cycles use their two reversal endpoints. Nested excursions remain
in the chronological parent window; this is a stated envelope convention, not
a claim that multiaxial cycle association is unique or experimentally qualified.
Counter chunks share state; independent LoadBlocks remain independent records.
The tensor history is still retained per block; bounded-memory shear streaming
and periodic counting are not claimed.

Acceptance: hand-worked returning-branch and fractional closure cases, nested
loops, rare shear pulse exclusion, proportional/shear phase histories, full and
half cycles, chunk invariance, same-plane block accumulation, extraction and
tensor-completeness regressions, serialization/audit, missing-owner errors,
old-owner compatibility, installed wheels and cross-repository CI.

Use logarithmic range queries to avoid repeatedly scanning nested windows.
Record exact evidence below as it becomes available. No results yet.

## Final policy refinement and implementation

Before acceptance testing, the half-cycle policy was refined: repeated
constant-amplitude records can appear as adjacent 0.5-count rows in three-point
rainflow. Using only each half interval would miss the full out-of-phase shear
range on an otherwise complete loop. Therefore **exactly matching adjacent half
cycles** share their complete chronological window, while retaining both 0.5
counts. Pairing is greedy in chronological order, requires shared reversal
identity and exact return to the same sampled normal level, never crosses a
LoadBlock boundary and introduces no tolerance or fictitious cycles. Unmatched
halves retain their own interval and are labelled `half_interval`.

The final policy identifier is `chronological_first_return_paired_halves_v1`.
For a full cycle with start A, turn B and closing reversal C, find the first
original sample j after B that crosses the normal level at A. End the window
at that crossing, not automatically at C. With a crossing inside raw segment
(j-1,j), use linear interpolation of **both** shear and time at the fractional
closure. The fraction may round to an endpoint at float64 resolution. No
reversal-only downsampling is applied to the shear history. An exact crossing
ends at that first sample, not the far end of a subsequent plateau.

For `[0,4,2,3,1,5,0]`, the inner normal loop 2→3→2 has reversal indices (2,3)
and closing reversal 4. Its physical closure is at sample coordinate 3.5. The
outer full loop 4→1→4 closes at 4.75. A pulse at sample 6 is outside both
windows; it can still legitimately belong to the overall boundary-half pair.

On each fixed plane and block:

```text
normal cycles = indexed ANYtimeseries rainflow with record endpoints
shear_range[i] = max - min over the defined chronological support
range_for_SN[i] = gamma * hypot(normal_range[i], 0.9 * shear_range[i])
damage = exposure_scale * sum(count[i] / N(range_for_SN[i]))
```

SCF/thickness corrections and DFF retain the existing single-application path.
The existing floating-point normal/shear range resolution guard remains visible
in per-part output. Damage still accumulates across blocks on each fixed plane
before selecting the maximum. The ordinary weld-normal baseline is retained.

Min/max trees use O(N) storage/build work and O(log N) interval/first-crossing
queries. This avoids rescanning all samples for each nested window. The normal
counter is genuinely chunkable; this first consumer still retains the block's
stress history and constructs its trees. It does **not** claim bounded-memory
shear streaming. A `cycle_chunk_size` changes only how that same normal history
is fed to the counter; all indices remain block-local and results are invariant.

## Usage and dependency boundary

Install the owner and consumer development checkouts together, from their
respective indexed-rainflow and cycle-associated-shear branches:

```bash
python -m pip install -e ./ANYtimeseries
python -m pip install -e "./ANYfatigue[dev]"
```

Both package version numbers are unchanged development versions. Capability is
checked using `INDEXED_CYCLES_VERSION == 1`, not guessed from a version number.
The published pre-feature `anytimes 1.1.0` supports the legacy/global paths but
fails explicitly on `cycle_associated`; it never falls back to an envelope.
Reports include the indexed implementation's source-file SHA-256 when accessible.
CI pins the exact owner commit used by the consumer; see the workflow and PR.

Add the option to an existing, reviewed multidirectional configuration:

```python
from dataclasses import replace
from anyfatigue.report import write_json

# definition is your existing AnalysisDefinition, including parent selection,
# complete tensor source, reviewed gamma and directional S-N assignments.
updated_md = replace(
    definition.settings.multidirectional,
    shear_association="cycle_associated",
    cycle_chunk_size=4096,  # optional; no change to record boundary semantics
)
updated = replace(definition, settings=replace(definition.settings,
                                              multidirectional=updated_md))
result = updated.run(source, keep_series=False)
write_json(result, "cycles.json", include_cycles=True)
updated.save("analysis.json")
```

```bash
anyfatigue run analysis.json --stress stresses.npz --out reports/ --include-cycles
```

`MultidirectionalSettings.uniform(...)` also accepts `shear_association` and
`cycle_chunk_size`. Default association stays `global_range`. Old JSON without
a chunk-size key round-trips unchanged; global mode rejects an irrelevant
chunk-size argument. Different LoadBlocks remain independent finite records;
join physically continuous input into one block before counting.

## Audit and engineering limits

`PartResult.cycles` retains normal range, mean and count. `equivalent_ranges`
contains the pre-SCF/thickness interaction range; `cycle_damage` contains the
individual exposure-scaled Miner contributions, which sum to part damage.
`cycle_association` retains raw reversal/closing indices, support start, closure
segment/fraction, original cycle kind, support kind, pairing partner row,
shear min/max and applied range. Available timestamps are retained even when
`keep_series=False`. All indices refer to the saved ordered **block cases**,
not global source row numbers. JSON cycle audit is opt-in; the ordinary points
CSV and compact JSON summaries remain compatible. Only the governing plane's
full detailed cycle audit is retained, as in the previous implementation.

This is a reproducible normal-master **envelope convention**, not a unique
multiaxial decomposition or a claim of complete/current-edition DNV F.12
qualification. Nested physical excursions remain in parent windows and can
contribute to multiple counted loops. Equal-level/tie behaviour and unmatched
finite-record boundary halves are explicit; record-length and time-resolution
convergence remain engineering checks. No phase-angle correction is inferred.
Gamma and directional S-N classifications still require engineering review.
Pure shear on a plane without normal-master cycles is still not a stand-alone
shear-initiation model; the existing diagnostics remain. No new S-N edition,
crack-sector accumulation, periodic counting or GUI editor is added.

## Evidence

Local Linux/Python 3.13.5: combined ANYfatigue source suite **481 passed, 34
skipped**, including 54 new tests. The skips are 32 existing optional ANYfem
producer tests and two Qt modules unavailable locally. Hand-worked cases include
fractional/nonuniform-time closure, pulse exclusion, exact-return plateaus,
paired out-of-phase halves, unmatched halves, nested chronological envelopes,
chunk and independent-block semantics, uniaxial correction factors, parent-only
A+D/B+E extraction on both faces, frame invariance, persistence/CLI/JSON audit,
and missing tensor/owner-contract rejection. Random full-loop windows were also
checked against a separately implemented direct original-sample scan.

The entire pre-existing headless fatigue suite also passed against the new owner
before adding consumer tests (427 passed, 34 skipped). The old-owner compatibility
matrix and installed-wheel/CI evidence are recorded in the PR. Tests establish
the stated implementation, not experimental fatigue qualification. The
workstation was offline; its global instructions were unavailable, its files
were not modified, and accessible repository/governance instructions were used.

## Development installation and CI dependency

The paired owner is ANYtimeseries PR #306, frozen at
`6132cf15e010b13c231ed59f6ecc1c01e9ddea22` in `requirements-indexed.txt`.
From this ANYfatigue checkout, install the normal dependencies followed by the
explicit development owner:

```bash
python -m pip install -e ".[dev]"
python -m pip install --no-deps --force-reinstall -r requirements-indexed.txt
```

All eight ordinary Ubuntu/Windows consumer CI jobs use this exact owner and run
the complete suite. A separate job uses the original published anytimes 1.1.0,
runs legacy scalar/global regression, and verifies explicit new-mode rejection.
It excludes only the new-mode happy-path test file, which is executed by every
full-suite job. Package versions are unchanged; results also record indexed
contract version and the SHA-256 of the executed owner source module.

This consumer branch is stacked on unmerged ANYfatigue PR #2 (parent-plate
extraction). Merge that prerequisite first and retarget this PR to main before
merging it. No automatic merge or package publication is performed here.
