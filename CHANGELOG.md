# Changelog

## 0.1.0 (unreleased)

First implementation.

* S-N curves from DNV-RP-C203 (April 2010) Tables 2-1 to 2-3 (in air, seawater with
  cathodic protection, free corrosion), thickness correction, SCF, custom curves.
* Simplified Weibull fatigue (DNV-RP-C203 section 5, D.13-1) from one load case per
  condition; reproduces ANYstructure's `CalcFatigue`.
* Rainflow time-series fatigue from a series of quasi-static load cases, using
  ANYtimeseries for counting; optional non-welded mean-stress reduction (2.5.1).
* Stress locations: elements, lines of nodes, interpolated points; direct or DNV hot-spot
  (method A and B) read-out; weld-direction resolution and DNV effective hot-spot range.
* ANYfem adapter for linear-static results (in-process); stress exchange file (`.npz`)
  and stress-history CSV for use without ANYfem.
* Reader for saved ANYfem projects (`.anyres.h5` sidecars): lazy, global surface stresses,
  patch-recovered nodal stresses with unqualified-node fallback, times from the project's load
  cases; available in the GUI, the CLI (`--anyfem`) and the API.
* Results, reports, the CLI and the GUI state that the S-N data are from the April 2010
  edition and that a newer one (2024-10, amended 2025-10) has not been checked.
* MPL-2.0 licence.
* Qt GUI (`run_gui.py`) in ANYfem's workbench style, command line, saved analysis files,
  CSV / JSON / Markdown reports.
