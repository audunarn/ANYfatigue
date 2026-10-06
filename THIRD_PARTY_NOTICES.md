# Third-party notices

ANYfatigue uses the following direct dependencies. They are installed separately
and are not copied into ANYfatigue source or binary distributions. Each dependency
remains under its own license; transitive dependencies and binary-wheel notices
remain governed by the notices shipped by their distributors.

| Dependency | Scope | License | Upstream |
| --- | --- | --- | --- |
| anytimes (ANYtimeseries) | runtime (rainflow counting) | MIT | https://github.com/audunarn/ANYtimeseries |
| ANYfem | optional `anyfem` extra | MPL-2.0 | https://github.com/audunarn/ANYfem |
| build | development | MIT | https://github.com/pypa/build |
| h5py | optional `anyfem` extra | BSD-3-Clause | https://www.h5py.org/ |
| NumPy | runtime | BSD-3-Clause and bundled-component licenses | https://numpy.org/ |
| PySide6 / Qt | optional `gui` extra | LGPL-3.0-only route | https://doc.qt.io/qtforpython-6/commercial/index.html |
| pytest | development | MIT | https://pytest.org/ |
| SciPy | runtime | BSD-3-Clause and bundled-component licenses | https://scipy.org/ |

The Qt frontend uses a dynamically installed PySide6 under its LGPL route and
does not bundle Qt libraries in the ANYfatigue wheel. A standalone installer
would have to include the applicable notices, source access and replacement
instructions; this entry does not establish installer compliance.

## Standards

S-N curve parameters and the equations implemented here are transcribed from DNV-RP-C203
*Fatigue design of offshore steel structures*. The standard is published by DNV and is not
reproduced in this repository; only numerical parameters needed to evaluate its curves are
recorded, with their source edition. Obtain the standard from DNV for the authoritative text.
