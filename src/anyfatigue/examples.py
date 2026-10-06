"""Ready-made stress sources and analyses, for the GUI's example menu and tests.

``synthetic_plate``
    A closed-form stress field on a flat plate with a weld line, varying over
    an irregular load series.  It needs no ANYfem and is **not a finite-element
    result**; its provenance says so.

``anyfem_plate``
    A simply supported 1 m square plate under a pressure series, solved with
    ANYfem (needs the ``anyfem`` extra).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from . import sn_curves as sn
from .extract import LineLocation, PointLocation, ReadOut
from .model import ArrayStressSource, SurfaceMesh
from .project import AnalysisDefinition
from .simplified import LoadCondition, SimplifiedSettings
from .timeseries import LoadBlock, TimeSeriesSettings

__all__ = ["anyfem_plate", "synthetic_plate", "wave_factors"]


def wave_factors(n: int = 160, dt: float = 0.5, seed: int = 4) -> np.ndarray:
    """An irregular, zero-mean load factor series (sum of three sinusoids)."""

    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    periods = np.array([9.0, 5.5, 3.2])
    amplitudes = np.array([1.0, 0.55, 0.3])
    phases = rng.uniform(0.0, 2.0 * math.pi, size=3)
    return sum(a * np.sin(2.0 * math.pi * t / p + ph) for a, p, ph in zip(amplitudes, periods, phases))


def synthetic_plate(
    nx: int = 12, ny: int = 8, steps: int = 160, dt: float = 0.5
) -> Tuple[ArrayStressSource, AnalysisDefinition]:
    """A 1.2 x 0.8 m plate, weld along y at x = 0.6, stress peaking mid-weld."""

    lx, ly = 1.2, 0.8
    xs, ys = np.linspace(0.0, lx, nx + 1), np.linspace(0.0, ly, ny + 1)
    index = {}
    node_ids, xyz = [], []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nid = 1000 + j * (nx + 1) + i
            index[(i, j)] = nid
            node_ids.append(nid)
            xyz.append((x, y, 0.0))
    elements, corners = [], []
    for j in range(ny):
        for i in range(nx):
            elements.append(1 + j * nx + i)
            corners.append((index[(i, j)], index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]))
    mesh = SurfaceMesh(np.array(node_ids), np.array(xyz), np.array(elements), np.array(corners))

    def field(points: np.ndarray) -> np.ndarray:
        """Stress per unit load factor (MPa): sigma_xx peaks at the weld, dips off it.

        The amplitude is chosen so that the example sits just inside the
        criterion (D x DFF about 0.8), which makes both a pass and a small
        parameter change that fails visible in the GUI.
        """

        x, y = points[:, 0], points[:, 1]
        weld = np.exp(-(((x - 0.6) / 0.12) ** 2))
        along = 0.5 + 0.5 * np.exp(-(((y - 0.4) / 0.25) ** 2))
        out = np.zeros((len(points), 6))
        out[:, 0] = 8.5 * weld * along + 1.0
        out[:, 1] = 0.25 * out[:, 0]
        out[:, 3] = 1.5 * weld * along
        return out

    factors = wave_factors(steps, dt)
    base_el, base_nd = field(mesh.centroids()), field(mesh.node_xyz)
    el_top = factors[:, None, None] * base_el[None]
    nd_top = factors[:, None, None] * base_nd[None]
    labels = [f"t{k:03d}" for k in range(steps)]
    source = ArrayStressSource(
        mesh, labels, el_top, -el_top, times=[k * dt for k in range(steps)],
        node_top=nd_top, node_bottom=-nd_top, nodal_origin="closed-form field",
        provenance={"kind": "synthetic example (closed-form field, not an FE result)"},
    )
    weld_nodes = tuple(index[(nx // 2, j)] for j in range(ny + 1))
    locations = (
        LineLocation(weld_nodes, surfaces=("top",),
                     readout=ReadOut("hotspot_a", thickness_mm=20.0, side=1), name="weld"),
        PointLocation(((0.6, 0.4, 0.0),), surfaces=("top",), weld_direction=(0.0, 1.0, 0.0),
                      readout=ReadOut("hotspot_b", thickness_mm=20.0, side=1), name="toe"),
    )
    settings = TimeSeriesSettings(
        curve=sn.get_curve("D", "air"), thickness_mm=20.0, design_life_years=25.0, dff=3.0,
        blocks=(LoadBlock("sea state", tuple(labels), 1.0),),
    )
    return source, AnalysisDefinition("time_series", settings, locations, name="synthetic plate",
                                      source={"kind": "example", "name": "synthetic_plate"})


def simplified_for(source: ArrayStressSource, definition: AnalysisDefinition) -> AnalysisDefinition:
    """The same locations assessed with the simplified method on the peak case."""

    series_peak = int(np.argmax(np.abs(source.element_stress([0], "top")[:, 0, 0])))
    settings = SimplifiedSettings(
        curve=definition.settings.curve, thickness_mm=definition.settings.thickness_mm,
        design_life_years=definition.settings.design_life_years, dff=definition.settings.dff,
        n0=1.0e4, conditions=(LoadCondition("sea", source.case_labels[series_peak], 0.9, 8.0),),
    )
    return AnalysisDefinition("simplified", settings, definition.locations, name=definition.name,
                              source=definition.source)


def anyfem_plate(pressure_pa: float = 3_500.0, side: float = 1.0, thickness_m: float = 0.010,
                 target_size: float = 1.0 / 16.0):
    """Solve a simply supported plate under a pressure wave and return (source, definition).

    The default 3.5 kPa amplitude puts the governing point just inside the
    criterion (D x DFF about 0.8).  Patch recovery of 48 load cases makes this
    take about a minute on a 16 x 16 mesh; that cost is ANYfem's, not ANYfatigue's.
    """

    try:
        from anyfem import Project, pinned, steel
        from anyfem.solve.run import solve_linear_static_many
    except ImportError as error:
        from .errors import MissingDependencyError

        raise MissingDependencyError("the ANYfem example needs `pip install ANYfatigue[anyfem]`") from error
    from .anyfem_adapter import from_solutions, line_location

    factors = wave_factors(48, 0.5)
    project = Project(name="fatigue example plate")
    project.add_material(steel("S355", thickness_m))
    project.add_plate_section("plate", thickness=thickness_m, material="S355")
    geometry = project.geometry
    corners = geometry.add_points([(0, 0, 0), (side, 0, 0), (side, side, 0), (0, side, 0)])
    edges = geometry.add_polyline(corners, close=True)
    face = geometry.add_face(edges)
    project.assign_plate(face, "plate")
    for edge in edges:
        project.add_support(pinned(project.edge(edge)))
    names = []
    for k, f in enumerate(factors):
        names.append(f"t{k:03d}")
        project.load_case(names[-1]).add_pressure(project.face(face), pressure_pa * float(f))
    mesh = project.generate_mesh(target_size)
    batch = solve_linear_static_many(project, mesh=mesh, load_cases=names)
    source = from_solutions(batch, times=[0.5 * k for k in range(len(names))], project=project)
    weld = line_location(batch, project.edge(edges[0]), surfaces=("top",),
                         readout=ReadOut("hotspot_a", thickness_mm=thickness_m * 1000.0, side=1),
                         name="supported edge")
    centre = PointLocation(((side / 2, side / 2, 0.0),), surfaces=("top", "bottom"),
                           weld_direction=(0.0, 1.0, 0.0), name="plate centre")
    settings = TimeSeriesSettings(
        curve=sn.get_curve("D", "air"), thickness_mm=thickness_m * 1000.0, design_life_years=25.0,
        dff=3.0, blocks=(LoadBlock("wave", tuple(names), 1.0),),
    )
    return source, AnalysisDefinition("time_series", settings, (weld, centre), name="ANYfem plate",
                                      source={"kind": "example", "name": "anyfem_plate"})
