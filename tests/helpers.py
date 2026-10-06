"""Synthetic meshes and closed-form stress fields for tests."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from anyfatigue.model import ArrayStressSource, SurfaceMesh

Field = Callable[[np.ndarray], np.ndarray]   # (n, 3) positions -> (n, 6) tensor


def plate_mesh(
    nx: int = 6,
    ny: int = 4,
    lx: float = 0.6,
    ly: float = 0.4,
    *,
    triangles: bool = False,
    skew: float = 0.0,
    flip: bool = False,
) -> SurfaceMesh:
    """A flat plate in the xy plane, quads (or two triangles per cell).

    ``skew`` shifts each node in x by ``skew * y`` to make non-rectangular
    quads; ``flip`` reverses the corner order (normal -> -z).
    """

    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    node_ids, xyz = [], []
    index = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nid = 100 + j * (nx + 1) + i
            index[(i, j)] = nid
            node_ids.append(nid)
            xyz.append((x + skew * y, y, 0.0))
    elements, corners = [], []
    eid = 1
    for j in range(ny):
        for i in range(nx):
            a, b = index[(i, j)], index[(i + 1, j)]
            c, d = index[(i + 1, j + 1)], index[(i, j + 1)]
            cells = [(a, b, c, -1), (a, c, d, -1)] if triangles else [(a, b, c, d)]
            for cell in cells:
                if flip:
                    cell = cell[:3][::-1] + (cell[3],) if cell[3] < 0 else cell[::-1]
                elements.append(eid)
                corners.append(cell)
                eid += 1
    return SurfaceMesh(np.array(node_ids), np.array(xyz), np.array(elements), np.array(corners))


def tensor_field(coefficients: np.ndarray) -> Field:
    """Linear field ``S = c0 + c1 x + c2 y`` with ``c`` of shape (3, 6)."""

    c = np.asarray(coefficients, dtype=float)

    def field(p: np.ndarray) -> np.ndarray:
        p = np.atleast_2d(p)
        return c[0] + np.outer(p[:, 0], c[1]) + np.outer(p[:, 1], c[2])

    return field


def scaled(field: Field, factor: float) -> Field:
    """The same field multiplied by a constant (a second load case)."""

    return lambda p: factor * field(p)


def source_from_fields(
    mesh: SurfaceMesh,
    fields: Sequence[Field],
    *,
    bottom_factor: float = -1.0,
    labels: Sequence[str] = (),
    times: Sequence[float] = (),
    with_nodal: bool = True,
) -> ArrayStressSource:
    """Top stress = field, bottom = ``bottom_factor * field`` (bending-like)."""

    n = len(fields)
    labels = list(labels) or [f"case{k}" for k in range(n)]
    centroids = mesh.centroids()
    el_top = np.stack([f(centroids) for f in fields])
    nd_top = np.stack([f(mesh.node_xyz) for f in fields])
    kwargs = {}
    if with_nodal:
        kwargs = dict(node_top=nd_top, node_bottom=bottom_factor * nd_top)
    return ArrayStressSource(
        mesh, labels, el_top, bottom_factor * el_top,
        times=list(times) or None, **kwargs,
    )
