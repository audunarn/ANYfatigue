"""GUI text parsing and the built-in examples (no Qt needed)."""

from __future__ import annotations

import numpy as np
import pytest

from anyfatigue import examples
from anyfatigue.errors import InputError
from anyfatigue.gui import parsing as ps


@pytest.mark.parametrize("text,expected", [
    ("3", (3,)), ("3, 5, 8-11", (3, 5, 8, 9, 10, 11)), ("1 2;3", (1, 2, 3)),
    ("5, 5, 4", (5, 4)), (" 10 - 12 ", (10, 11, 12)),
])
def test_parse_ids(text, expected):
    assert ps.parse_ids(text) == expected


@pytest.mark.parametrize("text", ["", "a", "1.5", "9-3", "1-"])
def test_parse_ids_rejects(text):
    with pytest.raises(InputError):
        ps.parse_ids(text)


def test_format_ids_round_trips():
    for ids in [(1,), (1, 2), (1, 2, 3, 4, 9), (7, 3, 4, 5), (10, 20, 21)]:
        assert ps.parse_ids(ps.format_ids(ids)) == ids


def test_parse_vector_and_points():
    assert ps.parse_vector("0, 1, 0") == (0.0, 1.0, 0.0)
    assert ps.parse_vector("   ") is None
    assert ps.parse_points("0.1 0.2 0; 0.3, 0.2, 0\n1 1 1") == (
        (0.1, 0.2, 0.0), (0.3, 0.2, 0.0), (1.0, 1.0, 1.0))
    for bad in ("1, 2", "a b c"):
        with pytest.raises(InputError):
            ps.parse_vector(bad)
    with pytest.raises(InputError):
        ps.parse_points("")


def test_parse_float_accepts_decimal_comma_and_checks_domain():
    assert ps.parse_float("2,5", "x") == 2.5
    with pytest.raises(InputError):
        ps.parse_float("0", "x", positive=True)
    with pytest.raises(InputError):
        ps.parse_float("nan", "x")
    with pytest.raises(InputError):
        ps.parse_float("-1", "x", minimum=0.0)
    with pytest.raises(InputError):
        ps.parse_float("abc", "x")


def test_synthetic_example_runs_both_methods():
    source, definition = examples.synthetic_plate(steps=80)
    assert "not an FE result" in source.provenance["kind"]
    assert np.isfinite(source.element_stress([0], "top")).all()
    ts = definition.run(source)
    assert ts.points and ts.critical.damage > 0.0
    simplified = examples.simplified_for(source, definition).run(source)
    assert simplified.critical.damage > 0.0
    # the weld line peaks mid-length
    line = [p for p in ts.points if p.kind == "line"]
    middle = max(line, key=lambda p: p.damage)
    assert 0.2 < middle.xyz[1] < 0.6


@pytest.mark.anyfem
def test_anyfem_example_builds_and_runs():
    source, definition = examples.anyfem_plate(target_size=1.0 / 8.0)
    assert source.n_cases == 48 and source.nodal_origin.startswith("ANYfem")
    result = definition.run(source)
    assert result.critical.damage > 0.0
