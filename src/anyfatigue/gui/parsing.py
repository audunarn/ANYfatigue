"""Text-field parsing for the GUI (no Qt, so it is testable on its own)."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..errors import InputError

__all__ = ["format_ids", "parse_float", "parse_ids", "parse_points", "parse_vector"]


def parse_float(text: str, name: str, *, positive: bool = False, minimum: Optional[float] = None) -> float:
    try:
        value = float(text.strip().replace(",", "."))
    except ValueError:
        raise InputError(f"{name}: '{text}' is not a number") from None
    if value != value or value in (float("inf"), float("-inf")):
        raise InputError(f"{name}: must be finite")
    if positive and value <= 0.0:
        raise InputError(f"{name}: must be positive")
    if minimum is not None and value < minimum:
        raise InputError(f"{name}: must be at least {minimum:g}")
    return value


def parse_ids(text: str, name: str = "ids") -> Tuple[int, ...]:
    """``"3, 5, 8-11"`` -> ``(3, 5, 8, 9, 10, 11)``; order is kept, repeats removed."""

    out: List[int] = []
    # ids are positive: "10 - 12" is the range 10-12, never 10 and -12
    text = re.sub(r"(\d)\s*-\s*(\d)", r"\1-\2", text)
    for token in re.split(r"[,\s;]+", text.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(-?\d+)\s*-\s*(\d+)", token)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if high < low:
                raise InputError(f"{name}: range '{token}' runs backwards")
            if high - low > 1_000_000:
                raise InputError(f"{name}: range '{token}' is too large")
            out.extend(range(low, high + 1))
        elif re.fullmatch(r"-?\d+", token):
            out.append(int(token))
        else:
            raise InputError(f"{name}: '{token}' is not an integer or a range")
    unique = tuple(dict.fromkeys(out))
    if not unique:
        raise InputError(f"{name}: give at least one id")
    return unique


def format_ids(ids) -> str:
    """The inverse of :func:`parse_ids`, collapsing consecutive runs."""

    ids = list(ids)
    parts: List[str] = []
    i = 0
    while i < len(ids):
        j = i
        while j + 1 < len(ids) and ids[j + 1] == ids[j] + 1:
            j += 1
        parts.append(str(ids[i]) if j - i < 2 else f"{ids[i]}-{ids[j]}")
        if j - i == 1:
            parts.append(str(ids[j]))
        i = j + 1
    return ", ".join(parts)


def parse_vector(text: str, name: str = "vector") -> Optional[Tuple[float, float, float]]:
    """``"0, 1, 0"`` -> ``(0.0, 1.0, 0.0)``; empty text is ``None``."""

    if not text.strip():
        return None
    parts = [p for p in re.split(r"[,\s;]+", text.strip()) if p]
    if len(parts) != 3:
        raise InputError(f"{name}: give three numbers x, y, z")
    return tuple(parse_float(p, name) for p in parts)  # type: ignore[return-value]


def parse_points(text: str, name: str = "points") -> Tuple[Tuple[float, float, float], ...]:
    """Points separated by ``;`` or new lines: ``"0.1 0.2 0; 0.3 0.2 0"``."""

    rows = [r for r in re.split(r"[;\n]+", text.strip()) if r.strip()]
    if not rows:
        raise InputError(f"{name}: give at least one point")
    return tuple(parse_vector(r, name) for r in rows)  # type: ignore[misc]
