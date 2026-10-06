"""Typed errors.

Ambiguous or unsupported input fails closed with one of these, never with a
silent fallback.  ``FatigueError`` is the common base so a caller (the GUI, a
script) can catch everything this package raises on purpose.
"""

from __future__ import annotations

__all__ = [
    "ExtractionError",
    "FatigueError",
    "InputError",
    "MissingDependencyError",
    "UnsupportedInputError",
]


class FatigueError(Exception):
    """Base class for every deliberate ANYfatigue failure."""


class InputError(FatigueError, ValueError):
    """An input value is invalid, inconsistent or outside its stated domain."""


class UnsupportedInputError(InputError):
    """The request is well formed but this release does not support it."""


class ExtractionError(FatigueError):
    """A stress location cannot be resolved or the source cannot supply it."""


class MissingDependencyError(FatigueError, ImportError):
    """An optional producer or owner package is required but not importable."""
