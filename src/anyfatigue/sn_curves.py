"""S-N curves.

Parameters are transcribed from DNV-RP-C203 *Fatigue design of offshore steel
structures*, April 2010 edition: Table 2-1 (in air), Table 2-2 (seawater with
cathodic protection) and Table 2-3 (seawater, free corrosion).  Only that
edition has been checked against the printed tables; the edition is carried on
every curve (``SNCurve.source``) and into every report, because a later
edition may change a value and a result must say which one it used.

Conventions (DNV-RP-C203 section 2.4)::

    log N = log a - m log( dS * (t / t_ref)**k )            (2.4.3)

``dS`` is the stress range in MPa, ``t`` the thickness through which a crack
will most likely grow, with ``t = t_ref`` used for ``t < t_ref``.  A bi-linear
curve changes slope from ``m1`` to ``m2`` at ``n_switch`` cycles (10^7 in air,
10^6 in seawater with cathodic protection).  The stress range at the change of
slope is, equation (D.13-5),::

    S1 = (a1 / n_switch) ** (1 / m1)

Stress concentration factors and the thickness factor both multiply the stress
range, so they are combined into one *effective* range before the curve is
entered; :meth:`SNCurve.effective_range` is that single place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .errors import InputError

__all__ = [
    "CURRENT_EDITION",
    "ENVIRONMENTS",
    "STANDARD_EDITION",
    "SNCurve",
    "curve_names",
    "edition_warnings",
    "get_curve",
    "parse_anystructure_name",
]

STANDARD_EDITION = "DNV-RP-C203, April 2010"
# The edition DNV currently publishes (dnv.com, 2026-10-06).  It sits behind DNV's
# subscription portal, so its tables have NOT been read or checked here.
CURRENT_EDITION = "DNV-RP-C203, edition 2024-10, amended 2025-10"

AIR = "air"
SEAWATER_CP = "seawater_cp"
FREE_CORROSION = "free_corrosion"
CUSTOM = "custom"
ENVIRONMENTS = (AIR, SEAWATER_CP, FREE_CORROSION)

_TABLE = {
    AIR: "Table 2-1 (S-N curves in air)",
    SEAWATER_CP: "Table 2-2 (S-N curves in seawater with cathodic protection)",
    FREE_CORROSION: "Table 2-3 (S-N curves in seawater for free corrosion)",
}
_KNEE = {AIR: 1.0e7, SEAWATER_CP: 1.0e6, FREE_CORROSION: None}

# name: (m1, log a1 in air, log a2 (air and cathodic protection), thickness
#        exponent k, log a1 with cathodic protection, log a (free corrosion))
_DATA: Dict[str, Tuple[float, float, float, float, float, float]] = {
    "B1": (4.0, 15.117, 17.146, 0.00, 14.917, 12.436),
    "B2": (4.0, 14.885, 16.856, 0.00, 14.685, 12.262),
    "C": (3.0, 12.592, 16.320, 0.15, 12.192, 12.115),
    "C1": (3.0, 12.449, 16.081, 0.15, 12.049, 11.972),
    "C2": (3.0, 12.301, 15.835, 0.15, 11.901, 11.824),
    "D": (3.0, 12.164, 15.606, 0.20, 11.764, 11.687),
    "E": (3.0, 12.010, 15.350, 0.20, 11.610, 11.533),
    "F": (3.0, 11.855, 15.091, 0.25, 11.455, 11.378),
    "F1": (3.0, 11.699, 14.832, 0.25, 11.299, 11.222),
    "F3": (3.0, 11.546, 14.576, 0.25, 11.146, 11.068),
    "G": (3.0, 11.398, 14.330, 0.25, 10.998, 10.921),
    "W1": (3.0, 11.261, 14.101, 0.25, 10.861, 10.784),
    "W2": (3.0, 11.107, 13.845, 0.25, 10.707, 10.630),
    "W3": (3.0, 10.970, 13.617, 0.25, 10.570, 10.493),
    # Tubular joints: the thickness exponent is 0.25 for SCF <= 10 and 0.30
    # above, and the reference thickness is 32 mm (section 2.4.3).
    "T": (3.0, 12.164, 15.606, 0.25, 11.764, 11.687),
}
_T_CURVE_K_HIGH_SCF = 0.30
_T_CURVE_SCF_LIMIT = 10.0
_T_CURVE_REF_THICKNESS = 32.0
_DEFAULT_REF_THICKNESS = 25.0


@dataclass(frozen=True)
class SNCurve:
    """A single- or two-slope S-N curve with DNV thickness correction."""

    name: str
    environment: str
    m1: float
    log_a1: float
    m2: Optional[float] = None
    log_a2: Optional[float] = None
    n_switch: Optional[float] = None
    k: float = 0.0
    t_ref_mm: float = _DEFAULT_REF_THICKNESS
    k_high_scf: Optional[float] = None
    scf_limit: float = _T_CURVE_SCF_LIMIT
    source: str = ""

    def __post_init__(self) -> None:
        for field_name in ("m1", "log_a1", "k", "t_ref_mm"):
            if not math.isfinite(getattr(self, field_name)):
                raise InputError(f"S-N curve {self.name!r}: {field_name} must be finite")
        if self.m1 <= 0.0:
            raise InputError(f"S-N curve {self.name!r}: m1 must be positive")
        if self.t_ref_mm <= 0.0:
            raise InputError(f"S-N curve {self.name!r}: reference thickness must be positive")
        if self.k < 0.0:
            raise InputError(f"S-N curve {self.name!r}: thickness exponent must be >= 0")
        second = (self.m2, self.log_a2, self.n_switch)
        if any(value is not None for value in second) and any(
            value is None for value in second
        ):
            raise InputError(
                f"S-N curve {self.name!r}: a bi-linear curve needs m2, log_a2 "
                "and n_switch together"
            )
        if self.m2 is not None:
            if not (math.isfinite(self.m2) and self.m2 > 0.0):
                raise InputError(f"S-N curve {self.name!r}: m2 must be positive")
            if not math.isfinite(self.log_a2):
                raise InputError(f"S-N curve {self.name!r}: log_a2 must be finite")
            if not (math.isfinite(self.n_switch) and self.n_switch > 1.0):
                raise InputError(f"S-N curve {self.name!r}: n_switch must exceed 1")

    # ------------------------------------------------------------------
    @property
    def bilinear(self) -> bool:
        return self.m2 is not None

    @property
    def a1(self) -> float:
        return 10.0 ** self.log_a1

    @property
    def a2(self) -> Optional[float]:
        return None if self.log_a2 is None else 10.0 ** self.log_a2

    @property
    def stress_switch(self) -> Optional[float]:
        """Effective stress range at the change of slope, MPa (D.13-5)."""

        if not self.bilinear:
            return None
        return 10.0 ** ((self.log_a1 - math.log10(self.n_switch)) / self.m1)

    # ------------------------------------------------------------------
    def thickness_exponent(self, scf: float = 1.0) -> float:
        if self.k_high_scf is not None and scf > self.scf_limit:
            return self.k_high_scf
        return self.k

    def thickness_factor(
        self, thickness_mm: Optional[float], scf: float = 1.0
    ) -> float:
        """``(max(t, t_ref) / t_ref) ** k``; 1.0 when no thickness is given.

        ``None`` means "the reference thickness", which is the DNV default for
        a thickness at or below ``t_ref``.  Analyses in this package require an
        explicit thickness so the choice is always visible.
        """

        if thickness_mm is None:
            return 1.0
        if not (math.isfinite(thickness_mm) and thickness_mm > 0.0):
            raise InputError(f"thickness must be positive, got {thickness_mm!r} mm")
        exponent = self.thickness_exponent(scf)
        return (max(thickness_mm, self.t_ref_mm) / self.t_ref_mm) ** exponent

    def effective_range(
        self,
        stress_range,
        *,
        scf: float = 1.0,
        thickness_mm: Optional[float] = None,
    ):
        """Stress range as entered into the curve: ``dS * SCF * (t/t_ref)^k``."""

        if not (math.isfinite(scf) and scf > 0.0):
            raise InputError(f"SCF must be positive, got {scf!r}")
        factor = scf * self.thickness_factor(thickness_mm, scf)
        return np.asarray(stress_range, dtype=float) * factor

    def cycles_to_failure(self, effective_range):
        """Cycles to failure for an effective stress range (MPa).

        A zero range never fails (``inf``).  A negative range is an input
        error: callers pass ranges, not signed stresses.
        """

        s = np.asarray(effective_range, dtype=float)
        if np.any(~np.isfinite(s)) or np.any(s < 0.0):
            raise InputError("stress ranges must be finite and non-negative")
        with np.errstate(divide="ignore", over="ignore"):
            log_s = np.log10(np.where(s > 0.0, s, 1.0))
            log_n1 = self.log_a1 - self.m1 * log_s
            if self.bilinear:
                log_n2 = self.log_a2 - self.m2 * log_s
                log_n = np.where(s >= self.stress_switch, log_n1, log_n2)
            else:
                log_n = log_n1
            n = np.where(s > 0.0, 10.0 ** log_n, np.inf)
        return n if n.ndim else float(n)

    def allowable_range(
        self,
        cycles: float,
        *,
        scf: float = 1.0,
        thickness_mm: Optional[float] = None,
    ) -> float:
        """Applied stress range (before SCF and thickness) that fails at ``cycles``."""

        if not (math.isfinite(cycles) and cycles > 0.0):
            raise InputError("cycles must be positive")
        if self.bilinear and cycles > self.n_switch:
            log_s = (self.log_a2 - math.log10(cycles)) / self.m2
        else:
            log_s = (self.log_a1 - math.log10(cycles)) / self.m1
        return float(10.0 ** log_s) / (scf * self.thickness_factor(thickness_mm, scf))

    # ------------------------------------------------------------------
    def describe(self) -> str:
        parts = [f"{self.name} ({self.environment})", f"m1={self.m1:g}", f"log a1={self.log_a1:g}"]
        if self.bilinear:
            parts += [
                f"m2={self.m2:g}",
                f"log a2={self.log_a2:g}",
                f"knee at {self.n_switch:.0e} cycles",
            ]
        parts += [f"k={self.k:g}", f"t_ref={self.t_ref_mm:g} mm"]
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "environment": self.environment,
            "m1": self.m1,
            "log_a1": self.log_a1,
            "m2": self.m2,
            "log_a2": self.log_a2,
            "n_switch": self.n_switch,
            "k": self.k,
            "t_ref_mm": self.t_ref_mm,
            "k_high_scf": self.k_high_scf,
            "scf_limit": self.scf_limit,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SNCurve":
        allowed = {
            "name", "environment", "m1", "log_a1", "m2", "log_a2", "n_switch",
            "k", "t_ref_mm", "k_high_scf", "scf_limit", "source",
        }
        unknown = set(data) - allowed
        if unknown:
            raise InputError(f"unknown S-N curve fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def custom(
        cls,
        name: str,
        *,
        m1: float,
        log_a1: float,
        m2: Optional[float] = None,
        log_a2: Optional[float] = None,
        n_switch: Optional[float] = None,
        k: float = 0.0,
        t_ref_mm: float = _DEFAULT_REF_THICKNESS,
    ) -> "SNCurve":
        """A user-defined curve; its source says so in every report."""

        return cls(
            name=name,
            environment=CUSTOM,
            m1=m1,
            log_a1=log_a1,
            m2=m2,
            log_a2=log_a2,
            n_switch=n_switch,
            k=k,
            t_ref_mm=t_ref_mm,
            source="user-defined",
        )


def edition_warnings(curve: SNCurve) -> Tuple[str, ...]:
    """What a result must say when a built-in curve is not from the current edition.

    Empty for user-defined curves, whose ``source`` is the user's responsibility.
    """

    if not curve.source.startswith(STANDARD_EDITION):
        return ()
    out = [
        f"S-N data are transcribed from {STANDARD_EDITION}; DNV now publishes "
        f"{CURRENT_EDITION}, which has not been checked here. Compare the tables "
        "and equations with the edition your project requires before design use."
    ]
    if curve.name == "T":
        out.append(
            "The T curve's reference thickness (32 mm) and thickness exponent are those of "
            f"{STANDARD_EDITION}; later editions may define them differently."
        )
    return tuple(out)


def curve_names() -> Tuple[str, ...]:
    return tuple(_DATA)


def get_curve(name: str, environment: str = AIR) -> SNCurve:
    """The DNV-RP-C203 curve ``name`` in ``environment``."""

    key = str(name).strip()
    if key not in _DATA:
        raise InputError(
            f"unknown S-N curve {name!r}; available: {', '.join(_DATA)}"
        )
    if environment not in ENVIRONMENTS:
        raise InputError(
            f"unknown environment {environment!r}; available: {', '.join(ENVIRONMENTS)}"
        )
    m1, log_a1_air, log_a2, k, log_a1_cp, log_a_free = _DATA[key]
    is_t = key == "T"
    common = dict(
        name=key,
        environment=environment,
        k=k,
        t_ref_mm=_T_CURVE_REF_THICKNESS if is_t else _DEFAULT_REF_THICKNESS,
        k_high_scf=_T_CURVE_K_HIGH_SCF if is_t else None,
        source=f"{STANDARD_EDITION}, {_TABLE[environment]}",
    )
    if environment == FREE_CORROSION:
        # Single slope m = 3.0 for all N, so B1 and B2 change slope too.
        return SNCurve(m1=3.0, log_a1=log_a_free, **common)
    log_a1 = log_a1_air if environment == AIR else log_a1_cp
    return SNCurve(
        m1=m1,
        log_a1=log_a1,
        m2=5.0,
        log_a2=log_a2,
        n_switch=_KNEE[environment],
        **common,
    )


def parse_anystructure_name(name: str) -> SNCurve:
    """Resolve an ANYstructure curve key such as ``"Ec"``.

    ANYstructure appends ``c`` for the cathodic-protection variant of a curve
    (``"B1c"``, ``"F3c"``); a bare name is the in-air curve.
    """

    key = str(name).strip()
    if key in _DATA:
        return get_curve(key, AIR)
    if key.endswith("c") and key[:-1] in _DATA:
        return get_curve(key[:-1], SEAWATER_CP)
    raise InputError(f"unknown ANYstructure S-N curve key {name!r}")


def iter_curves(environment: str = AIR) -> Iterable[SNCurve]:
    for name in _DATA:
        yield get_curve(name, environment)
