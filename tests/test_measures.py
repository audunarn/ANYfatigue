"""Resolving surface stress against a weld, and the DNV effective stress."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anyfatigue import measures as ms
from anyfatigue.errors import InputError, UnsupportedInputError

Z = np.array([0.0, 0.0, 1.0])


def test_normal_to_weld_for_a_weld_along_y():
    # tensor order: xx yy zz xy yz xz
    t = np.array([[100.0, 40.0, 7.0, 25.0, 3.0, 2.0]])
    s_perp, s_par, tau = ms.in_plane_components(t, Z, [0.0, 1.0, 0.0])
    assert s_perp == pytest.approx(100.0)   # stress across a weld along y is sigma_xx
    assert s_par == pytest.approx(40.0)
    assert abs(tau) == pytest.approx(25.0)   # sign depends on the n x t convention


def test_rotation_by_theta_matches_the_mohr_circle():
    sxx, syy, sxy = 120.0, 30.0, 45.0
    t = np.array([sxx, syy, 0.0, sxy, 0.0, 0.0])
    for theta in np.radians([0.0, 17.0, 45.0, 90.0, 133.0]):
        weld = [math.cos(theta), math.sin(theta), 0.0]       # weld direction
        perp, par, tau = ms.in_plane_components(t, Z, weld)
        # Mohr: stress on a line whose normal is (sin, -cos)
        nx, ny = math.sin(theta), -math.cos(theta)
        assert perp == pytest.approx(sxx * nx * nx + syy * ny * ny + 2 * sxy * nx * ny)
        assert perp + par == pytest.approx(sxx + syy)          # invariant
        assert perp * par - tau ** 2 == pytest.approx(sxx * syy - sxy ** 2)


def test_out_of_plane_components_never_enter_the_surface_stress():
    base = np.array([50.0, 10.0, 0.0, 5.0, 0.0, 0.0])
    loaded = base + np.array([0.0, 0.0, 400.0, 0.0, 90.0, -70.0])
    a = ms.in_plane_components(base, Z, [1.0, 0.0, 0.0])
    b = ms.in_plane_components(loaded, Z, [1.0, 0.0, 0.0])
    assert np.allclose(a, b)


def test_projection_on_an_inclined_plate_uses_its_own_normal():
    # Plate normal along global x: its surface is the yz plane.  A weld along z.
    n = np.array([1.0, 0.0, 0.0])
    t = np.array([999.0, 60.0, 20.0, 0.0, 0.0, 0.0])   # xx is out of plane
    perp, par, _tau = ms.in_plane_components(t, n, [0.0, 0.0, 1.0])
    assert par == pytest.approx(20.0)     # along z: sigma_zz
    assert perp == pytest.approx(60.0)    # across the weld, in-plane: sigma_yy


def test_in_plane_principal_values():
    t = np.array([[80.0, 20.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 50.0, 0.0, 0.0]])
    s1, s2 = ms.in_plane_principal(t, Z)
    assert s1 == pytest.approx([80.0, 50.0]) and s2 == pytest.approx([20.0, -50.0])
    assert np.all(s1 >= s2)


def test_principal_values_do_not_depend_on_the_plate_orientation():
    t_local = np.array([90.0, -30.0, 0.0, 40.0, 0.0, 0.0])
    s1, s2 = ms.in_plane_principal(t_local, Z)
    # Same state expressed in global axes for a plate whose normal is y: swap yy<->zz, xy->xz.
    t_global = np.array([90.0, 0.0, -30.0, 0.0, 0.0, 40.0])
    g1, g2 = ms.in_plane_principal(t_global, np.array([0.0, 1.0, 0.0]))
    assert (g1, g2) == (pytest.approx(s1), pytest.approx(s2))


def test_effective_hot_spot_uniaxial_across_the_weld_is_the_stress_itself():
    # Pure normal stress: sqrt(p^2) = p dominates 0.72*p (alpha) for alpha<=1.
    assert ms.effective_hot_spot_range(100.0, 0.0, 0.0) == pytest.approx(100.0)
    assert ms.effective_hot_spot_range(100.0, 0.0, 0.0, method="B") == pytest.approx(112.0)


def test_effective_hot_spot_parallel_stress_is_reduced_by_alpha():
    # Stress parallel to the weld only: the first term is 0, principal = par.
    assert ms.effective_hot_spot_range(0.0, 100.0, 0.0, alpha=0.72) == pytest.approx(72.0)
    assert ms.effective_hot_spot_range(0.0, 100.0, 0.0, alpha=0.9) == pytest.approx(90.0)


def test_effective_hot_spot_shear_term_has_the_081_weight():
    assert ms.effective_hot_spot_range(0.0, 0.0, 100.0) == pytest.approx(
        max(math.sqrt(0.81) * 100.0, 100.0)
    )   # principal range of pure shear is 100, which governs with alpha = 1
    got = ms.effective_hot_spot_range(60.0, 0.0, 30.0, alpha=0.72)
    s1 = 30.0 + 0.5 * math.sqrt(60.0 ** 2 + 4 * 30.0 ** 2)
    s2 = 30.0 - 0.5 * math.sqrt(60.0 ** 2 + 4 * 30.0 ** 2)
    assert got == pytest.approx(max(math.sqrt(60.0 ** 2 + 0.81 * 30.0 ** 2),
                                    0.72 * s1, 0.72 * abs(s2)))


def test_effective_hot_spot_rejects_bad_options():
    with pytest.raises(InputError):
        ms.effective_hot_spot_range(1.0, 0.0, 0.0, method="C")
    with pytest.raises(InputError):
        ms.effective_hot_spot_range(1.0, 0.0, 0.0, alpha=0.0)


def test_measure_text_round_trip_and_validation():
    for text in ("normal_to_weld", "principal_abs_max", "component:xy", "dnv_effective"):
        assert ms.StressMeasure.parse(text).to_text() == text
    with pytest.raises(InputError):
        ms.StressMeasure("component", "ab")
    with pytest.raises(InputError):
        ms.StressMeasure("normal_to_weld", "xx")
    with pytest.raises(InputError):
        ms.StressMeasure("von_mises")


def test_signed_series_for_each_measure():
    history = np.array([[100.0, 0.0, 0, 0, 0, 0], [-60.0, 0.0, 0, 0, 0, 0], [0.0, 0.0, 0, 0, 0, 0]])
    assert ms.StressMeasure("component", "xx").signed_series(history, Z) == pytest.approx(
        [100.0, -60.0, 0.0])
    assert ms.StressMeasure("normal_to_weld").signed_series(history, Z, [0, 1, 0]) == pytest.approx(
        [100.0, -60.0, 0.0])
    # largest-magnitude principal keeps the sign of the dominant stress
    assert ms.StressMeasure("principal_abs_max").signed_series(history, Z) == pytest.approx(
        [100.0, -60.0, 0.0])


def test_signed_series_needs_a_weld_direction_and_refuses_effective_stress():
    t = np.zeros((2, 6))
    with pytest.raises(InputError):
        ms.StressMeasure("normal_to_weld").signed_series(t, Z, None)
    with pytest.raises(UnsupportedInputError):
        ms.StressMeasure("dnv_effective").signed_series(t, Z, [1, 0, 0])


def test_range_of_a_difference_tensor():
    delta = np.array([[80.0, 20.0, 0.0, 0.0, 0.0, 0.0]])
    assert ms.StressMeasure("normal_to_weld").range_of(delta, Z, [0, 1, 0]) == pytest.approx([80.0])
    assert ms.StressMeasure("principal_abs_max").range_of(delta, Z) == pytest.approx([80.0])
    assert ms.StressMeasure("component", "yy").range_of(-delta, Z) == pytest.approx([20.0])
    eff = ms.StressMeasure("dnv_effective", alpha=0.72).range_of(delta, Z, [0, 1, 0])
    assert eff == pytest.approx(ms.effective_hot_spot_range(80.0, 20.0, 0.0, alpha=0.72))


def test_weld_direction_parallel_to_normal_is_rejected():
    with pytest.raises(InputError):
        ms.plane_basis(Z, [0.0, 0.0, 2.0])
