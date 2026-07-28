"""Unit tests for the physics core (``physics.py``).

Each test pins down a physical property of phase-modulation sidebands rather
than an implementation detail, so the suite doubles as executable documentation
of what the model guarantees. Run it with ``pytest`` (see ``requirements-dev.txt``).
"""

import numpy as np
import pytest
from scipy.special import jv

from physics import (DB_FLOOR, captured_power, color_for_n, intensity_vs_beta,
                     orders_for_power, sideband_intensities, to_decibels)


def test_energy_conservation():
    """Over enough orders the intensities sum to 1 (power is only redistributed)."""
    for beta in [0.1, 1.0, 2.405, 5.0, 8.0]:
        Jn2 = sideband_intensities(beta, N=40)
        assert captured_power(Jn2) == pytest.approx(1.0, abs=1e-6)


def test_captured_power_matches_direct_sum():
    """captured_power equals an explicit sum over -N..N of |J_n(beta)|^2."""
    beta, N = 3.0, 6
    Jn2 = sideband_intensities(beta, N)
    direct = sum(jv(n, beta)**2 for n in range(-N, N + 1))
    assert captured_power(Jn2) == pytest.approx(direct, rel=1e-9)


def test_symmetry_jn_squared():
    """The +n and -n orders carry equal intensity: |J_{-n}|^2 == |J_n|^2."""
    beta = 2.7
    for n in range(0, 8):
        assert jv(-n, beta)**2 == pytest.approx(jv(n, beta)**2, rel=1e-9)


def test_carrier_null_at_first_zero_of_j0():
    """The carrier vanishes at beta ~= 2.4048, the first zero of J_0."""
    Jn2 = sideband_intensities(2.4048255577, N=0)
    assert Jn2[0] == pytest.approx(0.0, abs=1e-6)


def test_sideband_intensities_shape_and_bounds():
    """Intensities have the expected shape and lie in [0, 1]."""
    Jn2 = sideband_intensities(1.0, N=5)
    assert Jn2.shape == (6,)
    assert np.all(Jn2 >= 0.0)
    assert np.all(Jn2 <= 1.0)


def test_intensity_vs_beta_agrees_with_sideband_intensities():
    """Fig. 2's curves pass exactly through Fig. 1's peak heights.

    The two figures are only consistent because both read the same
    ``|J_n(beta)|^2``: the curve sampled at the operating point must equal the
    per-line intensity used for the peak, or the markers would sit off-curve.
    """
    beta, N = 2.7, 6
    Jn2 = sideband_intensities(beta, N)
    for n in range(N + 1):
        assert intensity_vs_beta(n, beta) == pytest.approx(Jn2[n], rel=1e-12)


def test_intensity_vs_beta_ignores_order_sign_and_keeps_grid_shape():
    """The sign of the order is irrelevant, and a beta grid maps element-wise."""
    grid = np.linspace(0.0, 10.0, 64)
    for n in range(4):
        assert np.allclose(intensity_vs_beta(-n, grid), intensity_vs_beta(n, grid))
    curve = intensity_vs_beta(2, grid)
    assert curve.shape == grid.shape
    assert np.all(curve >= 0.0)
    assert np.all(curve <= 1.0)


def test_intensity_vs_beta_carrier_starts_at_full_power():
    """With no modulation all power is in the carrier and none in the sidebands."""
    assert intensity_vs_beta(0, 0.0) == pytest.approx(1.0)
    for n in range(1, 5):
        assert intensity_vs_beta(n, 0.0) == pytest.approx(0.0)


def test_orders_for_power_is_the_first_order_reaching_the_threshold():
    """The returned order clears the threshold and the one below it does not."""
    for beta in [0.5, 1.0, 2.405, 5.0, 7.82, 10.0]:
        n = orders_for_power(beta, 0.98)
        assert captured_power(sideband_intensities(beta, n)) >= 0.98
        assert captured_power(sideband_intensities(beta, n - 1)) < 0.98


def test_orders_for_power_grows_with_modulation_depth():
    """Deeper modulation spreads power outwards, so more orders are needed."""
    counts = [orders_for_power(beta, 0.98) for beta in np.arange(0.0, 10.1, 0.5)]
    assert counts[0] == 0                       # unmodulated: the carrier alone
    assert np.all(np.diff(counts) >= 0)
    assert counts[-1] > counts[0]


def test_orders_for_power_caps_at_the_search_limit():
    """An unreachable threshold returns the cap instead of searching forever."""
    assert orders_for_power(3.0, threshold=1.5, search_limit=8) == 8


def test_to_decibels_reference_values():
    """Unit power is 0 dB, half power is -3 dB, and a factor 100 down is -20 dB."""
    assert to_decibels(1.0) == pytest.approx(0.0)
    assert to_decibels(0.5) == pytest.approx(-3.0103, abs=1e-4)
    assert to_decibels(0.01) == pytest.approx(-20.0)


def test_to_decibels_clamps_zero_and_sub_floor_values():
    """Zero and anything under the floor clamp to it rather than diverging."""
    assert to_decibels(0.0) == pytest.approx(DB_FLOOR)
    assert to_decibels(1e-9) == pytest.approx(DB_FLOOR)
    out = to_decibels(np.array([1.0, 0.0, 1e-9]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(DB_FLOOR)
    assert out[2] == pytest.approx(DB_FLOOR)


def test_to_decibels_keeps_bessel_nulls_on_scale():
    """The carrier null is exactly zero intensity, so it must stay finite."""
    assert np.all(np.isfinite(to_decibels(sideband_intensities(2.4048255577, N=6))))


def test_to_decibels_is_monotonic_above_the_floor():
    """Ordering is preserved, so the dB view ranks sidebands like the linear one."""
    x = np.linspace(10 ** (DB_FLOOR / 10), 1.0, 500)
    assert np.all(np.diff(to_decibels(x)) > 0)


def test_to_decibels_never_exceeds_zero_for_physical_intensities():
    """Power conservation caps any single line at unit power, i.e. 0 dB."""
    for beta in [0.0, 0.5, 1.0, 2.405, 5.0]:
        assert np.all(to_decibels(sideband_intensities(beta, N=8)) <= 0.0)


def test_color_for_n_matches_negative_order_and_never_raises():
    """Colours ignore the sign of the order and the carrier is the theme colour."""
    for n in range(0, 20):
        assert color_for_n(n) == color_for_n(-n)
    assert color_for_n(0) == 'black'
