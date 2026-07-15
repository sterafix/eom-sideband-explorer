"""Unit tests for the Bessel sideband core (``core/bessel.py``).

Each test pins down a physical property of phase-modulation sidebands rather
than an implementation detail, so the suite doubles as executable documentation
of what the model guarantees. Run it with ``pytest`` (see ``requirements-dev.txt``).
"""

import numpy as np
import pytest
from scipy.special import jv

from core.bessel import captured_power, sideband_intensities
from core.theming import color_for_n


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


def test_color_for_n_matches_negative_order_and_never_raises():
    """Colours ignore the sign of the order and the carrier is the theme colour."""
    for n in range(0, 20):
        assert color_for_n(n) == color_for_n(-n)
    assert color_for_n(0) == 'black'
