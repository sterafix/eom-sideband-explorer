import numpy as np
import pytest
from scipy.special import jv

from physics import captured_power, color_for_n, sideband_intensities


def test_energy_conservation():
    for beta in [0.1, 1.0, 2.405, 5.0, 8.0]:
        Jn2 = sideband_intensities(beta, N=40)
        assert captured_power(Jn2) == pytest.approx(1.0, abs=1e-6)


def test_captured_power_matches_direct_sum():
    beta, N = 3.0, 6
    Jn2 = sideband_intensities(beta, N)
    direct = sum(jv(n, beta)**2 for n in range(-N, N + 1))
    assert captured_power(Jn2) == pytest.approx(direct, rel=1e-9)


def test_symmetry_jn_squared():
    beta = 2.7
    for n in range(0, 8):
        assert jv(-n, beta)**2 == pytest.approx(jv(n, beta)**2, rel=1e-9)


def test_carrier_null_at_first_zero_of_j0():
    Jn2 = sideband_intensities(2.4048255577, N=0)
    assert Jn2[0] == pytest.approx(0.0, abs=1e-6)


def test_sideband_intensities_shape_and_bounds():
    Jn2 = sideband_intensities(1.0, N=5)
    assert Jn2.shape == (6,)
    assert np.all(Jn2 >= 0.0)
    assert np.all(Jn2 <= 1.0)


def test_color_for_n_matches_negative_order_and_never_raises():
    for n in range(0, 20):
        assert color_for_n(n) == color_for_n(-n)
    assert color_for_n(0) == 'black'
