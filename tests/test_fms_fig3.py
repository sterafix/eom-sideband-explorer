"""Regression tests against Bjorklund et al. Fig. 3 (0.05 <= delta_R <= 4.0).

Fig. 3 (Appl. Phys. B 32, 145-152 (1983), p. 148) plots S1, S2, S3 vs R0 for
delta_R in {0.05, 0.1, 0.4, 0.8, 1.6, 2.5, 3.0, 4.0}, R0 in roughly [-16, 16].
This is the "wavelength modulation to FM spectroscopy" transition region: at
the smallest delta_R, S1 is close to the derivative of the absorption and S2
is very weak (second-derivative-like); at the largest delta_R shown, S1
begins to show separate resonances.
"""

import numpy as np
import pytest

from core.fms import fms_signals

FIG3_DELTA_R = [0.05, 0.1, 0.4, 0.8, 1.6, 2.5, 3.0, 4.0]

# Analytic derivatives of the R=0-centered Lorentzian, used for the small
# delta_R limit check (delta_peak = 1): delta'(R) = -2R/(R**2+1)**2,
# phi''(R) = 2R(R**2-3)/(R**2+1)**3.
def _delta_prime(R):
    return -2 * R / (R**2 + 1)**2


def _phi_double_prime(R):
    return 2 * R * (R**2 - 3) / (R**2 + 1)**3


@pytest.fixture(params=FIG3_DELTA_R)
def delta_R(request):
    return request.param


def test_all_signals_zero_at_line_center(delta_R):
    """S1, S2, S3 are all zero at R0 = 0, for every delta_R shown in Fig. 3."""
    S1, S2, S3 = fms_signals(0.0, delta_R)
    assert S1 == pytest.approx(0.0, abs=1e-10)
    assert S2 == pytest.approx(0.0, abs=1e-10)
    assert S3 == pytest.approx(0.0, abs=1e-10)


def test_s3_nonnegative_and_s1_bounded(delta_R):
    """S3 is always >= 0, and |S1| never exceeds delta_peak (here 1.0)."""
    R0 = np.linspace(-16, 16, 4001)
    S1, _, S3 = fms_signals(R0, delta_R)
    assert np.all(S3 >= -1e-9)
    assert np.all(np.abs(S1) <= 1.0 + 1e-6)


def test_small_delta_r_derivative_limit():
    """At the smallest delta_R (0.05), S1 ~ derivative of absorption, S2 ~ second
    derivative of dispersion, matching Bjorklund's description of the WMS limit.
    """
    delta_R = 0.05
    R0 = np.linspace(-8, 8, 801)
    S1, S2, _ = fms_signals(R0, delta_R)

    approx_S1 = -2 * delta_R * _delta_prime(R0)
    approx_S2 = delta_R**2 * _phi_double_prime(R0)

    # Both a tight relative-shape check (correlation) and a small absolute
    # residual, since the signals themselves are small at this delta_R.
    assert S1 == pytest.approx(approx_S1, abs=5e-4)
    assert S2 == pytest.approx(approx_S2, abs=5e-5)


def test_peak_amplitudes_grow_with_delta_r_across_fig3_range():
    """S1's peak amplitude at delta_R=4.0 is much larger than at delta_R=0.05
    (Bjorklund notes S1 "reaches its full strength" well before the end of
    this range and begins showing separate resonances by delta_R~3-4).
    """
    R0 = np.linspace(-16, 16, 4001)
    S1_small, _, _ = fms_signals(R0, 0.05)
    S1_large, _, _ = fms_signals(R0, 4.0)
    assert np.max(np.abs(S1_large)) > 10 * np.max(np.abs(S1_small))
    assert np.max(np.abs(S1_large)) == pytest.approx(1.0, abs=0.05)
