"""Regression tests against Bjorklund et al. Fig. 4 (0.1 <= delta_R <= 50.0).

Fig. 4 (Appl. Phys. B 32, 145-152 (1983), p. 149) plots S1, S2, S3 vs R0 for
delta_R in {0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 50.0}, with R0 spanning a
window that scales with delta_R so the resolved sideband features are
visible. For delta_R >= 5, Bjorklund notes the S1 resonances become close to
the Lorentzian absorption lineshape itself (i.e. approach +-1), and S3 shows
resolved peaks at these resonances.
"""

import numpy as np
import pytest

from core.fms import fms_signals

FIG4_DELTA_R = [0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 50.0]


def _r0_grid(delta_R, n=8001):
    """An R0 window wide enough to show both sideband resonances plus margin."""
    span = 2 * delta_R + 16
    return np.linspace(-span, span, n)


@pytest.fixture(params=FIG4_DELTA_R)
def delta_R(request):
    return request.param


def test_all_signals_zero_at_line_center(delta_R):
    """S1, S2, S3 are all zero at R0 = 0, for every delta_R shown in Fig. 4."""
    S1, S2, S3 = fms_signals(0.0, delta_R)
    assert S1 == pytest.approx(0.0, abs=1e-10)
    assert S2 == pytest.approx(0.0, abs=1e-10)
    assert S3 == pytest.approx(0.0, abs=1e-10)


def test_s3_nonnegative_and_s1_bounded(delta_R):
    """S3 is always >= 0, and |S1| never exceeds delta_peak (here 1.0)."""
    R0 = _r0_grid(delta_R)
    S1, _, S3 = fms_signals(R0, delta_R)
    assert np.all(S3 >= -1e-9)
    assert np.all(np.abs(S1) <= 1.0 + 1e-6)


def test_resolved_sidebands_approach_unit_lorentzian_peaks():
    """For delta_R >= 10 (fully resolved sidebands), S1's extrema approach
    +-1 and S3's peak approaches 1, matching the isolated-Lorentzian limit
    Bjorklund describes for large delta_R.
    """
    for delta_R in [10.0, 20.0, 30.0, 50.0]:
        R0 = _r0_grid(delta_R)
        S1, _, S3 = fms_signals(R0, delta_R)
        assert S1.max() == pytest.approx(1.0, abs=0.05)
        assert S1.min() == pytest.approx(-1.0, abs=0.05)
        assert S3.max() == pytest.approx(1.0, abs=0.05)


def test_s1_extrema_grow_monotonically_with_delta_r():
    """S1's peak amplitude increases monotonically across the Fig. 4 delta_R
    range, approaching (but never exceeding) delta_peak = 1.
    """
    peaks = []
    for delta_R in FIG4_DELTA_R:
        R0 = _r0_grid(delta_R)
        S1, _, _ = fms_signals(R0, delta_R)
        peaks.append(np.max(np.abs(S1)))
    assert np.all(np.diff(peaks) > 0)
    assert peaks[-1] <= 1.0 + 1e-6
