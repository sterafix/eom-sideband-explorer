"""Unit tests for the small-M FMS signal model (``core/fms.py``).

Each test pins down a property from Bjorklund et al., Appl. Phys. B 32,
145-152 (1983), rather than an implementation detail. Figure-specific
regression tests against Figs. 3 and 4 live in ``test_fms_fig3.py`` and
``test_fms_fig4.py``.
"""

import numpy as np
import pytest

from core.demod import mixed_signal, s1_component, s2_component, s3_magnitude
from core.fms import fms_discriminant_slope, fms_signal_mixed, fms_signals
from core.lineshapes import delta, phi


def test_fms_signals_matches_manual_composition():
    """fms_signals equals building S1/S2/S3 by hand from delta/phi + demod."""
    R0, delta_R = 2.3, 1.7
    S1, S2, S3 = fms_signals(R0, delta_R)

    dm, dp = delta(R0 - delta_R), delta(R0 + delta_R)
    pm, p0, pp = phi(R0 - delta_R), phi(R0), phi(R0 + delta_R)
    expected_S1 = s1_component(dm, dp)
    expected_S2 = s2_component(pm, pp, p0)
    expected_S3 = s3_magnitude(expected_S1, expected_S2)

    assert S1 == pytest.approx(expected_S1)
    assert S2 == pytest.approx(expected_S2)
    assert S3 == pytest.approx(expected_S3)


def test_fms_signals_vanish_at_line_center():
    """S1 == S2 == S3 == 0 at R0 == 0, for any delta_R (Bjorklund's key property)."""
    for delta_R in [0.05, 0.1, 1.0, 1 / np.sqrt(3), 5.0, 20.0, 50.0]:
        S1, S2, S3 = fms_signals(0.0, delta_R)
        assert S1 == pytest.approx(0.0, abs=1e-10)
        assert S2 == pytest.approx(0.0, abs=1e-10)
        assert S3 == pytest.approx(0.0, abs=1e-10)


def test_s1_bounded_and_s3_nonnegative():
    """|S1| <= delta_peak everywhere, and S3 >= 0 everywhere."""
    R0 = np.linspace(-50, 50, 2001)
    for delta_R in [0.1, 1.0, 5.0, 20.0]:
        S1, _, S3 = fms_signals(R0, delta_R)
        assert np.all(np.abs(S1) <= 1.0 + 1e-9)
        assert np.all(S3 >= -1e-12)


def test_delta_peak_scales_linearly():
    """fms_signals scales linearly with delta_peak (delta/phi both do)."""
    R0, delta_R = 3.0, 0.8
    S1a, S2a, _ = fms_signals(R0, delta_R, delta_peak=1.0)
    S1b, S2b, _ = fms_signals(R0, delta_R, delta_peak=2.5)
    assert S1b == pytest.approx(2.5 * S1a)
    assert S2b == pytest.approx(2.5 * S2a)


def test_fms_signal_mixed_recovers_s1_and_s2():
    """theta=0 recovers S1; theta=pi/2 recovers S2; general theta matches mixed_signal."""
    R0, delta_R = 1.5, 2.0
    S1, S2, _ = fms_signals(R0, delta_R)
    assert fms_signal_mixed(R0, delta_R, 0.0) == pytest.approx(S1)
    assert fms_signal_mixed(R0, delta_R, np.pi / 2) == pytest.approx(S2)
    theta = 0.73
    assert fms_signal_mixed(R0, delta_R, theta) == pytest.approx(mixed_signal(S1, S2, theta))


def test_discriminant_slope_matches_finite_difference_at_center():
    """fms_discriminant_slope matches a central finite difference of S1/S2 at R0=0."""
    h = 1e-6
    for delta_R in [0.05, 0.3, 1 / np.sqrt(3), 2.0, np.sqrt(3), 10.0]:
        S1_plus, S2_plus, _ = fms_signals(h, delta_R)
        S1_minus, S2_minus, _ = fms_signals(-h, delta_R)
        num_slope_S1 = (S1_plus - S1_minus) / (2 * h)
        num_slope_S2 = (S2_plus - S2_minus) / (2 * h)

        slope_S1, slope_S2 = fms_discriminant_slope(delta_R)
        assert slope_S1 == pytest.approx(num_slope_S1, abs=1e-4)
        assert slope_S2 == pytest.approx(num_slope_S2, abs=1e-4)


def test_discriminant_slope_s1_peaks_at_one_over_sqrt3():
    """slope_S1 is maximized at delta_R = 1/sqrt(3), with peak value 3*sqrt(3)/4."""
    optimum = 1 / np.sqrt(3)
    slope_at_optimum, _ = fms_discriminant_slope(optimum)
    assert slope_at_optimum == pytest.approx(3 * np.sqrt(3) / 4)

    nearby = np.linspace(0.01, 5.0, 5000)
    slopes, _ = fms_discriminant_slope(nearby)
    assert slope_at_optimum >= slopes.max() - 1e-9


def test_discriminant_slope_s2_overshoots_before_settling():
    """|slope_S2| is 0 at delta_R=0, reaches 2.25 at sqrt(3), then decays toward 2.

    The expansion plan's claim that slope_S2 "monotonically increases,
    saturating at 2" does not hold: it overshoots. This test pins the
    actual (verified) behavior instead.
    """
    _, slope_at_zero = fms_discriminant_slope(0.0)
    assert slope_at_zero == pytest.approx(0.0)

    _, slope_at_sqrt3 = fms_discriminant_slope(np.sqrt(3))
    assert slope_at_sqrt3 == pytest.approx(-2.25)

    _, slope_at_large = fms_discriminant_slope(1e6)
    assert slope_at_large == pytest.approx(-2.0, abs=1e-6)

    # The overshoot: |slope_S2| at delta_R=sqrt(3) exceeds the delta_R -> infinity asymptote.
    assert abs(slope_at_sqrt3) > abs(slope_at_large)


def test_functions_vectorize_over_arrays():
    """fms_signals, fms_signal_mixed, and fms_discriminant_slope preserve array shape."""
    R0 = np.linspace(-10, 10, 41)
    S1, S2, S3 = fms_signals(R0, 1.0)
    assert S1.shape == R0.shape and S2.shape == R0.shape and S3.shape == R0.shape

    mixed = fms_signal_mixed(R0, 1.0, np.pi / 4)
    assert mixed.shape == R0.shape

    delta_R = np.linspace(0.1, 10, 23)
    slope_S1, slope_S2 = fms_discriminant_slope(delta_R)
    assert slope_S1.shape == delta_R.shape and slope_S2.shape == delta_R.shape
