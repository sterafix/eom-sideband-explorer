"""Regression tests against Black Fig. 7 (fast-modulation PDH error signal).

Fig. 7 (Am. J. Phys. 69, 79 (2001), p. 83) plots the error signal for a cavity
of finesse 500 with the modulation frequency high: "about 20 linewidths:
roughly 4% of a free spectral range." In this fast regime the sine
(quadrature) component dominates near resonance, and the signal shows the
characteristic three-feature PDH shape: a steep central dispersive feature at
resonance plus two sideband-reflection features at x = 1 +- x_mod.
"""

import numpy as np
import pytest

from core.pdh import pdh_error_signal, r_from_finesse

FINESSE = 500.0
X_MOD = 0.04        # ~20 linewidths (1/finesse = 2e-3), per Black's Fig. 7 caption
R = r_from_finesse(FINESSE)
LINEWIDTH = 1 / FINESSE


def test_sine_component_dominates_near_resonance():
    """Within half a linewidth of resonance, the sine component strongly
    dominates the cosine one (globally their peak magnitudes are comparable)."""
    x = np.linspace(1 - LINEWIDTH / 2, 1 + LINEWIDTH / 2, 5001)
    cos_c, sin_c = pdh_error_signal(x, X_MOD, R, 1.0)
    assert np.max(np.abs(sin_c)) > 10 * np.max(np.abs(cos_c))


def test_three_feature_shape():
    """|sin| shows three separated features: a central one at x=1 and two
    sideband features at x = 1 +- x_mod, each rising well above the gaps."""
    x = np.linspace(0.9, 1.1, 40001)
    _, sin_c = pdh_error_signal(x, X_MOD, R, 1.0)
    a = np.abs(sin_c)

    def peak_near(center, half=0.01):
        window = (x > center - half) & (x < center + half)
        return a[window].max()

    central = peak_near(1.0)
    lower = peak_near(1 - X_MOD)
    upper = peak_near(1 + X_MOD)

    # A gap between the central and a sideband feature.
    gap_window = (x > 1 + X_MOD / 2 - 0.005) & (x < 1 + X_MOD / 2 + 0.005)
    gap = a[gap_window].max()

    assert central > 2 * gap
    assert lower > 2 * gap
    assert upper > 2 * gap
    assert lower == pytest.approx(upper, rel=1e-6)     # symmetric sidebands
    assert central > lower                             # central feature is tallest


def test_exact_null_at_resonance_for_various_conditions():
    """The error signal is exactly zero at x=1 for every finesse and x_mod
    (a direct consequence of F(1, r) = 0)."""
    for finesse in [100.0, 500.0, 2000.0]:
        r = r_from_finesse(finesse)
        for x_mod in [0.01, 0.04, 0.1]:
            cos_c, sin_c = pdh_error_signal(1.0, x_mod, r, 1.0)
            assert cos_c == pytest.approx(0.0, abs=1e-10)
            assert sin_c == pytest.approx(0.0, abs=1e-10)
