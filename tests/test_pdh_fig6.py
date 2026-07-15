"""Regression tests against Black Fig. 6 (slow-modulation PDH error signal).

Fig. 6 (Am. J. Phys. 69, 79 (2001), p. 83) plots the error signal for a cavity
of finesse 500 with the modulation frequency low: "about half a linewidth:
about 10^-3 of a free spectral range." In this slow regime the cosine
(in-phase) component dominates and the signal resembles the derivative of the
cavity lineshape, antisymmetric about resonance.

Note: the expansion plan quoted x_mod ~= 5e-4; the paper's caption says
~10^-3 (half a linewidth at finesse 500), which is used here.
"""

import numpy as np
import pytest

from core.pdh import pdh_error_signal, r_from_finesse

FINESSE = 500.0
X_MOD = 1e-3        # half a linewidth (1/finesse = 2e-3), per Black's Fig. 6 caption
R = r_from_finesse(FINESSE)


def test_cosine_component_dominates():
    """In the slow regime the cosine component exceeds the sine one in peak
    magnitude over the plotted window."""
    x = np.linspace(0.995, 1.005, 20001)
    cos_c, sin_c = pdh_error_signal(x, X_MOD, R, 1.0)
    assert np.max(np.abs(cos_c)) > np.max(np.abs(sin_c))


def test_cosine_component_odd_about_resonance():
    """The Fig. 6 (cosine) signal is exactly antisymmetric about x=1."""
    u = np.linspace(0.0002, 0.005, 500)
    cos_p, _ = pdh_error_signal(1 + u, X_MOD, R, 1.0)
    cos_m, _ = pdh_error_signal(1 - u, X_MOD, R, 1.0)
    assert cos_p == pytest.approx(-cos_m, abs=1e-10)


def test_single_zero_crossing_at_resonance():
    """The cosine signal has opposite signs just below and just above x=1,
    i.e. a single dispersive zero crossing sitting exactly on resonance."""
    delta = 1e-4        # a fraction of the linewidth
    cos_below, _ = pdh_error_signal(1 - delta, X_MOD, R, 1.0)
    cos_above, _ = pdh_error_signal(1 + delta, X_MOD, R, 1.0)
    assert cos_below * cos_above < 0
    cos_at, _ = pdh_error_signal(1.0, X_MOD, R, 1.0)
    assert cos_at == pytest.approx(0.0, abs=1e-10)
