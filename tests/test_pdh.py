"""Unit tests for the PDH cavity error-signal model (``core/pdh.py``).

Each test pins down a property from Black, Am. J. Phys. 69, 79 (2001), rather
than an implementation detail. Figure-specific regression tests against Figs.
6 and 7 live in ``test_pdh_fig6.py`` and ``test_pdh_fig7.py``.
"""

import numpy as np
import pytest
from scipy.special import jv

from core.demod import mixed_signal
from core.pdh import (
    cavity_reflection,
    cavity_reflection_general,
    finesse_from_r,
    pdh_error_signal,
    pdh_error_signal_mixed,
    pdh_frequency_discriminant,
    pdh_optimum_modulation_depth,
    r_from_finesse,
    sideband_powers,
)


def test_cavity_reflection_zero_at_resonance():
    """F(x, r) == 0 at integer x (exact cavity resonance), for any r."""
    # Residual is pure floating-point round-off in exp(2j*pi*x), which grows
    # slowly with x; 1e-9 is far below any physical scale here.
    for r in [0.5, 0.9, 0.99, 0.999]:
        for x_int in [0, 1, 2, 5, 10]:
            assert cavity_reflection(x_int, r) == pytest.approx(0.0, abs=1e-9)


def test_general_cavity_reduces_to_symmetric():
    """cavity_reflection_general reduces to cavity_reflection for a symmetric,
    lossless cavity (r1 == r2 == r, t1 == sqrt(1 - r**2))."""
    r = 0.98
    t1 = np.sqrt(1 - r**2)
    x = np.linspace(0.4, 1.6, 41)
    assert cavity_reflection_general(x, r, t1, r) == pytest.approx(cavity_reflection(x, r))


def test_finesse_r_round_trip_and_known_values():
    """finesse_from_r and r_from_finesse invert each other; check textbook values."""
    for finesse in [10.0, 100.0, 500.0, 1e5]:
        r = r_from_finesse(finesse)
        assert finesse_from_r(r) == pytest.approx(finesse)
    # finesse = pi / (1 - r**2): at r**2 = 1 - pi/100, finesse == 100.
    assert finesse_from_r(np.sqrt(1 - np.pi / 100)) == pytest.approx(100.0)


def test_sideband_powers_conserve_for_small_beta():
    """Pc + 2*Ps approaches P0 for small beta (most power in carrier + 1st order)."""
    for beta in [0.01, 0.05, 0.1]:
        Pc, Ps = sideband_powers(beta, P0=1.0)
        assert Pc + 2 * Ps == pytest.approx(1.0, abs=1e-3)


def test_error_signal_exact_null_at_resonance():
    """Both error-signal components are exactly 0 at x=1, for any finesse/x_mod/beta."""
    for finesse in [50.0, 500.0, 5000.0]:
        r = r_from_finesse(finesse)
        for x_mod in [1e-3, 0.04, 0.2]:
            for beta in [0.5, 1.0, 1.5]:
                cos_c, sin_c = pdh_error_signal(1.0, x_mod, r, beta)
                assert cos_c == pytest.approx(0.0, abs=1e-10)
                assert sin_c == pytest.approx(0.0, abs=1e-10)


def test_error_signal_components_antisymmetric_about_resonance():
    """Both components are exactly odd about x=1: f(1+u) == -f(1-u).

    A consequence of F(2-x) == conj(F(x)) for the symmetric lossless cavity,
    proven independent of finesse, x_mod, and beta. This is the property that
    guarantees the lock's zero crossing sits exactly on resonance.
    """
    r = r_from_finesse(500.0)
    u = np.linspace(0.0005, 0.08, 400)
    for x_mod in [1e-3, 0.04]:
        cp, sp = pdh_error_signal(1 + u, x_mod, r, 1.0)
        cm, sm = pdh_error_signal(1 - u, x_mod, r, 1.0)
        assert cp == pytest.approx(-cm, abs=1e-10)
        assert sp == pytest.approx(-sm, abs=1e-10)


def test_error_signal_mixed_recovers_components():
    """theta=0 recovers the cosine component; theta=pi/2 recovers the sine one."""
    r = r_from_finesse(500.0)
    x = np.linspace(0.95, 1.05, 51)
    cos_c, sin_c = pdh_error_signal(x, 0.04, r, 1.0)
    assert pdh_error_signal_mixed(x, 0.04, r, 1.0, 0.0) == pytest.approx(cos_c)
    assert pdh_error_signal_mixed(x, 0.04, r, 1.0, np.pi / 2) == pytest.approx(sin_c)
    theta = 0.6
    assert pdh_error_signal_mixed(x, 0.04, r, 1.0, theta) == pytest.approx(
        mixed_signal(cos_c, sin_c, theta))


def test_slow_modulation_cosine_dominates():
    """Deep in the slow regime (x_mod much less than the linewidth), the cosine
    component strongly dominates the sine one near resonance (Black Sec. IV.A)."""
    finesse = 500.0
    r = r_from_finesse(finesse)
    linewidth = 1 / finesse
    x_mod = linewidth / 20          # deep slow regime
    x = np.linspace(1 - linewidth, 1 + linewidth, 4001)
    cos_c, sin_c = pdh_error_signal(x, x_mod, r, 1.0)
    assert np.max(np.abs(cos_c)) > 5 * np.max(np.abs(sin_c))


def test_fast_modulation_sine_dominates_near_resonance():
    """Deep in the fast regime, the sine component strongly dominates near
    resonance (Black Sec. IV.B, the standard PDH error signal)."""
    finesse = 500.0
    r = r_from_finesse(finesse)
    linewidth = 1 / finesse
    x_mod = 0.04                     # ~20 linewidths, fast regime
    x = np.linspace(1 - linewidth / 2, 1 + linewidth / 2, 4001)
    cos_c, sin_c = pdh_error_signal(x, x_mod, r, 1.0)
    assert np.max(np.abs(sin_c)) > 5 * np.max(np.abs(cos_c))


def test_optimum_modulation_depth_maximizes_sqrt_pc_ps():
    """pdh_optimum_modulation_depth returns 1.08, which maximizes sqrt(Pc*Ps)."""
    beta_opt = pdh_optimum_modulation_depth()
    assert beta_opt == pytest.approx(1.08, abs=0.005)

    betas = np.linspace(0.2, 3.0, 5000)
    product = np.abs(jv(0, betas) * jv(1, betas))
    assert abs(jv(0, beta_opt) * jv(1, beta_opt)) >= product.max() - 1e-4


def test_frequency_discriminant_formula_and_sign():
    """pdh_frequency_discriminant equals -8*sqrt(Pc*Ps)/delta_nu and is negative."""
    finesse = 500.0
    r = r_from_finesse(finesse)
    delta_nu_fsr = 1.0
    Pc, Ps = sideband_powers(1.0)
    D = pdh_frequency_discriminant(r, delta_nu_fsr, Pc, Ps)
    expected = -8 * np.sqrt(Pc * Ps) / (delta_nu_fsr / finesse_from_r(r))
    assert D == pytest.approx(expected)
    assert D < 0
