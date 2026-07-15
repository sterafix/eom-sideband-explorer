"""Unit tests for the shared Lorentzian lineshape pair (``core/lineshapes.py``).

Each test pins down a property from Bjorklund et al., Appl. Phys. B 32,
145-152 (1983), Eqs. (5)-(7), rather than an implementation detail.
"""

import numpy as np
import pytest

from core.lineshapes import R_of_omega, delta, phi


def test_delta_peaks_at_delta_peak_on_resonance():
    """delta(0) == delta_peak, for any delta_peak."""
    for delta_peak in [0.5, 1.0, 3.2]:
        assert delta(0.0, delta_peak) == pytest.approx(delta_peak)


def test_phi_vanishes_on_resonance():
    """phi(0) == 0, regardless of delta_peak."""
    for delta_peak in [0.5, 1.0, 3.2]:
        assert phi(0.0, delta_peak) == pytest.approx(0.0, abs=1e-12)


def test_delta_is_even_phi_is_odd():
    """delta(R) == delta(-R) and phi(R) == -phi(-R) for a range of detunings."""
    R = np.array([0.1, 0.5, 1.0, 2.5, 7.0])
    assert delta(R) == pytest.approx(delta(-R))
    assert phi(R) == pytest.approx(-phi(-R))


def test_delta_and_phi_decay_far_from_resonance():
    """Both lineshapes vanish far from resonance (|R| -> infinity)."""
    R_far = 1e6
    assert delta(R_far) == pytest.approx(0.0, abs=1e-6)
    assert phi(R_far) == pytest.approx(0.0, abs=1e-6)


def test_absorption_dispersion_invariant():
    """delta(R)**2 + phi(R)**2 == delta_peak * delta(R) for all R (algebraic identity)."""
    R = np.linspace(-20, 20, 501)
    delta_peak = 1.7
    lhs = delta(R, delta_peak)**2 + phi(R, delta_peak)**2
    rhs = delta_peak * delta(R, delta_peak)
    assert lhs == pytest.approx(rhs)


def test_delta_peak_scales_linearly():
    """delta and phi both scale linearly with delta_peak."""
    R = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert delta(R, 2.0) == pytest.approx(2.0 * delta(R, 1.0))
    assert phi(R, 2.0) == pytest.approx(2.0 * phi(R, 1.0))


def test_R_of_omega_reference_points():
    """R_of_omega is 0 on resonance and 1 at the half-linewidth point."""
    Omega, fwhm = 2 * np.pi * 5e6, 2 * np.pi * 1e6
    assert R_of_omega(Omega, Omega, fwhm) == pytest.approx(0.0)
    assert R_of_omega(Omega + fwhm / 2, Omega, fwhm) == pytest.approx(1.0)
    assert R_of_omega(Omega - fwhm / 2, Omega, fwhm) == pytest.approx(-1.0)


def test_functions_vectorize_over_arrays():
    """delta, phi, and R_of_omega accept arrays and preserve shape."""
    R = np.linspace(-10, 10, 37)
    assert delta(R).shape == R.shape
    assert phi(R).shape == R.shape
    omega = np.linspace(0, 10, 37)
    assert R_of_omega(omega, 5.0, 2.0).shape == omega.shape
