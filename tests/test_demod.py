"""Unit tests for the phase-sensitive demodulation core (``core/demod.py``).

Each test pins down a property from Bjorklund et al., Appl. Phys. B 32,
145-152 (1983), Eq. (4), rather than an implementation detail. These helpers
are shared verbatim by the (not yet implemented) FMS and PDH pages.
"""

import numpy as np
import pytest

from core.demod import mixed_signal, s1_component, s2_component, s3_magnitude


def test_s1_antisymmetric_in_its_arguments():
    """s1_component(a, b) == -s1_component(b, a), and 0 when a == b."""
    a, b = 0.7, 0.3
    assert s1_component(a, b) == pytest.approx(-s1_component(b, a))
    assert s1_component(a, a) == pytest.approx(0.0)


def test_s2_vanishes_when_all_inputs_equal():
    """s2_component(x, x, x) == 0 for any x (on-resonance carrier == sidebands)."""
    for x in [0.0, 0.5, -1.2]:
        assert s2_component(x, x, x) == pytest.approx(0.0, abs=1e-12)


def test_s2_matches_definition():
    """s2_component(phi_minus, phi_plus, phi_0) == phi_plus + phi_minus - 2*phi_0."""
    phi_minus, phi_plus, phi_0 = 0.3, -0.1, 0.05
    expected = phi_plus + phi_minus - 2 * phi_0
    assert s2_component(phi_minus, phi_plus, phi_0) == pytest.approx(expected)


def test_s3_magnitude_nonnegative_and_reduces_to_abs_s1():
    """s3_magnitude >= 0 always, and equals |s1| when s2 == 0."""
    s1 = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    s2 = np.zeros_like(s1)
    assert np.all(s3_magnitude(s1, s2) >= 0.0)
    assert s3_magnitude(s1, s2) == pytest.approx(np.abs(s1))


def test_s3_magnitude_pythagorean():
    """s3_magnitude(s1, s2) == sqrt(s1**2 + s2**2)."""
    s1, s2 = 3.0, 4.0
    assert s3_magnitude(s1, s2) == pytest.approx(5.0)


def test_mixed_signal_recovers_s1_and_s2_at_reference_angles():
    """theta=0 recovers S1; theta=pi/2 recovers S2."""
    s1, s2 = 1.3, -0.7
    assert mixed_signal(s1, s2, 0.0) == pytest.approx(s1)
    assert mixed_signal(s1, s2, np.pi / 2) == pytest.approx(s2)


def test_mixed_signal_is_linear_in_s1_s2():
    """mixed_signal is linear in (s1, s2) for fixed theta."""
    theta = 0.9
    s1a, s2a = 1.0, 2.0
    s1b, s2b = -0.5, 0.3
    lhs = mixed_signal(s1a + s1b, s2a + s2b, theta)
    rhs = mixed_signal(s1a, s2a, theta) + mixed_signal(s1b, s2b, theta)
    assert lhs == pytest.approx(rhs)


def test_helpers_vectorize_over_arrays():
    """All four helpers accept array inputs and preserve shape."""
    a = np.linspace(-1, 1, 11)
    b = np.linspace(1, -1, 11)
    c = np.zeros(11)
    assert s1_component(a, b).shape == a.shape
    assert s2_component(a, b, c).shape == a.shape
    s1, s2 = s1_component(a, b), s2_component(a, b, c)
    assert s3_magnitude(s1, s2).shape == a.shape
    assert mixed_signal(s1, s2, np.linspace(0, np.pi, 11)).shape == a.shape
