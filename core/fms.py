"""Small-modulation-index (M << 1) FM spectroscopy (FMS) signal model.

Implements Bjorklund, Levenson, Lenth & Ortiz, "Frequency Modulation (FM)
Spectroscopy: Theory of Lineshapes and Signal-to-Noise Analysis," Appl.
Phys. B 32, 145-152 (1983), Eqs. (1)-(7): a strong carrier plus two weak
sidebands probe an isolated Lorentzian spectral feature, and the
carrier-sideband beat note at the modulation frequency is demodulated into
in-phase (``S1``), quadrature (``S2``), and total-magnitude (``S3``)
components.

This module is deliberately thin: it evaluates the shared Lorentzian
lineshape (:mod:`core.lineshapes`) at the carrier and both sidebands, then
composes the result with the shared demodulation helpers
(:mod:`core.demod`), which the (upcoming) PDH page reuses for its own
phase-mixing. All the FMS-specific physics is just *which* detunings get
plugged into ``delta``/``phi`` and how ``delta_R`` (the normalized sideband
spacing) enters that.

All functions accept and return ``numpy`` arrays (or scalars) so they
vectorize over a detuning axis (``R0``) without a Python loop.

Stretch goal (not implemented here): ``fms_signals_arbitrary_M``, summing
Bessel sideband orders from ``core/bessel.py`` to go beyond the M << 1
limit used throughout this module. Left for a later pass.
"""

import numpy as np

from core.demod import mixed_signal, s1_component, s2_component, s3_magnitude
from core.lineshapes import delta, phi


def fms_signals(R0, delta_R, delta_peak=1.0):
    """Return the demodulated FMS signal triple ``(S1, S2, S3)`` at ``R0``.

    Evaluates the shared Lorentzian absorption/dispersion pair
    (:func:`core.lineshapes.delta`, :func:`core.lineshapes.phi`) at the
    lower sideband (``R0 - delta_R``), the carrier (``R0``), and the upper
    sideband (``R0 + delta_R``), then composes the beat-signal components
    via :mod:`core.demod`, reproducing Bjorklund et al., Eqs. (3)-(7).

    Parameters
    ----------
    R0 : array_like
        Normalized carrier detuning (the x-axis of Bjorklund Figs. 3-4).
    delta_R : float
        Normalized sideband spacing, ``omega_m / (delta_omega_fwhm / 2)``.
    delta_peak : float, optional
        Peak absorption at line center. Defaults to ``1.0``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(S1, S2, S3)``, each the same shape as ``R0``. ``S1`` and ``S2``
        are signed (in-phase/quadrature); ``S3`` is their magnitude and is
        always non-negative. All three vanish at ``R0 == 0`` for every
        ``delta_R``.
    """
    R0 = np.asarray(R0, dtype=float)
    delta_minus, delta_plus = delta(R0 - delta_R, delta_peak), delta(R0 + delta_R, delta_peak)
    phi_minus, phi_0, phi_plus = (
        phi(R0 - delta_R, delta_peak), phi(R0, delta_peak), phi(R0 + delta_R, delta_peak))

    S1 = s1_component(delta_minus, delta_plus)
    S2 = s2_component(phi_minus, phi_plus, phi_0)
    S3 = s3_magnitude(S1, S2)
    return S1, S2, S3


def fms_signal_mixed(R0, delta_R, theta, delta_peak=1.0):
    """Return the FMS signal observed with demodulation phase ``theta``.

    Wraps :func:`fms_signals` with :func:`core.demod.mixed_signal`, backing
    the "phase adjuster" slider on the FMS page. ``theta = 0`` recovers
    ``S1``; ``theta = pi/2`` recovers ``S2``.

    Parameters
    ----------
    R0 : array_like
        Normalized carrier detuning.
    delta_R : float
        Normalized sideband spacing.
    theta : array_like
        Demodulation phase, in radians.
    delta_peak : float, optional
        Peak absorption at line center. Defaults to ``1.0``.

    Returns
    -------
    numpy.ndarray or float
        The phase-mixed signal, same shape as ``R0`` (broadcast with
        ``theta`` if that is also an array).
    """
    S1, S2, _ = fms_signals(R0, delta_R, delta_peak)
    return mixed_signal(S1, S2, theta)


def fms_discriminant_slope(delta_R):
    """Return the analytic slopes of ``S1`` and ``S2`` at line center.

    Closed forms for ``dS/dR0`` at ``R0 = 0`` (``delta_peak = 1``), used by
    the "how do I choose omega_m" panel:

    - ``slope_S1 = 4 * delta_R / (delta_R**2 + 1)**2``, maximized at
      ``delta_R = 1/sqrt(3)`` with peak value ``3*sqrt(3)/4``.
    - ``slope_S2 = 2 * (1 - delta_R**2) / (1 + delta_R**2)**2 - 2``, a
      *signed* quantity: it is 0 at ``delta_R = 0``, reaches a minimum of
      ``-2.25`` at ``delta_R = sqrt(3)``, and approaches ``-2`` as
      ``delta_R`` grows further (it does not monotonically saturate at
      ``-2``; ``abs(slope_S2)`` overshoots to 2.25 before settling back).

    Parameters
    ----------
    delta_R : array_like
        Normalized sideband spacing.

    Returns
    -------
    tuple of numpy.ndarray
        ``(slope_S1, slope_S2)``, same shape as ``delta_R``.
    """
    delta_R = np.asarray(delta_R, dtype=float)
    slope_S1 = 4 * delta_R / (delta_R**2 + 1)**2
    slope_S2 = 2 * (1 - delta_R**2) / (1 + delta_R**2)**2 - 2
    return slope_S1, slope_S2
