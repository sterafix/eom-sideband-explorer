"""Shared Lorentzian absorption/dispersion lineshape pair.

This module is the atomic building block for the FM spectroscopy (FMS) page:
one place that defines the normalized Lorentzian absorption ``delta`` and
dispersion ``phi`` used throughout Bjorklund, Levenson, Lenth & Ortiz,
"Frequency Modulation (FM) Spectroscopy: Theory of Lineshapes and Signal to
Noise Analysis," Appl. Phys. B 32, 145-152 (1983) (their Eqs. 5-7).

These functions carry no notion of sidebands or modulation frequency; they
are composed by :mod:`core.demod` and, on top of that, :mod:`core.fms` into
the actual FMS error-signal model. The PDH page does not use this module
directly — it has its own cavity reflection lineshape in ``core/pdh.py``,
since a Fabry-Perot cavity resonance is not a Lorentzian in reflected
amplitude the way an isolated absorption feature is.

All functions accept and return ``numpy`` arrays (or scalars) so they
vectorize over a detuning axis without a Python loop.

Stretch goal (not implemented here): Doppler-broadened ``voigt_delta`` /
``voigt_phi`` variants using ``scipy.special.wofz``, for a more realistic
FMS case. Left for a later pass once the small-M Lorentzian case is in place
and tested.
"""

import numpy as np


def delta(R, delta_peak=1.0):
    """Return the normalized Lorentzian absorption at detuning ``R``.

    Bjorklund et al., Eq. (5): ``delta(R) = delta_peak / (R**2 + 1)``. This is
    an even function of ``R``, peaking at ``delta_peak`` on resonance
    (``R == 0``) and decaying to zero far from resonance.

    Parameters
    ----------
    R : array_like
        Normalized detuning (see :func:`R_of_omega`), dimensionless.
    delta_peak : float, optional
        Peak absorption at line center. Defaults to ``1.0``.

    Returns
    -------
    numpy.ndarray or float
        The absorption ``delta(R)``, same shape as ``R``.
    """
    R = np.asarray(R, dtype=float)
    return delta_peak / (R**2 + 1)


def phi(R, delta_peak=1.0):
    """Return the normalized Lorentzian dispersion at detuning ``R``.

    Bjorklund et al., Eq. (6): ``phi(R) = delta_peak * R / (R**2 + 1)``. This
    is an odd function of ``R``, vanishing on resonance (``R == 0``) and
    decaying to zero far from resonance, with extrema of
    ``+-delta_peak / 2`` at ``R = -+1``.

    Parameters
    ----------
    R : array_like
        Normalized detuning (see :func:`R_of_omega`), dimensionless.
    delta_peak : float, optional
        Peak absorption of the corresponding :func:`delta`. Defaults to
        ``1.0``.

    Returns
    -------
    numpy.ndarray or float
        The dispersion ``phi(R)``, same shape as ``R``.
    """
    R = np.asarray(R, dtype=float)
    return delta_peak * R / (R**2 + 1)


def R_of_omega(omega, Omega, delta_omega_fwhm):
    """Return the normalized detuning ``R`` for optical frequency ``omega``.

    Bjorklund et al., Eq. (7): ``R = (omega - Omega) / (delta_omega_fwhm / 2)``,
    i.e. the detuning from line center ``Omega``, expressed in half-linewidths
    so that ``R = +-1`` sits at the half-maximum points of :func:`delta`.

    Parameters
    ----------
    omega : array_like
        Optical (angular) frequency at which to evaluate the detuning.
    Omega : float
        Line center (angular) frequency.
    delta_omega_fwhm : float
        Full width at half maximum of the absorption line, in the same
        (angular) frequency units as ``omega`` and ``Omega``.

    Returns
    -------
    numpy.ndarray or float
        The normalized detuning ``R``, same shape as ``omega``.
    """
    omega = np.asarray(omega, dtype=float)
    return (omega - Omega) / (delta_omega_fwhm / 2)
