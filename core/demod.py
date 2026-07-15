"""Phase-sensitive demodulation of per-component lineshape values.

This module turns a set of per-component (carrier, upper sideband, lower
sideband) absorption/dispersion values into the demodulated error-signal
components ``S1``/``S2``/``S3``, and mixes them by an arbitrary demodulation
phase. It implements Bjorklund et al., Appl. Phys. B 32, 145-152 (1983),
Eq. (4).

This is the one physics module genuinely shared between the FMS and PDH
pages: both demodulate a carrier-plus-two-sidebands beat note at the
modulation frequency, and both expose a demodulation-phase slider backed by
:func:`mixed_signal`. What differs between the two pages is *which*
lineshape produces the underlying attenuation/phase values (an isolated
Lorentzian absorber for FMS, a Fabry-Perot cavity reflection coefficient for
PDH) -- that physics lives in ``core/fms.py`` and ``core/pdh.py``
respectively, not here.

All functions accept and return ``numpy`` arrays (or scalars) so they
vectorize over a detuning axis without a Python loop.
"""

import numpy as np


def s1_component(delta_minus, delta_plus):
    """Return the in-phase (``cos(w_m t)``) demodulated component ``S1``.

    Bjorklund et al., Eq. (4): ``S1 = delta_minus - delta_plus``, the
    background-subtracted difference in absorption between the lower and
    upper sidebands.

    Parameters
    ----------
    delta_minus : array_like
        Absorption at the lower sideband (see :func:`core.lineshapes.delta`).
    delta_plus : array_like
        Absorption at the upper sideband.

    Returns
    -------
    numpy.ndarray or float
        ``S1``, same shape as the (broadcast) inputs.
    """
    return np.asarray(delta_minus) - np.asarray(delta_plus)


def s2_component(phi_minus, phi_plus, phi_0):
    """Return the quadrature (``sin(w_m t)``) demodulated component ``S2``.

    Bjorklund et al., Eq. (4): ``S2 = phi_plus + phi_minus - 2 * phi_0``, the
    sideband dispersion sum offset by twice the carrier's own dispersion.

    Parameters
    ----------
    phi_minus : array_like
        Dispersion at the lower sideband (see :func:`core.lineshapes.phi`).
    phi_plus : array_like
        Dispersion at the upper sideband.
    phi_0 : array_like
        Dispersion at the carrier.

    Returns
    -------
    numpy.ndarray or float
        ``S2``, same shape as the (broadcast) inputs.
    """
    return np.asarray(phi_plus) + np.asarray(phi_minus) - 2 * np.asarray(phi_0)


def s3_magnitude(s1, s2):
    """Return the total demodulated signal magnitude ``S3 = sqrt(S1^2 + S2^2)``.

    Uses ``numpy.hypot`` for the usual numerical-overflow robustness of that
    function; always non-negative regardless of the signs of ``s1``/``s2``.

    Parameters
    ----------
    s1 : array_like
        In-phase component, e.g. from :func:`s1_component`.
    s2 : array_like
        Quadrature component, e.g. from :func:`s2_component`.

    Returns
    -------
    numpy.ndarray or float
        ``S3``, same shape as the (broadcast) inputs, always ``>= 0``.
    """
    return np.hypot(s1, s2)


def mixed_signal(s1, s2, theta):
    """Return the demodulated signal for an arbitrary demodulation phase.

    Implements the "phase adjuster" mentioned in Bjorklund et al. but never
    plotted there: ``mixed_signal = s1 * cos(theta) + s2 * sin(theta)``. This
    is the function that backs a demodulation-phase slider on both the FMS
    and PDH pages -- ``theta = 0`` recovers ``S1``, ``theta = pi/2``
    recovers ``S2``.

    Parameters
    ----------
    s1 : array_like
        In-phase component, e.g. from :func:`s1_component`.
    s2 : array_like
        Quadrature component, e.g. from :func:`s2_component`.
    theta : array_like
        Demodulation phase, in radians.

    Returns
    -------
    numpy.ndarray or float
        The phase-mixed signal, broadcast over ``s1``, ``s2``, and ``theta``.
    """
    return np.asarray(s1) * np.cos(theta) + np.asarray(s2) * np.sin(theta)
