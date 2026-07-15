"""Physics of electro-optic phase modulation.

This module contains the numerical core of the EOM Sideband Explorer, kept
deliberately free of any user-interface or plotting code so that it can be
imported, reused, and unit-tested on its own (see ``tests/test_bessel.py``).

Background
----------
An ideal electro-optic phase modulator imprints a sinusoidal phase onto a
monochromatic laser field::

    E(t) = E0 * exp(i[wc*t + beta*sin(wm*t)])

where ``wc`` is the optical carrier (angular) frequency, ``wm = 2*pi*f0`` is the
modulation (angular) frequency, and ``beta`` is the modulation depth in radians.
The Jacobi-Anger identity expands this into a carrier plus an infinite,
discrete set of sidebands spaced by the modulation frequency::

    E(t) = E0 * sum_n J_n(beta) * exp(i(wc + n*wm)*t)

The spectral line at detuning ``n * f0`` therefore has amplitude ``J_n(beta)``
and intensity ``|J_n(beta)|^2``, where ``J_n`` is the Bessel function of the
first kind. Two properties follow and are used throughout the app:

* Symmetry: ``J_{-n}(beta) = (-1)^n J_n(beta)``, so the +n and -n orders carry
  equal intensity ``|J_n(beta)|^2``.
* Energy conservation: ``sum_n |J_n(beta)|^2 = 1``, so phase modulation only
  redistributes optical power between the carrier and its sidebands.
"""

import numpy as np
from scipy.special import jv


def sideband_intensities(beta, N):
    """Return the per-line intensities ``|J_n(beta)|^2`` for orders ``0..N``.

    Only the non-negative orders are computed, since the negative orders are
    the mirror image: ``|J_{-n}(beta)|^2 == |J_n(beta)|^2``.

    Parameters
    ----------
    beta : float
        Modulation depth in radians.
    N : int
        Highest sideband order to include. The result covers orders
        ``0, 1, ..., N``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N + 1,)`` where element ``n`` is ``|J_n(beta)|^2``.
    """
    return jv(np.arange(N + 1), beta)**2


def captured_power(Jn2):
    """Return the total optical power contained in a set of sideband orders.

    Sums the intensities of the carrier and both (+n and -n) copies of every
    higher order. For a complete spectrum this approaches ``1.0`` (energy
    conservation); for a truncated set of orders it is the fraction of the
    optical power captured by those orders.

    Parameters
    ----------
    Jn2 : numpy.ndarray
        Per-line intensities for orders ``0..N``, as returned by
        :func:`sideband_intensities`. Element 0 is the carrier.

    Returns
    -------
    float
        ``Jn2[0] + 2 * sum(Jn2[1:])`` — the carrier plus both sides of each
        higher order.
    """
    return float(Jn2[0] + 2 * Jn2[1:].sum())
