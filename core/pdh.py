"""Pound-Drever-Hall (PDH) cavity-lock error signal model.

Implements Black, "An introduction to Pound-Drever-Hall laser frequency
stabilization," Am. J. Phys. 69, 79 (2001), Sec. III-IV: a phase-modulated
laser (carrier plus two sidebands) reflects off a Fabry-Perot cavity, and the
reflected power is demodulated at the modulation frequency into an in-phase
(cosine) and quadrature (sine) error signal.

Like the FMS page, PDH demodulates a carrier-plus-two-sidebands beat note and
exposes a demodulation-phase adjuster (:func:`core.demod.mixed_signal`, shared
verbatim). What differs is the lineshape: a Fabry-Perot cavity *reflection
coefficient* replaces the Lorentzian absorber, and the sideband bookkeeping
uses Bessel powers J0(beta)^2 / J1(beta)^2 rather than the small-M FMS
amplitudes.

**Normalized frequency convention.** Everything is expressed in the variable
``x = f / delta_nu_fsr`` (laser frequency in units of the cavity free spectral
range), matching Black's plot axes. One cavity round trip is a phase of
``2*pi*x``; Black's ``exp(i * omega / delta_nu_fsr)`` is exactly
``exp(2j*pi*x)`` once ``omega`` is read in free-spectral-range units. The
modulation frequency is likewise given as ``x_mod`` (a fraction of an FSR), so
the sidebands sit at ``x +- x_mod`` and cavity resonances at integer ``x``.

All functions accept and return ``numpy`` arrays (or scalars) and vectorize
over the frequency axis ``x`` without a Python loop.

Stretch goal (not implemented here): a shot-noise frequency-sensitivity
function mirroring Black Sec. V.B. Left for a later pass.
"""

import numpy as np
from scipy.special import jv

from core.demod import mixed_signal


def cavity_reflection(x, r):
    """Return the Fabry-Perot reflection coefficient ``F(x)`` (Black Eq. 3.1).

    Symmetric, lossless two-mirror cavity::

        F(x) = r * (exp(2j*pi*x) - 1) / (1 - r**2 * exp(2j*pi*x))

    where ``r`` is the amplitude reflection coefficient of each mirror. ``F``
    is complex; ``|F|**2`` is the reflected intensity (the Airy function) and
    ``arg(F)`` carries the which-side-of-resonance information the PDH lock
    exploits. ``F`` vanishes exactly at integer ``x`` (exact resonance).

    Parameters
    ----------
    x : array_like
        Laser frequency in units of the cavity free spectral range.
    r : float
        Mirror amplitude reflection coefficient, ``0 <= r < 1``.

    Returns
    -------
    numpy.ndarray or complex
        The complex reflection coefficient, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    phase = np.exp(2j * np.pi * x)
    return r * (phase - 1) / (1 - r**2 * phase)


def cavity_reflection_general(x, r1, t1, r2):
    """Return the reflection coefficient for a lossy/asymmetric cavity (Black App. A).

    General two-mirror case::

        F(x) = (-r1 + r2 * (r1**2 + t1**2) * exp(2j*pi*x))
               / (1 - r1 * r2 * exp(2j*pi*x))

    where ``r1``/``t1`` are the input mirror's amplitude reflection/transmission
    coefficients and ``r2`` the back mirror's reflection coefficient. Reduces
    to :func:`cavity_reflection` for a symmetric lossless cavity
    (``r1 == r2 == r``, ``r1**2 + t1**2 == 1``).

    Parameters
    ----------
    x : array_like
        Laser frequency in units of the cavity free spectral range.
    r1, t1 : float
        Input-mirror amplitude reflection and transmission coefficients.
    r2 : float
        Back-mirror amplitude reflection coefficient.

    Returns
    -------
    numpy.ndarray or complex
        The complex reflection coefficient, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    phase = np.exp(2j * np.pi * x)
    return (-r1 + r2 * (r1**2 + t1**2) * phase) / (1 - r1 * r2 * phase)


def finesse_from_r(r):
    """Return the cavity finesse ``pi / (1 - r**2)`` (Black Sec. IV.A).

    The high-finesse approximation relating the mirror reflectivity to the
    finesse. Used so the UI can expose finesse (the physically intuitive
    quantity) as its primary control.

    Parameters
    ----------
    r : array_like
        Mirror amplitude reflection coefficient, ``0 <= r < 1``.

    Returns
    -------
    numpy.ndarray or float
        The cavity finesse.
    """
    r = np.asarray(r, dtype=float)
    return np.pi / (1 - r**2)


def r_from_finesse(finesse):
    """Return the mirror amplitude reflectivity ``r`` for a given finesse.

    Inverse of :func:`finesse_from_r`: ``r = sqrt(1 - pi / finesse)``. Requires
    ``finesse > pi`` for a real, physical ``r``.

    Parameters
    ----------
    finesse : array_like
        Cavity finesse (must exceed ``pi``).

    Returns
    -------
    numpy.ndarray or float
        The mirror amplitude reflection coefficient.
    """
    finesse = np.asarray(finesse, dtype=float)
    return np.sqrt(1 - np.pi / finesse)


def sideband_powers(beta, P0=1.0):
    """Return the carrier and first-order-sideband powers ``(Pc, Ps)`` (Black Sec. III.C).

    ``Pc = J0(beta)**2 * P0`` and ``Ps = J1(beta)**2 * P0``, where ``beta`` is
    the modulation depth. For small ``beta`` almost all power is in the carrier
    and first-order sidebands (``Pc + 2*Ps -> P0``).

    Parameters
    ----------
    beta : float
        Modulation depth in radians.
    P0 : float, optional
        Total incident power. Defaults to ``1.0``.

    Returns
    -------
    tuple of float
        ``(Pc, Ps)``, the carrier power and the power in each first-order
        sideband.
    """
    Pc = jv(0, beta)**2 * P0
    Ps = jv(1, beta)**2 * P0
    return Pc, Ps


def pdh_error_signal(x, x_mod, r, beta, P0=1.0):
    """Return the PDH error-signal components ``(cos_component, sin_component)``.

    Implements Black's exact result, Eq. (3.3) (no slow/fast approximation
    baked in). The carrier at ``x`` and sidebands at ``x +- x_mod`` reflect off
    the cavity; the modulation-frequency beat term is::

        C = F0 * conj(Fp) - conj(F0) * Fm

    with ``F0 = F(x)``, ``Fp = F(x + x_mod)``, ``Fm = F(x - x_mod)``. The
    demodulated components are ``2*sqrt(Pc*Ps)`` times ``Re(C)`` and ``Im(C)``.

    The cosine component dominates in the slow-modulation regime
    (``x_mod`` much less than the cavity linewidth); the sine component is the
    usual PDH error signal in the fast-modulation, near-resonance regime.

    Parameters
    ----------
    x : array_like
        Laser frequency in units of the cavity free spectral range.
    x_mod : float
        Modulation frequency as a fraction of the free spectral range.
    r : float
        Mirror amplitude reflection coefficient.
    beta : float
        Modulation depth in radians.
    P0 : float, optional
        Total incident power. Defaults to ``1.0``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(cos_component, sin_component)``, each the same shape as ``x``. Both
        are exactly zero at integer ``x`` (cavity resonance) and exactly
        antisymmetric about each resonance.
    """
    F0 = cavity_reflection(x, r)
    Fp = cavity_reflection(np.asarray(x, dtype=float) + x_mod, r)
    Fm = cavity_reflection(np.asarray(x, dtype=float) - x_mod, r)
    C = F0 * np.conj(Fp) - np.conj(F0) * Fm

    Pc, Ps = sideband_powers(beta, P0)
    scale = 2 * np.sqrt(Pc * Ps)
    return scale * C.real, scale * C.imag


def pdh_error_signal_mixed(x, x_mod, r, beta, theta, P0=1.0):
    """Return the PDH error signal observed with demodulation phase ``theta``.

    Wraps :func:`pdh_error_signal` with :func:`core.demod.mixed_signal` (the
    same phase adjuster used on the FMS page). ``theta = 0`` recovers the
    cosine (in-phase) component; ``theta = pi/2`` recovers the sine
    (quadrature) component, the standard PDH error signal.

    Parameters
    ----------
    x : array_like
        Laser frequency in units of the cavity free spectral range.
    x_mod : float
        Modulation frequency as a fraction of the free spectral range.
    r : float
        Mirror amplitude reflection coefficient.
    beta : float
        Modulation depth in radians.
    theta : array_like
        Demodulation phase in radians.
    P0 : float, optional
        Total incident power. Defaults to ``1.0``.

    Returns
    -------
    numpy.ndarray or float
        The phase-mixed error signal, same shape as ``x``.
    """
    cos_component, sin_component = pdh_error_signal(x, x_mod, r, beta, P0)
    return mixed_signal(cos_component, sin_component, theta)


def pdh_frequency_discriminant(r, delta_nu_fsr, Pc, Ps):
    """Return the near-resonance frequency discriminant ``D`` (Black Eq. 4.2).

    ``D = -8 * sqrt(Pc * Ps) / delta_nu``, where the cavity linewidth is
    ``delta_nu = delta_nu_fsr / finesse_from_r(r)``. ``D`` is the slope of the
    (fast-modulation) error signal with respect to laser-frequency deviation
    from resonance: ``epsilon = D * delta_f``. It is the PDH analogue of the
    FMS discriminant slope.

    Parameters
    ----------
    r : float
        Mirror amplitude reflection coefficient.
    delta_nu_fsr : float
        Cavity free spectral range (in the frequency units of the desired ``D``).
    Pc, Ps : float
        Carrier and first-order-sideband powers (see :func:`sideband_powers`).

    Returns
    -------
    float
        The frequency discriminant ``D`` (negative).
    """
    delta_nu = delta_nu_fsr / finesse_from_r(r)
    return -8 * np.sqrt(Pc * Ps) / delta_nu


def pdh_optimum_modulation_depth():
    """Return the modulation depth that maximizes the discriminant (Black App. B).

    Black's Appendix B result: ``beta ~= 1.08`` maximizes ``sqrt(Pc*Ps) =
    |J0(beta) * J1(beta)|`` (and hence the discriminant ``D``) at fixed total
    power, giving a sideband-to-carrier power ratio ``Ps/Pc ~= 0.42``.

    Returns
    -------
    float
        The optimum modulation depth, ``1.08``.
    """
    return 1.08
