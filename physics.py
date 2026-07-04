import numpy as np
from scipy.special import jv

# ----------------------------------------------------------------------
#  Pure phase modulation:  E(t) = E0 exp(i[wc t + beta sin(wm t)])
#                               = E0 * sum_n J_n(beta) exp(i(wc + n wm)t)
#  -> spectral line at n*f0 has intensity J_n(beta)^2   (Jacobi-Anger)
# ----------------------------------------------------------------------

# Colorblind-safe palette (adjacent orders stay distinguishable under CVD);
# the dark variants are the same hues re-stepped for a dark chart surface.
_ORDER_COLORS = {
    "light": ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948'],
    "dark":  ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767'],
}
_CARRIER_COLOR = {"light": "black", "dark": "#fafafa"}

def color_for_n(n, mode="light"):
    n = abs(n)
    if n == 0:
        return _CARRIER_COLOR[mode]
    palette = _ORDER_COLORS[mode]
    return palette[(n - 1) % len(palette)]

def sideband_intensities(beta, N):
    """|J_n(beta)|^2 for n = 0..N; J_{-n}(beta)^2 == J_n(beta)^2 covers the negative orders."""
    return jv(np.arange(N + 1), beta)**2

def captured_power(Jn2):
    return float(Jn2[0] + 2 * Jn2[1:].sum())
