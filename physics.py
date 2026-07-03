import numpy as np
from scipy.special import jv

# ----------------------------------------------------------------------
#  Pure phase modulation:  E(t) = E0 exp(i[wc t + beta sin(wm t)])
#                               = E0 * sum_n J_n(beta) exp(i(wc + n wm)t)
#  -> spectral line at n*f0 has intensity J_n(beta)^2   (Jacobi-Anger)
# ----------------------------------------------------------------------

_ORDER_COLORS = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

def color_for_n(n):
    n = abs(n)
    return 'black' if n == 0 else _ORDER_COLORS[(n - 1) % len(_ORDER_COLORS)]

def sideband_intensities(beta, N):
    """|J_n(beta)|^2 for n = 0..N; J_{-n}(beta)^2 == J_n(beta)^2 covers the negative orders."""
    return jv(np.arange(N + 1), beta)**2

def captured_power(Jn2):
    return float(Jn2[0] + 2 * Jn2[1:].sum())
