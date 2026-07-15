"""Shared color scheme and matplotlib theming for the EOM Sideband Explorer.

All pages (Sidebands, FMS, PDH) render matplotlib figures on a transparent
background and pick up the active Streamlit light/dark theme. This module is
the single place that defines those colors and the rcParams tweaks needed to
match them, so every page reads as one product instead of styling itself
independently.
"""

import streamlit as st

# Colorblind-safe palette for sideband orders 1, 2, 3, ... (index n-1).
# Adjacent orders stay distinguishable under common colour-vision deficiencies;
# the "dark" variants are the same hues re-stepped for a dark chart surface.
_ORDER_COLORS = {
    "light": ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948'],
    "dark":  ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767'],
}

# The carrier (order 0) is drawn in the theme's foreground colour instead.
_CARRIER_COLOR = {"light": "black", "dark": "#fafafa"}


def color_for_n(n, mode="light"):
    """Return the plotting colour for sideband order ``n``.

    The carrier (``n == 0``) uses the theme foreground colour; every other
    order is assigned a colour from a colourblind-safe palette. Because the
    +n and -n orders are physically identical, the sign of ``n`` is ignored.
    The palette cycles if ``|n|`` exceeds its length.

    Parameters
    ----------
    n : int
        Sideband order. Positive, negative, and zero are all accepted;
        ``color_for_n(n) == color_for_n(-n)``.
    mode : {"light", "dark"}, optional
        Chart surface for which to pick the colour variant. Defaults to
        ``"light"``.

    Returns
    -------
    str
        A matplotlib-compatible colour (hex string or named colour).
    """
    n = abs(n)
    if n == 0:
        return _CARRIER_COLOR[mode]
    palette = _ORDER_COLORS[mode]
    return palette[(n - 1) % len(palette)]


def resolve_mode():
    """Return ``"dark"`` or ``"light"`` for the currently active Streamlit theme."""
    return "dark" if st.context.theme.type == "dark" else "light"


def theme_ink(is_dark):
    """Return the theme-dependent ink colours for a matplotlib figure.

    Parameters
    ----------
    is_dark : bool
        Whether the active Streamlit theme is dark.

    Returns
    -------
    dict
        Colours keyed by role: ``grey`` (axes/labels), ``text`` (titles,
        matches the Streamlit ``textColor``), ``hline`` (dashed helper lines),
        and ``box_fc`` / ``box_ec`` (annotation-box face and edge).
    """
    return {
        "grey":   '0.65' if is_dark else '0.45',
        "text":   '#fafafa' if is_dark else '#262730',
        "hline":  '0.5' if is_dark else '0.7',
        "box_fc": '#262730' if is_dark else 'white',
        "box_ec": '0.4' if is_dark else '0.8',
    }


def apply_plot_style(ink):
    """Apply shared matplotlib rcParams so a figure's axes match the Streamlit theme."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": ink["grey"], "axes.labelcolor": ink["grey"],
        "xtick.color": ink["grey"], "ytick.color": ink["grey"],
        "text.color": ink["text"],
        "axes.linewidth": 1.0, "font.size": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    })
