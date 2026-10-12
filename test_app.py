"""Tests for the figure-drawing layer (``app.py``).

These cover the one property the two panels depend on and that no physics test
can catch: Fig. 1 and Fig. 2 must stay on a single shared vertical scale, so
the dashed guide line drawn at a peak in Fig. 1 lands exactly on that order's
marker in Fig. 2. The linear/dB toggle is the thing most able to break it, so
it is exercised on both settings.

The spectral-truncation message is checked here too, since its advice must
never point at a setting the sidebar cannot reach.

The figures are built without pyplot and rendered off-screen, so the suite
needs no display or browser.
"""

import io

import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from PIL import Image

from app import (NMAX, PRESETS, YLIM, YLIM_DB, build_figure, figure_png,
                 theme_ink, truncation_message)
from physics import sideband_intensities

# Every preset operating point, on both y-scales.
CASES = [(beta, N, db) for beta, N in PRESETS.values() for db in (False, True)]


def _figure(beta, N, db_scale, mode='light'):
    """Build the two-panel figure for one operating point, y-scale and theme."""
    return build_figure(beta=beta, f0=1000.0, N=N,
                        Jn2=sideband_intensities(beta, N), width_frac=0.012,
                        noise=0.004, color_peaks=True, fixed_xaxis=False,
                        db_scale=db_scale, mode=mode)


def _luminance(color):
    """Relative luminance (0 black .. 1 white) of a matplotlib colour."""
    r, g, b, _ = to_rgba(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _guide_line_heights(ax):
    """Return the y-positions of the dashed horizontal guide lines on ``ax``."""
    return sorted(line.get_ydata()[0] for line in ax.get_lines()
                  if line.get_linestyle() == '--' and len(line.get_xdata()) == 2)


def _marker_heights(ax):
    """Return the y-positions of the operating-point markers on ``ax``."""
    return sorted(coll.get_offsets()[0][1] for coll in ax.collections)


@pytest.mark.parametrize("beta,N,db_scale", CASES)
def test_both_panels_share_one_y_range(beta, N, db_scale):
    """The guide lines only read across the two panels if their ranges match."""
    fig = _figure(beta, N, db_scale)
    ax1, ax2 = fig.axes
    assert ax1.get_ylim() == ax2.get_ylim()
    assert ax1.get_ylim() == (YLIM_DB if db_scale else YLIM)


@pytest.mark.parametrize("beta,N,db_scale", CASES)
def test_guide_lines_land_on_the_bessel_markers(beta, N, db_scale):
    """Each peak's guide line in Fig. 1 meets its order's marker in Fig. 2."""
    fig = _figure(beta, N, db_scale)
    ax1, ax2 = fig.axes
    heights, markers = _guide_line_heights(ax1), _marker_heights(ax2)
    assert len(heights) == N + 1        # one per displayed order, carrier included
    assert len(markers) == N + 1
    assert heights == pytest.approx(markers)


@pytest.mark.parametrize("beta,N,db_scale", CASES)
def test_nothing_drawn_is_infinite(beta, N, db_scale):
    """Bessel nulls are exactly zero, so the dB path must never emit -inf."""
    fig = _figure(beta, N, db_scale)
    for ax in fig.axes:
        for line in ax.get_lines():
            assert np.all(np.isfinite(np.atleast_1d(line.get_ydata())))


@pytest.mark.parametrize("N", range(1, NMAX + 1))
def test_truncation_message_only_advises_while_the_slider_has_room(N):
    """The "display more orders" advice must disappear once the slider is maxed.

    Below the maximum it has to be present, since it is the whole reason the
    warning is worth reading.
    """
    message = truncation_message(0.75, N)
    assert message.startswith("**Spectral truncation:** 25% of the optical power")
    assert ("Increase the number of displayed sideband orders" in message) == (N < NMAX)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_figure_carries_its_own_opaque_background(mode):
    """The rendered figure is opaque and its ink contrasts with its own background.

    Streamlit can report the previous theme after a viewer switches it in the
    menu, so the figure must never rely on the page behind it: with its own
    background it stays readable even when it was drawn for the other theme.
    """
    fig = _figure(1.0, 2, False, mode)
    png = Image.open(io.BytesIO(figure_png(fig))).convert("RGBA")
    assert png.getextrema()[3] == (255, 255), "the figure has transparent pixels"

    ink = theme_ink(mode)
    assert png.getpixel((5, 5))[:3] == tuple(round(255 * c) for c in to_rgba(ink["bg"])[:3])
    for role in ("text", "grey"):
        assert abs(_luminance(ink[role]) - _luminance(ink["bg"])) > 0.35, role


def test_dark_figure_sets_every_colour_itself():
    """No text or frame on a dark figure falls back to matplotlib's black ink.

    Streamlit serves concurrent visitors from one process, so the figure must
    not take its theme colours from the global rcParams: every artist gets its
    colour passed in directly, which this checks on the dark theme.
    """
    fig = _figure(1.0, 2, False, 'dark')
    figure_png(fig)                     # draw, so every tick label exists
    bg = _luminance(theme_ink('dark')["bg"])
    texts = [t for t in fig.findobj(Text) if t.get_text()]
    spines = [s for ax in fig.axes for s in ax.spines.values()]
    assert texts
    for t in texts:
        assert _luminance(t.get_color()) - bg > 0.35, t.get_text()
    for spine in spines:
        assert _luminance(spine.get_edgecolor()) - bg > 0.35
