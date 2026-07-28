"""Tests for the figure-drawing layer (``app.py``).

These cover the one property the two panels depend on and that no physics test
can catch: Fig. 1 and Fig. 2 must stay on a single shared vertical scale, so
the dashed guide line drawn at a peak in Fig. 1 lands exactly on that order's
marker in Fig. 2. The linear/dB toggle is the thing most able to break it, so
it is exercised on both settings.

Rendering uses the non-interactive Agg backend, so the suite still needs no
display or browser.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pytest                            # noqa: E402

from app import (PRESETS, YLIM, YLIM_DB, apply_plot_style,  # noqa: E402
                 build_figure, theme_ink)
from physics import sideband_intensities  # noqa: E402

# Every preset operating point, on both y-scales.
CASES = [(beta, N, db) for beta, N in PRESETS.values() for db in (False, True)]


def _figure(beta, N, db_scale):
    """Build the two-panel figure for one operating point and y-scale."""
    ink = theme_ink(False)
    apply_plot_style(ink)
    return build_figure(beta=beta, f0=1000.0, N=N,
                        Jn2=sideband_intensities(beta, N), width_frac=0.012,
                        noise=0.004, color_peaks=True, fixed_xaxis=False,
                        db_scale=db_scale, mode='light', ink=ink)


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
    plt.close(fig)


@pytest.mark.parametrize("beta,N,db_scale", CASES)
def test_guide_lines_land_on_the_bessel_markers(beta, N, db_scale):
    """Each peak's guide line in Fig. 1 meets its order's marker in Fig. 2."""
    fig = _figure(beta, N, db_scale)
    ax1, ax2 = fig.axes
    heights, markers = _guide_line_heights(ax1), _marker_heights(ax2)
    assert len(heights) == N + 1        # one per displayed order, carrier included
    assert len(markers) == N + 1
    assert heights == pytest.approx(markers)
    plt.close(fig)


@pytest.mark.parametrize("beta,N,db_scale", CASES)
def test_nothing_drawn_is_infinite(beta, N, db_scale):
    """Bessel nulls are exactly zero, so the dB path must never emit -inf."""
    fig = _figure(beta, N, db_scale)
    for ax in fig.axes:
        for line in ax.get_lines():
            assert np.all(np.isfinite(np.atleast_1d(line.get_ydata())))
    plt.close(fig)
