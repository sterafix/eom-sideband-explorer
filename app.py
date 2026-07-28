"""Streamlit UI for the EOM Sideband Explorer.

This module is the presentation layer only: it reads the user's controls,
calls into :mod:`physics` for the numbers, and draws the two figures. All of
the physics lives in :mod:`physics` so it can be tested independently.

The two figures shown side by side are:

* **Fig. 1 - Optical spectrum:** each sideband order drawn at ``n * f0`` with
  height ``|J_n(beta)|^2``.
* **Fig. 2 - Bessel curves:** ``|J_n(beta)|^2`` versus modulation depth, with a
  marker at the chosen ``beta`` showing that every peak height in Fig. 1 is a
  slice through these curves.

Run locally with ``streamlit run app.py``.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from physics import (color_for_n, sideband_intensities, intensity_vs_beta,
                     captured_power, to_decibels, DB_FLOOR)

NMAX = 6        # max selectable sideband orders to display

# Share of the optical power the displayed orders must hold for the spectrum to
# count as essentially complete; below it the truncation notice takes over.
POWER_COMPLETE = 0.98

# Shared y-limits for both figures. The two panels show the same intensities
# (|J_n(beta)|^2), so their axes must span the same range for the dashed
# helper lines to align visually across panels. The headroom above 1.0 leaves
# room for the order labels in Fig. 1.
YLIM = (-0.06, 1.14)

# Shared y-limits for the dB view, analogous to YLIM above: the bottom sits a
# little below the to_decibels() clipping floor (DB_FLOOR) and the top leaves
# headroom for the order labels above the 0 dB carrier line.
YLIM_DB = (DB_FLOOR - 4.0, 5.0)

# Preset operating points, each mapping a label to (modulation depth beta,
# number of orders to display). The carrier-null preset sits at the first
# zero of J_0, a standard laboratory reference point.
PRESETS = {
    "Low β (0.5)":            (0.5,   2),
    "Moderate β (1.0)":       (1.0,   2),
    "Carrier null (2.405)":   (2.405, 3),
    "High β (3.0)":           (3.0,   4),
    "Deep modulation (5.0)":  (5.0,   6),
}


# ---------------------------------------------------------------------------
# Widget-synchronisation callbacks
#
# The modulation depth beta is edited by two widgets (a slider and a number
# input) that share one value, plus a set of preset "pills". Each callback
# copies its own widget's state to the counterparts before the next rerun
# redraws them, so the three controls always agree.
# ---------------------------------------------------------------------------
def _sync_preset_selection():
    """Light up the preset pill that matches the current beta, else clear it.

    The pills advertise specific operating points, so the highlight must track
    beta: it lights up the matching preset and clears once beta moves away.
    """
    beta = st.session_state.beta
    st.session_state.preset = next(
        (label for label, (b, _) in PRESETS.items()
         if abs(b - beta) < 1e-9), None)


def _beta_from_slider():
    """Push the slider's beta to the number input and refresh the preset pill."""
    st.session_state.beta_input = st.session_state.beta
    _sync_preset_selection()


def _beta_from_input():
    """Push the number input's beta to the slider and refresh the preset pill."""
    st.session_state.beta = st.session_state.beta_input
    _sync_preset_selection()


def _apply_preset():
    """Apply the selected preset's beta and displayed-order count to the widgets."""
    if st.session_state.preset is None:
        return          # pill was deselected; keep the current values
    b, n = PRESETS[st.session_state.preset]
    st.session_state.beta = b
    st.session_state.beta_input = b
    st.session_state.N = n


# ---------------------------------------------------------------------------
# Theming and plotting
#
# Both figures are rendered on a transparent background, so only the ink
# colours depend on the active Streamlit theme (light or dark).
# ---------------------------------------------------------------------------
def theme_ink(is_dark):
    """Return the theme-dependent ink colours for the matplotlib figure.

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
    """Apply shared matplotlib rcParams so both axes match the Streamlit theme."""
    plt.rcParams.update({
        "axes.edgecolor": ink["grey"], "axes.labelcolor": ink["grey"],
        "xtick.color": ink["grey"], "ytick.color": ink["grey"],
        "text.color": ink["text"],
        "axes.linewidth": 1.0, "font.size": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    })


# ---------------------------------------------------------------------------
# Y-scale plumbing
#
# The two figures line up only because they plot the same intensities against
# the same vertical range: that is what lets a dashed guide line join a peak in
# Fig. 1 to its marker in Fig. 2. To keep that true under the linear/dB toggle,
# every plotted intensity goes through _scale_y and every axis takes its limits
# from _ylim, so neither figure can pick up a scale the other one did not.
# ---------------------------------------------------------------------------
def _scale_y(values, db_scale):
    """Map linear intensities onto the active y-scale.

    Accepts a scalar or an array and returns the same shape, so peak heights,
    marker positions, and whole traces all convert through one call.
    """
    return to_decibels(values) if db_scale else values


def _ylim(db_scale):
    """Return the y-limits shared by both figures for the active y-scale."""
    return YLIM_DB if db_scale else YLIM


def draw_spectrum(ax, *, beta, f0, N, Jn2, width_frac, noise, color_peaks, db_scale, mode, ink):
    """Draw Fig. 1: the optical power spectrum.

    Each order ``n`` is placed at detuning ``n * f0`` with height
    ``|J_n(beta)|^2``. The mathematically sharp lines are rendered as narrow
    Gaussian peaks on a small synthetic noise floor so they are visible.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    beta : float
        Modulation depth in radians (shown in the annotation box).
    f0 : float
        Modulation frequency in MHz (sets the line spacing).
    N : int
        Highest sideband order to display.
    Jn2 : numpy.ndarray
        Per-line intensities for orders ``0..N``.
    width_frac : float
        Gaussian peak width as a fraction of the plotted frequency span.
    noise : float
        Amplitude of the synthetic noise floor.
    color_peaks : bool
        Whether to overpaint each peak in its order's colour.
    db_scale : bool
        If ``True``, plot the trace and peak heights in decibels
        (``10*log10``) instead of linear intensity.
    mode : {"light", "dark"}
        Theme variant used to pick peak colours.
    ink : dict
        Theme ink colours from :func:`theme_ink`.
    """
    span = (N + 0.8) * f0
    freq = np.linspace(-span, span, 6000)
    width = width_frac * span
    trace = np.zeros_like(freq)
    for n in range(-N, N + 1):
        trace += Jn2[abs(n)] * np.exp(-0.5 * ((freq - n * f0) / width)**2)
    trace += noise * np.abs(np.random.default_rng(0).normal(size=freq.shape))
    trace = _scale_y(trace, db_scale)

    ax.plot(freq, trace, color=color_for_n(0, mode), lw=0.9, zorder=1)
    for n in range(0, N + 1):
        h = _scale_y(Jn2[n], db_scale)
        centers = [0.0] if n == 0 else [-n * f0, n * f0]
        for c0 in centers:
            # Only the color highlight is skipped for vanishingly small orders
            # (there is no visible peak in the noise floor to paint); the order
            # label is shown regardless, so it always matches Fig. 2's curves
            # and markers, which are drawn for every order irrespective of h.
            # The test is on the linear intensity, not on h: under the dB scale
            # h is negative for every order, so comparing it here would switch
            # the highlight off entirely.
            if color_peaks and Jn2[n] > 1e-3:
                sel = np.abs(freq - c0) < 3 * width
                ax.plot(freq[sel], trace[sel], color=color_for_n(n, mode), lw=1.5, zorder=2)
            order_lbl = "0" if n == 0 else (f"+{n}" if c0 > 0 else f"-{n}")
            ax.annotate(order_lbl, (c0, h), textcoords="offset points",
                        xytext=(0, 8), ha='center', fontsize=8,
                        fontweight='bold', color=color_for_n(n, mode), zorder=4)
        ax.axhline(h, ls='--', color=ink["hline"], lw=0.8, zorder=0)

    ax.set_xlim(-span, span)
    ax.set_ylim(*_ylim(db_scale))
    ax.set_xlabel('Optical frequency detuning [MHz]', fontweight='bold')
    ax.set_ylabel('Intensity [dB]' if db_scale else 'Intensity [arb. units]', fontweight='bold')
    ax.set_title('Fig. 1: Optical spectrum', fontweight='bold', color=ink["text"], pad=12)
    ax.text(0.04, 0.95,
            f"$f_0$ = {f0:.0f} MHz\n"
            f"$\\beta$ = {beta:.3f} rad",
            transform=ax.transAxes, va='top', ha='left', fontsize=10, color=ink["grey"],
            bbox=dict(boxstyle='round', fc=ink["box_fc"], ec=ink["box_ec"]))


def draw_bessel_curves(ax, *, beta, N, Jn2, fixed_xaxis, db_scale, mode, ink):
    """Draw Fig. 2: ``|J_n(beta)|^2`` versus modulation depth.

    A marker at the chosen ``beta`` sits on each curve, showing that the peak
    heights in Fig. 1 are slices through these Bessel curves at that ``beta``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    beta : float
        Modulation depth in radians (marker position).
    N : int
        Highest sideband order to plot.
    Jn2 : numpy.ndarray
        Per-line intensities for orders ``0..N`` (marker heights).
    fixed_xaxis : bool
        If ``True``, pin the beta-axis to 0-10; otherwise scale it to beta.
    db_scale : bool
        If ``True``, plot the curves and markers in decibels (``10*log10``)
        instead of linear intensity.
    mode : {"light", "dark"}
        Theme variant used to pick curve colours.
    ink : dict
        Theme ink colours from :func:`theme_ink`.
    """
    bmax = 10.0 if fixed_xaxis else max(3.0, beta * 1.1)
    bgrid = np.linspace(0, bmax, 800)
    for n in range(0, N + 1):
        curve = _scale_y(intensity_vs_beta(n, bgrid), db_scale)
        h = _scale_y(Jn2[n], db_scale)
        ax.plot(bgrid, curve, color=color_for_n(n, mode), lw=2.2,
                label=f"$|J_{{{n}}}|^2$")
        ax.plot([0, beta], [h, h], ls='--', color=ink["hline"], lw=0.8, zorder=0)
        ax.scatter(beta, h, color=color_for_n(n, mode), s=20, zorder=3)   # operating-point marker
    ax.axvline(beta, ls='--', color=ink["hline"], lw=0.8, zorder=0)

    ax.set_xlim(0, bmax)
    ax.set_ylim(*_ylim(db_scale))
    ax.set_xlabel('Modulation depth  $\\beta$ [rad]', fontweight='bold')
    ax.set_title('Fig. 2: Bessel curves $|J_n(\\beta)|^2$', fontweight='bold',
                 color=ink["text"], pad=12)
    # move the y-axis to the right, like the reference figure
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.set_ylabel('Relative intensity [dB]' if db_scale else 'Relative intensity',
                  fontweight='bold')
    ax.legend(title='Bessel functions', loc='upper right', frameon=False)


def build_figure(*, beta, f0, N, Jn2, width_frac, noise, color_peaks,
                 fixed_xaxis, db_scale, mode, ink):
    """Build and return the two-panel matplotlib figure (spectrum + Bessel curves)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150)
    fig.subplots_adjust(wspace=0.18, top=0.86, bottom=0.13, left=0.07, right=0.93)
    fig.patch.set_alpha(0.0)
    for ax in (ax1, ax2):
        ax.set_facecolor('none')

    draw_spectrum(ax1, beta=beta, f0=f0, N=N, Jn2=Jn2, width_frac=width_frac,
                  noise=noise, color_peaks=color_peaks, db_scale=db_scale,
                  mode=mode, ink=ink)
    draw_bessel_curves(ax2, beta=beta, N=N, Jn2=Jn2, fixed_xaxis=fixed_xaxis,
                       db_scale=db_scale, mode=mode, ink=ink)
    return fig


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------
def render_sidebar():
    """Draw the sidebar controls and return the current parameter values.

    Returns
    -------
    tuple
        ``(beta, f0, N, width_frac, noise, color_peaks, fixed_xaxis, db_scale)``.
    """
    st.sidebar.title("EOM parameters")
    beta = st.sidebar.slider("Modulation depth  β [rad]", 0.0, 10.0, step=0.005,
                             key="beta", format="%.3f", on_change=_beta_from_slider)
    st.sidebar.number_input("Exact β [rad]", 0.0, 10.0, step=0.005, key="beta_input",
                            format="%.3f", on_change=_beta_from_input,
                            help="Type a precise modulation depth; the slider follows.")
    f0 = st.sidebar.slider("Modulation frequency  f₀ [MHz]", 50.0, 3000.0, 1000.0, 5.0,
                           format="%.0f")
    N = st.sidebar.slider(
        "Displayed sideband orders", 1, NMAX, step=1, key="N",
        help="How many sideband orders are drawn on each side of the carrier. "
             "All orders exist physically; this only sets how many are shown.")

    st.sidebar.pills("Presets", list(PRESETS), key="preset", on_change=_apply_preset,
                     help="Common operating points. The highlight follows β: it "
                          "clears when the sliders leave the preset value.")

    st.sidebar.markdown("---")
    scale_choice = st.sidebar.radio(
        "Y-axis scale", ["Linear", "dB"], horizontal=True, key="scale_choice",
        help="Linear shows intensity on 0-1; dB (10·log₁₀, referenced to the "
             "unmodulated carrier) compresses the huge range between the carrier "
             "and the higher-order sidebands, the way a spectrum analyzer "
             "displays a spectrum. Applies to both figures, so the dashed guide "
             "lines stay aligned.")
    db_scale = scale_choice == "dB"

    with st.sidebar.expander("Display options"):
        width_frac = st.slider("Peak width", 0.003, 0.040, 0.012, 0.001, format="%.3f")
        noise = st.slider("Noise level", 0.0, 0.030, 0.004, 0.001, format="%.3f")
        color_peaks = st.checkbox("Color-code peaks", True)
        fixed_xaxis = st.checkbox("Fix β-axis to 0–10", False)

    return beta, f0, N, width_frac, noise, color_peaks, fixed_xaxis, db_scale


def truncation_message(cap, N):
    """Return the spectral-truncation warning for a captured-power fraction.

    The advice to display more orders is appended only while the slider has
    room left: at ``N == NMAX`` there is nothing to increase, so the message
    states the shortfall and stops rather than pointing at a setting that does
    not exist.

    Parameters
    ----------
    cap : float
        Fraction of the optical power held by the displayed orders, from
        :func:`physics.captured_power`.
    N : int
        Highest displayed sideband order, i.e. the current slider position.

    Returns
    -------
    str
        The warning body, as Markdown.
    """
    message = (f"**Spectral truncation:** {(1 - cap) * 100:.0f}% of the optical "
               f"power falls outside the displayed orders.")
    if N < NMAX:
        message += (" Increase the number of displayed sideband orders to view "
                    "the entire relevant spectrum.")
    return message


def render_metrics(Jn2, N):
    """Show the captured-power metric and how complete the displayed spectrum is."""
    with st.container(border=True):
        col_left, col_right = st.columns([1, 2])
        with col_left:
            cap = captured_power(Jn2)
            st.metric("Captured optical power", f"{cap*100:.1f} %",
                      help="Fraction of the total optical power contained in the "
                           "displayed sideband orders.")
        with col_right:
            if cap < POWER_COMPLETE:
                st.warning(truncation_message(cap, N), icon=":material/warning:")
            else:
                st.success("**Spectrum essentially complete:** the displayed sideband "
                           "orders account for the full optical power.",
                           icon=":material/check_circle:")


def render_intensity_table(Jn2, f0, N):
    """Show the exact per-line intensities and combined power shares as a table."""
    with st.container(border=True):
        st.subheader("Sideband intensities")
        orders = list(range(0, N + 1))
        table = {
            "Order": ["0 (carrier)" if n == 0 else f"±{n}" for n in orders],
            "Detuning [MHz]": ["0" if n == 0 else f"±{n * f0:.0f}" for n in orders],
            "Jₙ(β)² per line": [f"{Jn2[n]:.4f}" for n in orders],
            "Power share (±n)": [
                f"{Jn2[n] * (1 if n == 0 else 2) * 100:.2f} %" for n in orders],
        }
        col_t, _ = st.columns([3, 2])
        with col_t:
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("Intensity is given per spectral line; the power share combines the "
                       "+n and -n orders, so the shares sum to the captured optical power "
                       "shown above.")


def render_physics_details():
    """Render the expandable 'Physics & model details' write-up."""
    with st.expander("Physics & model details"):
        st.markdown(
            "This tool models pure sinusoidal **phase modulation** of a "
            "monochromatic laser, the regime of an ideal electro-optic phase "
            "modulator. The modulator imprints a time-varying phase "
            "on the optical carrier:")
        st.latex(r"E(t)=E_0\,e^{\,i[\omega_c t+\beta\sin(\omega_m t)]}")
        st.markdown(
            "Expanding with the **Jacobi-Anger identity** decomposes this into the "
            "carrier plus an infinite, discrete set of sidebands spaced by the "
            "modulation frequency $\\omega_m$:")
        st.latex(r"E(t)=E_0\sum_{n=-\infty}^{\infty} J_n(\beta)\,e^{\,i(\omega_c+n\omega_m)t},")
        st.markdown(
            r"where $J_n(\beta)$ is the Bessel function (of the first kind). The line at detuning $n f_0$ has amplitude $J_n(\beta)$ and therefore "
            r"intensity $|J_n(\beta)|^2$. Two consequences shape the figures:")
        st.markdown(
            r"- **Symmetry.** Since $J_{-n}(\beta)=(-1)^n J_n(\beta)$, the power "
            r"spectrum is symmetric about the carrier: the $\pm n$ orders have equal "
            r"intensity." "\n"
            r"- **Energy conservation.** $\sum_{n} |J_n(\beta)|^2 = 1$, so phase "
            r"modulation only redistributes power between carrier and sidebands. The "
            r"*Captured optical power* readout is the partial sum of $|J_n(\beta)|^2$ "
            r"over the displayed orders.")
        st.markdown(
            r"The carrier first vanishes at $\beta\approx 2.4048$, the first zero of "
            r"$J_0$. This carrier-null point is a standard laboratory reference for "
            r"measuring $V_\pi$: at the null, $\beta = 2.4048$ is known exactly, so "
            r"the half-wave voltage follows directly from the RF drive level,")
        st.latex(r"V_\pi = \frac{\pi\,V_\text{peak}}{2.4048}.")
        st.markdown(
            r"This in turn allows the modulation depth to be calculated approximately for any "
            r"drive voltage, provided the temperature, wavelength, and drive frequency remain close to the calibration conditions and we "
            r"ignore effects like piezoelectric resonances.")
        st.markdown("**What the figures show**")
        st.markdown(
            r"- **Fig. 1** plots each order at $n f_0$ with height $|J_n(\beta)|^2$, "
            r"annotated with its order number. The ideal lines are mathematically "
            r"sharp; for visibility they are drawn as narrow Gaussian peaks on a small "
            r"synthetic noise floor (both adjustable under Display options). The dashed "
            r"guide line marks the exact $|J_n(\beta)|^2$ value, so the peak can sit a "
            r"touch above it once noise is added." "\n"
            r"- **Fig. 2** plots $|J_n(\beta)|^2$ versus modulation depth. The marker "
            r"at your chosen $\beta$ shows that every peak height in Fig. 1 is a slice "
            r"through these Bessel curves at that $\beta$." "\n"
            r"- **Y-axis scale.** The *dB* setting plots $10\log_{10}|J_n(\beta)|^2$ on "
            r"both figures, referenced to unit power ($0$ dB, the whole beam in the "
            r"carrier before modulation). Sideband intensity falls off steeply with "
            r"order, so on a linear axis the higher orders sit indistinguishably on "
            r"zero; in dB each is placed at its true height, which is what makes the "
            r"weaker orders and the Fig. 2 curves readable. In Fig. 1 an order that "
            r"falls below the synthetic noise floor stays buried in it, just as it "
            r"would be on a real analyzer. Values are floored at "
            rf"${DB_FLOOR:.0f}$ dB so the Bessel nulls, where the intensity is exactly "
            r"zero, stay on scale.")
        st.markdown(
            r"You set $\beta$ (the peak phase deviation), the line spacing $f_0$, and "
            r"how many orders to display. On real hardware $\beta$ follows from the RF "
            r"drive via $\beta=\pi V_\text{peak}/V_\pi$, where $V_\pi$ is the device- "
            r"and wavelength-dependent half-wave voltage (not used in this idealized "
            r"simulation).")
        st.markdown("**Beyond the ideal case: residual amplitude modulation**")
        st.markdown(
            r"Real modulators never achieve *pure* phase modulation. Small imperfections "
            r"(e.g., facet reflections and polarization misalignment) add a little "
            r"**residual amplitude modulation (RAM)** on top of the phase modulation, "
            r"breaking the strict carrier-sideband quadrature assumed above. In precision "
            r"applications such as Pound-Drever-Hall laser stabilization, RAM shows up as a "
            r"small, temperature-dependent offset in the error signal that can drift the "
            r"lock point over time. This is not captured by the idealized model here."
        )
        st.markdown(
            "**References and further reading**\n"
            "- [Jacobi-Anger expansion](https://en.wikipedia.org/wiki/Jacobi%E2%80%93Anger_expansion)\n"
            "- [Bessel function](https://en.wikipedia.org/wiki/Bessel_function)\n"
            "- [Electro-optic modulator](https://en.wikipedia.org/wiki/Electro-optic_modulator)\n"
            "- [Pockels effect](https://en.wikipedia.org/wiki/Pockels_effect)\n"
            "- [Phase modulation](https://en.wikipedia.org/wiki/Phase_modulation)\n"
            "- [Pound-Drever-Hall technique](https://en.wikipedia.org/wiki/Pound%E2%80%93Drever%E2%80%93Hall_technique)\n"
            "- B. E. A. Saleh and M. C. Teich, *Fundamentals of Photonics* (electro-optics chapter)")
        st.caption(
            "Idealizations: pure phase modulation (no residual amplitude modulation), "
            "a single RF tone, and no loss.")


def init_session_state():
    """Seed the session-state defaults the widgets rely on before they are created."""
    if "beta" not in st.session_state:
        st.session_state.beta = 1.0
    if "beta_input" not in st.session_state:
        st.session_state.beta_input = st.session_state.beta
    if "N" not in st.session_state:
        st.session_state.N = 2


def main():
    """Configure the page, read the controls, and render the whole app."""
    st.set_page_config(page_title="EOM Sideband Explorer",
                       page_icon="〰️", layout="wide")

    # Resolve the active Streamlit theme so the matplotlib figure matches it.
    mode = "dark" if st.context.theme.type == "dark" else "light"
    ink = theme_ink(mode == "dark")

    init_session_state()
    beta, f0, N, width_frac, noise, color_peaks, fixed_xaxis, db_scale = render_sidebar()

    # |J_n(beta)|^2 for n = 0..N, shared by both figures and the table.
    Jn2 = sideband_intensities(beta, N)

    apply_plot_style(ink)
    fig = build_figure(beta=beta, f0=f0, N=N, Jn2=Jn2, width_frac=width_frac,
                       noise=noise, color_peaks=color_peaks,
                       fixed_xaxis=fixed_xaxis, db_scale=db_scale, mode=mode, ink=ink)

    st.header("EOM Sideband Explorer")
    st.caption("Phase-modulation sideband spectrum and its Bessel-function "
               "decomposition · line intensity $= J_n(\\beta)^2$ (Jacobi-Anger expansion)")

    render_metrics(Jn2, N)

    with st.container(border=True):
        st.pyplot(fig, use_container_width=True, bbox_inches=None)
    plt.close(fig)      # release the figure so reruns do not accumulate in memory

    render_intensity_table(Jn2, f0, N)
    render_physics_details()

    st.caption("Source code on GitHub: "
               "https://github.com/sterafix/eom-sideband-explorer")


if __name__ == "__main__":
    main()
