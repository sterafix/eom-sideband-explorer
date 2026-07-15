"""Streamlit page: FM spectroscopy (FMS) error signal.

This module is the presentation layer only: it reads the user's controls,
calls into :mod:`core.fms` for the numbers, and draws the figures. All of
the physics lives in :mod:`core.fms`, :mod:`core.lineshapes`, and
:mod:`core.demod` so it can be tested independently (see
``tests/test_fms.py``, ``tests/test_fms_fig3.py``, ``tests/test_fms_fig4.py``).

Reproduces the demodulated heterodyne beat signals S1 (in-phase), S2
(quadrature), and S3 (magnitude) of Bjorklund, Levenson, Lenth & Ortiz,
"Frequency Modulation (FM) Spectroscopy," Appl. Phys. B 32, 145 (1983),
Figs. 3-4: a single isolated Lorentzian spectral feature probed by a small
modulation index (M << 1) FM optical spectrum, as the normalized carrier
detuning R0 is scanned for a fixed normalized sideband spacing delta_R.

The three figures shown are:

* **Main plot** - S1, S2, S3 vs. R0, stacked, redrawn live at the current
  delta_R: the interactive equivalent of one row of Bjorklund's Fig. 3/4.
* **Discriminant slope plot** - the analytic slope of S1 and S2 at line
  center vs. delta_R, with the current delta_R and the S1-optimum marked.
* **Demodulation phase plot** - the signal an actual phase-sensitive
  detector would observe at the chosen demodulation phase theta, morphing
  continuously between S1 (theta=0) and S2 (theta=90 deg).
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from core.fms import fms_discriminant_slope, fms_signal_mixed, fms_signals
from core.theming import apply_plot_style, color_for_n, resolve_mode, theme_ink

# Bounds for the normalized sideband spacing delta_R, matching the range
# spanned by Bjorklund's Fig. 3 (0.05-4.0) and Fig. 4 (0.1-50.0) combined.
DELTA_R_MIN, DELTA_R_MAX = 0.05, 50.0

# Preset operating points, each mapping a label to a delta_R value. "Optimal
# S1 slope" sits at the analytic maximum of the S1 discriminant slope (see
# core.fms.fms_discriminant_slope); it is also this page's default, since it
# is the most commonly useful single operating point.
OPTIMAL_DELTA_R = 1 / np.sqrt(3)
PRESETS = {
    "WMS limit (ΔR=0.1)":        0.1,
    "Optimal S1 slope (ΔR=1/√3)": OPTIMAL_DELTA_R,
    "Fully resolved (ΔR=20)":     20.0,
}


# ---------------------------------------------------------------------------
# Widget-synchronisation callbacks
#
# delta_R is edited by two widgets that share one value: a log-scaled slider
# (the useful range spans nearly three decades) and a number input for the
# exact value, plus a set of preset pills. This mirrors the beta/beta_input
# pattern on the Sidebands page (views/sidebands.py).
# ---------------------------------------------------------------------------
def _sync_preset_selection():
    """Light up the preset pill that matches the current delta_R, else clear it."""
    delta_R = st.session_state.delta_R_input
    st.session_state.fms_preset = next(
        (label for label, dr in PRESETS.items() if abs(dr - delta_R) < 1e-9), None)


def _delta_R_from_log_slider():
    """Push the log-slider's delta_R to the number input and refresh the preset pill."""
    delta_R = 10 ** st.session_state.log_delta_R
    st.session_state.delta_R_input = delta_R
    _sync_preset_selection()


def _delta_R_from_input():
    """Push the number input's delta_R to the log slider and refresh the preset pill."""
    st.session_state.log_delta_R = np.log10(st.session_state.delta_R_input)
    _sync_preset_selection()


def _apply_fms_preset():
    """Apply the selected preset's delta_R to both widgets."""
    if st.session_state.fms_preset is None:
        return          # pill was deselected; keep the current values
    delta_R = PRESETS[st.session_state.fms_preset]
    st.session_state.delta_R_input = delta_R
    st.session_state.log_delta_R = np.log10(delta_R)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def r0_grid(delta_R, n=4001):
    """Return an R0 axis wide enough to show both sidebands' resonances.

    The window grows with delta_R (so widely-spaced sidebands are both
    visible) but never shrinks below a fixed floor (so the line-center
    structure stays visible at small delta_R).

    Parameters
    ----------
    delta_R : float
        Normalized sideband spacing.
    n : int, optional
        Number of points; dense enough to resolve the sharp S3 notch at
        R0=0. Defaults to 4001.

    Returns
    -------
    numpy.ndarray
        R0 values, symmetric about zero.
    """
    span = max(8.0, 1.8 * delta_R + 8.0)
    return np.linspace(-span, span, n)


def draw_signal_panels(axes, *, R0, S1, S2, S3, mode, ink):
    """Draw the stacked S1/S2/S3 vs. R0 panels (the main figure).

    Parameters
    ----------
    axes : sequence of matplotlib.axes.Axes
        Three axes, one each for S1, S2, S3, sharing the R0 x-axis.
    R0 : numpy.ndarray
        Normalized carrier detuning.
    S1, S2, S3 : numpy.ndarray
        The demodulated signal components from :func:`core.fms.fms_signals`.
    mode : {"light", "dark"}
        Theme variant used to pick trace colours.
    ink : dict
        Theme ink colours from :func:`core.theming.theme_ink`.
    """
    labels = ["$S_1$ (absorption)", "$S_2$ (dispersion)", "$S_3 = \\sqrt{S_1^2+S_2^2}$"]
    traces = [S1, S2, S3]
    for ax, trace, label, n in zip(axes, traces, labels, (1, 2, 3)):
        ax.axhline(0, color=ink["hline"], lw=0.8, zorder=0)
        ax.plot(R0, trace, color=color_for_n(n, mode), lw=1.8, zorder=2)
        ax.set_ylabel(label, fontweight='bold')
        margin = 0.15 * max(np.max(np.abs(trace)), 1e-3)
        ax.set_ylim(np.min(trace) - margin, np.max(trace) + margin)
    axes[0].set_title('FMS heterodyne beat signals vs. carrier detuning $R_0$',
                      fontweight='bold', color=ink["text"], pad=12)
    axes[-1].set_xlabel('Normalized carrier detuning  $R_0$', fontweight='bold')


def build_main_figure(*, R0, S1, S2, S3, mode, ink):
    """Build and return the three-panel matplotlib figure (S1, S2, S3 vs. R0)."""
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.5), dpi=150, sharex=True)
    fig.subplots_adjust(hspace=0.12, top=0.93, bottom=0.09, left=0.14, right=0.96)
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor('none')
    draw_signal_panels(axes, R0=R0, S1=S1, S2=S2, S3=S3, mode=mode, ink=ink)
    return fig


def build_slope_figure(*, delta_R, mode, ink):
    """Build and return the discriminant-slope-vs.-delta_R figure.

    Plots ``|slope_S1|`` and ``|slope_S2|`` (see
    :func:`core.fms.fms_discriminant_slope`) against delta_R on a log axis,
    with the current operating point and the S1-optimum marked.
    """
    fig, ax = plt.subplots(figsize=(9.0, 3.2), dpi=150)
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.11, right=0.96)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    grid = np.logspace(np.log10(DELTA_R_MIN), np.log10(DELTA_R_MAX), 600)
    slope_S1, slope_S2 = fms_discriminant_slope(grid)
    ax.plot(grid, np.abs(slope_S1), color=color_for_n(1, mode), lw=2.0, label='$|dS_1/dR_0|$')
    ax.plot(grid, np.abs(slope_S2), color=color_for_n(2, mode), lw=2.0, label='$|dS_2/dR_0|$')

    ax.axvline(OPTIMAL_DELTA_R, ls='--', color=ink["hline"], lw=0.8, zorder=0)
    ax.axvline(delta_R, color=ink["text"], lw=1.2, zorder=1)

    ax.set_xscale('log')
    ax.set_xlim(DELTA_R_MIN, DELTA_R_MAX)
    ax.set_xlabel('Normalized sideband spacing  $\\Delta R$', fontweight='bold')
    ax.set_ylabel('|Slope| at $R_0=0$', fontweight='bold')
    ax.set_title('Discriminant slope vs. modulation frequency', fontweight='bold',
                 color=ink["text"], pad=10)
    ax.legend(loc='upper right', frameon=False)
    return fig


def build_mixed_figure(*, R0, mixed, theta_deg, mode, ink):
    """Build and return the demodulation-phase-mixed signal figure."""
    fig, ax = plt.subplots(figsize=(9.0, 3.2), dpi=150)
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.11, right=0.96)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    ax.axhline(0, color=ink["hline"], lw=0.8, zorder=0)
    ax.plot(R0, mixed, color=color_for_n(4, mode), lw=1.8, zorder=2)
    ax.set_xlabel('Normalized carrier detuning  $R_0$', fontweight='bold')
    ax.set_ylabel('Demodulated signal', fontweight='bold')
    ax.set_title(f'Signal at demodulation phase $\\theta$ = {theta_deg:.0f}°',
                 fontweight='bold', color=ink["text"], pad=10)
    return fig


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------
def render_sidebar():
    """Draw the sidebar controls and return the current parameter values.

    Returns
    -------
    tuple
        ``(delta_R, theta_deg)``.
    """
    st.sidebar.title("FMS parameters")
    st.sidebar.slider(
        "Modulation frequency  ΔR (log scale)",
        np.log10(DELTA_R_MIN), np.log10(DELTA_R_MAX), step=0.01,
        key="log_delta_R", format="%.2f", on_change=_delta_R_from_log_slider,
        help="Sideband spacing normalized to the half-linewidth: "
             "ΔR = ω_m / (Δω_FWHM / 2).")
    st.sidebar.number_input(
        "Exact ΔR", DELTA_R_MIN, DELTA_R_MAX, step=0.001, key="delta_R_input",
        format="%.3f", on_change=_delta_R_from_input,
        help="Type a precise ΔR; the slider follows.")
    theta_deg = st.sidebar.slider(
        "Demodulation phase  θ [deg]", 0.0, 360.0, step=1.0, key="theta_deg",
        help="Phase-adjuster setting on the rf mixer. θ=0° observes S1 "
             "(absorption); θ=90° observes S2 (dispersion).")

    st.sidebar.pills("Presets", list(PRESETS), key="fms_preset",
                     on_change=_apply_fms_preset,
                     help="Common operating points. The highlight follows ΔR: "
                          "it clears when the sliders leave the preset value.")

    return st.session_state.delta_R_input, theta_deg


def render_slope_metrics(delta_R):
    """Show the current discriminant slope readouts, side by side."""
    with st.container(border=True):
        slope_S1, slope_S2 = fms_discriminant_slope(delta_R)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("S1 slope at line center", f"{slope_S1:.3f}",
                      help="dS1/dR0 at R0=0. Maximized at ΔR=1/√3 "
                           "(value 3√3/4 ≈ 1.299).")
        with col2:
            st.metric("|S2 slope| at line center", f"{abs(slope_S2):.3f}",
                      help="|dS2/dR0| at R0=0. Reaches a local maximum of "
                           "2.25 at ΔR=√3 before settling toward 2.")


def render_physics_details():
    """Render the expandable 'Physics & model details' write-up."""
    with st.expander("Physics & model details"):
        st.markdown(
            "This page models **FM spectroscopy (FMS)** in the small "
            "modulation index limit (M ≪ 1): a strong optical carrier plus "
            "two weak sidebands, spaced by the rf modulation frequency "
            "$\\omega_m$, probe a single **isolated Lorentzian** spectral "
            "feature with no background absorption. This follows "
            "Bjorklund, Levenson, Lenth & Ortiz, *Appl. Phys. B* **32**, "
            "145 (1983).")
        st.markdown(
            "The absorption and dispersion of the feature are normalized "
            "Lorentzians in the detuning $R = (\\omega-\\Omega)/(\\Delta\\omega_{FWHM}/2)$:")
        st.latex(r"\delta(R)=\frac{\delta_{peak}}{R^2+1},\qquad "
                r"\phi(R)=\delta_{peak}\frac{R}{R^2+1}.")
        st.markdown(
            "The photodetector beat note at $\\omega_m$ decomposes into an "
            "in-phase and quadrature component, evaluated at the lower "
            "sideband, carrier, and upper sideband detunings "
            "$R_0-\\Delta R,\\ R_0,\\ R_0+\\Delta R$:")
        st.latex(r"S_1=\delta_{-}-\delta_{+},\qquad "
                r"S_2=\phi_{+}+\phi_{-}-2\phi_0,\qquad S_3=\sqrt{S_1^2+S_2^2}.")
        st.markdown(
            r"- **$S_1$** (in-phase, $\cos\omega_m t$) is proportional to the "
            r"*absorption* the feature induces." "\n"
            r"- **$S_2$** (quadrature, $\sin\omega_m t$) is proportional to "
            r"the *dispersion*." "\n"
            r"- **$S_3$** is the magnitude a phase-*insensitive* detector "
            r"(e.g. a spectrum analyzer) would see." "\n"
            r"- The **demodulation phase** $\theta$ slider mixes the two: "
            r"$S_1\cos\theta+S_2\sin\theta$, matching the \"phase adjuster\" "
            r"mentioned (but never plotted) in the original paper.")
        st.markdown(
            "**Two regimes, one slider.** For $\\Delta R \\ll 1$ (\"WMS "
            "limit\" preset), $S_1$ resembles the *derivative* of the "
            "absorption and $S_2$ the *second derivative* of the dispersion "
            "— this is ordinary wavelength modulation spectroscopy. For "
            "$\\Delta R \\gg 1$ (\"Fully resolved\" preset), each sideband "
            "individually resolves the Lorentzian feature as it is tuned "
            "through resonance, and $S_1$'s extrema approach $\\pm\\delta_{peak}$. "
            "In between sits the \"Optimal S1 slope\" preset, "
            "$\\Delta R=1/\\sqrt{3}$, which maximizes the discriminant slope "
            "used for frequency locking.")
        st.markdown(
            "**A property useful for locking:** $S_1=S_2=S_3=0$ at $R_0=0$ "
            "for *every* $\\Delta R$ — the carrier on exact resonance always "
            "gives a null, regardless of modulation frequency.")
        st.caption(
            "Idealizations: small modulation index (M ≪ 1), a single "
            "isolated Lorentzian feature, no background absorption. See "
            "the PDH Lock page for the Fabry-Perot cavity case, which "
            "reuses this page's demodulation machinery but not this "
            "Lorentzian lineshape.")
        st.markdown(
            "**Reference:** G. C. Bjorklund, M. D. Levenson, W. Lenth, "
            "C. Ortiz, \"Frequency Modulation (FM) Spectroscopy: Theory of "
            "Lineshapes and Signal-to-Noise Analysis,\" "
            "*Appl. Phys. B* **32**, 145-152 (1983).")


def init_session_state():
    """Seed the session-state defaults the widgets rely on before they are created."""
    if "delta_R_input" not in st.session_state:
        st.session_state.delta_R_input = OPTIMAL_DELTA_R
    if "log_delta_R" not in st.session_state:
        st.session_state.log_delta_R = np.log10(st.session_state.delta_R_input)
    if "theta_deg" not in st.session_state:
        st.session_state.theta_deg = 0.0


def render():
    """Render the FMS page: sidebar controls, three figures, and notes."""
    mode = resolve_mode()
    ink = theme_ink(mode == "dark")

    init_session_state()
    delta_R, theta_deg = render_sidebar()

    R0 = r0_grid(delta_R)
    S1, S2, S3 = fms_signals(R0, delta_R)
    mixed = fms_signal_mixed(R0, delta_R, np.deg2rad(theta_deg))

    apply_plot_style(ink)

    st.header("FMS Error Signal")
    st.caption("Frequency modulation (FM) spectroscopy error signal for an isolated "
               "Lorentzian feature · small modulation index (M ≪ 1) limit "
               "(Bjorklund et al., 1983)")

    render_slope_metrics(delta_R)

    with st.container(border=True):
        fig_main = build_main_figure(R0=R0, S1=S1, S2=S2, S3=S3, mode=mode, ink=ink)
        st.pyplot(fig_main, use_container_width=True, bbox_inches=None)
    plt.close(fig_main)

    with st.container(border=True):
        fig_slope = build_slope_figure(delta_R=delta_R, mode=mode, ink=ink)
        st.pyplot(fig_slope, use_container_width=True, bbox_inches=None)
    plt.close(fig_slope)

    with st.container(border=True):
        fig_mixed = build_mixed_figure(R0=R0, mixed=mixed, theta_deg=theta_deg,
                                       mode=mode, ink=ink)
        st.pyplot(fig_mixed, use_container_width=True, bbox_inches=None)
    plt.close(fig_mixed)

    render_physics_details()
