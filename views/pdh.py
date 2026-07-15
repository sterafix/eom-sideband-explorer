"""Streamlit page: Pound-Drever-Hall (PDH) locking error signal.

This module is the presentation layer only: it reads the user's controls,
calls into :mod:`core.pdh` for the numbers, and draws the figures. All of the
physics lives in :mod:`core.pdh` and :mod:`core.demod` so it can be tested
independently (see ``tests/test_pdh.py``, ``tests/test_pdh_fig6.py``,
``tests/test_pdh_fig7.py``).

Reproduces the demodulated PDH cavity-lock error signal of Black, "An
introduction to Pound-Drever-Hall laser frequency stabilization," Am. J. Phys.
69, 79 (2001), Figs. 6-7: a phase-modulated laser reflects off a Fabry-Perot
cavity, and the reflected power is demodulated at the modulation frequency
into an in-phase (cosine) and quadrature (sine) error signal, plotted against
the laser frequency x = f / delta_nu_fsr.

The three figures shown are:

* **Main plot** - the cosine and sine error-signal components vs. x, redrawn
  live: slow modulation reproduces Fig. 6 (cosine), fast modulation Fig. 7
  (sine, the standard PDH curve).
* **Discriminant plot** - the frequency discriminant |D| vs. modulation depth
  beta, with the optimum beta = 1.08 and the current beta marked.
* **Demodulation phase plot** - the signal an actual mixer would produce at
  the chosen demodulation phase theta, morphing between the cosine (theta=0)
  and sine (theta=90 deg) components.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from core.pdh import (
    cavity_reflection_general,
    finesse_from_r,
    pdh_error_signal,
    pdh_error_signal_mixed,
    pdh_frequency_discriminant,
    pdh_optimum_modulation_depth,
    r_from_finesse,
    sideband_powers,
)
from core.theming import apply_plot_style, color_for_n, resolve_mode, theme_ink

# Bounds for the primary controls. Finesse must exceed pi for a real r; the
# range spans typical reference cavities (tens) to ultra-high-finesse ones.
FINESSE_MIN, FINESSE_MAX = 10.0, 1e5
# x_mod ranges up to ~1 FSR so the sideband-resonance pitfall (sidebands
# landing on adjacent cavity resonances, x_mod near an integer) is reachable.
X_MOD_MIN, X_MOD_MAX = 1e-4, 1.2
BETA_MAX = 3.0

OPTIMUM_BETA = pdh_optimum_modulation_depth()

# Preset operating points. Each maps a label to a dict of parameter values;
# unlike the single-value FMS presets, PDH presets set several controls at
# once (finesse, x_mod, beta, theta), so the sync/apply callbacks handle a
# dict. theta selects which component the "observed signal" panel shows: 0deg
# = cosine (Fig. 6), 90deg = sine (the standard PDH curve, Fig. 7).
PRESETS = {
    "Typical PDH (fast)":       {"finesse": 500.0, "x_mod": 0.04,  "beta": OPTIMUM_BETA, "theta": 90.0},
    "Slow modulation (Fig. 6)": {"finesse": 500.0, "x_mod": 1e-3,  "beta": 1.0,          "theta": 0.0},
    "Optimum depth (β=1.08)":   {"finesse": 500.0, "x_mod": 0.04,  "beta": OPTIMUM_BETA, "theta": 90.0},
    "Sideband resonance pitfall": {"finesse": 500.0, "x_mod": 0.95, "beta": 1.0,         "theta": 90.0},
}


# ---------------------------------------------------------------------------
# Widget-synchronisation callbacks
#
# finesse and x_mod are each edited by a log-scaled slider plus an exact
# number input (the useful ranges span several decades), mirroring the FMS
# delta_R triad. Presets set all three physical parameters at once.
# ---------------------------------------------------------------------------
def _current_params():
    """Return the current (finesse, x_mod, beta, theta) from the widgets."""
    return (st.session_state.finesse_input,
            st.session_state.x_mod_input,
            st.session_state.beta_pdh,
            st.session_state.theta_deg_pdh)


def _sync_preset_selection():
    """Light up the preset pill whose full parameter set matches, else clear it."""
    finesse, x_mod, beta, theta = _current_params()
    st.session_state.pdh_preset = next(
        (label for label, p in PRESETS.items()
         if abs(p["finesse"] - finesse) < 1e-6
         and abs(p["x_mod"] - x_mod) < 1e-9
         and abs(p["beta"] - beta) < 1e-9
         and abs(p["theta"] - theta) < 1e-9), None)


def _finesse_from_log_slider():
    st.session_state.finesse_input = float(10 ** st.session_state.log_finesse)
    _sync_preset_selection()


def _finesse_from_input():
    st.session_state.log_finesse = float(np.log10(st.session_state.finesse_input))
    _sync_preset_selection()


def _x_mod_from_log_slider():
    st.session_state.x_mod_input = float(10 ** st.session_state.log_x_mod)
    _sync_preset_selection()


def _x_mod_from_input():
    st.session_state.log_x_mod = float(np.log10(st.session_state.x_mod_input))
    _sync_preset_selection()


def _param_changed():
    """Refresh the preset highlight when beta or theta is edited directly."""
    _sync_preset_selection()


def _apply_pdh_preset():
    """Apply the selected preset's parameters to all widgets."""
    if st.session_state.pdh_preset is None:
        return
    p = PRESETS[st.session_state.pdh_preset]
    st.session_state.finesse_input = p["finesse"]
    st.session_state.log_finesse = float(np.log10(p["finesse"]))
    st.session_state.x_mod_input = p["x_mod"]
    st.session_state.log_x_mod = float(np.log10(p["x_mod"]))
    st.session_state.beta_pdh = p["beta"]
    st.session_state.theta_deg_pdh = p["theta"]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def x_grid(x_mod, finesse):
    """Return an x axis zoomed on one cavity resonance (x=1).

    The window grows with x_mod (so both sideband features at x = 1 +- x_mod
    are visible) but never below a few linewidths (so the central feature is
    visible). Grid density is adaptive so the linewidth stays resolved even at
    very high finesse, where a fixed-count grid would miss the narrow central
    feature.

    Parameters
    ----------
    x_mod : float
        Modulation frequency as a fraction of the free spectral range.
    finesse : float
        Cavity finesse (sets the linewidth ``1 / finesse``).

    Returns
    -------
    numpy.ndarray
        x values, symmetric about the resonance at x = 1.
    """
    linewidth = 1 / finesse
    half_width = max(3 * linewidth, 1.5 * x_mod + 3 * linewidth)
    n = int(np.clip(half_width * 2 / (linewidth / 8), 6001, 120001))
    return np.linspace(1 - half_width, 1 + half_width, n)


def draw_component_panels(axes, *, x, cos_c, sin_c, mode, ink):
    """Draw the stacked cosine/sine error-signal component panels."""
    labels = ["In-phase ($\\cos$) component", "Quadrature ($\\sin$) component"]
    traces = [cos_c, sin_c]
    for ax, trace, label, n in zip(axes, traces, labels, (1, 2)):
        ax.axhline(0, color=ink["hline"], lw=0.8, zorder=0)
        ax.axvline(1.0, ls='--', color=ink["hline"], lw=0.8, zorder=0)
        ax.plot(x, trace, color=color_for_n(n, mode), lw=1.8, zorder=2)
        ax.set_ylabel(label, fontweight='bold')
        margin = 0.15 * max(np.max(np.abs(trace)), 1e-6)
        ax.set_ylim(np.min(trace) - margin, np.max(trace) + margin)
    axes[0].set_title('PDH error-signal components vs. laser frequency $x$',
                      fontweight='bold', color=ink["text"], pad=12)
    axes[-1].set_xlabel('Laser frequency  $x = f / \\Delta\\nu_\\mathrm{FSR}$',
                        fontweight='bold')


def build_main_figure(*, x, cos_c, sin_c, mode, ink):
    """Build the two-panel figure (cosine and sine components vs. x)."""
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), dpi=150, sharex=True)
    fig.subplots_adjust(hspace=0.12, top=0.91, bottom=0.11, left=0.14, right=0.96)
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor('none')
    draw_component_panels(axes, x=x, cos_c=cos_c, sin_c=sin_c, mode=mode, ink=ink)
    return fig


def build_discriminant_figure(*, r, beta, mode, ink):
    """Build the |D| vs. beta figure, marking the optimum and current beta."""
    fig, ax = plt.subplots(figsize=(9.0, 3.2), dpi=150)
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.12, right=0.96)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    betas = np.linspace(0.01, BETA_MAX, 600)
    Pc, Ps = sideband_powers(betas)
    D = pdh_frequency_discriminant(r, 1.0, Pc, Ps)      # delta_nu_fsr = 1 (x units)
    ax.plot(betas, np.abs(D), color=color_for_n(3, mode), lw=2.0)

    ax.axvline(OPTIMUM_BETA, ls='--', color=ink["hline"], lw=0.8, zorder=0)
    ax.axvline(beta, color=ink["text"], lw=1.2, zorder=1)
    ax.annotate(f"optimum β = {OPTIMUM_BETA:.2f}", (OPTIMUM_BETA, 0),
                xytext=(6, 8), textcoords="offset points", fontsize=9,
                color=ink["grey"])

    ax.set_xlim(0, BETA_MAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Modulation depth  $\\beta$ [rad]', fontweight='bold')
    ax.set_ylabel('$|D|$  (per FSR)', fontweight='bold')
    ax.set_title('Frequency discriminant vs. modulation depth', fontweight='bold',
                 color=ink["text"], pad=10)
    return fig


def build_mixed_figure(*, x, mixed, theta_deg, mode, ink):
    """Build the demodulation-phase-mixed error-signal figure."""
    fig, ax = plt.subplots(figsize=(9.0, 3.2), dpi=150)
    fig.subplots_adjust(top=0.86, bottom=0.2, left=0.12, right=0.96)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor('none')

    ax.axhline(0, color=ink["hline"], lw=0.8, zorder=0)
    ax.axvline(1.0, ls='--', color=ink["hline"], lw=0.8, zorder=0)
    ax.plot(x, mixed, color=color_for_n(4, mode), lw=1.8, zorder=2)
    ax.set_xlabel('Laser frequency  $x = f / \\Delta\\nu_\\mathrm{FSR}$', fontweight='bold')
    ax.set_ylabel('Error signal', fontweight='bold')
    ax.set_title(f'Error signal at demodulation phase $\\theta$ = {theta_deg:.0f}°',
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
        ``(finesse, x_mod, beta, theta_deg, advanced)`` where ``advanced`` is a
        dict of general-cavity parameters (or ``None`` when the default
        symmetric cavity is selected).
    """
    st.sidebar.title("PDH parameters")
    st.sidebar.slider(
        "Cavity finesse (log scale)",
        float(np.log10(FINESSE_MIN)), float(np.log10(FINESSE_MAX)), step=0.01,
        key="log_finesse", format="%.2f", on_change=_finesse_from_log_slider,
        help="Cavity finesse F ≈ π/(1−r²). The linewidth is 1/F of a free "
             "spectral range.")
    st.sidebar.number_input(
        "Exact finesse", FINESSE_MIN, FINESSE_MAX, step=1.0, key="finesse_input",
        format="%.1f", on_change=_finesse_from_input,
        help="Type a precise finesse; the slider follows.")
    st.sidebar.slider(
        "Modulation frequency  x_mod (log scale)",
        float(np.log10(X_MOD_MIN)), float(np.log10(X_MOD_MAX)), step=0.01,
        key="log_x_mod", format="%.2f", on_change=_x_mod_from_log_slider,
        help="Modulation frequency as a fraction of a free spectral range. "
             "Slow (≪ linewidth) → cosine signal; fast (≫ linewidth) → sine.")
    st.sidebar.number_input(
        "Exact x_mod", X_MOD_MIN, X_MOD_MAX, step=1e-4, key="x_mod_input",
        format="%.4f", on_change=_x_mod_from_input,
        help="Type a precise x_mod; the slider follows.")
    beta = st.sidebar.slider(
        "Modulation depth  β [rad]", 0.0, BETA_MAX, step=0.01, key="beta_pdh",
        on_change=_param_changed,
        help="Phase modulation depth. β ≈ 1.08 maximizes the discriminant.")
    theta_deg = st.sidebar.slider(
        "Demodulation phase  θ [deg]", 0.0, 360.0, step=1.0, key="theta_deg_pdh",
        on_change=_param_changed,
        help="Phase-adjuster setting on the rf mixer. θ=0° observes the cosine "
             "component; θ=90° the sine (standard PDH) component.")

    st.sidebar.pills("Presets", list(PRESETS), key="pdh_preset",
                     on_change=_apply_pdh_preset,
                     help="Common operating points. The highlight clears when "
                          "any control leaves the preset value.")

    advanced = None
    with st.sidebar.expander("Advanced: general (lossy/asymmetric) cavity"):
        if st.checkbox("Use general two-mirror model (Appendix A)", key="pdh_advanced"):
            r1 = st.slider("Input mirror r₁", 0.5, 0.9999, 0.99, 0.0005, format="%.4f")
            t1 = st.slider("Input mirror t₁", 0.0, 0.5, float(np.sqrt(1 - 0.99**2)),
                           0.0005, format="%.4f")
            r2 = st.slider("Back mirror r₂", 0.5, 1.0, 0.999, 0.0005, format="%.4f")
            advanced = {"r1": r1, "t1": t1, "r2": r2}
            st.caption("Reflected power = |F|²; losses when r₁²+t₁² < 1.")

    return (st.session_state.finesse_input, st.session_state.x_mod_input,
            beta, theta_deg, advanced)


def render_discriminant_metric(r, beta):
    """Show the current frequency discriminant and cavity linewidth."""
    with st.container(border=True):
        Pc, Ps = sideband_powers(beta)
        D = pdh_frequency_discriminant(r, 1.0, Pc, Ps)
        linewidth = 1 / finesse_from_r(r)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Frequency discriminant |D|", f"{abs(D):.1f}",
                      help="|D| = 8√(Pc·Ps)/δν per FSR (Black Eq. 4.2), the "
                           "near-resonance error-signal slope. Maximized at β≈1.08.")
        with col2:
            st.metric("Cavity linewidth δν", f"{linewidth:.2e} FSR",
                      help="δν = Δν_FSR / finesse. The error-signal features "
                           "are this wide.")


def render_pitfall_warning(x_mod):
    """Warn when the sidebands land near an adjacent cavity resonance."""
    nearest_int = round(x_mod)
    if nearest_int >= 1 and abs(x_mod - nearest_int) < 0.1:
        st.warning(
            f"**Sideband-resonance pitfall:** x_mod ≈ {x_mod:.3f} places the "
            f"modulation sidebands close to a cavity resonance (integer multiple "
            f"of the FSR). The sidebands are then no longer fully reflected, the "
            f"F(x±x_mod) ≈ −1 assumption breaks down, and the error signal "
            f"distorts. Keep x_mod well below 1 for a clean lock.")


def render_physics_details():
    """Render the expandable 'Physics & model details' write-up."""
    with st.expander("Physics & model details"):
        st.markdown(
            "This page models **Pound-Drever-Hall (PDH)** laser frequency "
            "stabilization, following Black, *Am. J. Phys.* **69**, 79 (2001). "
            "A phase-modulated laser (carrier + two sidebands) reflects off a "
            "**Fabry-Perot cavity**; the reflected power, demodulated at the "
            "modulation frequency, gives an error signal that tells the servo "
            "which side of resonance the laser is on.")
        st.markdown(
            "Everything is in the normalized frequency $x = f/\\Delta\\nu_"
            "\\mathrm{FSR}$ (laser frequency in units of the cavity free "
            "spectral range), so cavity resonances sit at integer $x$ and one "
            "round trip is a phase $2\\pi x$. The symmetric, lossless cavity "
            "reflection coefficient is (Eq. 3.1):")
        st.latex(r"F(x)=\frac{r\left(e^{2\pi i x}-1\right)}{1-r^2 e^{2\pi i x}},")
        st.markdown(
            "with $r$ the mirror amplitude reflectivity and finesse "
            "$\\mathcal{F}\\approx\\pi/(1-r^2)$. The demodulated error signal "
            "has an in-phase and a quadrature component built from the beat "
            "term $C = F(x)F^*(x{+}x_\\mathrm{mod}) - F^*(x)F(x{-}x_\\mathrm{mod})$ "
            "(Eq. 3.3):")
        st.latex(r"\epsilon_{\cos}=2\sqrt{P_c P_s}\,\mathrm{Re}(C),\qquad "
                 r"\epsilon_{\sin}=2\sqrt{P_c P_s}\,\mathrm{Im}(C),")
        st.markdown(
            r"where $P_c=J_0(\beta)^2 P_0$ and $P_s=J_1(\beta)^2 P_0$ are the "
            r"carrier and sideband powers. Two regimes:")
        st.markdown(
            r"- **Slow modulation** ($x_\mathrm{mod}\ll$ linewidth, *Fig. 6* / "
            r"\"Slow modulation\" preset): the **cosine** component dominates; "
            r"the signal looks like the derivative of the cavity dip." "\n"
            r"- **Fast modulation near resonance** ($x_\mathrm{mod}$ many "
            r"linewidths but $\ll 1$ FSR, *Fig. 7* / \"Typical PDH\" preset): "
            r"the **sine** component dominates, giving the classic three-feature "
            r"PDH curve — a steep central discriminant plus two sideband "
            r"features at $x=1\pm x_\mathrm{mod}$." "\n"
            r"- The **demodulation phase** $\theta$ selects the observed "
            r"combination $\epsilon_{\cos}\cos\theta+\epsilon_{\sin}\sin\theta$.")
        st.markdown(
            r"**Locking.** The error signal is **exactly zero at resonance** "
            r"($x=1$) and antisymmetric about it, for every finesse, "
            r"$x_\mathrm{mod}$, and $\beta$ — the servo's set point. Its "
            r"near-resonance slope is the frequency discriminant (Eq. 4.2), "
            r"$D=-8\sqrt{P_c P_s}/\delta\nu$, maximized at $\beta\approx1.08$ "
            r"(Appendix B).")
        st.caption(
            "Default model: symmetric, lossless two-mirror cavity (Eq. 3.1). "
            "The general lossy/asymmetric case (Appendix A) is available under "
            "\"Advanced\" in the sidebar. Compare with the FMS page, which "
            "reuses this same demodulation machinery for a Lorentzian absorber "
            "instead of a cavity.")
        st.markdown(
            "**Reference:** E. D. Black, \"An introduction to Pound-Drever-Hall "
            "laser frequency stabilization,\" *Am. J. Phys.* **69**, 79-87 (2001).")


def init_session_state():
    """Seed the session-state defaults the widgets rely on before creation."""
    defaults = {
        "finesse_input": 500.0,
        "x_mod_input": 0.04,
        "beta_pdh": OPTIMUM_BETA,
        "theta_deg_pdh": 90.0,      # default to the standard PDH (sine) signal
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "log_finesse" not in st.session_state:
        st.session_state.log_finesse = float(np.log10(st.session_state.finesse_input))
    if "log_x_mod" not in st.session_state:
        st.session_state.log_x_mod = float(np.log10(st.session_state.x_mod_input))


def render():
    """Render the PDH page: sidebar controls, three figures, and notes."""
    mode = resolve_mode()
    ink = theme_ink(mode == "dark")

    init_session_state()
    finesse, x_mod, beta, theta_deg, advanced = render_sidebar()
    r = r_from_finesse(finesse)

    x = x_grid(x_mod, finesse)
    if advanced is None:
        cos_c, sin_c = pdh_error_signal(x, x_mod, r, beta)
        mixed = pdh_error_signal_mixed(x, x_mod, r, beta, np.deg2rad(theta_deg))
    else:
        # General-cavity model: build the beat term from cavity_reflection_general.
        from core.demod import mixed_signal
        r1, t1, r2 = advanced["r1"], advanced["t1"], advanced["r2"]
        F0 = cavity_reflection_general(x, r1, t1, r2)
        Fp = cavity_reflection_general(x + x_mod, r1, t1, r2)
        Fm = cavity_reflection_general(x - x_mod, r1, t1, r2)
        C = F0 * np.conj(Fp) - np.conj(F0) * Fm
        Pc, Ps = sideband_powers(beta)
        scale = 2 * np.sqrt(Pc * Ps)
        cos_c, sin_c = scale * C.real, scale * C.imag
        mixed = mixed_signal(cos_c, sin_c, np.deg2rad(theta_deg))

    apply_plot_style(ink)

    st.header("PDH Lock")
    st.caption("Pound-Drever-Hall cavity-lock error signal · phase-modulated laser "
               "reflected from a Fabry-Perot cavity (Black, 2001)")

    render_discriminant_metric(r, beta)
    render_pitfall_warning(x_mod)

    with st.container(border=True):
        fig_main = build_main_figure(x=x, cos_c=cos_c, sin_c=sin_c, mode=mode, ink=ink)
        st.pyplot(fig_main, use_container_width=True, bbox_inches=None)
    plt.close(fig_main)

    with st.container(border=True):
        fig_disc = build_discriminant_figure(r=r, beta=beta, mode=mode, ink=ink)
        st.pyplot(fig_disc, use_container_width=True, bbox_inches=None)
    plt.close(fig_disc)

    with st.container(border=True):
        fig_mixed = build_mixed_figure(x=x, mixed=mixed, theta_deg=theta_deg,
                                       mode=mode, ink=ink)
        st.pyplot(fig_mixed, use_container_width=True, bbox_inches=None)
    plt.close(fig_mixed)

    render_physics_details()
