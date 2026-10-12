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

import io

import matplotlib
import numpy as np
import streamlit as st
from matplotlib.figure import Figure

from physics import (color_for_n, sideband_intensities, intensity_vs_beta,
                     captured_power, to_decibels, DB_FLOOR)

NMAX = 6        # max selectable sideband orders to display

# Share of the optical power the displayed orders must hold for the spectrum to
# count as essentially complete; below it the truncation warning is shown.
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
# The figure is drawn for the active Streamlit theme (light or dark) and
# carries that theme's background colour itself, rather than being transparent.
# Streamlit only reports the theme reliably after a rerun: when a viewer
# switches the theme in the app menu, nothing reruns, so the figure keeps the
# previous theme until the next interaction (streamlit/streamlit#11920). With
# its own background the figure then still reads as a light or dark panel,
# instead of light ink on a light page or dark ink on a dark one.
#
# Streamlit serves every visitor from threads of one process, and pyplot and
# matplotlib's rcParams are global to that process. So the figure is built
# without pyplot, and every theme colour is passed to the artists directly:
# concurrent visitors (one in light mode, one in dark) can then never pick up
# each other's colours. Only settings that are the same for everyone live in
# rcParams.
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "axes.linewidth": 1.0,
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
})


def theme_ink(mode):
    """Return the theme-dependent ink colours for the matplotlib figure.

    Parameters
    ----------
    mode : {"light", "dark"}
        The active Streamlit theme.

    Returns
    -------
    dict
        Colours keyed by role: ``bg`` (figure background, matches the
        Streamlit ``backgroundColor``), ``grey`` (axes/labels), ``text``
        (titles, matches the Streamlit ``textColor``), ``hline`` (dashed helper
        lines), and ``box_fc`` / ``box_ec`` (annotation-box face and edge).
    """
    is_dark = mode == "dark"
    return {
        "bg":     '#0e1117' if is_dark else '#ffffff',
        "grey":   '0.65' if is_dark else '0.45',
        "text":   '#fafafa' if is_dark else '#262730',
        "hline":  '0.5' if is_dark else '0.7',
        "box_fc": '#262730' if is_dark else 'white',
        "box_ec": '0.4' if is_dark else '0.8',
    }


def _color_axes(ax, ink):
    """Give the frame, ticks and axis labels of ``ax`` the theme's grey ink."""
    for spine in ax.spines.values():
        spine.set_edgecolor(ink["grey"])
    ax.tick_params(colors=ink["grey"])
    ax.xaxis.label.set_color(ink["grey"])
    ax.yaxis.label.set_color(ink["grey"])


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
    ax.set_xlabel('Frequency offset from carrier [MHz]', fontweight='bold')
    ax.set_ylabel('Relative intensity [dB]' if db_scale else 'Relative intensity',
                  fontweight='bold')
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
    legend = ax.legend(title='Bessel functions', loc='upper right', frameon=False,
                       labelcolor=ink["text"])
    legend.get_title().set_color(ink["text"])


def build_figure(*, beta, f0, N, Jn2, width_frac, noise, color_peaks,
                 fixed_xaxis, db_scale, mode):
    """Build and return the two-panel matplotlib figure (spectrum + Bessel curves).

    ``mode`` (``"light"`` or ``"dark"``) selects the theme colours; the other
    arguments are passed on to :func:`draw_spectrum` and
    :func:`draw_bessel_curves`.
    """
    ink = theme_ink(mode)
    fig = Figure(figsize=(12.5, 5.2))
    ax1, ax2 = fig.subplots(1, 2)
    fig.subplots_adjust(wspace=0.18, top=0.86, bottom=0.13, left=0.07, right=0.93)
    fig.patch.set_facecolor(ink["bg"])

    draw_spectrum(ax1, beta=beta, f0=f0, N=N, Jn2=Jn2, width_frac=width_frac,
                  noise=noise, color_peaks=color_peaks, db_scale=db_scale,
                  mode=mode, ink=ink)
    draw_bessel_curves(ax2, beta=beta, N=N, Jn2=Jn2, fixed_xaxis=fixed_xaxis,
                       db_scale=db_scale, mode=mode, ink=ink)
    for ax in (ax1, ax2):
        ax.set_facecolor('none')
        _color_axes(ax, ink)
    return fig


def figure_png(fig):
    """Render ``fig`` to PNG bytes, uncropped, on its own background.

    The canvas is saved exactly as laid out by ``subplots_adjust`` instead of
    being cropped to its content, at 200 dpi, with the figure's own background
    colour (see :func:`theme_ink`). Saving it here, rather than passing savefig
    options to ``st.pyplot`` (now deprecated), keeps that control in our hands.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    return buf.getvalue()


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
    # Slider and exact-value box share one row, so they read as a single control.
    col_slider, col_box = st.sidebar.columns([2.2, 1], vertical_alignment="bottom")
    beta = col_slider.slider("Modulation depth  β [rad]", 0.0, 10.0, step=0.005,
                             key="beta", format="%.3f", on_change=_beta_from_slider,
                             help="Peak phase shift the EOM gives the light, in radians. "
                                  "A stronger RF drive gives a larger β. Small β puts "
                                  "most power in the carrier; large β spreads it over "
                                  "many sidebands.")
    col_box.number_input("Exact β [rad]", 0.0, 10.0, step=0.005, key="beta_input",
                         format="%.3f", on_change=_beta_from_input,
                         label_visibility="collapsed")
    f0 = st.sidebar.slider("Modulation frequency  f₀ [MHz]", 50.0, 3000.0, 1000.0, 5.0,
                           format="%.0f",
                           help="Frequency of the RF signal driving the EOM. It sets "
                                "the spacing between neighbouring lines, but not "
                                "their heights.")
    N = st.sidebar.slider(
        "Displayed sideband orders", 1, NMAX, step=1, key="N",
        help="How many sideband orders are drawn on each side of the carrier. "
             "Order n sits at n·f₀ from the carrier. All orders exist "
             "physically; this only sets how many are shown.")

    st.sidebar.pills("Presets", list(PRESETS), key="preset", on_change=_apply_preset,
                     help="Common operating points. At the carrier null the carrier "
                          "vanishes completely, which is used to calibrate EOMs "
                          "(see Physics & model details). The highlight clears "
                          "when β leaves the preset value.")

    with st.sidebar.expander("Display options"):
        scale_choice = st.radio(
            "Y-axis scale", ["Linear", "dB"], horizontal=True, key="scale_choice",
            help="dB (10·log₁₀, referenced to the unmodulated carrier) makes the "
                 "weak higher-order sidebands visible, the way a spectrum analyzer "
                 "shows them. Applies to both figures.")
        width_frac = st.slider("Peak width", 0.003, 0.040, 0.012, 0.001, format="%.3f",
                               help="Cosmetic only. The ideal lines are infinitely "
                                    "narrow; a width just makes them visible.")
        noise = st.slider("Noise level", 0.0, 0.030, 0.004, 0.001, format="%.3f",
                          help="Cosmetic only. Adds a fake detector noise floor to "
                               "Fig. 1, as on a real measurement.")
        color_peaks = st.checkbox("Color-code peaks", True)
        fixed_xaxis = st.checkbox("Fix β-axis to 0 to 10", False)
    db_scale = scale_choice == "dB"

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
    """Show the captured-power metric, plus a warning if the spectrum is truncated."""
    with st.container(border=True):
        col_left, col_right = st.columns([1, 2])
        with col_left:
            cap = captured_power(Jn2)
            st.metric("Captured optical power", f"{cap*100:.1f} %",
                      help="Share of the total optical power in the lines shown. "
                           "The rest sits in higher orders that are not drawn.")
        with col_right:
            # Only speak up when something needs doing; a complete spectrum
            # is already evident from the metric itself.
            if cap < POWER_COMPLETE:
                st.warning(truncation_message(cap, N), icon=":material/warning:")


def render_intensity_table(Jn2, f0, N):
    """Show the exact intensity of each displayed sideband as a table."""
    with st.container(border=True):
        st.subheader("Sideband intensities")
        orders = list(range(0, N + 1))
        table = {
            "Order": ["0 (carrier)" if n == 0 else f"±{n}" for n in orders],
            "Offset from carrier [MHz]": ["0" if n == 0 else f"±{n * f0:.0f}" for n in orders],
            "Power fraction Jₙ(β)²": [f"{Jn2[n]:.4f}" for n in orders],
        }
        col_t, _ = st.columns([3, 2])
        with col_t:
            st.dataframe(table, hide_index=True, width="stretch")
            st.caption("Power in one line as a fraction of the laser power going in "
                       "(1 = all of it). Values are per sideband: the +n and -n "
                       "sidebands are equally strong, so the pair holds twice this.")


def render_physics_details():
    """Render the expandable 'Physics & model details' write-up."""
    with st.expander("Physics & model details"):
        st.markdown(
            "**How an EOM works.** An EOM is a crystal whose refractive index changes "
            "with the voltage across it (the Pockels effect). An RF voltage at "
            "frequency $f_0$ therefore delays the light a little more, then a little "
            "less, in step with the RF: it modulates the optical **phase**. This tool "
            "models that sinusoidal phase modulation of a single-frequency laser, for "
            "an ideal modulator:")
        st.latex(r"E(t)=E_0\,e^{\,i[\omega_c t+\beta\sin(\omega_m t)]}")
        st.markdown(
            r"Here $\omega_c$ is the laser (carrier) angular frequency, $\omega_m = 2\pi f_0$ "
            r"the modulation frequency, and $\beta$ the modulation depth, the peak "
            r"phase shift in radians. The **Jacobi-Anger identity** rewrites this "
            r"wiggling phase as a sum of steady laser lines, the carrier plus an "
            r"infinite set of sidebands spaced by $f_0$:")
        st.latex(r"E(t)=E_0\sum_{n=-\infty}^{\infty} J_n(\beta)\,e^{\,i(\omega_c+n\omega_m)t},")
        st.markdown(
            r"where $J_n$ is the Bessel function of the first kind of order $n$. The "
            r"line $n f_0$ away from the carrier has amplitude $J_n(\beta)$ and "
            r"therefore a power fraction $|J_n(\beta)|^2$. Two consequences shape the "
            r"figures:")
        st.markdown(
            r"- **Symmetry.** Since $J_{-n}(\beta)=(-1)^n J_n(\beta)$, the $+n$ and "
            r"$-n$ sidebands are equally strong." "\n"
            r"- **Energy conservation.** $\sum_{n} |J_n(\beta)|^2 = 1$: phase "
            r"modulation adds no power, it only moves it from the carrier into the "
            r"sidebands. The *Captured optical power* readout adds up the lines shown.")
        st.markdown("**Useful operating points**")
        st.markdown(
            r"- **Small $\beta$** (below about 0.5): each first-order sideband holds "
            r"about $\beta^2/4$ of the power, and higher orders are negligible. Typical "
            r"for laser locking and FM spectroscopy." "\n"
            r"- **$\beta \approx 1.08$** maximizes $J_0 J_1$, the carrier-sideband beat "
            r"that gives the strongest Pound-Drever-Hall error signal." "\n"
            r"- **$\beta \approx 1.84$** puts the most power into each first-order "
            r"sideband (about 34 %), for example to make a repump frequency in laser "
            r"cooling." "\n"
            r"- **$\beta \approx 2.405$** (first zero of $J_0$) removes the carrier "
            r"completely: the *carrier null*.")
        st.markdown(
            r"**From voltage to $\beta$.** On real hardware $\beta=\pi V_\text{peak}/V_\pi$, "
            r"where $V_\text{peak}$ is the RF voltage amplitude (half the peak-to-peak "
            r"value) and $V_\pi$ is the modulator's half-wave voltage, which depends on "
            r"the device, the wavelength and the RF frequency. The carrier null gives a "
            r"simple way to measure it: raise the RF drive until the carrier "
            r"disappears; there $\beta = 2.4048$, so")
        st.latex(r"V_\pi = \frac{\pi\,V_\text{peak}}{2.4048}.")
        st.markdown(
            r"With $V_\pi$ known, $\beta$ can be estimated for any drive voltage, as long "
            r"as wavelength, RF frequency and temperature stay close to the calibration "
            r"(effects such as piezoelectric resonances in the crystal can still spoil "
            r"this). The simulation itself takes $\beta$ directly and needs no $V_\pi$.")
        st.markdown("**What the figures show**")
        st.markdown(
            r"- **Fig. 1** is the optical spectrum: each line at $n f_0$ from the "
            r"carrier, with height $|J_n(\beta)|^2$ and its order number on top. Real "
            r"lines are infinitely narrow, so they are drawn as narrow peaks on a small "
            r"fake noise floor (both adjustable under Display options). The dashed line "
            r"marks the exact value; with noise added, a peak can poke slightly above it." "\n"
            r"- **Fig. 2** shows how each line's power changes with $\beta$. The dots at "
            r"your chosen $\beta$ have the same heights as the peaks in Fig. 1, linked by "
            r"the dashed lines." "\n"
            r"- **Y-axis scale.** *dB* plots $10\log_{10}|J_n(\beta)|^2$ on both "
            r"figures, where 0 dB is the full laser power. On the linear axis the higher "
            r"orders are too small to see; the logarithmic dB axis spreads small values "
            r"apart so they become visible. In Fig. 1 a line weaker than the noise floor "
            r"stays hidden in it, just as on a real measurement. Values are cut off at "
            rf"${DB_FLOOR:.0f}$ dB so the zeros of the Bessel functions stay on scale.")
        st.markdown("**Beyond the ideal case: residual amplitude modulation**")
        st.markdown(
            r"Real modulators never produce *pure* phase modulation. Imperfections such "
            r"as reflections at the crystal faces or light polarized not exactly along "
            r"the crystal axis add a small **residual amplitude modulation (RAM)**: the "
            r"laser power itself wobbles at $f_0$, which pure phase modulation never "
            r"does. In laser locking schemes such as Pound-Drever-Hall, RAM appears as "
            r"a small, temperature-dependent offset in the error signal that slowly "
            r"shifts the lock point. The ideal model here does not include it."
        )
        st.markdown(
            "**References and further reading**\n"
            "- [Jacobi-Anger expansion](https://en.wikipedia.org/wiki/Jacobi%E2%80%93Anger_expansion)\n"
            "- [Bessel function](https://en.wikipedia.org/wiki/Bessel_function)\n"
            "- [Electro-optic modulator](https://en.wikipedia.org/wiki/Electro-optic_modulator)\n"
            "- [Pockels effect](https://en.wikipedia.org/wiki/Pockels_effect)\n"
            "- [Phase modulation](https://en.wikipedia.org/wiki/Phase_modulation)\n"
            "- [Pound-Drever-Hall technique](https://en.wikipedia.org/wiki/Pound%E2%80%93Drever%E2%80%93Hall_technique)\n"
            "- E. D. Black, [An introduction to Pound-Drever-Hall laser frequency "
            "stabilization](https://doi.org/10.1119/1.1286663), Am. J. Phys. 69, 79 (2001)\n"
            "- B. E. A. Saleh and M. C. Teich, *Fundamentals of Photonics* (electro-optics chapter)")
        st.caption(
            "Idealizations: pure phase modulation (no residual amplitude modulation), "
            "a single-frequency laser, a single RF tone, and no loss.")


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

    init_session_state()
    beta, f0, N, width_frac, noise, color_peaks, fixed_xaxis, db_scale = render_sidebar()

    # |J_n(beta)|^2 for n = 0..N, shared by both figures and the table.
    Jn2 = sideband_intensities(beta, N)

    fig = build_figure(beta=beta, f0=f0, N=N, Jn2=Jn2, width_frac=width_frac,
                       noise=noise, color_peaks=color_peaks,
                       fixed_xaxis=fixed_xaxis, db_scale=db_scale, mode=mode)

    st.header("EOM Sideband Explorer")
    st.markdown(
        "An **electro-optic modulator (EOM)** shifts the phase of a laser beam back "
        "and forth, driven by an RF signal of frequency $f_0$. This creates new "
        "laser frequencies, called **sidebands**, spaced by $f_0$ above and below "
        "the original laser line (the **carrier**). The **modulation depth** "
        "$\\beta$ sets how much power moves from the carrier into the sidebands.")
    st.caption("Try it: drag the β slider in the sidebar and watch the carrier "
               "(order 0) shrink as the sidebands (±1, ±2, ...) grow. "
               "On a phone, open the sidebar with the arrow at the top left.")

    render_metrics(Jn2, N)

    with st.container(border=True):
        st.image(figure_png(fig), width="stretch")

    render_intensity_table(Jn2, f0, N)
    render_physics_details()

    st.caption("Source code on GitHub: "
               "https://github.com/sterafix/eom-sideband-explorer")


if __name__ == "__main__":
    main()
