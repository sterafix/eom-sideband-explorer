import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import streamlit as st

# ----------------------------------------------------------------------
#  EOM Sideband Calculator
#  Pure phase modulation:  E(t) = E0 exp(i[wc t + beta sin(wm t)])
#                               = E0 * sum_n J_n(beta) exp(i(wc + n wm)t)
#  -> spectral line at n*f0 has intensity J_n(beta)^2   (Jacobi-Anger)
# ----------------------------------------------------------------------

NMAX = 6        # max selectable sideband orders to display

st.set_page_config(page_title="EOM Sideband Calculator",
                   page_icon="〰️", layout="wide")

# ---------- physics helpers ----------
def color_for_n(n):
    return {0:'black', 1:'#1f77b4', 2:'#2ca02c', 3:'#d62728',
            4:'#9467bd', 5:'#8c564b', 6:'#e377c2'}[abs(n)]

def captured_power(beta, N):
    return float(sum(jv(n, beta)**2 for n in range(-N, N + 1)))

# ---------- preset operating points (beta, displayed orders) ----------
PRESETS = {
    "Low β (0.5)":            (0.5,   2),
    "Moderate β (1.0)":       (1.0,   2),
    "Carrier null (2.405)":   (2.405, 3),
    "High β (3.0)":           (3.0,   4),
    "Deep modulation (5.0)":  (5.0,   6),
}

# session-state defaults must exist before the widgets are created
if "beta" not in st.session_state:
    st.session_state.beta = 1.0
if "N" not in st.session_state:
    st.session_state.N = 2

def apply_preset(b, n):
    st.session_state.beta = b
    st.session_state.N = n

# ---------- sidebar controls ----------
st.sidebar.title("EOM parameters")
beta = st.sidebar.slider("Modulation depth  β [rad]", 0.0, 10.0, step=0.005, key="beta")
f0   = st.sidebar.slider("Modulation frequency  f₀ [MHz]", 50.0, 3000.0, 1000.0, 5.0)
N    = st.sidebar.slider(
    "Displayed sideband orders", 1, NMAX, step=1, key="N",
    help="How many sideband orders are drawn on each side of the carrier. "
         "All orders exist physically; this only sets how many are shown.")

st.sidebar.markdown("**Presets**")
for label, (b, n) in PRESETS.items():
    st.sidebar.button(label, on_click=apply_preset, args=(b, n),
                      use_container_width=True)

st.sidebar.markdown("---")
with st.sidebar.expander("Display options"):
    width_frac  = st.slider("Peak width", 0.003, 0.040, 0.012, 0.001, format="%.3f")
    noise       = st.slider("Noise level", 0.0, 0.030, 0.004, 0.001, format="%.3f")
    color_peaks = st.checkbox("Color-code peaks", True)
    fixed_xaxis = st.checkbox("Fix β-axis to 0–10", False)

# ---------- styling to match the reference figure ----------
GREY = '0.45'
plt.rcParams.update({
    "axes.edgecolor": GREY, "axes.labelcolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.linewidth": 1.0, "font.size": 11,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150)
fig.subplots_adjust(wspace=0.18, top=0.86, bottom=0.13, left=0.07, right=0.93)

# ===================== Fig. 1: spectrum =====================
span  = (N + 0.8) * f0
freq  = np.linspace(-span, span, 6000)
width = width_frac * span
trace = np.zeros_like(freq)
for n in range(-N, N + 1):
    trace += jv(n, beta)**2 * np.exp(-0.5 * ((freq - n * f0) / width)**2)
trace += noise * np.abs(np.random.default_rng(0).normal(size=freq.shape))

ax1.plot(freq, trace, color='black', lw=0.9, zorder=1)
for n in range(0, N + 1):
    h = jv(n, beta)**2
    if h <= 1e-3:
        continue
    centers = [0.0] if n == 0 else [-n * f0, n * f0]
    for c0 in centers:
        if color_peaks:
            sel = np.abs(freq - c0) < 3 * width
            ax1.plot(freq[sel], trace[sel], color=color_for_n(n), lw=1.5, zorder=2)
        # order label above each peak
        order_lbl = "0" if n == 0 else (f"+{n}" if c0 > 0 else f"-{n}")
        ax1.annotate(order_lbl, (c0, h), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8,
                     fontweight='bold', color=color_for_n(n), zorder=4)
    ax1.axhline(h, ls='--', color='0.7', lw=0.8, zorder=0)

ax1.set_xlim(-span, span)
ax1.set_ylim(-0.06, 1.14)        # headroom for the order labels
ax1.set_xlabel('Optical frequency detuning [MHz]', fontweight='bold')
ax1.set_ylabel('Intensity [arb. units]', fontweight='bold')
ax1.set_title('Fig. 1: Optical spectrum', fontweight='bold', color='black', pad=12)

ax1.text(0.04, 0.95,
         f"$f_0$ = {f0:.0f} MHz\n"
         f"$\\beta$ = {beta:.2f} rad",
         transform=ax1.transAxes, va='top', ha='left', fontsize=10, color=GREY,
         bbox=dict(boxstyle='round', fc='white', ec='0.8'))

# ===================== Fig. 2: Bessel curves =====================
bmax  = 10.0 if fixed_xaxis else max(3.0, beta * 1.1)
bgrid = np.linspace(0, bmax, 800)
for n in range(0, N + 1):
    h = jv(n, beta)**2
    ax2.plot(bgrid, jv(n, bgrid)**2, color=color_for_n(n), lw=2.2,
             label=f"$|J_{{{n}}}|^2$")
    ax2.plot([0, beta], [h, h], ls='--', color='0.7', lw=0.8, zorder=0)
    ax2.scatter(beta, h, color=color_for_n(n), s=20, zorder=3)   # operating-point marker
ax2.axvline(beta, ls='--', color='0.7', lw=0.8, zorder=0)

ax2.set_xlim(0, bmax)
ax2.set_ylim(-0.06, 1.08)
ax2.set_xlabel('Modulation depth  $\\beta$ [rad]', fontweight='bold')
ax2.set_title('Fig. 2: Carrier/sideband ratio', fontweight='bold', color='black', pad=12)
# move the y-axis to the right, like the reference figure
ax2.yaxis.set_label_position('right')
ax2.yaxis.tick_right()
ax2.set_ylabel('rel. Intensity', fontweight='bold')
ax2.legend(title='Bessel Fct.', loc='upper right', frameon=False)

# ===================== render =====================
st.header("EOM Sideband Calculator")
st.caption("Phase-modulation sideband spectrum and its Bessel-function "
           "decomposition · line intensity $= J_n(\\beta)^2$ (Jacobi-Anger expansion)")

st.divider()

col_left, _ = st.columns([2, 3])
with col_left:
    cap = captured_power(beta, N)
    st.metric("Captured optical power", f"{cap*100:.1f} %",
              help="Fraction of the total optical power contained in the displayed "
                   "sideband orders. Raise the displayed-orders count at high beta "
                   "to account for the full spectrum.")
    if cap < 0.98:
        st.warning(f"**Spectral truncation:** {(1-cap)*100:.0f}% of the optical "
                   f"power falls outside the displayed orders. Increase the number "
                   f"of displayed sidebands to account for the full spectrum.")
    else:
        st.caption("The displayed orders account for essentially the full "
                   "optical power.")

st.pyplot(fig, use_container_width=True)
plt.close(fig)      # release the figure so reruns do not accumulate in memory

# ---------- exact sideband intensities (table) ----------
st.subheader("Sideband intensities")
orders = list(range(0, N + 1))
table = {
    "Order": ["0 (carrier)" if n == 0 else f"±{n}" for n in orders],
    "Detuning [MHz]": ["0" if n == 0 else f"±{n * f0:.0f}" for n in orders],
    "Intensity per line  Jₙ(β)²": [f"{jv(n, beta)**2:.4f}" for n in orders],
    "Power share (both ±n)": [
        f"{jv(n, beta)**2 * (1 if n == 0 else 2) * 100:.2f} %" for n in orders],
}
col_t, _ = st.columns([3, 2])
with col_t:
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Intensity is given per spectral line; the power share combines the "
               "+n and -n orders, so the shares sum to the captured optical power "
               "shown above.")

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
    st.latex(r"E(t)=E_0\sum_{n=-\infty}^{\infty} J_n(\beta)\,e^{\,i(\omega_c+n\omega_m)t}")
    st.markdown(
        r"The line at detuning $n f_0$ has amplitude $J_n(\beta)$ and therefore "
        r"intensity $J_n(\beta)^2$. Two consequences shape the figures:")
    st.markdown(
        r"- **Symmetry.** Since $J_{-n}(\beta)=(-1)^n J_n(\beta)$, the power "
        r"spectrum is symmetric about the carrier: the $\pm n$ orders have equal "
        r"intensity." "\n"
        r"- **Energy conservation.** $\sum_{n} J_n(\beta)^2 = 1$, so phase "
        r"modulation only redistributes power between carrier and sidebands. The "
        r"*Captured optical power* readout is the partial sum of $J_n(\beta)^2$ "
        r"over the displayed orders.")
    st.markdown(
        r"The carrier first vanishes at $\beta\approx 2.4048$, the first zero of "
        r"$J_0$. This carrier-null point is a standard laboratory reference for "
        r"measuring $V_\pi$: at the null, $\beta = 2.4048$ is known exactly, so "
        r"the half-wave voltage follows directly from the RF drive level,")
    st.latex(r"V_\pi = \frac{\pi\,V_\text{peak}}{2.4048}.")
    st.markdown(
        r"This in turn allows the modulation depth to be calculated for any "
        r"drive voltage.")
    st.markdown("**What the figures show**")
    st.markdown(
        r"- **Fig. 1** plots each order at $n f_0$ with height $J_n(\beta)^2$, "
        r"annotated with its order number. The ideal lines are mathematically "
        r"sharp; for visibility they are drawn as narrow Gaussian peaks on a small "
        r"synthetic noise floor (both adjustable under Display options)." "\n"
        r"- **Fig. 2** plots $|J_n(\beta)|^2$ versus modulation depth. The marker "
        r"at your chosen $\beta$ shows that every peak height in Fig. 1 is a slice "
        r"through these Bessel curves at that $\beta$.")
    st.markdown(
        r"You set $\beta$ (the peak phase deviation), the line spacing $f_0$, and "
        r"how many orders to display. On real hardware $\beta$ follows from the RF "
        r"drive via $\beta=\pi V_\text{peak}/V_\pi$, where $V_\pi$ is the device- "
        r"and wavelength-dependent half-wave voltage (not used in this idealized "
        r"simulation).")
    st.markdown(
        "**References and further reading**\n"
        "- [Jacobi-Anger expansion](https://en.wikipedia.org/wiki/Jacobi%E2%80%93Anger_expansion)\n"
        "- [Bessel function](https://en.wikipedia.org/wiki/Bessel_function)\n"
        "- [Electro-optic modulator](https://en.wikipedia.org/wiki/Electro-optic_modulator)\n"
        "- [Pockels effect](https://en.wikipedia.org/wiki/Pockels_effect)\n"
        "- [Phase modulation](https://en.wikipedia.org/wiki/Phase_modulation)\n"
        "- B. E. A. Saleh and M. C. Teich, *Fundamentals of Photonics* (electro-optics chapter)")
    st.caption(
        "Idealizations: pure phase modulation (no residual amplitude modulation), "
        "a single RF tone, and no loss.")

st.caption("Interactive model of electro-optic phase modulation · "
           "idealized pure phase modulator.")

st.caption("🛠️ Code's open on GitHub if you want to peek under the hood or build "
           "on it: https://github.com/sterafix/eom-sideband-explorer")