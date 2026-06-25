import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
import streamlit as st

# ----------------------------------------------------------------------
#  EOM Sideband Calculator  —  a little weekend project
#  Pure phase modulation:  E(t) = E0 exp(i[wc t + beta sin(wm t)])
#                               = E0 * sum_n J_n(beta) exp(i(wc + n wm)t)
#  -> spectral line at n*f0 has intensity J_n(beta)^2   (Jacobi-Anger)
# ----------------------------------------------------------------------

# --- generic illustrative constants (edit for your own device) ---
VPI0   = 20.0     # half-wave voltage [V] at reference wavelength
LAM0   = 1064.0   # reference wavelength [nm]  (common Nd:YAG line)
R_LOAD = 50.0     # RF input impedance [ohm]
NMAX   = 6        # max selectable sidebands

st.set_page_config(page_title="EOM Sideband Calculator",
                   page_icon="🔬", layout="wide")

# ---------- physics helpers ----------
def color_for_n(n):
    return {0:'black', 1:'#1f77b4', 2:'#2ca02c', 3:'#d62728',
            4:'#9467bd', 5:'#8c564b', 6:'#e377c2'}[abs(n)]

def beta_of(Vpeak, lam):
    Vpi = VPI0 * (lam / LAM0)                 # Vpi proportional to lambda
    return np.pi * Vpeak / Vpi, Vpi

def rf_dbm(Vpeak):
    if Vpeak <= 0:
        return None
    return 10.0 * np.log10(((Vpeak / np.sqrt(2))**2 / R_LOAD) / 1e-3)

def captured_power(beta, N):
    return float(sum(jv(n, beta)**2 for n in range(-N, N + 1)))

# ---------- sidebar controls ----------
st.sidebar.title("EOM parameters")
Vpeak = st.sidebar.slider("RF drive voltage  Vₚₑₐₖ [V]", 0.0, 40.0, 6.37, 0.1)
f0    = st.sidebar.slider("Modulation frequency  f₀ [MHz]", 50.0, 3000.0, 1000.0, 5.0)
lam   = st.sidebar.slider("Wavelength  λ [nm]", 200.0, 1600.0, 1064.0, 1.0)
N     = st.sidebar.slider("Number of sidebands", 1, NMAX, 2)
st.sidebar.markdown("---")
with st.sidebar.expander("Display options"):
    width_frac  = st.slider("Peak width", 0.003, 0.040, 0.012, 0.001, format="%.3f")
    noise       = st.slider("Noise level", 0.0, 0.030, 0.004, 0.001, format="%.3f")
    color_peaks = st.checkbox("Color-code peaks", True)
    fixed_xaxis = st.checkbox("Fix β-axis to 0–10", False)

beta, Vpi = beta_of(Vpeak, lam)

# ---------- styling to match the reference figure ----------
GREY = '0.45'
plt.rcParams.update({
    "axes.edgecolor": GREY, "axes.labelcolor": GREY,
    "xtick.color": GREY, "ytick.color": GREY,
    "axes.linewidth": 1.0, "font.size": 11,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
fig.subplots_adjust(wspace=0.32, top=0.86, bottom=0.13, left=0.07, right=0.93)

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
    if color_peaks:
        for c0 in ([0.0] if n == 0 else [-n * f0, n * f0]):
            sel = np.abs(freq - c0) < 3 * width
            ax1.plot(freq[sel], trace[sel], color=color_for_n(n), lw=1.5, zorder=2)
    ax1.axhline(h, ls='--', color='0.7', lw=0.8, zorder=0)

ax1.set_xlim(-span, span)
ax1.set_ylim(-0.06, 1.08)
ax1.set_xlabel('Optical frequency detuning [MHz]', fontweight='bold')
ax1.set_ylabel('Intensity [arb. units]', fontweight='bold')
ax1.set_title('Fig. 1: Optical spectrum', fontweight='bold', color='black', pad=12)

rf = rf_dbm(Vpeak)
rf_str = "—" if rf is None else f"{rf:.1f} dBm"
ax1.text(0.04, 0.95,
         f"$f_0$ = {f0:.0f} MHz\n"
         f"$\\beta$ = {beta:.2f} rad\n"
         f"$V_{{peak}}$ = {Vpeak:.2f} V\n"
         f"$V_\\pi$ = {Vpi:.1f} V\n"
         f"$\\lambda_{{test}}$ = {lam:.0f} nm\n"
         f"RF$_{{in}}$ = {rf_str}",
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
left, right = st.columns([4, 1])
with left:
    st.title("EOM Sideband Calculator")
    st.caption("A little side project for visualizing phase-modulation sidebands · "
               "intensity = $J_n(\\beta)^2$ (Jacobi–Anger expansion)")
with right:
    cap = captured_power(beta, N)
    st.metric("Captured power", f"{cap*100:.1f} %",
              help="Fraction of total optical power inside the displayed ±N sidebands. "
                   "Increase N at high β to capture more.")
    if cap < 0.98:
        st.warning(f"{(1-cap)*100:.0f}% of power is outside the view — raise N.")

st.pyplot(fig)
plt.close(fig)      # release the figure so reruns don't pile up in memory

with st.expander("ℹ️  Physics & model details"):
    st.markdown("A sinusoidally driven **electro-optic phase modulator** imprints "
                "$E(t)=E_0 e^{i[\\omega_c t+\\beta\\sin(\\omega_m t)]}$. "
                "The Jacobi–Anger expansion gives:")
    st.latex(r"E(t)=E_0\sum_{n=-\infty}^{\infty} J_n(\beta)\,"
             r"e^{i(\omega_c+n\omega_m)t}")
    st.markdown(r"so the line at detuning $n f_0$ carries intensity $J_n(\beta)^2$. "
                r"The modulation depth follows")
    st.latex(r"\beta=\pi\,\frac{V_{\text{peak}}}{V_\pi(\lambda)},"
             r"\qquad V_\pi\propto\lambda.")
    st.markdown(r"The carrier vanishes at the first zero of $J_0$, "
                r"$\beta\approx 2.405$.")
    st.caption("Constants: $V_{\\pi,0}=20.0$ V at $\\lambda_0=1064$ nm, RF into "
               "$50\\,\\Omega$ — generic illustrative numbers; tweak them at the "
               "top of `app.py` for your own device.")
st.caption("Hacked together for fun — no guarantees. Calibrate the constants to your own EOM.")
