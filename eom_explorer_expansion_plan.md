# EOM Sideband Explorer: Expansion Plan (FMS, PDH)

## Purpose of this document

This is an implementation plan for extending the existing `eom-sideband-explorer`
Streamlit app (https://github.com/sterafix/eom-sideband-explorer) with two new
pages that simulate demodulated error signals: FM spectroscopy (FMS) and
Pound Drever Hall locking (PDH).

A third technique, modulation transfer spectroscopy (MTS), was considered
during planning but is explicitly out of scope for now. It is nonlinear
pump-probe physics, meaningfully different in kind from the linear,
single-pass treatment used for FMS and PDH, and would need its own modeling
effort. Do not implement an MTS page or an `mts.py` module as part of this
plan. If it becomes relevant later, it should get its own planning pass
rather than being retrofitted here.

The existing app already covers EOM sideband generation via Bessel functions
(carrier and sideband power vs. modulation index M, including presets such as
the carrier null at M ~ 2.405, RAM documented as an idealization, colorblind
safe colors, light/dark themes, MIT license, modular codebase with tests and
CI). This plan extends that same app and should follow its existing coding
style, plotting library, color scheme, and test conventions rather than
introducing new ones.

Before writing any code, Claude Code should inspect the current repo
structure (module layout, plotting library used, theme/color constants,
existing test setup) and adapt file paths below to match. The paths given
here are a proposal, not a hard requirement.

References:

- FMS math: Bjorklund, Levenson, Lenth, Ortiz, "Frequency Modulation (FM)
  Spectroscopy: Theory of Lineshapes and Signal to Noise Analysis," Appl.
  Phys. B 32, 145 to 152 (1983). Figures 3 and 4 of that paper are used
  below as regression test targets.
- PDH math: Black, "An introduction to Pound Drever Hall laser frequency
  stabilization," Am. J. Phys. 69, 79 (2001). This is the primary source
  for the PDH page; equation numbers below (Eq. 3.1, Eq. 3.3, Eq. 4.1,
  Eq. 4.2, and Appendix A/B) refer to this paper unless stated otherwise.
  Figures 6 and 7 of this paper are used below as regression test targets.
  Every formula quoted below was checked numerically against those two
  figures before being written into this plan, so they can be implemented
  as given rather than re-derived.

## Guiding principles

1. Physics and UI are separated. All lineshape math lives in pure, tested
   functions with no Streamlit calls. Pages import from the physics core and
   only handle sliders, layout, and plotting calls.
2. The two new pages (FMS, PDH) share one physics core module where the
   math actually overlaps (phase demodulation, Bessel sideband amplitudes).
   They do not share code where the physics is genuinely different
   (Lorentzian absorber for FMS, cavity reflection coefficient for PDH).
3. Every physics function gets at least one regression test against a known
   analytic result or a digitized reference curve. The Bjorklund paper
   figures are well suited for this because they are a citable, external
   ground truth, not just "the code agrees with itself."
4. Each page should work in both a small M (Bjorklund) limit and, where
   practical, an arbitrary M (Bessel) mode, consistent with the existing
   sideband explorer's treatment of M.
5. UI conventions (slider ranges, color scheme, light/dark theme, layout
   pattern, preset buttons) should match the existing sidebands page so the
   app reads as one product, not separate bolted together tools.

## Proposed repository structure

```
eom-sideband-explorer/
├── app.py                        # navigation entry point (st.Page / st.navigation)
├── core/
│   ├── bessel.py                 # existing: J_n(M) sideband amplitudes
│   ├── lineshapes.py             # NEW: shared Lorentzian delta, phi (FMS)
│   ├── demod.py                  # NEW: phase mixing, S1/S2/S3 helpers
│   ├── fms.py                    # NEW: Bjorklund small-M FMS signal model
│   └── pdh.py                    # NEW: cavity reflection + PDH error signal
├── pages/                        # or pages_src/, matching existing convention
│   ├── 1_Sidebands.py            # existing app, renamed/moved if needed
│   ├── 2_FMS_Error_Signal.py     # NEW
│   └── 3_PDH_Lock.py             # NEW
└── tests/
    ├── test_bessel.py            # existing
    ├── test_lineshapes.py        # NEW
    ├── test_fms_fig3.py          # NEW: regression against Bjorklund Fig. 3
    ├── test_fms_fig4.py          # NEW: regression against Bjorklund Fig. 4
    ├── test_pdh_fig6.py          # NEW: regression against Black Fig. 6
    └── test_pdh_fig7.py          # NEW: regression against Black Fig. 7
```

## Shared core module: `core/lineshapes.py`

Purpose: one place that defines the Lorentzian absorption/dispersion pair
used by FMS (and, approximately, near resonance, related to the PDH cavity
lineshape, though PDH uses its own cavity reflection coefficient in
`core/pdh.py` rather than this module directly).

Functions to implement:

- `delta(R, delta_peak=1.0)` returns `delta_peak / (R**2 + 1)`
- `phi(R, delta_peak=1.0)` returns `delta_peak * R / (R**2 + 1)`
- `R_of_omega(omega, Omega, delta_omega_fwhm)` returns the normalized
  detuning `(omega - Omega) / (delta_omega_fwhm / 2)`, i.e. Eq. (7) of
  Bjorklund et al.
- Optionally, `voigt_delta(R, ...)` and `voigt_phi(R, ...)` using the
  Faddeeva function (`scipy.special.wofz`) for a Doppler broadened line,
  useful for a more realistic FMS case later. Stretch goal, not required
  for first release.

These functions carry no notion of sidebands or modulation frequency. They
are the atomic building block everything else composes.

## Shared core module: `core/demod.py`

Purpose: turn a set of per-component (carrier, upper sideband, lower
sideband) attenuation/phase values into the demodulated error signal
components, and mix them by demodulation phase.

Functions to implement:

- `s1_component(delta_minus, delta_plus)` returns `delta_minus - delta_plus`
  (Eq. 4, cos(w_m t) term, using background subtracted Delta-delta values)
- `s2_component(phi_minus, phi_plus, phi_0)` returns
  `phi_plus + phi_minus - 2 * phi_0` (Eq. 4, sin(w_m t) term)
- `s3_magnitude(s1, s2)` returns `sqrt(s1**2 + s2**2)`
- `mixed_signal(s1, s2, theta)` returns `s1 * cos(theta) + s2 * sin(theta)`,
  implementing the "phase adjuster" mentioned in the paper but never plotted
  there. This is the function that should back a demodulation phase slider
  on the FMS and PDH pages.

All functions should accept numpy arrays so they vectorize over a detuning
axis without a Python loop.

## Page: FMS Error Signal (`pages/2_FMS_Error_Signal.py`)

### Physics source

`core/fms.py`, built directly on `core/lineshapes.py` and `core/demod.py`.
Implements the small modulation index (M much less than 1) limit as derived
in Bjorklund et al., Eqs. (1) to (7).

Required functions in `core/fms.py`:

- `fms_signals(R0, delta_R, delta_peak=1.0)` returns a tuple `(S1, S2, S3)`
  evaluated at carrier detuning `R0` for a fixed normalized sideband spacing
  `delta_R = omega_m / (delta_omega_fwhm / 2)`. Internally this calls
  `delta()` and `phi()` at `R0 - delta_R`, `R0`, and `R0 + delta_R`, then
  `s1_component`, `s2_component`, `s3_magnitude`.
- `fms_signal_mixed(R0, delta_R, theta, delta_peak=1.0)` wraps the above
  with `mixed_signal` for the phase adjuster slider.
- `fms_discriminant_slope(delta_R)` returns the analytic slope of S1 and S2
  at line center (R0 = 0), for the "how do I choose omega_m" panel. Closed
  forms:
  - S1 slope at R0=0: `4 * delta_R / (delta_R**2 + 1)**2`, maximized at
    `delta_R = 1/sqrt(3)`.
  - S2 slope at R0=0: `abs(2 * (1 - delta_R**2) / (1 + delta_R**2)**2 - 2)`,
    monotonically increasing, saturating at 2 for large delta_R.
  These two closed forms should each get a unit test checking the known
  maximum/asymptote.

### Optional arbitrary M extension

If time allows, a second function `fms_signals_arbitrary_M(R0, delta_R, M,
n_max=5)` that sums over Bessel sideband orders using the existing
`core/bessel.py` amplitudes, reducing to `fms_signals` in the M much less
than 1 limit. This connects the new page back to the existing Bessel based
sideband explorer and should be flagged as a stretch goal, not required for
first release.

### UI content

- Two sliders: `R0` range (e.g. minus 100 to 100, matching the paper's axes)
  is the x-axis, not a slider; the actual sliders are `delta_R` (modulation
  frequency relative to linewidth, e.g. 0.05 to 50, log scale) and,
  optionally, a demodulation phase `theta` (0 to 360 degrees, or 0 to 2 pi).
- Main plot: S1, S2, S3 vs R0, three traces or three small multiples,
  matching the layout of Bjorklund Fig. 4 (one row per delta_R would be too
  much for an interactive app; instead use a single delta_R value driven by
  slider, redrawn live, which is the natural interactive equivalent of the
  whole figure).
- A discriminant slope panel or annotation: show the local slope at R0=0
  for the current delta_R, and optionally a small secondary plot of slope
  vs delta_R with the current value marked, so the person can see where
  they sit relative to the optimum.
- Presets: "WMS limit" (delta_R around 0.1), "Optimal S1 slope" (delta_R =
  1/sqrt(3)), "Fully resolved sidebands" (delta_R = 10 or 20), mirroring the
  preset button pattern already used on the sidebands page.
- Short caption/expander explaining that this is the small M, single
  isolated Lorentzian feature, no background absorption limit, with a
  pointer to the PDH page for the cavity case.

### Tests

- `test_fms_fig3.py` and `test_fms_fig4.py`: for each delta_R value shown
  in the paper's Fig. 3 (0.05, 0.1, 0.4, 0.8, 1.6, 2.5, 3.0, 4.0) and Fig. 4
  (0.1, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 50.0), evaluate `fms_signals` over
  a dense R0 grid and assert known structural properties: S1, S2, S3 all
  zero at R0=0 for every delta_R; S3 always nonnegative; peak amplitudes
  approach 1 for large delta_R; S1 and S2 approach the derivative/second
  derivative relationship for small delta_R (can be checked by comparing to
  a finite difference derivative of `delta(R)` and `phi(R)` at delta_R to
  0.05).
- `test_lineshapes.py`: `delta(0) == delta_peak`, `phi(0) == 0`, both decay
  to zero as R goes to infinity, `phi` is odd, `delta` is even.

## Page: PDH Lock (`pages/3_PDH_Lock.py`)

### Physics source

`core/pdh.py`. PDH reuses the demodulation/phase-mixing pattern from FMS
but replaces the Lorentzian absorber with a Fabry-Perot cavity reflection
coefficient, and it needs its own sideband bookkeeping (Bessel powers
rather than the small-M FMS amplitudes), since Black's derivation is
carried through in terms of J0/J1 sideband powers from the start rather
than the M much less than 1 expansion Bjorklund uses.

**Normalized frequency convention.** Black plots everything against
"frequency (free spectral ranges)," i.e. ordinary frequency divided by
the cavity's free spectral range. Implement `core/pdh.py` in that same
normalized variable throughout, call it `x = f / delta_nu_fsr`, so that
one round trip in the cavity corresponds to a phase of `2 * pi * x`. This
sidesteps a unit-consistency trap in Eq. (3.1): Black writes
`exp(i * omega / delta_nu_fsr)`, which only makes sense once `omega` is
understood as already expressed in these free-spectral-range units, not
as a literal angular frequency in rad/s divided by a frequency in Hz.
Working in `x` directly avoids that ambiguity and matches the paper's own
plot axes, which makes the regression tests against Figs. 6 and 7
straightforward.

Required functions in `core/pdh.py`:

- `cavity_reflection(x, r)` implements Black's Eq. (3.1) for the
  symmetric, lossless cavity:
  `F(x) = r * (exp(2j * pi * x) - 1) / (1 - r**2 * exp(2j * pi * x))`,
  where `r` is the amplitude reflection coefficient of each mirror
  (0 to 1). This is the main-text formula and should be the default used
  by the page.
- `cavity_reflection_general(x, r1, t1, r2)` implements the general lossy,
  asymmetric two-mirror case from Appendix A:
  `F(x) = (-r1 + r2 * (r1**2 + t1**2) * exp(2j * pi * x)) /
  (1 - r1 * r2 * exp(2j * pi * x))`.
  Offer this as an "advanced" option behind an expander, not the default,
  since most users only need the symmetric case.
- `finesse_from_r(r)` returns `pi / (1 - r**2)`, Black's high-finesse
  approximation (Sec. IV.A), used to let the UI expose finesse as the
  primary slider instead of a raw reflectivity value, consistent with how
  the FMS page exposes `delta_R` instead of raw frequencies.
- `sideband_powers(beta, P0=1.0)` returns `(Pc, Ps)` where
  `Pc = J0(beta)**2 * P0` and `Ps = J1(beta)**2 * P0` (Sec. III.C), using
  the existing Bessel machinery from `core/bessel.py` if it already
  exposes `J0`/`J1`, otherwise `scipy.special.jv`.
- `pdh_error_signal(x, x_mod, r, beta, P0=1.0)` is the primary function
  and implements Black's general result, Eq. (3.3), which is exact (no
  slow/fast approximation baked in):
  ```
  F0 = cavity_reflection(x, r)
  Fp = cavity_reflection(x + x_mod, r)
  Fm = cavity_reflection(x - x_mod, r)
  C = F0 * conj(Fp) - conj(F0) * Fm
  Pc, Ps = sideband_powers(beta, P0)
  cos_component = 2 * sqrt(Pc * Ps) * Re(C)
  sin_component = 2 * sqrt(Pc * Ps) * Im(C)
  ```
  Returns `(cos_component, sin_component)`. The sine component is the
  usual PDH error signal in the fast-modulation, near-resonance regime;
  the cosine component dominates in the slow-modulation regime. Both are
  needed to reproduce Figs. 6 and 7 and to support the demodulation phase
  slider below.
- `pdh_error_signal_mixed(x, x_mod, r, beta, theta, P0=1.0)` wraps the
  above with `demod.mixed_signal` (reuse the same helper written for the
  FMS page) so the phase adjuster slider works identically on both pages.
- `pdh_frequency_discriminant(r, delta_nu_fsr, Pc, Ps)` implements Eq.
  (4.2), `D = -8 * sqrt(Pc * Ps) / delta_nu`, where
  `delta_nu = delta_nu_fsr / finesse_from_r(r)` is the cavity linewidth.
  This is the near-resonance slope of the error signal with respect to
  ordinary frequency deviation, and is the PDH analogue of the FMS
  discriminant slope panel.
- `pdh_optimum_modulation_depth()` returns the constant `1.08` (Appendix
  B), the modulation depth that maximizes `D` for fixed total power, with
  the corresponding sideband-to-carrier power ratio `Ps / Pc = 0.42`.
  Expose this as a preset button, mirroring the FMS "Optimal S1 slope"
  preset.
- Optional, if there is appetite for the noise panel to mirror Bjorklund's
  S/N section on the FMS side: `pdh_shot_noise_frequency_sensitivity(...)`
  implementing Eq. (5.1)-adjacent results from Sec. V.B. Treat as a
  stretch goal, not required for first release.

### UI content

- Sliders: finesse (primary control, mapped internally to `r` via
  `finesse_from_r`), modulation depth `beta`, modulation frequency
  expressed as `x_mod` (fraction of a free spectral range, log scale so
  both the slow-modulation and fast-modulation regimes from Figs. 6 and 7
  are reachable), demodulation phase `theta`.
- Main plot: PDH error signal (`pdh_error_signal_mixed`) vs. laser
  frequency `x`, scanned across at least one full free spectral range so
  the periodic structure and the characteristic dispersive S-curve at each
  resonance are both visible, matching the style of Black's Figs. 6 and 7.
- A note or annotation distinguishing the two regimes shown in Black's
  Figs. 6 and 7: slow modulation (`x_mod` much less than the cavity
  linewidth, cosine term dominates, resembles the direct derivative of the
  cavity lineshape) versus fast modulation near resonance (`x_mod` many
  linewidths but still much less than one free spectral range, sine term
  dominates, the standard PDH error signal shape).
- Secondary plot or annotation: `D` vs. `x_mod` or vs. `beta`, so the
  person can see how modulation frequency and depth choice trade off
  against discriminant slope, with the optimum `beta = 1.08` marked.
- A toggle or warning annotation for the sideband resonance condition
  (`x_mod` close to 1, i.e. modulation frequency close to a multiple of
  the free spectral range), a known practical PDH pitfall worth calling
  out explicitly.
- Presets: "typical PDH setup" (moderate finesse, `x_mod` in the fast
  near-resonance regime), "slow modulation / WMS-like" (reproduces Fig. 6
  conditions), "optimum modulation depth" (`beta = 1.08`), "sideband
  resonance pitfall" (`x_mod` near 1).
- Caption noting the default uses the symmetric, lossless cavity
  approximation (Eq. 3.1), with the general lossy/asymmetric case
  (Appendix A) available as an advanced option.

### Tests

- `test_pdh_fig6.py`: at finesse about 500 (`r = sqrt(1 - pi/500)`) and
  `x_mod` about `5e-4` (half a linewidth, matching Black's Fig. 6
  caption), assert the cosine component dominates the sine component in
  peak magnitude near resonance, and that the signal has the antisymmetric
  derivative-like shape (odd about `x = 1`, single zero crossing at exact
  resonance).
- `test_pdh_fig7.py`: at the same finesse and `x_mod` about `0.04` (about
  20 linewidths, roughly 4 percent of a free spectral range, matching
  Black's Fig. 7 caption), assert the sine component dominates near
  resonance and shows the characteristic three-feature PDH shape (central
  dispersive feature plus two sideband-reflection shoulders), and that the
  error signal is exactly zero at `x = 1` for every finesse and `x_mod`
  (on-resonance null, a direct consequence of `F(1, r) = 0`).
- `test_pdh.py` (general): `cavity_reflection(x, r)` reduces to `0` at
  integer `x` (exact resonance) for any `r`; `finesse_from_r` reproduces
  the textbook relation for a couple of known `r` values; `sideband_powers`
  satisfies `Pc + 2 * Ps` approaches `P0` for small `beta`; the exact
  `pdh_error_signal` reduces to the slow-modulation and fast-modulation
  limits described in Black Secs. IV.A and IV.B when `x_mod` is chosen
  deep in each regime (use this as an internal consistency check between
  the general Eq. 3.3 formula and the two limiting cases, not just against
  the figures).

## Navigation and shared UI

- Use whichever Streamlit navigation mechanism the existing repo already
  uses (`st.navigation`/`st.Page`, or the `pages/` folder auto-detection
  convention). Do not introduce a second mechanism.
- All three pages (Sidebands, FMS, PDH) should sit in the same sidebar
  navigation, in that order, since it mirrors the pedagogical progression
  from "what sidebands does an EOM make" to "what do I do with them."
  Keep everything under the existing repository, URL, and license (MIT).
- Reuse existing color scheme, theme handling (light/dark), and any shared
  plotting helper functions rather than duplicating styling code per page.

## Suggested implementation order

1. `core/lineshapes.py` and `core/demod.py`, with tests. These are the
   foundation the FMS page depends on and are the easiest to verify
   independently. `core/demod.py`'s `mixed_signal` helper is also reused
   directly by the PDH page.
2. `core/fms.py` plus the FMS page, since it directly reproduces a
   published, citable figure (Bjorklund Fig. 3 and Fig. 4), which makes
   correctness verification straightforward before moving on.
3. `core/pdh.py` plus the PDH page, reusing `demod.py`'s `mixed_signal`
   and adding the cavity reflection physics from Black (2001). This is
   likely the highest practical value page for QUBIG's customers. It
   reproduces Black's Fig. 6 and Fig. 7 as its own regression targets, the
   same pattern used for the FMS page against Bjorklund.
4. Optional stretch goals, in any order once the above is stable: arbitrary
   M Bessel extension for the FMS page (reusing `core/bessel.py`), RAM
   modeling on the FMS page, the general lossy/asymmetric cavity case
   (Appendix A of Black) as an advanced option on the PDH page, the PDH
   shot noise sensitivity panel (Black Sec. V.B).

## Open questions for Claude Code to resolve by inspecting the repo

- Which plotting library does the existing app use (matplotlib, plotly,
  altair, or Streamlit's native chart functions), and do the new pages
  need to match it exactly.
- What testing framework and CI configuration already exist, so the new
  tests slot into the same pattern.
- Whether `scipy` is already a dependency (needed for `sideband_powers` in
  `core/pdh.py` if `core/bessel.py` does not already expose a usable
  `J0`/`J1`, and useful more generally if complex exponentials or special
  functions are handled via scipy elsewhere in the app).
- Whether the existing repo has a shared theme/color constants module that
  the new pages should import from directly.
