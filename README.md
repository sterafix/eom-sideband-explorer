# EOM Sideband Explorer

A small interactive tool for visualizing the optical sideband spectrum of an
**electro-optic phase modulator**. Phase-modulating a laser at frequency *f₀*
with depth *β* produces a carrier plus sidebands at ±*n·f₀*, where each line's
intensity is *Jₙ(β)²* (Jacobi–Anger). Drag the sliders to see the spectrum and
the Bessel functions update side by side.

https://eom-sideband-explorer.streamlit.app/

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
