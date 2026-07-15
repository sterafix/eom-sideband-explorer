"""Streamlit page: FM spectroscopy (FMS) error signal.

Placeholder page. The physics core (``core/fms.py``, built on
``core/lineshapes.py`` and ``core/demod.py``) and the interactive controls
described in ``eom_explorer_expansion_plan.md`` land in a later step; this
stub only wires the page into navigation so the three-page sidebar (Sidebands
-> FMS -> PDH) is real and launchable now.
"""

import streamlit as st


def render():
    """Render the FMS page (currently a placeholder)."""
    st.header("FMS Error Signal")
    st.caption("Frequency modulation (FM) spectroscopy error signal, following "
               "Bjorklund et al., Appl. Phys. B 32, 145 (1983).")
    st.info("🚧 Under development — the FMS physics core and interactive "
            "controls are not implemented yet. See "
            "`eom_explorer_expansion_plan.md` for the plan.")
