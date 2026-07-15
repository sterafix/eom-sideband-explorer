"""Streamlit page: Pound-Drever-Hall (PDH) locking error signal.

Placeholder page. The physics core (``core/pdh.py``, reusing
``core/demod.py``'s ``mixed_signal`` and adding the Fabry-Perot cavity
reflection physics) and the interactive controls described in
``eom_explorer_expansion_plan.md`` land in a later step; this stub only
wires the page into navigation so the three-page sidebar (Sidebands -> FMS
-> PDH) is real and launchable now.
"""

import streamlit as st


def render():
    """Render the PDH page (currently a placeholder)."""
    st.header("PDH Lock")
    st.caption("Pound-Drever-Hall laser frequency stabilization error signal, "
               "following Black, Am. J. Phys. 69, 79 (2001).")
    st.info("🚧 Under development — the PDH physics core and interactive "
            "controls are not implemented yet. See "
            "`eom_explorer_expansion_plan.md` for the plan.")
