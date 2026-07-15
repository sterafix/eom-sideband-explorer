"""Navigation entry point for the EOM Sideband Explorer.

This module only configures the page and wires up the sidebar navigation
across the app's pages, in the pedagogical order: what sidebands an EOM
makes (Sidebands), what to do with them for spectroscopy (FMS), and for
laser locking (PDH). All presentation logic lives in ``views/``; all physics
lives in ``core/``.

Run locally with ``streamlit run app.py``.
"""

import streamlit as st

from views import sidebands, fms, pdh


def main():
    """Configure the page and run the sidebar navigation."""
    st.set_page_config(page_title="EOM Sideband Explorer",
                       page_icon="〰️", layout="wide")

    # Each page module exposes a same-named `render` function, so the URL
    # pathname (which Streamlit would otherwise infer from that callable
    # name) must be given explicitly to keep the three pages distinct.
    pg = st.navigation([
        st.Page(sidebands.render, title="Sidebands", icon="〰️",
                url_path="sidebands", default=True),
        st.Page(fms.render, title="FMS Error Signal", icon="📈",
                url_path="fms"),
        st.Page(pdh.render, title="PDH Lock", icon="🔒",
                url_path="pdh"),
    ])
    pg.run()


if __name__ == "__main__":
    main()
