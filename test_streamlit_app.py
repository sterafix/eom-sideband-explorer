"""Smoke tests that run the whole Streamlit app headlessly.

The other suites call the physics and drawing functions directly, so they
never touch the Streamlit runtime. That left a gap: a Streamlit release can
deprecate an argument the app passes, and the app then shows the deprecation
notice to every visitor while all unit tests stay green. ``AppTest`` runs
``app.py`` as a real script, so anything Streamlit renders as an error or a
warning shows up here.
"""

import pytest
from streamlit.testing.v1 import AppTest

from app import PRESETS

TIMEOUT = 30    # seconds; the first run imports matplotlib and scipy


def _run(**state):
    """Run the app once with the given session-state values preset."""
    at = AppTest.from_file("app.py", default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def _assert_clean(at):
    """Fail if the run raised or showed anything but the app's own alerts."""
    assert not at.exception, [e.value for e in at.exception]
    # The truncation warning is the app's only intended st.warning; anything
    # else (for example a deprecation notice from Streamlit) is a regression.
    stray = [w.value for w in at.warning
             if not w.value.startswith("**Spectral truncation:**")]
    assert not stray, stray


def test_default_view_renders_without_errors_or_warnings():
    """The landing page runs cleanly and draws the figure."""
    at = _run()
    _assert_clean(at)
    assert at.get("image"), "the figure image was not rendered"


@pytest.mark.parametrize("label", list(PRESETS))
@pytest.mark.parametrize("scale", ["Linear", "dB"])
def test_every_preset_renders_on_both_scales(label, scale):
    """Each preset operating point runs cleanly on either y-axis scale."""
    beta, N = PRESETS[label]
    _assert_clean(_run(beta=beta, beta_input=beta, N=N, scale_choice=scale))
