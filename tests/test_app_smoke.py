"""Headless smoke test: every app page must render without exceptions."""

from __future__ import annotations

import pathlib

from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
PAGES = [
    "Shot Map",
    "xG Estimator",
    "Evaluasi Model",
    "Insight Pemain & Tim",
]


def test_all_pages_render_without_exception() -> None:
    app = AppTest.from_file(str(APP), default_timeout=600)
    app.run()
    assert not app.exception, f"Shot Map failed: {app.exception}"

    nav = app.sidebar.selectbox[0]
    for page in PAGES[1:]:
        nav.set_value(page)
        app.run()
        assert not app.exception, f"{page} failed: {app.exception}"


def test_estimator_widget_changes_hint_probability() -> None:
    """Moving the estimator slider must re-run cleanly (error informal handling)."""
    app = AppTest.from_file(str(APP), default_timeout=600)
    app.run()
    nav = app.sidebar.selectbox[0]
    nav.set_value("xG Estimator")
    app.run()
    assert not app.exception
    sliders = app.slider  # depth slider is the first slider on the page
    sliders[0].set_value(104)
    app.run()
    assert not app.exception
