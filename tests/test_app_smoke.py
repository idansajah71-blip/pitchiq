"""Headless smoke test: every app page must render without exceptions."""

from __future__ import annotations

import pathlib

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
PAGES = [
    "Front Page",
    "Shot Map",
    "Match Timeline",
    "xG Estimator",
    "Model Evaluation",
    "Beat the Model",
    "Team & Player Insights",
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


def test_estimator_slider_updates_distance_and_renders_cleanly() -> None:
    """Moving the shot location updates the visible distance output."""
    app = AppTest.from_file(str(APP), default_timeout=600)
    app.run()
    nav = app.sidebar.selectbox[0]
    nav.set_value("xG Estimator")
    app.run()
    assert not app.exception
    initial_distance = next(
        metric.value for metric in app.metric if metric.label == "Distance to goal"
    )
    sliders = app.slider  # depth slider is the first slider on the page
    sliders[0].set_value(104)
    app.run()
    assert not app.exception
    updated_distance = next(
        metric.value for metric in app.metric if metric.label == "Distance to goal"
    )
    assert initial_distance != updated_distance
