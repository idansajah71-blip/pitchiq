"""Headless smoke test: every app page must render without exceptions."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

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
    """In-process AppTest sweep across every page."""
    app = AppTest.from_file(str(APP), default_timeout=600)
    app.run()
    assert not app.exception, f"Front Page failed: {app.exception}"

    nav = app.sidebar.selectbox[0]
    for page in PAGES[1:]:
        nav.set_value(page)
        app.run()
        assert not app.exception, f"{page} failed: {app.exception}"


def test_all_pages_via_subprocess() -> None:
    """Isolated-interpreter sweep across every page.

    pytest's in-process environment has been observed to mask Streamlit page
    exceptions (page exceptions recorded but the sweep still passing), so this
    test mirrors a real browser session in a separate interpreter: navigate
    each sidebar page, rerun, and fail loudly on any script exception.
    """
    script = f"""\
import sys
from streamlit.testing.v1 import AppTest

at = AppTest.from_file({str(APP)!r}, default_timeout=600)
at.run()
if at.exception:
    print(f"Front Page failed: {{at.exception[0].message}}", file=sys.stderr)
    sys.exit(1)
for page in {PAGES[1:]!r}:
    at.sidebar.selectbox[0].set_value(page)
    at.run()
    if at.exception:
        print(f"{{page}} failed: {{at.exception[0].message}}", file=sys.stderr)
        sys.exit(1)
"""
    holder = pathlib.Path(tempfile.gettempdir()) / "_shotworth_smoke_sub.py"
    holder.write_text(script, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(holder)],
        cwd=str(ROOT),
        timeout=600,
        capture_output=True,
        text=True,
    )
    holder.unlink(missing_ok=True)
    assert proc.returncode == 0, (
        f"page sweep crashed (rc={proc.returncode})\n{proc.stderr or proc.stdout}"
    )


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
