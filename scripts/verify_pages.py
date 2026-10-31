"""Deeper end-to-end verification: every page must render AND carry its key
content markers (buttons, metrics, sections, charts). Run directly:

    python scripts/verify_pages.py
"""

from __future__ import annotations

import sys
import pathlib

from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"

CHECKS: dict[str, dict[str, list[str]]] = {
    "Front Page": {"md": ["SHOT", "VOL.", "Pick a door"], "button": [], "metric": [], "selectbox": []},
    "Shot Map": {"md": [], "button": [], "metric": [], "selectbox": ["Team", "Player"]},
    "Match Timeline": {"md": [], "button": [], "metric": ["Minute"], "selectbox": ["Pick a match"]},
    "xG Estimator": {"md": [], "button": [], "metric": ["Distance to goal"], "selectbox": []},
    "Model Evaluation": {"md": ["ROC-AUC"], "button": [], "metric": [], "selectbox": []},
    "Beat the Model": {"md": ["unseen"], "button": [], "metric": [], "selectbox": []},
    "Team & Player Insights": {"md": ["Regression watchlist", "Sample-size control"], "button": [], "metric": [], "selectbox": []},
}


def main() -> int:
    failures: list[str] = []
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()

    md_all = " ".join(getattr(m, "value", "") or "" for m in at.markdown)
    buttons = [b.label for b in at.button]
    metrics = [m.label for m in at.metric]

    def report(page: str, ok: bool, detail: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {page:<22} {detail}")
        if not ok:
            failures.append(page)

    def sweep(page: str) -> None:
        at.sidebar.selectbox[0].set_value(page)
        at.run()
        if at.exception:
            report(page, False, f"exception: {at.exception[0].message}")
            return
        spec = CHECKS[page]
        md = "  ".join(getattr(m, "value", "") or "" for m in at.markdown)
        mlbls = [m.label for m in at.metric]
        sbls = [getattr(s, "label", "") for s in at.selectbox]
        blbls = [b.label for b in at.button]
        missing = (
            [f"md:{n}" for n in spec["md"] if n not in md]
            + [f"metric:{n}" for n in spec["metric"] if n not in mlbls]
            + [f"selectbox:{n}" for n in spec["selectbox"] if n not in sbls]
            + [f"button:{n}" for n in spec["button"] if n not in blbls]
        )
        counts = (
            f"md={len(at.markdown)} metrics={len(at.metric)} "
            f"buttons={len(at.button)} tables={len(at.dataframe)}"
        )
        report(page, not missing, counts + ("" if not missing else f" MISSING={missing}"))

    report("Front Page", (not at.exception) and "VOL." in md_all and len(buttons) >= 6,
           f"buttons={buttons} metrics={metrics}")
    if not at.exception:
        for page in CHECKS:
            if page != "Front Page":
                sweep(page)
    else:
        report("Front Page", False, "initial run exception")

    print("\nRESULT:", "ALL OK" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
