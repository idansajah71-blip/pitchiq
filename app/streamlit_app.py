"""ShotWorth — Expected Goals Intelligence (Streamlit UI).

Sportswear-poster design language: giant display typography, diagonal
navy/green bands, dashed stat tiles, numbered sections and real inline-SVG
icons (no emoji). Pages: shot map, xG estimator, model audit, team insight.
All numbers come from the saved pipeline (train.py + reports); StatsBomb xG is
a benchmark only, never an input feature.

Run from the project root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR, FEATURES_PARQUET, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TEST_SPLIT_SEED
from src.features import FEATURE_COLUMNS, distance_to_goal, shot_angle_deg

MIN_SHOTS_DEFAULT = 15
MIN_ASSISTS = 10
ASSETS_DIR = ROOT / "app" / "assets"

REPO_URL = "https://github.com/idansajah71-blip/shotworth"
DEMO_URL = "https://shotworth-pitchiq-xg.streamlit.app"

CREDIT = (
    "Data: StatsBomb Open Data (via statsbombpy). ShotWorth is an independent "
    "hackathon project and is not affiliated with StatsBomb."
)

page_titles: dict[str, str] = {
    "Front Page": "ShotWorth — every shot has a price",
    "Shot Map": "Shot Map — the quality of every shot",
    "Match Timeline": "Match Timeline — the xG race, replayed minute by minute",
    "xG Estimator": "xG Estimator — live goal probability",
    "Model Evaluation": "Model Evaluation — how honest is it",
    "Beat the Model": "Beat the Model — can you out-score it",
    "Team & Player Insights": "Team & Player Insights — leaderboards & recommendations",
}

# Poster configuration per page: index letter, display words, subtitle.
page_posters: dict[str, dict[str, str]] = {
    "Front Page": dict(
        index="00", word_1="SHOT", word_2="WORTH",
        subtitle="12,241 real shots priced by a transparent model, built on 479 "
                 "matches of StatsBomb Open Data — pick a door and step inside.",
        asset="hero_front.png",
    ),
    "Shot Map": dict(
        index="A", word_1="SHOT", word_2="MAP",
        subtitle="Every point is one shot, attacks facing the goal on the right. "
                 "Size & colour = xG: the bigger and the greener, the better the chance.",
        asset="hero_shotmap.png",
    ),
    "Match Timeline": dict(
        index="B", word_1="MATCH", word_2="TIMELINE",
        subtitle="Replay any of the 479 matches like a live broadcast: the xG race "
                 "builds minute by minute, goals land on the curve, and each chance "
                 "is priced the moment it happens.",
        asset="hero_timeline.png",
    ),
    "xG Estimator": dict(
        index="C", word_1="xG", word_2="ESTIMATOR",
        subtitle="Place any shot, in any situation — the goal probability is scored "
                 "in real time by the same model trained on real matches.",
        asset="hero_estimator.png",
    ),
    "Model Evaluation": dict(
        index="D", word_1="MODEL", word_2="AUDIT",
        subtitle="A match-grouped holdout with no leakage, probability metrics that "
                 "do not fake accuracy, and StatsBomb xG scored on identical data.",
        asset="hero_evaluation.png",
    ),
    "Beat the Model": dict(
        index="E", word_1="BEAT", word_2="MODEL",
        subtitle="Five unseen shots from the model's untouched holdout. Price them "
                 "yourself, then log loss settles it — you vs the model, same data.",
        asset="hero_quiz.png",
    ),
    "Team & Player Insights": dict(
        index="F", word_1="TEAM", word_2="INSIGHT",
        subtitle="Leaderboards scoped by player–team–competition, zone patterns and "
                 "manager recommendations with labelled assumptions — not causal claims.",
        asset="hero_insights.png",
    ),
}

# --- Design tokens ------------------------------------------------------------
INK = "#0b0e14"
NAVY = "#101a2c"
NAVY_2 = "#152944"
BLUE_WEDGE = "rgba(27, 79, 156, 0.5)"
PANEL = "#111a26"
LINE = "#26364a"
GREEN = "#2ec27e"
GREEN_DIM = "rgba(46, 194, 126, 0.14)"
RED = "#ff4655"
AMBER = "#f59e0b"
TEXT = "#e6edf3"
TEXT_DIM = "#93a4b5"

DISPLAY = "'Barlow Condensed','Oswald','Arial Narrow',Impact,sans-serif"
BODY = "'Inter','Source Sans Pro',system-ui,sans-serif"
BLUE_DEEP = "#1b4f9c"

XG_SCALE = [
    [0.0, "#7f1d1d"],
    [0.15, "#b91c1c"],
    [0.35, "#ea580c"],
    [0.55, "#d9a406"],
    [0.75, "#65a30d"],
    [1.0, "#16a34a"],
]
HOVERLABEL = dict(bgcolor="#0e1520", bordercolor="#26364a", font=dict(color=TEXT, size=13))

PITCH_LINE = "rgba(230, 237, 243, 0.62)"

PITCH_SHAPES = [
    dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=PITCH_LINE, width=2), layer="below"),
    dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=PITCH_LINE, width=1.4), layer="below"),
    dict(type="circle", x0=52, y0=32, x1=68, y1=48, line=dict(color=PITCH_LINE, width=1.4), layer="below"),
    dict(type="circle", x0=59.4, y0=39.4, x1=60.6, y1=40.6, line=dict(color=PITCH_LINE, width=1.4), layer="below"),
    dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=PITCH_LINE, width=1.4), layer="below"),
    dict(type="rect", x0=114, y0=34, x1=120, y1=46, line=dict(color=PITCH_LINE, width=1.4), layer="below"),
    dict(type="path", path="M 108.5 29.5 A 12 12 0 0 1 108.5 50.5",
         line=dict(color=PITCH_LINE, width=1.4), layer="below"),
    dict(type="circle", x0=106.4, y0=39.4, x1=107.6, y1=40.6, line=dict(color=PITCH_LINE, width=1.4), layer="below"),
    dict(type="line", x0=120, y0=36, x1=120, y1=44, line=dict(color="#ffffff", width=4), layer="below"),
]


# --- Inline SVG icons (stroke=currentColor, no emoji) -------------------------
ICONS: dict[str, str] = {
    "pitch": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
        '<rect x="2.5" y="4.5" width="19" height="15" rx="1.5"/>'
        '<line x1="12" y1="4.5" x2="12" y2="19.5"/>'
        '<circle cx="12" cy="12" r="2.6"/>'
        '<rect x="17.5" y="9" width="4" height="6"/></svg>'
    ),
    "target": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
        '<circle cx="12" cy="12" r="8.5"/><circle cx="12" cy="12" r="4.5"/>'
        '<circle cx="12" cy="12" r="0.8" fill="currentColor"/></svg>'
    ),
    "chart": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
        '<path d="M3.5 20.5h17"/><path d="M4.5 15.5l4-5 3.5 3 5.5-6.5"/>'
        '<circle cx="8.5" cy="10.5" r="0.7" fill="currentColor"/>'
        '<circle cx="12" cy="13.5" r="0.7" fill="currentColor"/></svg>'
    ),
    "bolt": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
        '<path d="M13 2.5L5 13.5h5l-1 8 8-11h-5l1-8z"/></svg>'
    ),
    "arrow": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" '
        'stroke-linecap="round" stroke-linejoin="round" width="14" height="14">'
        '<path d="M4.5 12h15"/><path d="M13.5 6l6 6-6 6"/></svg>'
    ),
    "github": (
        '<svg viewBox="0 0 24 24" fill="currentColor" width="16" height="16">'
        '<path d="M12 2C6.48 2 2 6.58 2 12.25c0 4.53 2.87 8.37 6.84 9.72.5.1.68-.22.68-.49'
        ' 0-.24-.01-.87-.01-1.7-2.78.62-3.37-1.37-3.37-1.37-.45-1.18-1.11-1.5-1.11-1.5'
        '-.91-.64.07-.62.07-.62 1 .07 1.53 1.05 1.53 1.05.89 1.57 2.34 1.11 2.91.85'
        '.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75'
        '-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05a9.36 9.36 0 0 1 5 0c1.91-1.33'
        ' 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94'
        '-2.34 4.8-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.6.69.49'
        'A10.02 10.02 0 0 0 22 12.25C22 6.58 17.52 2 12 2z"/></svg>'
    ),
    "users": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
        '<circle cx="9" cy="8" r="3.2"/><path d="M2.5 20c0-3.3 2.9-5.5 6.5-5.5s6.5 2.2 6.5 5.5"/>'
        '<circle cx="17.5" cy="9.5" r="2.6"/><path d="M16.5 14.7c2.9.2 5 2.1 5 4.8"/></svg>'
    ),
    "zone": (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
        'stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
        '<rect x="3" y="3" width="18" height="18" rx="1.5"/>'
        '<path d="M3 12h18M12 3v18"/><rect x="15" y="15" width="6" height="6"/>'
        '<path d="M15 15l6 6M21 15l-6 6"/></svg>'
    ),
}


def icon(name: str, size: int = 16) -> str:
    """Return an inline SVG icon sized for inline use."""
    return ICONS[name].replace('width="16" height="16"', f'width="{size}" height="{size}"')


BODY_PARTS = ["Right Foot", "Left Foot", "Head", "Other"]
SHOT_TYPES = ["Open Play", "Free Kick"]


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model...")
def load_model() -> dict[str, Any]:
    """Load the artifact: {model_name, pipeline, calibrator, feature_columns, metadata}."""
    return joblib.load(MODELS_DIR / "best_model.joblib")


@st.cache_data(show_spinner="Loading shots...")
def load_shots() -> pd.DataFrame:
    """Load features and score every shot with the saved pipeline."""
    features = pd.read_parquet(FEATURES_PARQUET)
    artifact = load_model()
    probabilities = artifact["pipeline"].predict_proba(features[FEATURE_COLUMNS])[:, 1]
    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        probabilities = calibrator.predict(probabilities)
    features["shotworth_xg"] = probabilities
    return features


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    """Load one of the saved report CSVs."""
    return pd.read_csv(REPORTS_DIR / name)


@st.cache_data(show_spinner=False)
def load_markdown(name: str) -> str:
    """Load one of the saved report markdown files."""
    return (REPORTS_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Design system helpers
# ---------------------------------------------------------------------------


def global_css() -> None:
    """Inject fonts + component styling once per run."""
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {{ color-scheme: dark; }}

/* Balanced composition: constrain and center the main column */
div[data-testid="stMainBlockContainer"] {{
    max-width: 1220px;
    margin: 0 auto;
    padding-top: 2.4rem;
    padding-left: 2rem;
    padding-right: 2rem;
}}

/* Strip Streamlit framework chrome (menu, toolbar, footer, decorations) */
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, div[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stToolbar"], [data-testid="stDeployButton"] {{ visibility: hidden; }}

/* Splash atmosphere: every page lives in the same deep gradient world */
div[data-testid="stMain"] {{
    background:
        radial-gradient(1100px 520px at 82% -12%, {tint(BLUE_DEEP, 0.40)}, transparent 70%),
        radial-gradient(900px 460px at -10% 115%, {tint(GREEN, 0.16)}, transparent 70%),
        linear-gradient(150deg, {NAVY} 0%, {INK} 82%);
}}

/* Poster hero band */
.piq-hero {{
    position: relative; overflow: hidden;
    background: linear-gradient(102deg, {NAVY} 0%, {NAVY_2} 46%, rgba(23,94,54,0.62) 100%);
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 22px 26px 18px 26px;
    margin-bottom: 14px;
}}
.piq-wedge {{
    position: absolute; top: -55%; right: 3%;
    width: 250px; height: 190%;
    background: linear-gradient(180deg, rgba(46,194,126,0.26), rgba(46,194,126,0));
    transform: rotate(16deg);
}}
.piq-wedge2 {{
    position: absolute; top: -30%; right: 20%;
    width: 180px; height: 160%;
    background: linear-gradient(180deg, {BLUE_WEDGE}, rgba(27,79,156,0));
    transform: rotate(-13deg);
}}
.piq-hero .edge {{
    position: absolute; top: 0; bottom: 0; left: 0; width: 3px;
    background: linear-gradient(180deg, {GREEN}, transparent);
}}
.piq-node {{
    position: absolute; width: 30px; height: 30px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    background: {GREEN_DIM}; border: 1.4px dashed rgba(46,194,126,0.55);
    color: {GREEN};
}}
.piq-hero-img {{
    position: absolute; right: 14px; bottom: -6px; height: 142px;
    filter: drop-shadow(0 10px 22px rgba(0,0,0,0.55));
    pointer-events: none;
}}
.piq-index {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 25px; height: 25px; border-radius: 50%;
    border: 1.5px solid {TEXT_DIM}; color: {TEXT_DIM};
    font: 600 12px/1 {BODY}; vertical-align: middle; margin-right: 9px;
}}
.piq-word {{
    display: block;
    font-family: {DISPLAY}; font-weight: 700;
    font-size: clamp(44px, 5.4vw, 72px);
    line-height: 0.94; letter-spacing: 1.5px;
    text-transform: uppercase; color: {TEXT};
    margin: 8px 0 10px 0; white-space: nowrap;
}}
.piq-word .ghost {{
    color: transparent;
    -webkit-text-stroke: 1.3px rgba(230,237,243,0.40);
    margin-left: 0.18em;
}}
.piq-sub {{
    font: 400 13px/1.5 {BODY}; color: {TEXT_DIM};
    max-width: 560px; margin: 0 0 8px 0;
}}
.piq-crumbs {{ margin-top: 4px; }}
.piq-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 11px; margin-right: 7px;
    border: 1px solid {LINE}; border-radius: 999px;
    background: rgba(255,255,255,0.03);
    color: {TEXT_DIM}; font: 600 10.5px/1.9 {BODY};
    letter-spacing: 0.14em; text-transform: uppercase;
}}
.piq-chip svg {{ color: {GREEN}; }}
.piq-chip.solid {{ border-color: rgba(46,194,126,0.55); color: {GREEN}; }}

/* Section headings: letter circle + spaced caps + rule */
.piq-section {{
    display: flex; align-items: center; gap: 12px;
    margin: 16px 0 8px 0;
}}
.piq-section .idx {{
    flex: 0 0 auto; width: 26px; height: 26px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    background: {GREEN_DIM}; color: {GREEN};
    font: 700 12px/1 {DISPLAY}; border: 1px solid rgba(46,194,126,0.4);
}}
.piq-section .t {{
    flex: 0 0 auto; white-space: nowrap;
    font: 700 12px/1.6 {BODY}; letter-spacing: 0.16em;
    text-transform: uppercase; color: {TEXT};
}}
.piq-section .t svg {{ color: {GREEN}; vertical-align: -2px; }}
.piq-section .rule {{ flex: 1 1 auto; height: 1px; background: {LINE}; }}

/* Metric tiles: dashed border, display-type value, red micro tag */
div[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {PANEL} 0%, #0f1622 100%);
    border: 1.5px dashed {LINE};
    border-radius: 12px;
    padding: 14px 16px 10px 16px;
}}
div[data-testid="stMetricLabel"] p {{
    margin-bottom: 2px;
    font: 700 10px/1.7 {BODY} !important;
    letter-spacing: 0.14em; text-transform: uppercase; color: {TEXT_DIM} !important;
}}
div[data-testid="stMetricValue"] {{ font-family: {DISPLAY}; font-weight: 700; letter-spacing: 1px; }}
div[data-testid="stMetricDelta"] svg {{ color: {GREEN} !important; }}

/* Front page: open-air splash composition (no hero band) */
.piq-page-head {{ margin-top: 4px; }}
.piq-page-brand {{ display: flex; align-items: center; gap: 10px; color: {TEXT_DIM};
    font: 600 11px/2.2 {BODY}; letter-spacing: .24em; text-transform: uppercase; margin: 6px 0 0 0; }}
.piq-giant {{ font-family: {DISPLAY}; font-weight: 700; font-size: clamp(64px, 8.6vw, 112px);
    line-height: .94; letter-spacing: 2px; color: {TEXT}; text-transform: uppercase;
    margin: 10px 0 0 0; white-space: nowrap; }}
.piq-giant .ghost {{ color: transparent; -webkit-text-stroke: 1.6px rgba(230,237,243,.5); }}
.piq-rule-sweep {{ height: 3px; width: 130px; background: {GREEN}; border-radius: 2px;
    margin: 16px 0 12px 0; }}
.piq-page-sub {{ font: 400 15.5px/1.65 {BODY}; color: {TEXT_DIM}; max-width: 640px; margin: 0 0 8px 0; }}
.piq-page-stats {{ display: flex; gap: 46px; flex-wrap: wrap; margin: 22px 0 4px 0; }}
.piq-page-stats .cell .num {{ font-family: {DISPLAY}; font-weight: 700;
    font-size: clamp(44px, 5vw, 62px); line-height: 1; color: {TEXT}; }}
.piq-page-stats .cell .lbl {{ font: 600 9.5px/2.4 {BODY}; letter-spacing: .18em;
    text-transform: uppercase; color: {TEXT_DIM}; }}

/* Door index chips */
.piq-door-toprow {{ display: flex; align-items: center; justify-content: space-between;
    margin: 2px 0 2px 0; }}
.piq-door-idx {{ display: inline-flex; align-items: center; min-width: 30px; height: 24px;
    padding: 0 9px; border-radius: 999px; border: 1.4px solid {tint(GREEN, 0.45)};
    color: {GREEN}; font: 700 12px/1 {DISPLAY}; letter-spacing: .1em; }}
.piq-door-arrow svg {{ color: {TEXT_DIM}; }}

/* Landing door tiles */
[class*="st-key-door-card"] {{
    border: 1.4px dashed {LINE};
    border-radius: 16px;
    background: linear-gradient(150deg, rgba(13,17,26,0.62) 0%, rgba(13,17,26,0.42) 100%);
    padding: 14px 18px 12px 18px;
    margin-bottom: 4px;
    transition: transform .32s cubic-bezier(.34,1.56,.64,1), border-color .2s ease, box-shadow .32s ease;
}}
[class*="st-key-door-card"]:hover {{
    border-color: {tint(GREEN, 0.55)};
    transform: translateY(-4px);
    box-shadow: 0 18px 38px rgba(0,0,0,0.5);
}}
[class*="st-key-door-card"] button {{
    width: 100%;
    background: transparent;
    border: none;
    color: {TEXT};
    font-family: {DISPLAY}; font-weight: 700; font-size: 24px;
    letter-spacing: 0.09em; text-transform: uppercase;
    padding: 8px 6px 4px 6px;
}}
[class*="st-key-door-card"] button:hover {{
    color: {GREEN};
}}
[class*="st-key-door-card"] button p {{ font-size: inherit !important; font-family: inherit !important; }}

/* Poster panels around key charts */
.st-key-map-panel, .st-key-goal-panel {{
    background: linear-gradient(118deg, #0d1420 0%, {NAVY} 48%, rgba(28,94,54,0.4) 115%);
    border: 1px solid {LINE};
    border-radius: 14px;
    padding: 12px 10px 8px 10px;
    margin-bottom: 2px;
}}

/* Dataframe panels */
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; }}

/* Pill buttons (sidebar) */
.piq-pill {{
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 14px; border-radius: 999px; margin: 3px 6px 3px 0;
    border: 1.4px solid {LINE}; background: {PANEL};
    color: {TEXT} !important; text-decoration: none !important;
    font: 600 10.5px/2 {BODY}; letter-spacing: 0.14em; text-transform: uppercase;
    transition: border-color .15s ease, color .15s ease;
}}
.piq-pill:hover {{ border-color: {GREEN}; color: {GREEN} !important; }}
.piq-pill svg {{ color: {GREEN}; }}
.piq-pill.solid {{
    background: {GREEN}; border-color: {GREEN}; color: {INK} !important;
}}
.piq-pill.solid svg {{ color: {INK}; }}
.piq-pill.solid:hover {{ background: #37d68f; color: {INK} !important; }}

/* Sidebar brand */
.piq-brand {{ display: flex; align-items: center; gap: 11px; margin: 6px 0 2px 0; }}
.piq-brand .mark {{ flex: 0 0 auto; }}
.piq-brand .name {{
    font-family: {DISPLAY}; font-weight: 700; letter-spacing: 0.28em;
    font-size: 19px; color: {TEXT}; text-transform: uppercase;
}}
.piq-brand .tag {{ font: 500 10px/1.5 {BODY}; letter-spacing: 0.18em; color: {GREEN}; text-transform: uppercase; }}

/* Expander polish */
div[data-testid="stExpander"] {{
    border: 1px solid {LINE} !important; border-radius: 12px !important;
    background: {PANEL} !important;
}}

/* Poster footer band */
.piq-band {{
    position: relative; overflow: hidden;
    border: 1px solid {LINE}; border-radius: 14px;
    background: linear-gradient(92deg, {NAVY} 0%, #0e1622 60%, rgba(28,94,54,0.25) 100%);
    padding: 16px 20px; margin-top: 18px;
}}
.piq-band .ghost {{
    position: absolute; right: 14px; top: 50%; transform: translateY(-50%);
    font-family: {DISPLAY}; font-weight: 700; letter-spacing: 2px;
    font-size: 44px; color: rgba(230,237,243,0.05);
    text-transform: uppercase; white-space: nowrap; pointer-events: none;
}}
.piq-band .lead {{
    font: 700 11px/1.7 {BODY}; letter-spacing: 0.16em; text-transform: uppercase;
    color: {TEXT};
}}
.piq-band .sub {{ font: 400 11.5px/1.6 {BODY}; color: {TEXT_DIM}; max-width: 640px; }}

/* Motion system: entrances animate on page mount only (React keeps unchanged
   DOM nodes across reruns, so slider drags do not replay them). */
@keyframes piqRise {{ from {{ opacity: 0; transform: translateY(16px) scale(.985); }} to {{ opacity: 1; transform: none; }} }}
@keyframes piqFade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
@keyframes piqSweep {{ from {{ width: 0; }} to {{ width: 130px; }} }}
@keyframes piqGlow {{ 0%, 100% {{ box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
                     50% {{ box-shadow: 0 0 18px 2px rgba(46,194,126,0.28); }} }}
@keyframes piqGhost {{ from {{ transform: translate(8px, -50%); }} to {{ transform: translate(-32px, -50%); }} }}
@keyframes piqFloatA {{ 0%, 100% {{ transform: rotate(16deg) translateY(0); }}
                       50% {{ transform: rotate(14deg) translateY(-12px); }} }}
@keyframes piqFloatB {{ 0%, 100% {{ transform: rotate(-13deg) translateY(0); }}
                       50% {{ transform: rotate(-11deg) translateY(10px); }} }}
@keyframes piqSpin {{ to {{ transform: rotate(360deg); }} }}

.piq-hero {{ animation: piqRise .65s cubic-bezier(.22,1,.36,1) both; }}
.piq-hero .piq-wedge {{ animation: piqFloatA 10s ease-in-out infinite; }}
.piq-hero .piq-wedge2 {{ animation: piqFloatB 12s ease-in-out infinite; }}
.piq-hero .piq-orb {{
    position: absolute; top: 26px; right: 108px; width: 92px; height: 92px;
    border: 1.4px dashed rgba(230,237,243,0.16); border-radius: 50%;
    animation: piqSpin 44s linear infinite; pointer-events: none;
}}
.piq-hero .piq-orb::after {{
    content: ""; position: absolute; top: -4px; right: 50%;
    width: 5px; height: 5px; border-radius: 50%; background: {GREEN};
}}
.piq-section {{ animation: piqRise .6s cubic-bezier(.22,1,.36,1) both; animation-delay: .08s; }}
div[data-testid="stMetric"] {{ animation: piqRise .6s cubic-bezier(.22,1,.36,1) both; animation-delay: .1s; }}
div[data-testid="stDataFrame"], div[data-testid="stExpander"] {{ animation: piqFade .5s ease both; animation-delay: .12s; }}
.st-key-map-panel, .st-key-goal-panel, .st-key-timeline-panel, .st-key-quiz-panel {{
    animation: piqRise .65s cubic-bezier(.22,1,.36,1) both; animation-delay: .14s;
}}
.stPlotlyChart {{ animation: piqFade .5s ease both; animation-delay: .12s; }}

/* Front page choreography */
.piq-page-brand {{ animation: piqRise .8s cubic-bezier(.22,1,.36,1) both; animation-delay: .04s; }}
.piq-giant {{ animation: piqRise .8s cubic-bezier(.22,1,.36,1) both; animation-delay: .18s; }}
.piq-rule-sweep {{ animation: piqSweep .9s .5s cubic-bezier(.22,1,.36,1) both; }}
.piq-page-sub {{ animation: piqRise .8s cubic-bezier(.22,1,.36,1) both; animation-delay: .34s; }}
.piq-page-stats {{ animation: piqRise .8s cubic-bezier(.22,1,.36,1) both; animation-delay: .46s; }}
.piq-door-toprow {{ animation: piqRise .6s cubic-bezier(.22,1,.36,1) both; }}
[class*="st-key-door-card"] {{ animation: piqRise .6s cubic-bezier(.22,1,.36,1) backwards; }}
.st-key-door-card-0 {{ animation-delay: .05s; }}
.st-key-door-card-1 {{ animation-delay: .1s; }}
.st-key-door-card-2 {{ animation-delay: .15s; }}
.st-key-door-card-3 {{ animation-delay: .2s; }}
.st-key-door-card-4 {{ animation-delay: .25s; }}
.st-key-door-card-5 {{ animation-delay: .3s; }}

/* Magazine corner ticks + barcode (front page atmosphere) */
.piq-page-head {{ position: relative; }}
.piq-tick {{ position: absolute; width: 15px; height: 15px; opacity: .55;
    border-color: rgba(230,237,243,0.3); pointer-events: none; }}
.piq-tick.p1 {{ top: -6px; left: -10px; border-top: 1.6px solid; border-left: 1.6px solid; }}
.piq-tick.p2 {{ top: -6px; right: -10px; border-top: 1.6px solid; border-right: 1.6px solid; }}
.piq-tick.p3 {{ bottom: -6px; left: -10px; border-bottom: 1.6px solid; border-left: 1.6px solid; }}
.piq-tick.p4 {{ bottom: -6px; right: -10px; border-bottom: 1.6px solid; border-right: 1.6px solid; }}
.piq-barcode {{ height: 24px; width: 210px; opacity: .5; margin-top: 18px;
    background: repeating-linear-gradient(90deg,
        rgba(230,237,243,0.55) 0 2px, transparent 2px 5px,
        rgba(230,237,243,0.55) 5px 6px, transparent 6px 11px); }}
.piq-vol {{ font: 600 9.5px/2.4 {BODY}; letter-spacing: .2em; text-transform: uppercase;
    color: {TEXT_DIM}; margin-top: 2px; }}

/* Micro-interactions */
.piq-pill.solid {{ animation: piqGlow 2.8s ease-in-out infinite; }}
div[data-testid="stMetric"]:hover {{ border-color: {tint(GREEN, 0.5)}; transition: border-color .2s ease; }}
[class*="st-key-door-card"]:hover .piq-door-arrow svg {{
    transform: translateX(6px);
    transition: transform .35s cubic-bezier(.34,1.56,.64,1);
}}
.piq-band .ghost {{ animation: piqGhost 18s ease-in-out infinite alternate; }}

/* Audit ticker: audited facts scroll like a poster footnote strip */
.piq-ticker {{ overflow: hidden; border-top: 1.4px dashed {LINE}; border-bottom: 1.4px dashed {LINE};
    padding: 9px 0; margin: 28px 0 0 0; }}
.piq-ticker .track {{ display: inline-flex; gap: 34px; white-space: nowrap;
    animation: piqMarquee 48s linear infinite; }}
.piq-ticker:hover .track {{ animation-play-state: paused; }}
.piq-ticker .fact {{ font: 600 11.5px/2 {BODY}; letter-spacing: .13em; text-transform: uppercase; color: {TEXT_DIM}; }}
.piq-ticker .fact b {{ color: {TEXT}; font-family: {DISPLAY}; font-weight: 700; letter-spacing: .09em;
    font-size: 13.5px; margin-right: 6px; }}
@keyframes piqMarquee {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}

/* Broadcast LIVE badge (timeline replay) */
.piq-live-badge {{ display: inline-flex; align-items: center; gap: 9px; padding: 4px 14px;
    border: 1.4px dashed rgba(222,66,64,0.55); border-radius: 999px; color: {TEXT};
    font: 600 10.5px/2.2 {BODY}; letter-spacing: .16em; text-transform: uppercase; margin: 0 0 4px 0; }}
.piq-live-badge .dot {{ width: 8px; height: 8px; border-radius: 50%; background: {RED};
    animation: piqPulse 1.2s ease-in-out infinite; }}
@keyframes piqPulse {{ 0%, 100% {{ opacity: .35; transform: scale(.82); }} 50% {{ opacity: 1; transform: scale(1); }} }}

/* Poster panel around the quiz round score */
.st-key-quiz-panel {{ border: 1.4px solid {LINE}; border-radius: 16px;
    background: linear-gradient(150deg, rgba(13,17,26,0.55) 0%, rgba(13,17,26,0.32) 100%);
    padding: 16px 18px 12px 18px; }}

/* Quiz verdict (splash-setone) */
.piq-verdict {{ animation: piqRise .75s cubic-bezier(.22,1,.36,1) both; margin-top: 2px; }}
.piq-verdict .word {{ font-family: {DISPLAY}; font-weight: 700; font-size: clamp(38px, 4.6vw, 60px);
    line-height: 1; letter-spacing: 1.5px; text-transform: uppercase; color: {GREEN}; }}
.piq-verdict .word.m {{ color: {RED}; }}
.piq-verdict .rule {{ height: 3px; width: 110px; background: {GREEN}; border-radius: 2px; margin: 13px 0 9px 0;
    animation: piqSweep .7s .1s cubic-bezier(.22,1,.36,1) both; }}
.piq-verdict .sub {{ font: 500 13.5px/1.7 {BODY}; color: {TEXT_DIM}; max-width: 660px; }}
</style>
""",
        unsafe_allow_html=True,
    )


PAGE_ACCENTS: dict[str, str] = {
    "Shot Map": GREEN,
    "Match Timeline": "#4ea1ff",
    "xG Estimator": AMBER,
    "Model Evaluation": RED,
    "Beat the Model": GREEN,
    "Team & Player Insights": AMBER,
    "Front Page": "#4ea1ff",
}


def tint(color: str, alpha: float) -> str:
    """Hex color to rgba(...,alpha) for translucent poster tints."""
    hex_value = color.lstrip("#")
    r, g, b = int(hex_value[0:2], 16), int(hex_value[2:4], 16), int(hex_value[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def hero(page: str) -> None:
    """Poster band: giant display words + diagonal wedges + numbered index.

    Each page gets its own accent colour — a one-page identity injection so
    pages do not all wear the same gradient.
    """
    cfg = page_posters[page]
    accent = PAGE_ACCENTS.get(page, GREEN)
    st.markdown(
        f"""<style>
.piq-hero {{ background: linear-gradient(102deg, {NAVY} 0%, {NAVY_2} 46%, {tint(accent, 0.20)} 100%); }}
.piq-hero .piq-wedge {{ background: linear-gradient(180deg, {tint(accent, 0.30)}, {tint(accent, 0)}); }}
.piq-hero .piq-wedge2 {{ background: linear-gradient(180deg, {tint(accent, 0.20)}, {tint(accent, 0)}); }}
.piq-hero .edge {{ background: linear-gradient(180deg, {accent}, transparent); }}
.piq-hero .piq-node {{ border-color: {tint(accent, 0.55)}; color: {accent}; background: {tint(accent, 0.12)}; }}
.piq-hero .piq-index {{ border-color: {tint(accent, 0.60)}; color: {accent}; }}
.piq-hero .piq-chip.solid {{ border-color: {tint(accent, 0.55)}; color: {accent}; }}
.piq-hero .piq-chip svg {{ color: {accent}; }}
.piq-hero .piq-word .amp {{ color: {accent}; }}
.piq-section .idx {{ color: {accent}; border-color: {tint(accent, 0.40)}; background: {tint(accent, 0.10)}; }}
.piq-section .t svg {{ color: {accent}; }}
</style>""",
        unsafe_allow_html=True,
    )
    img = ASSETS_DIR / cfg["asset"]
    img_html = (
        f'<img class="piq-hero-img" src="data:image/png;base64,{base64.b64encode(img.read_bytes()).decode()}" alt="">'
        if img.exists() else ""
    )
    word = f"<span>{cfg['word_1']}</span><span class='ghost'>{cfg['word_2']}</span>"
    chips = "".join(
        f'<span class="piq-chip">{icon(ic, 13)} {txt}</span>'
        for ic, txt in (
            ("pitch", "12,241 shots"),
            ("zone", "479 matches"),
            ("chart", "3 competitions"),
        )
    )
    st.markdown(
        f"""
<div class="piq-hero">
  <span class="piq-wedge"></span><span class="piq-wedge2"></span>
  <span class="piq-orb"></span><span class="edge"></span>
  {img_html}
  <span class="piq-node" style="top:18px; right:34px;">{icon('target', 15)}</span>
  <span class="piq-node" style="bottom:16px; right:64px;">{icon('bolt', 15)}</span>
  <span class="piq-index">{cfg['index']}</span>
  <span class="piq-chip solid">{icon('users', 13)} SHOTWORTH xG INTELLIGENCE</span>
  <div class="piq-word">{word}</div>
  <p class="piq-sub">{cfg['subtitle']}</p>
  <div class="piq-crumbs">{chips}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def section(letter: str, title: str, icon_name: str | None = None) -> None:
    """Numbered section heading: circle letter + spaced caps + thin rule."""
    ic = f"{icon(icon_name, 14)} " if icon_name else ""
    st.markdown(
        f"""<div class="piq-section">
<span class="idx">{letter}</span>
<span class="t">{ic}{title}</span>
<span class="rule"></span>
</div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


def footer_band() -> None:
    """Poster footer band with ghost display text."""
    st.markdown(
        f"""<div class="piq-band">
<span class="ghost">STATSBOMB OPEN DATA</span>
<div class="lead">Built on StatsBomb Open Data</div>
<div class="sub">{CREDIT}</div>
</div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 0. Front page (magazine cover)
# ---------------------------------------------------------------------------


def _nav_to(page: str) -> None:
    """Button callback: jump the sidebar selectbox to another page."""
    st.session_state["menu"] = page


def front_page() -> None:
    """Magazine-cover landing in the splash's open-air poster language."""
    data = load_shots()
    shots_count = int((~data["is_penalty"].fillna(False)).sum())
    matches_count = int(data["match_id"].nunique())
    try:
        metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
        test_auc = f"{metadata['test_metrics']['roc_auc']:.3f}"
    except (OSError, KeyError, json.JSONDecodeError):
        test_auc = "0.796"

    logo = ASSETS_DIR / "logo.svg"
    logo_dot = (
        f'<img src="data:image/svg+xml;base64,'
        f'{base64.b64encode(logo.read_bytes()).decode()}" width="26" height="26" alt="">'
        if logo.exists() else ""
    )
    stat_cells = [
        ("SHOTS", f"{shots_count:,}"),
        ("MATCHES", f"{matches_count}"),
        ("COMPETITIONS", "3"),
        ("TEST AUC", test_auc),
    ]
    cells = "".join(
        f"""<div class="cell"><div class="num">{value}</div><div class="lbl">{label}</div></div>"""
        for label, value in stat_cells
    )
    st.markdown(
        f"""<div class="piq-page-head">
  <span class="piq-tick p1"></span><span class="piq-tick p2"></span>
  <span class="piq-tick p3"></span><span class="piq-tick p4"></span>
  <div class="piq-page-brand">{logo_dot}<span>SHOTWORTH · xG INTELLIGENCE</span></div>
  <div class="piq-giant">SHOT <span class="ghost">WORTH</span></div>
  <div class="piq-rule-sweep"></div>
  <p class="piq-page-sub">Every shot has a price. {shots_count:,} real attempts from
  {matches_count} matches of StatsBomb Open Data, priced by a transparent model —
  an honest audit for every number that follows.</p>
  <div class="piq-page-stats">{cells}</div>
  <div class="piq-barcode"></div>
  <div class="piq-vol">VOL. {shots_count:,} — POST-AUDIT EDITION · STATSBOMB OPEN DATA REPACKED</div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    section("1", "Pick a door", "pitch")
    doors = [
        ("Shot Map", "pitch", "Every one of the 12,241 shots on a live pitch, size and colour = xG."),
        ("Match Timeline", "bolt", "Replay any match like a broadcast: the xG race builds minute by minute."),
        ("xG Estimator", "target", "Price any shot in real time — and see why each factor moves it."),
        ("Model Evaluation", "chart", "The honest audit: AUC, log loss, Brier and calibration, explained."),
        ("Beat the Model", "bolt", "Five unseen shots, your odds vs the model's — log loss decides."),
        ("Team & Player Insights", "users", "Head-to-head radar, scout cards, zones and manager recommendations."),
    ]
    for door_row_start in range(0, len(doors), 2):
        pair = st.columns(2, gap="medium")
        for slot, door_number, (page_name, page_icon, blurb) in zip(
            pair,
            range(door_row_start + 1, door_row_start + 3),
            doors[door_row_start:door_row_start + 2],
        ):
            with slot:
                with st.container(key=f"door-card-{door_number - 1}"):
                    st.markdown(
                        f"""<div class="piq-door-toprow">
<span class="piq-door-idx">0{door_number}</span>
<span class="piq-door-arrow">{icon('arrow', 14)}</span>
</div>""",
                        unsafe_allow_html=True,
                    )
                    st.button(
                        page_name,
                        on_click=_nav_to,
                        kwargs={"page": page_name},
                        use_container_width=True,
                    )
                    st.caption(blurb)
    facts = [
        ("Messi", "27 goals on 19.94 xG — +7.06 finishing premium, 190 shots"),
        ("Özil", "142 assisted shots turned into 14.5 xG"),
        ("Aston Villa", "54% of shots below 0.05 xG — scenario adds +5.2 xG"),
        ("Stoke City", "59 long-range shots, 0 goals — 0.017 xG per shot"),
        ("Counter-attacks", "0.146 xG per shot vs 0.097 open play, 558 shots"),
        ("Audit", f"AUC {test_auc} · log loss 0.257 · Brier 0.0732 · ECE 0.0204"),
    ]
    fact_items = "".join(
        f'<span class="fact"><b>{name}</b>{value}</span>' for name, value in facts
    )
    st.markdown(
        f"""<div class="piq-ticker"><div class="track">{fact_items}{fact_items}</div></div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.caption(
        "All numbers on this cover are computed live from the audited pipeline — the "
        "model never sees StatsBomb xG as an input and penalties are excluded."
    )


# ---------------------------------------------------------------------------
# 1. Shot map
# ---------------------------------------------------------------------------


def filter_and_display(data: pd.DataFrame, show_penalties: bool) -> pd.DataFrame:
    """Return the visible shot frame plus the xG column used for color/size."""
    view = data.copy()
    if show_penalties:
        # Penalty rows are outside the model; show the StatsBomb benchmark there.
        view["xg_display"] = np.where(view["is_penalty"], view["xg_statsbomb"], view["shotworth_xg"])
    else:
        view = view[~view["is_penalty"].fillna(False)]
        view["xg_display"] = view["shotworth_xg"]
    return view


def shot_map_page() -> None:
    data = load_shots()
    hero("Shot Map")

    section("1", "Filter the shots", "target")
    head = st.columns([1.6, 1.6, 1.15, 1.15])
    team = head[0].selectbox("Team", ["All"] + sorted(data["team"].astype(str).unique().tolist()))
    team_players = sorted(data.loc[data["team"] == team, "player"].astype(str).unique())
    all_players = sorted(data["player"].value_counts().loc[lambda c: c >= 15].index.astype(str).tolist())
    player = head[1].selectbox("Player", ["All"] + (team_players if team != "All" else all_players))
    body = head[2].selectbox("Body part", ["All"] + sorted(data["body_part"].dropna().astype(str).unique().tolist()))
    shot_type = head[3].selectbox("Shot type", ["All"] + sorted(data["shot_type"].dropna().astype(str).unique().tolist()))
    tail = st.columns([2.4, 2.4, 6])
    with tail[0]:
        goal_filter = st.radio("Outcome", ["All", "Goals", "No goals"], horizontal=True)
    with tail[1]:
        show_penalties = st.toggle("Include penalties", value=False)

    view = filter_and_display(data, show_penalties)
    if team != "All":
        view = view[view["team"].astype(str) == team]
    if player != "All":
        view = view[view["player"].astype(str) == player]
    if body != "All":
        view = view[view["body_part"].astype(str) == body]
    if shot_type != "All":
        view = view[view["shot_type"].astype(str) == shot_type]
    if goal_filter == "Goals":
        view = view[view["is_goal"]]
    elif goal_filter == "No goals":
        view = view[~view["is_goal"]]

    if view.empty:
        st.info("No shots match this filter combination — loosen one of the filters.")
        return

    section("2", "Live sample", "users")
    metrics = st.columns(3)
    conversion = view["is_goal"].mean()
    metrics[0].metric("Shots displayed", f"{len(view):,}")
    metrics[1].metric("Goals", f"{int(view['is_goal'].sum()):,}")
    mean_label = "Mean displayed xG" if show_penalties else "Mean ShotWorth xG"
    metrics[2].metric(mean_label, f"{view['xg_display'].mean():.3f}", delta=f"conversion {conversion:.1%}")
    if show_penalties:
        st.caption("Penalty xG comes from the StatsBomb benchmark while non-penalties use ShotWorth, so the combined mean mixes two sources — not one homogeneous model.")

    import plotly.graph_objects as go

    section("3", "The shot map", "pitch")

    minutes = view["minute"].astype(int).astype(str)
    seconds = view["second"].astype(int).map(lambda s: f"{s:02d}")
    custom = np.stack(
        [
            view["player"].astype(str).values,
            (minutes + "'" + seconds).values,
            view["team"].astype(str).values,
            view["body_part"].astype(str).values,
            view["shot_type"].astype(str).values,
            view["xg_display"].astype(float).values,
            np.where(view["is_goal"], "GOAL", view["outcome"].astype(str)).astype(object),
        ],
        axis=1,
    )
    sizes = 7 + 26 * np.sqrt(view["xg_display"].to_numpy(float))

    fig = go.Figure(
        go.Scattergl(
            x=view["x"].astype(float),
            y=view["y"].astype(float),
            mode="markers",
            marker=dict(
                size=sizes,
                color=view["xg_display"].astype(float),
                colorscale=XG_SCALE,
                cmin=0.0,
                cmax=float(np.percentile(view["xg_display"], 98)),
                opacity=0.8,
                line=dict(width=0.7, color="#0b0e14"),
                colorbar=dict(title=dict(text="xG", side="right"), thickness=10, len=0.68,
                              outlinewidth=0, tickfont=dict(size=10), tickformat=".2f"),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> • minute %{customdata[1]}<br>"
                "Team: %{customdata[2]}<br>"
                "Body: %{customdata[3]} • Type: %{customdata[4]}<br>"
                "xG: %{customdata[5]:.3f} • Outcome: %{customdata[6]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=HOVERLABEL,
        font=dict(family=BODY, color=TEXT_DIM),
        shapes=PITCH_SHAPES,
        height=520,
        margin=dict(l=10, r=58, t=10, b=10),
        xaxis=dict(range=[-3, 126], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="y", scaleratio=1.0),
        yaxis=dict(range=[-3, 83], showgrid=False, zeroline=False, visible=False),
    )

    # Poster echo of the empty half: restrained display-type watermark.
    fig.add_annotation(
        x=30, y=42, text="<b>12,241</b><br><span style='font-size:12px'>SHOTS</span>",
        showarrow=False, font=dict(family=DISPLAY, size=44, color="rgba(230,237,243,0.08)"),
        align="center",
    )
    fig.add_annotation(
        x=30, y=31, text="mean ShotWorth xG 0.098",
        showarrow=False, font=dict(family=BODY, size=11, color="rgba(46,194,126,0.32)"),
        align="center",
    )

    with st.container(key="map-panel"):
        st.plotly_chart(
            fig, width="stretch",
            config=dict(displayModeBar=False, displaylogo=False),
        )

    st.caption(
        "**How to read:** big green points = good chances (high xG); small red "
        "points = low-probability attempts. Use the filters above to explore "
        "teams, players and shot context."
    )


# ---------------------------------------------------------------------------
# 1b. Match timeline (xG race + live replay)
# ---------------------------------------------------------------------------


def timeline_page() -> None:
    """Broadcast-style cumulative xG race for one match, replayable minute by minute."""
    import plotly.graph_objects as go

    hero("Match Timeline")
    data = load_shots()

    # Match catalog: most eventful matches first (by total ShotWorth xG).
    totals = data.groupby("match_id")["shotworth_xg"].sum().sort_values(ascending=False)
    teams_by_match = data.groupby("match_id")["team"].agg(
        lambda s: s.astype(str).value_counts().index.tolist()
    )
    shot_counts = data["match_id"].value_counts()

    labels: list[str] = []
    match_ids: list[Any] = []
    for match_id, total in totals.items():
        teams_lbl = teams_by_match.get(match_id, [])
        label = "  vs  ".join(teams_lbl[:2]) if len(teams_lbl) >= 2 else str(teams_lbl[0])
        labels.append(f"{label}   ·   {shot_counts[match_id]} shots, xG {total:.1f}")
        match_ids.append(match_id)
    if not labels:
        st.info("No matches available in the features table.")
        return

    pick = st.selectbox("Pick a match", labels, index=0)
    md = data[data["match_id"] == match_ids[labels.index(pick)]].sort_values(
        ["minute", "second"]
    ).reset_index(drop=True)
    teams = [str(t) for t in md["team"].astype(str).value_counts().index[:2]]
    if not teams:
        st.info("This match has no recorded shots in the features table.")
        return
    colors = {teams[0]: GREEN}
    if len(teams) > 1:
        colors[teams[1]] = RED
    max_minute = int(md["minute"].max())
    non_pen_xg = md.loc[~md["is_penalty"].fillna(False), "shotworth_xg"]
    xg_cmax = float(np.percentile(non_pen_xg, 98)) if len(non_pen_xg) else 0.5

    ss = st.session_state
    ss.setdefault("mt_playing", False)
    ss.setdefault("mt_minute", max_minute)

    # Replay controls live outside the fragment so pressing them always
    # triggers a full script run (needed to mount/unmount the timer).
    ctl = st.columns([1, 1, 6])
    replaying = bool(ss["mt_playing"])
    if ctl[0].button("Replay from 0'", type="secondary" if replaying else "primary"):
        ss["mt_playing"] = True
        ss["mt_minute"] = 0
        st.rerun()
    if ctl[1].button("Jump to full match", disabled=not replaying):
        ss["mt_playing"] = False
        ss["mt_minute"] = max_minute
        st.rerun()
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    def race_render(current: int, live: bool) -> None:
        """Draw scoreline tiles, the masked xG race and the event feed."""
        upto = md[md["minute"] <= current]

        if live:
            st.markdown(
                '<div class="piq-live-badge"><span class="dot"></span>LIVE REPLAY — '
                "the match builds minute by minute</div>",
                unsafe_allow_html=True,
            )

        tiles = st.columns([1, 1.35, 1.35, 1.1])
        tiles[0].metric("Minute", f"{current}'")
        for i, team in enumerate(teams[:2]):
            part = upto[upto["team"].astype(str) == team]
            goals = int(part["is_goal"].sum())
            xg = float(part.loc[~part["is_penalty"].fillna(False), "shotworth_xg"].sum())
            tiles[i + 1].metric(
                team,
                f"{xg:.2f} xG",
                delta=f"{goals} goals · {len(part)} shots",
                delta_color="off",
            )
        if live:
            tiles[3].metric("Mode", "LIVE")

        race_fig = go.Figure()
        for team in teams:
            color = colors[team]
            part = upto[
                (upto["team"].astype(str) == team)
                & (~upto["is_penalty"].fillna(False))
            ]
            cum = part.groupby("minute")["shotworth_xg"].sum().cumsum()
            xs, ys = (list(cum.index), list(cum.values)) if len(cum) else ([0], [0.0])
            race_fig.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines", legendgroup=team,
                    name=f"{team} — ShotWorth",
                    line=dict(shape="hv", width=2.6, color=color),
                    customdata=xs,
                    hovertemplate=f"<b>{team}</b><br>Minute %{{x}}'<br>Cumulative ShotWorth xG %{{y:.2f}}<extra></extra>",
                )
            )
            sb = part.groupby("minute")["xg_statsbomb"].sum().cumsum()
            sbx, sby = (list(sb.index), list(sb.values)) if len(sb) else ([0], [0.0])
            race_fig.add_trace(
                go.Scatter(
                    x=sbx, y=sby, mode="lines", showlegend=False,
                    legendgroup=team, hoverinfo="skip",
                    line=dict(shape="hv", width=1.3, color=color, dash="dot"),
                    opacity=0.55,
                )
            )
            goal_pts = part[part["is_goal"]]
            if not goal_pts.empty:
                gx = goal_pts["minute"].astype(int).tolist()
                gy = [float(cum.loc[m]) if m in cum.index else 0.0 for m in gx]
                race_fig.add_trace(
                    go.Scatter(
                        x=gx, y=gy, mode="markers", legendgroup=team,
                        name=f"{team} — goals",
                        marker=dict(symbol="star", size=13, color=color,
                                    line=dict(width=1.2, color="#0b0e14")),
                        customdata=[[str(p), f"{float(x):.2f}"] for p, x in zip(goal_pts["player"].astype(str), goal_pts["shotworth_xg"])],
                        hovertemplate="<b>%{customdata[0]}</b> scores · xG %{customdata[1]}<extra></extra>",
                    )
                )
        race_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=HOVERLABEL, font=dict(family=BODY, color=TEXT_DIM),
            xaxis=dict(title="Minute", range=[0, max_minute + 2], gridcolor="#1a2434", zeroline=False,
                       dtick=15),
            yaxis=dict(title="Cumulative xG", gridcolor="#1a2434", zeroline=False),
            height=390, margin=dict(l=52, r=16, t=18, b=44),
            legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        )

        with st.container(key="timeline-panel"):
            race_cols = st.columns([1.5, 1], gap="small")
            with race_cols[0]:
                st.plotly_chart(race_fig, width="stretch",
                                config=dict(displayModeBar=False, displaylogo=False))
            with race_cols[1]:
                pitch_fig = go.Figure()
                dots = upto[~upto["is_penalty"].fillna(False)]
                pitch_fig.add_trace(go.Scattergl(
                    x=dots["x"].astype(float), y=dots["y"].astype(float),
                    mode="markers",
                    marker=dict(
                        size=7 + 16 * np.sqrt(dots["shotworth_xg"].astype(float)),
                        color=dots["shotworth_xg"].astype(float),
                        colorscale=XG_SCALE, cmin=0.0, cmax=max(xg_cmax, 0.05),
                        opacity=0.85, line=dict(width=0.6, color="#0b0e14"),
                    ),
                    customdata=np.stack([
                        dots["player"].astype(str).values,
                        dots["minute"].astype(int).astype(str).values,
                        dots["shotworth_xg"].astype(float).values,
                        np.where(dots["is_goal"], "GOAL", dots["outcome"].astype(str)).astype(object),
                    ], axis=1),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> · %{customdata[1]}'<br>"
                        "xG %{customdata[2]:.3f} · %{customdata[3]}<extra></extra>"
                    ),
                ))
                pitch_fig.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    hoverlabel=HOVERLABEL, font=dict(family=BODY, color=TEXT_DIM),
                    shapes=PITCH_SHAPES, height=392, margin=dict(l=6, r=6, t=6, b=6),
                    xaxis=dict(range=[-3, 126], showgrid=False, zeroline=False, visible=False,
                               scaleanchor="y", scaleratio=1.0),
                    yaxis=dict(range=[-3, 83], showgrid=False, zeroline=False, visible=False),
                )
                st.plotly_chart(pitch_fig, width="stretch",
                                config=dict(displayModeBar=False, displaylogo=False))
        st.caption(
            "Left: cumulative ShotWorth xG race (solid) with the StatsBomb benchmark dotted. "
            "Right: every shot so far pops in on the pitch as the replay advances — size and "
            "colour = xG, stars mark goals in the race. Penalties are excluded from the "
            "ShotWorth curve and scored separately."
        )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if upto.empty:
            st.caption("Kick-off — no shots yet.")
        else:
            recent = upto.tail(5).iloc[::-1]
            chips = "".join(
                f'<span class="piq-chip{" solid" if row.is_goal else ""}">'
                f'{int(row.minute)}&#39; · {str(row.player)[:22]} · '
                f'{float(row.shotworth_xg):.2f} xG · '
                f'{"GOAL" if row.is_goal else str(row.outcome)[:12]}'
                f'</span>'
                for row in recent.itertuples()
            )
            st.markdown(f'<div class="piq-crumbs">{chips}</div>', unsafe_allow_html=True)

    if ss["mt_playing"]:
        @st.fragment(run_every=0.8)
        def mt_replay() -> None:
            nxt = min(int(st.session_state.get("mt_minute", 0)) + 1, max_minute)
            st.session_state["mt_minute"] = nxt
            if nxt >= max_minute:
                st.session_state["mt_playing"] = False
            race_render(int(st.session_state["mt_minute"]), live=nxt < max_minute)

        mt_replay()
    else:
        race_render(int(ss["mt_minute"]), live=False)
    if "mt_minute" in ss and int(ss["mt_minute"]) < max_minute and not ss["mt_playing"]:
        ctl[2].caption("Replay paused mid-match — press “Jump to full match” or hit Replay again.")


# ---------------------------------------------------------------------------
# 2. xG estimator
# ---------------------------------------------------------------------------


def build_estimator_row(x: float, y: float, body_part: str, shot_type: str,
                        under_pressure: bool, first_time: bool) -> pd.DataFrame:
    """Assemble one model input row from user choices.

    Freeze-frame / defender features are not available without a match context;
    they stay NaN so the saved imputer fills dataset medians, and
    `has_freeze_frame = 0` lets the model discount them.
    """
    distance = float(distance_to_goal(x, y))
    angle = float(shot_angle_deg(x, y))
    values: dict[str, Any] = {column: np.nan for column in FEATURE_COLUMNS}
    values.update(
        distance=distance,
        angle_deg=angle,
        log_distance=float(np.log1p(distance)),
        distance_to_goal_line=120.0 - float(x),
        abs_y_from_center=abs(float(y) - 40.0),
        minute=45,
        has_freeze_frame=0,
        under_pressure=int(under_pressure),
        first_time=int(first_time),
        open_goal=0,
        one_on_one=0,
        aerial_won=0,
        body_part=body_part,
        shot_type=shot_type,
        technique="Normal",
        play_pattern="Regular Play" if shot_type == "Open Play" else "From Free Kick",
    )
    return pd.DataFrame([values])[FEATURE_COLUMNS], distance, angle


def estimator_page() -> None:
    hero("xG Estimator")
    artifact = load_model()
    penalty_rate = load_shots().loc[lambda df: df["is_penalty"].fillna(False), "is_goal"].mean()

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        section("1", "Set the shot", "target")
        x = st.slider("Shot depth (x: 60 = halfway line, 120 = opponent's goal)", 0, 119, 110)
        y = st.slider("Lateral position (y: 40 = centre of goal)", 0, 80, 40)
        body_part = st.selectbox("Body part", BODY_PARTS)
        shot_type = st.selectbox("Shot type", SHOT_TYPES)
        under_pressure = st.checkbox("Taken under pressure")
        first_time = st.checkbox("First-time finish (no first touch)")

    with right:
        row, distance, angle = build_estimator_row(x, y, body_part, shot_type, under_pressure, first_time)
        probability = float(artifact["pipeline"].predict_proba(row)[:, 1][0])
        calibrator = artifact.get("calibrator")
        if calibrator is not None:
            probability = float(calibrator.predict(np.array([probability]))[0])

        if probability >= 0.30:
            prob_color = "#16a34a"
        elif probability >= 0.12:
            prob_color = "#d9a406"
        else:
            prob_color = RED

        section("2", "Model verdict", "bolt")
        prob_col, dist_col, ang_col = st.columns([1.3, 1, 1])
        with prob_col:
            st.markdown(
                f"""<div style="font-family:{DISPLAY};font-weight:700;letter-spacing:1px;
                font-size:44px;line-height:1.05;color:{prob_color};">{probability:.1%}</div>
                <div style="font:600 10px/1.6 {BODY};letter-spacing:0.16em;text-transform:uppercase;
                color:{TEXT_DIM};margin-top:2px;">Goal probability</div>""",
                unsafe_allow_html=True,
            )
            st.progress(min(probability, 1.0))
        dist_col.metric("Distance to goal", f"{distance:.1f}")
        ang_col.metric("Open goal angle", f"{angle:.1f}")

        import plotly.graph_objects as go

        goal_fig = go.Figure()
        goal_fig.add_trace(
            go.Scatter(
                x=[x, 120.0, 120.0, x],
                y=[y, 36.0, 44.0, y],
                fill="toself",
                fillcolor="rgba(46,194,126,0.22)",
                line=dict(color=GREEN, width=2),
                mode="lines",
                hoverinfo="skip",
            )
        )
        goal_fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode="markers",
                marker=dict(size=16, color=RED, symbol="x",
                            line=dict(width=0, color=RED)),
                name="Shot",
                hovertemplate=f"x={x:.0f}, y={y:.0f}, probability {probability:.0%}<extra></extra>",
            )
        )
        goal_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=HOVERLABEL,
            font=dict(family=BODY, color=TEXT_DIM),
            shapes=[
                dict(type="line", x0=120, y0=36, x1=120, y1=44, line=dict(color="#ffffff", width=5)),
                dict(type="rect", x0=114, y0=18, x1=120, y1=62, line=dict(color=PITCH_LINE, width=1.4)),
            ],
            height=360,
            margin=dict(l=10, r=10, t=0, b=10),
            xaxis=dict(range=[max(60, min(float(x), 118)) - 10, 123.0],
                       showgrid=False, zeroline=False, visible=False,
                       scaleanchor="y", scaleratio=1.0),
            yaxis=dict(range=[22, 58], showgrid=False, zeroline=False, visible=False),
        )

        with st.container(key="goal-panel"):
            st.plotly_chart(
                goal_fig, width="stretch",
                config=dict(displayModeBar=False, displaylogo=False),
            )

    with st.expander("Understand the number"):
        st.markdown(
            "- **Probability** = the share of similar shots that became goals in "
            "identical situations, learned from tens of thousands of real attempts.\n"
            "- The estimator **does not read defender/keeper positions** — "
            "freeze-frame features are median-imputed, so this is a 'quiet' situation. "
            "Real pressure usually lowers it further.\n"
            f"- For scale: penalty conversion in this dataset is {penalty_rate:.0%} — "
            "penalties are reported separately because a 12-yard one-on-one is a "
            "different task."
        )

    section("3", "Why this xG", "chart")
    cal_base = load_shots().loc[~load_shots()["is_penalty"].fillna(False)]
    base_distance = float(cal_base["distance"].median())
    base_angle = float(cal_base["angle_deg"].median())

    def neutral_prob(column: str, value: Any) -> float:
        """Probability with one factor swapped to a dataset-typical value."""
        row_copy = row.copy()
        row_copy[column] = value
        if column in ("distance", "angle_deg"):
            row_copy["log_distance"] = np.log1p(float(row_copy["distance"].iloc[0]))
        if column == "shot_type" and value == "Open Play":
            row_copy["play_pattern"] = "Regular Play"
        elif column == "shot_type":
            row_copy["play_pattern"] = "From Free Kick"
        p = float(artifact["pipeline"].predict_proba(row_copy[FEATURE_COLUMNS])[:, 1][0])
        cal = artifact.get("calibrator")
        return float(cal.predict(np.array([p]))[0]) if cal is not None else p

    contributions: list[tuple[str, float]] = [
        ("Distance (→ median)", neutral_prob("distance", base_distance)),
        ("Body part (→ right foot)", neutral_prob("body_part", "Right Foot")),
        ("Angle (→ median)", neutral_prob("angle_deg", base_angle)),
        ("Shot type (→ open play)", neutral_prob("shot_type", "Open Play")),
        ("Under pressure (→ no)", neutral_prob("under_pressure", 0)),
        ("First-time finish (→ no)", neutral_prob("first_time", 0)),
    ]

    import plotly.graph_objects as go

    deltas = [
        (name, probability - p_neutral)
        for name, p_neutral in contributions
    ]
    deltas.sort(key=lambda item: abs(item[1]))
    expl_fig = go.Figure()
    expl_fig.add_trace(go.Bar(
        x=[d for _, d in deltas],
        y=[name for name, _ in deltas],
        orientation="h",
        marker=dict(
            color=[GREEN if d >= 0 else RED for _, d in deltas],
            line=dict(width=0),
        ),
        hovertemplate="%{y}<br>Δ xG %{x:+.3f}<extra></extra>",
    ))
    expl_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY, color=TEXT_DIM), hoverlabel=HOVERLABEL,
        xaxis=dict(title="xG contribution (current situation → this factor)", zeroline=True,
                   zerolinecolor="#5b6b7e", gridcolor="#1a2434"),
        yaxis=dict(title=""),
        height=300, margin=dict(l=10, r=16, t=10, b=40),
    )
    contrib_col, note_col = st.columns([1.5, 1], gap="large")
    with contrib_col:
        st.plotly_chart(expl_fig, width="stretch")
    with note_col:
        st.markdown(
            f"**Current probability: {probability:.1%}** — each bar asks *\"what if one "
            "factor were dataset-typical?\"* The gap to that neutral world is the "
            "factor's influence on this specific shot.\n\n"
            f"- Shot is placed {distance:.0f} m out vs a typical {base_distance:.0f} m.\n"
            f"- Open angle {angle:.0f}° vs a typical {base_angle:.0f}°.\n\n"
            "Bars do **not** sum to the total — factors interact, so this is an "
            "approximate one-at-a-time diagnosis, not an exact Shapley decomposition."
        )


# ---------------------------------------------------------------------------
# 3. Model evaluation
# ---------------------------------------------------------------------------


def _test_split() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Reproduce the grouped 80/20 holdout used in training (same seed)."""
    from sklearn.model_selection import GroupShuffleSplit

    features = load_shots()
    X = features[FEATURE_COLUMNS]
    y = features["is_goal"].astype(int)
    mask = ~features["is_penalty"].fillna(False)
    X, y, groups = X.loc[mask], y.loc[mask], features.loc[mask, "match_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=TEST_SPLIT_SEED)
    train_index, test_index = next(splitter.split(X, y, groups))
    return X.iloc[test_index], y.iloc[test_index], features.loc[X.index[test_index], "xg_statsbomb"]


def evaluation_page() -> None:
    hero("Model Evaluation")
    artifact = load_model()
    metadata = artifact["metadata"]
    X_test, y_test, sb_xg = _test_split()
    probabilities = artifact["pipeline"].predict_proba(X_test)[:, 1]
    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        probabilities = calibrator.predict(probabilities)

    test_matches = metadata["split"]["test_matches"]
    st.markdown("".join([
        '<div class="piq-crumbs">',
        f'<span class="piq-chip">holdout: {int(test_matches)} matches</span>',
        '<span class="piq-chip">grouped seed 20261029</span>',
        '<span class="piq-chip">zero match overlap</span>',
        "</div>",
    ]), unsafe_allow_html=True)

    section("1", "Holdout scores", "chart")
    metrics_head = st.columns(4)
    metrics_head[0].metric("Deployment model", "Logistic Regression")
    metrics_head[1].metric("ROC-AUC (test)", f"{metadata['test_metrics']['roc_auc']:.3f}")
    metrics_head[2].metric("Log loss (test)", f"{metadata['test_metrics']['log_loss']:.3f}")
    metrics_head[3].metric("Brier (test)", f"{metadata['test_metrics']['brier']:.4f}")

    left, right = st.columns([1.15, 1], gap="large")

    with right:
        section("2", "Model comparison", "users")
        comparison = load_csv("model_comparison.csv")
        show = comparison[
            ["model", "test_roc_auc", "test_log_loss", "test_brier",
             "cv_roc_auc_mean", "cv_roc_auc_std"]
        ].copy()
        show.columns = ["Model", "Test AUC", "Log loss", "Brier", "CV AUC", "± CV"]
        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
            column_config={
                "Test AUC": st.column_config.NumberColumn(format="%.3f"),
                "Log loss": st.column_config.NumberColumn(format="%.3f"),
                "Brier": st.column_config.NumberColumn(format="%.4f"),
                "CV AUC": st.column_config.NumberColumn(format="%.3f"),
                "± CV": st.column_config.NumberColumn(format="±%.3f"),
            },
        )
        st.caption(
            "Tuned-tree CV numbers are exploratory (no nested CV yet). StatsBomb "
            "metrics are scored on identical data purely for context, they never "
            "opposed the models in training."
        )

        left_ablation, right_ablation = st.columns(2, gap="small")
        with left_ablation:
            st.markdown("##### LogReg ablation")
            ablation = load_csv("logreg_ablation.csv")
            st.dataframe(
                ablation[["stage", "roc_auc", "brier"]].rename(
                    columns={"stage": "Stage", "roc_auc": "AUC", "brier": "Brier"}
                ),
                width="stretch", hide_index=True,
                column_config={"AUC": st.column_config.NumberColumn(format="%.3f"),
                               "Brier": st.column_config.NumberColumn(format="%.4f")},
            )
        with right_ablation:
            st.markdown("##### Calibration")
            calib = load_csv("calibration_comparison.csv")
            st.dataframe(
                calib[["model", "variant", "brier", "calibration_slope"]].rename(
                    columns={"brier": "Brier", "calibration_slope": "Slope"}
                ),
                width="stretch", hide_index=True,
                column_config={"Brier": st.column_config.NumberColumn(format="%.4f"),
                               "Slope": st.column_config.NumberColumn(format="%.3f")},
            )

    with left:
        import plotly.graph_objects as go
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import roc_curve

        section("3", "Probability curves", "chart")
        fig = go.Figure()
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=artifact["model_name"],
            line=dict(color=GREEN, width=2.6),
            hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>",
        ))
        fpr_sb, tpr_sb, _ = roc_curve(y_test, sb_xg)
        fig.add_trace(go.Scatter(
            x=fpr_sb, y=tpr_sb, mode="lines", name="StatsBomb xG",
            line=dict(color=AMBER, width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color="#5b6b7e", width=1.2, dash="dash"),
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=HOVERLABEL, font=dict(family=BODY, color=TEXT_DIM),
            xaxis=dict(title="False positive rate", range=[0, 1], gridcolor="#1a2434", zeroline=False),
            yaxis=dict(title="True positive rate", range=[0, 1.02], gridcolor="#1a2434", zeroline=False),
            height=400, margin=dict(l=44, r=16, t=28, b=44),
            legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, width="stretch")

        observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
        obs_sb, pred_sb = calibration_curve(y_test, sb_xg, n_bins=10, strategy="quantile")
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(
            x=predicted, y=observed, mode="lines+markers",
            name=artifact["model_name"], line=dict(color=GREEN, width=2.2),
            marker=dict(size=7),
        ))
        cal_fig.add_trace(go.Scatter(
            x=pred_sb, y=obs_sb, mode="lines+markers", name="StatsBomb xG",
            line=dict(color=AMBER, width=2, dash="dot"), marker=dict(size=7),
        ))
        cal_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line=dict(color="#5b6b7e", width=1.2, dash="dash"),
        ))
        cal_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=HOVERLABEL, font=dict(family=BODY, color=TEXT_DIM),
            xaxis=dict(title="Mean predicted xG", range=[0, 0.5], gridcolor="#1a2434", zeroline=False),
            yaxis=dict(title="Observed goal rate", range=[0, 0.5], gridcolor="#1a2434", zeroline=False),
            height=400, margin=dict(l=44, r=16, t=28, b=44),
        )
        st.plotly_chart(cal_fig, width="stretch")

    with st.expander("How to read the metrics (plain language)"):
        st.markdown(
            "- **ROC-AUC** — how often the model ranks a random goal above a random "
            "non-goal. 0.5 = guessing, 1.0 = perfect.\n"
            "- **Log loss** — the penalty for wrong confidence; a 99% call on a miss "
            "is punished far harder than a 60% call. Lower = better.\n"
            "- **Brier** — mean squared gap between probability and outcome; honest "
            "across the full xG range. Lower = better.\n"
            "- **Calibration** — shots valued at 0.25 should score about a quarter "
            "of the time; a curve hugging the diagonal deserves trust."
        )


# ---------------------------------------------------------------------------
# 3b. Beat the model (mini-game)
# ---------------------------------------------------------------------------


def quiz_page() -> None:
    """Game mode: price five unseen holdout shots against the model, log loss decides."""
    import plotly.graph_objects as go

    hero("Beat the Model")
    artifact = load_model()

    @st.cache_resource(show_spinner="Drawing holdout shots...")
    def _quiz_pool() -> pd.DataFrame:
        """All non-penalty shots from the grouped test split, model-scored."""
        from sklearn.model_selection import GroupShuffleSplit

        features = load_shots()
        X = features[FEATURE_COLUMNS]
        y = features["is_goal"].astype(int)
        mask = ~features["is_penalty"].fillna(False)
        Xt, yt, groups = X.loc[mask], y.loc[mask], features.loc[mask, "match_id"]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=TEST_SPLIT_SEED)
        _train_index, test_index = next(splitter.split(Xt, yt, groups))
        pool = features.loc[Xt.index[test_index]].copy()
        probs = artifact["pipeline"].predict_proba(pool[FEATURE_COLUMNS])[:, 1]
        calibrator = artifact.get("calibrator")
        if calibrator is not None:
            probs = calibrator.predict(probs)
        pool["p_model"] = probs
        return pool

    pool = _quiz_pool()
    ss = st.session_state

    head_cols = st.columns([1.2, 3])
    if head_cols[0].button("Deal 5 new shots", type="primary"):
        ss["bt_round"] = pool.sample(5)
        for i in range(5):
            ss.pop(f"bt_g_{i}", None)
        ss.pop("bt_scored", None)
    if "bt_round" not in ss or not len(ss["bt_round"]):
        ss["bt_round"] = pool.sample(5)
    head_cols[1].caption(
        "Every deal draws from matches the model never trained on. Price the chance "
        "in your head, we grade your log loss against the model's on the same five shots."
    )

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
    round_df = ss["bt_round"]
    for i, (_idx, shot) in enumerate(round_df.iterrows(), start=1):
        c_info, c_guess = st.columns([3.2, 1], gap="medium")
        with c_info:
            st.markdown(f"##### Shot {i}")
            chips = "".join(
                f'<span class="piq-chip">' f"{float(shot['distance']):.0f} m</span>"
                f'<span class="piq-chip">angle {float(shot["angle_deg"]):.0f}&#176;</span>'
                f'<span class="piq-chip">{str(shot["body_part"])}</span>'
                f'<span class="piq-chip">{str(shot["shot_type"])}</span>'
                f'<span class="piq-chip">{"pressured" if bool(shot["under_pressure"]) else "no pressure"}</span>'
                f'<span class="piq-chip">{"first-time" if bool(shot["first_time"]) else "controlled touch"}</span>'
            )
            st.markdown(f'<div class="piq-crumbs">{chips}</div>', unsafe_allow_html=True)
        with c_guess:
            st.slider("Your price (%)", 0, 100, 50, key=f"bt_g_{i}", format="%d%%")

    with st.container(key="goal-panel"):
        overview_fig = go.Figure()
        gx = round_df["x"].astype(float)
        gy = round_df["y"].astype(float)
        overview_fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="markers+text", text=[str(i + 1) for i in range(len(round_df))],
            textposition="top center", textfont=dict(color=TEXT, size=13, family=DISPLAY),
            marker=dict(size=15, color=GREEN, opacity=0.9,
                        line=dict(width=0.8, color="#0b0e14")),
            customdata=[[float(d), float(a)] for d, a in zip(round_df["distance"], round_df["angle_deg"])],
            hovertemplate="Shot #%{text}<br>%{customdata[0]:.0f} m · angle %{customdata[1]:.0f}°<extra></extra>",
        ))
        overview_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=HOVERLABEL, font=dict(family=BODY, color=TEXT_DIM),
            shapes=PITCH_SHAPES, height=360, margin=dict(l=10, r=14, t=6, b=10),
            xaxis=dict(range=[-3, 126], showgrid=False, zeroline=False, visible=False,
                       scaleanchor="y", scaleratio=1.0),
            yaxis=dict(range=[-3, 83], showgrid=False, zeroline=False, visible=False),
        )
        st.plotly_chart(overview_fig, width="stretch",
                        config=dict(displayModeBar=False, displaylogo=False))

    if st.button("Score this round", type="secondary"):
        ss["bt_scored"] = True

    if ss.get("bt_scored", False):
        epsilon = 1e-6
        outcomes = round_df["is_goal"].astype(bool).to_numpy()
        items = []
        for i, (_idx, shot) in enumerate(round_df.iterrows()):
            items.append(
                dict(
                    i=i,
                    p_user=min(max(st.session_state.get(f"bt_g_{i}", 50) / 100.0, epsilon), 1 - epsilon),
                    p_model=float(shot["p_model"]),
                    goal=bool(shot["is_goal"]),
                )
            )
        ll_user = float(np.mean([-it["goal"] * np.log(it["p_user"]) - (not it["goal"]) * np.log(1 - it["p_user"]) for it in items]))
        ll_model = float(np.mean([-it["goal"] * np.log(it["p_model"]) - (not it["goal"]) * np.log(1 - it["p_model"]) for it in items]))
        brier_user = float(np.mean([(it["p_user"] - it["goal"]) ** 2 for it in items]))
        brier_model = float(np.mean([(it["p_model"] - it["goal"]) ** 2 for it in items]))

        section("1", "Round score", "chart")
        with st.container(key="quiz-panel"):
            if ll_user < ll_model:
                verdict, vcls = "YOU BEAT THE MODEL", ""
                verdict_sub = (
                    f"Log loss {ll_user:.3f} vs {ll_model:.3f} — smaller is better, and yours is "
                    "smaller over these five shots."
                )
            elif ll_user > ll_model:
                verdict, vcls = "MODEL WINS THIS ROUND", "m"
                verdict_sub = (
                    f"Log loss {ll_model:.3f} vs your {ll_user:.3f}. Confident misses cost you "
                    "the round — punishing them is exactly what the model minimises."
                )
            else:
                verdict, vcls = "DEAD HEAT", "m"
                verdict_sub = "Identical log loss over these five shots — deal another round."
            st.markdown(
                f"""<div class="piq-verdict">
<div class="word {vcls}">{verdict}</div>
<div class="rule"></div>
<div class="sub">{verdict_sub}</div>
</div>""",
                unsafe_allow_html=True,
            )
            lead = st.columns(4)
            lead[0].metric("Your log loss", f"{ll_user:.3f}")
            lead[1].metric("Model log loss", f"{ll_model:.3f}")
            lead[2].metric("Your Brier", f"{brier_user:.3f}")
            lead[3].metric("Model Brier", f"{brier_model:.3f}")

        section("2", "The reveal", "target")
        reveal = round_df.copy()
        reveal.insert(0, "Shot", [f"#{k + 1}" for k in range(len(reveal))])
        reveal["Your guess"] = [f"{int(st.session_state.get(f'bt_g_{i}', 50)):d}%" for i in range(len(reveal))]
        reveal["Model xG"] = [f"{v:.3f}" for v in reveal["p_model"]]
        reveal_columns = reveal[
            ["Shot", "player", "team", "distance", "angle_deg", "body_part", "is_goal",
             "Model xG", "Your guess"]
        ].copy()
        reveal_columns.columns = [
            "Shot", "Player", "Team", "Distance (m)", "Angle (deg)", "Body part",
            "Outcome", "Model xG", "Your guess",
        ]
        outcome_fmt = ["GOAL" if g else "No goal" for g in reveal_columns["Outcome"]]
        reveal_columns["Outcome"] = outcome_fmt
        st.dataframe(reveal_columns, width="stretch", hide_index=True)
        st.caption(
            "Shots come from the grouped holdout; the model never saw them in "
            "training. Five-shot samples are noisy — deal a few rounds before "
            "drawing conclusions."
        )


# ---------------------------------------------------------------------------
# 4. Insights
# ---------------------------------------------------------------------------


def insights_page() -> None:
    hero("Team & Player Insights")
    player_stats = load_csv("player_summary.csv").set_index("player")
    team_stats = load_csv("team_summary.csv")

    section("1", "Sample-size control", "target")
    control, _ = st.columns([1, 1.4])
    with control:
        min_shots = st.slider(
            "Minimum shots per player", 5, 100, MIN_SHOTS_DEFAULT, step=5
        )
    st.caption(
        "Goals − xG is noisy on small samples; this slider keeps signals readable "
        "without steering conclusions."
    )

    qualified = player_stats[player_stats["shots"] >= min_shots]

    section("2", "Players & providers", "users")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("##### Player leaderboard (own shots, non-penalty)")
        sort_metric = st.selectbox(
            "Sort by",
            ["xG/shot (quality)", "goals − xG (finishing)", "total xG (volume)"],
        )
        if sort_metric.startswith("xG/shot"):
            ranked = qualified.sort_values("xg_per_shot", ascending=False)
        elif sort_metric.startswith("goals"):
            ranked = qualified.sort_values("goals_minus_xg", ascending=False)
        else:
            ranked = qualified.sort_values("xg_shotworth", ascending=False)
        table = ranked.head(15)[
            ["team", "competition", "shots", "goals", "xg_shotworth", "xg_per_shot", "goals_minus_xg"]
        ].copy()
        table.insert(0, "Player", ranked.head(15).index.astype(str))
        table.columns = [
            "Player", "Team", "Competition", "Shots", "Goals",
            "Total xG", "xG/shot", "Goals − xG",
        ]
        st.dataframe(
            table,
            width="stretch", hide_index=True,
            column_config={
                "Shots": st.column_config.NumberColumn(format="int"),
                "Goals": st.column_config.NumberColumn(format="int"),
                "Total xG": st.column_config.NumberColumn(format="%.2f"),
                "xG/shot": st.column_config.NumberColumn(format="%.3f"),
                "Goals − xG": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

        st.markdown(f"##### Chance providers — min {MIN_ASSISTS} assisted shots (key-pass linked)")
        if "assists" in player_stats.columns and player_stats["assists"].notna().any():
            creators = (
                player_stats[player_stats["assists"] >= MIN_ASSISTS]
                .sort_values("assisted_xg", ascending=False)
                .head(10)
            )
            creator_table = creators[["team", "competition", "assists", "assisted_goals", "assisted_xg"]].copy()
            creator_table.insert(0, "Player", creators.index.astype(str))
            creator_table.columns = [
                "Player", "Team", "Competition", "Assisted shots", "Goals created", "Assisted xG"
            ]
            st.dataframe(
                creator_table,
                width="stretch", hide_index=True,
                column_config={
                    "Assisted shots": st.column_config.NumberColumn(format="int"),
                    "Goals created": st.column_config.NumberColumn(format="int"),
                    "Assisted xG": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption("Assisted shots = key passes linked directly to a shot, not the official assist statistic.")
        else:
            st.info("Creator data is not built yet. Run `python -m src.creators`.")

    with right:
        st.markdown("##### Teams: shot selection & xG differential")
        team_view = team_stats[team_stats["shots_for"] >= 100].sort_values("xg_for", ascending=False).copy()
        team_show = team_view.head(15)[
            ["team", "shots_for", "xg_for", "xg_per_shot_for", "low_quality_share", "xg_diff"]
        ].copy()
        team_show.columns = [
            "Team", "Shots", "xG for", "xG/shot",
            "Shots < 0.05 xG share", "xG differential (for−against)",
        ]
        st.dataframe(
            team_show,
            width="stretch", hide_index=True,
            column_config={
                "Shots": st.column_config.NumberColumn(format="int"),
                "xG for": st.column_config.NumberColumn(format="%.1f"),
                "xG/shot": st.column_config.NumberColumn(format="%.3f"),
                "Shots < 0.05 xG share": st.column_config.NumberColumn(format="%.1%"),
                "xG differential (for−against)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

        zone_fig = FIGURES_DIR / "fig10_team_shot_selection.png"
        if zone_fig.exists():
            st.markdown("##### Team shot selection")
            st.image(str(zone_fig))

    section("3", "Head to head", "users")
    team_list = sorted(team_stats["team"].astype(str).unique())
    volume_pair = (
        team_stats.sort_values("matches", ascending=False)["team"].astype(str).tolist()[:2]
        if len(team_stats) >= 2 else team_list[:1] * 2
    )
    hA, hB = st.columns(2)
    with hA:
        team_a = st.selectbox("Team A", team_list, index=team_list.index(volume_pair[0]) if volume_pair[0] in team_list else 0)
    with hB:
        team_b = st.selectbox(
            "Team B", team_list,
            index=(team_list.index(volume_pair[1]) if volume_pair[1] in team_list else 0),
        )
    if team_a == team_b and len(team_list) > 1:
        team_b = next(t for t in team_list if t != team_a)

    AXIS_LABELS = [
        "Shots per match", "xG per match", "xG per shot",
        "Goals − xG per match", "Low-quality shot share", "xG conceded per match",
    ]

    def team_axes(name: str) -> list[float]:
        r = team_stats.loc[team_stats["team"].astype(str) == name].iloc[0]
        matches = max(float(r["matches"]), 1.0)
        return [
            float(r["shots_for"]) / matches,
            float(r["xg_for"]) / matches,
            float(r["xg_per_shot_for"]),
            (float(r["goals_for"]) - float(r["xg_for"])) / matches,
            float(r["low_quality_share"]),
            float(r["xg_against"]) / matches,
        ]

    axes_a, axes_b = team_axes(team_a), team_axes(team_b)
    radar_vals_a, radar_vals_b = [], []
    for va, vb in zip(axes_a, axes_b):
        lo, hi = min(va, vb), max(va, vb)
        span = (hi - lo) or 1e-9
        radar_vals_a.append(30 + (va - lo) / span * 70)
        radar_vals_b.append(30 + (vb - lo) / span * 70)

    import plotly.graph_objects as go

    radar_fig = go.Figure()
    for name, values, color in ((team_a, radar_vals_a, GREEN), (team_b, radar_vals_b, RED)):
        radar_fig.add_trace(go.Scatterpolar(
            r=values, theta=AXIS_LABELS, name=name, fill="toself", opacity=0.16,
            line=dict(color=color, width=2.4),
            customdata=[[name] * len(AXIS_LABELS)],
            hovertemplate=f"<b>{name}</b><br>%{{theta}}<extra></extra>",
        ))
    radar_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY, color=TEXT_DIM), hoverlabel=HOVERLABEL,
        polar=dict(
            radialaxis=dict(range=[0, 100], showticklabels=False, ticks="",
                            gridcolor="#1a2434", linecolor="#1a2434"),
            angularaxis=dict(gridcolor="#1a2434", linecolor="#1a2434"),
        ),
        height=390, margin=dict(l=30, r=30, t=26, b=26),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        showlegend=True,
    )
    radar_col, read_col = st.columns([1.35, 1], gap="large")
    with radar_col:
        st.plotly_chart(radar_fig, width="stretch")
    with read_col:
        def higher(a: float, b: float) -> str:
            return team_a if a >= b else team_b
        verdicts = "".join([
            f"<div class='sub'>• <b>{AXIS_LABELS[2]}</b>: {higher(axes_a[2], axes_b[2])} picks better shots "
            f"({max(axes_a[2], axes_b[2]):.3f} vs {min(axes_a[2], axes_b[2]):.3f} xG per attempt).</div>",
            f"<div class='sub'>• <b>{AXIS_LABELS[3]}</b>: {higher(axes_a[3], axes_b[3])} overperforms its xG "
            f"({max(axes_a[3], axes_b[3]):+.2f} per match) — finishing, luck or keeper errors.</div>",
            f"<div class='sub'>• <b>{AXIS_LABELS[4]}</b>: {higher(axes_a[4], axes_b[4])} leans more on "
            f"low-quality shots ({max(axes_a[4], axes_b[4]):.0%} of attempts under 0.05 xG).</div>",
            f"<div class='sub'>• <b>{AXIS_LABELS[5]}</b>: {higher(axes_a[5], axes_b[5])} concedes more "
            f"({max(axes_a[5], axes_b[5]):.2f} xG per match faced) — the attack has room.</div>",
        ])
        st.markdown(
            f"""<div class="piq-band"><div class="lead">Read of the pairing</div>{verdicts}
<div class="sub" style="margin-top:6px">Every spoke is rescaled between the two teams (winner of the spoke = outer edge) — it compares style, not absolute strength.</div></div>""",
            unsafe_allow_html=True,
        )

    section("4", "Most valuable zones", "zone")
    zone_fig = FIGURES_DIR / "fig9_zone_map.png"
    left_zone, right_zone = st.columns([1.4, 1], gap="large")
    with left_zone:
        if zone_fig.exists():
            st.image(str(zone_fig))
    with right_zone:
        zone = load_csv("zone_summary.csv")
        zone_top = zone[zone["shots"] >= 60].sort_values("mean_shotworth_xg", ascending=False).head(8)
        zone_top = zone_top[["x_band", "y_band", "shots", "conversion", "mean_shotworth_xg"]]
        zone_top.columns = ["x band", "y band", "Shots", "Conversion", "Mean xG"]
        st.dataframe(
            zone_top,
            width="stretch", hide_index=True,
            column_config={
                "Shots": st.column_config.NumberColumn(format="int"),
                "Conversion": st.column_config.NumberColumn(format="%.1%"),
                "Mean xG": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("Mirrored lateral bands are paired together since the audit; the rows above are the best combined pairs.")

    section("5", "Player cards", "users")
    card_choices = sorted(qualified.index.astype(str).tolist())
    if not card_choices:
        st.caption("Lower the minimum-shots slider to draw cards.")
    else:
        default_player = "Lionel Messi" if "Lionel Messi" in card_choices else card_choices[0]
        chosen_player = st.selectbox("Draw a card", card_choices, index=card_choices.index(default_player))
        prow = qualified.loc[chosen_player]
        finishing_delta = float(prow["goals_minus_xg"])
        if finishing_delta >= 4.0:
            tier, tier_color = "CLINICAL", GREEN
        elif finishing_delta >= 0.5:
            tier, tier_color = "HOT", "#65a30d"
        elif finishing_delta >= -0.5:
            tier, tier_color = "STEADY", AMBER
        else:
            tier, tier_color = "COOL", RED
        assisted = prow.get("assisted_xg")
        bar_scale = max(int(prow["goals"]), float(prow["xg_shotworth"]), 1.0)
        goals_widths = [
            100 * int(prow["goals"]) / bar_scale,
            100 * float(prow["xg_shotworth"]) / bar_scale,
        ]
        stats_grid = [
            ("Shots", f"{int(prow['shots'])}"),
            ("Goals", f"{int(prow['goals'])}"),
            ("Total xG", f"{float(prow['xg_shotworth']):.1f}"),
            ("xG / shot", f"{float(prow['xg_per_shot']):.3f}"),
            ("Goals − xG", f"{finishing_delta:+.2f}"),
            ("Assisted xG", f"{float(assisted):.2f}" if pd.notna(assisted) else "—"),
        ]
        grid_cells = "".join(
            f"""<div style="border:1px dashed {LINE}; border-radius:10px; padding:8px 10px; background:rgba(255,255,255,0.02);">
<div style="font-family:{DISPLAY}; font-weight:700; font-size:26px; line-height:1.05; color:{TEXT};">{value}</div>
<div style="font:600 9.5px/2 {BODY}; letter-spacing:0.16em; text-transform:uppercase; color:{TEXT_DIM};">{label}</div>
</div>"""
            for label, value in stats_grid
        )
        st.markdown(
            f"""<div class="piq-band" style="background:linear-gradient(102deg,{NAVY} 0%,{NAVY_2} 52%, {tint(AMBER,0.16)} 115%);">
<div style="display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start;">
  <div style="flex:1 1 260px;">
    <div class="lead">PLAYER CARD</div>
    <div style="font-family:{DISPLAY}; font-weight:700; font-size:44px; letter-spacing:1px; color:{TEXT}; line-height:1; margin:6px 0 4px 0;">{chosen_player}</div>
    <div class="piq-crumbs">
      <span class="piq-chip">{str(prow['team'])}</span>
      <span class="piq-chip">{str(prow['competition'])}</span>
      <span class="piq-chip" style="border-color:{tint(tier_color,0.55)}; color:{tier_color};">{tier}</span>
    </div>
  </div>
  <div style="flex:2 1 420px; display:grid; grid-template-columns:repeat(3,1fr); gap:8px;">
{grid_cells}
  </div>
</div>
<div style="margin-top:10px;">
  <div style="font:600 9.5px/2 {BODY}; letter-spacing:0.16em; text-transform:uppercase; color:{TEXT_DIM};">Goals vs xG</div>
  {"".join(f"""<div style="display:flex; align-items:center; gap:8px; margin:2px 0;">
<span style="font:600 9.5px/1.6 {BODY}; letter-spacing:0.12em; text-transform:uppercase; color:{TEXT_DIM}; min-width:38px;">{label}</span>
<span style="flex:1; height:10px; border-radius:8px; background:{LINE}; display:inline-block; overflow:hidden;">
<span style="display:block; height:10px; width:{width:.0f}%; border-radius:8px; background:{bar};"></span>
</span>
<span style="font-family:{DISPLAY}; font-weight:700; color:{TEXT}; min-width:40px;">{num:.2f}</span>
</div>""" for label, num, width, bar in zip(("Goals", "xG"), (int(prow['goals']), float(prow['xg_shotworth'])), (goals_widths[0], goals_widths[1]), (GREEN, GREEN)))}
</div>
</div>""",
            unsafe_allow_html=True,
        )
        st.caption("Tier = finishing over/underperformance vs xG among sampled players (CLINICAL ≥ +4). Assisted xG counts key-pass-linked shots only.")

    section("6", "Regression watchlist", "chart")
    if qualified.empty:
        st.caption("Lower the minimum-shots slider for watchlist candidates.")
    else:
        wcol_left, wcol_right = st.columns(2, gap="large")
        with wcol_left:
            st.markdown("##### Likely to cool down — finished far above xG")
            cooled = qualified.sort_values("goals_minus_xg", ascending=False).head(8)[
                ["team", "shots", "goals", "xg_shotworth", "goals_minus_xg"]
            ].copy()
            cooled.insert(0, "Player", cooled.index.astype(str))
            cooled.columns = ["Player", "Team", "Shots", "Goals", "Total xG", "Goals − xG"]
            st.dataframe(
                cooled, width="stretch", hide_index=True,
                column_config={
                    "Shots": st.column_config.NumberColumn(format="int"),
                    "Goals": st.column_config.NumberColumn(format="int"),
                    "Total xG": st.column_config.NumberColumn(format="%.1f"),
                    "Goals − xG": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
        with wcol_right:
            st.markdown("##### Undervalued finishers — quality priced in, goals not yet")
            quality_bar = float(qualified["xg_per_shot"].quantile(0.75))
            undervalued = qualified[
                (qualified["xg_per_shot"] >= quality_bar)
                & (qualified["goals_minus_xg"] < -0.5)
            ].sort_values("goals_minus_xg").head(8)[
                ["team", "shots", "goals", "xg_shotworth", "xg_per_shot", "goals_minus_xg"]
            ].copy()
            if undervalued.empty:
                st.caption("No undervalued finishers at this sample threshold — lower the slider.")
            else:
                undervalued.insert(0, "Player", undervalued.index.astype(str))
                undervalued.columns = [
                    "Player", "Team", "Shots", "Goals", "Total xG", "xG/shot", "Goals − xG"
                ]
                st.dataframe(
                    undervalued, width="stretch", hide_index=True,
                    column_config={
                        "Shots": st.column_config.NumberColumn(format="int"),
                        "Goals": st.column_config.NumberColumn(format="int"),
                        "Total xG": st.column_config.NumberColumn(format="%.1f"),
                        "xG/shot": st.column_config.NumberColumn(format="%.3f"),
                        "Goals − xG": st.column_config.NumberColumn(format="%+.2f"),
                    },
                )
        st.caption(
            "Left: finishing tends to mean-revert toward xG — hot streaks above "
            "expectation often cool; hot form is a scouting flag, not a guarantee. "
            "Right: players priced high xG/shot but finishing below expectation — "
            "labelled assumptions, small samples (see the slider) stay noisy."
        )

    section("7", "Scout card", "target")
    league_xg_ps = float(team_stats["xg_per_shot_for"].mean())
    league_lowq = float(team_stats["low_quality_share"].mean())
    scout_team = st.selectbox("Scout which team?", team_list, index=team_list.index(volume_pair[0]) if volume_pair[0] in team_list else 0)
    srow = team_stats.loc[team_stats["team"].astype(str) == scout_team].iloc[0]
    smatches = max(float(srow["matches"]), 1.0)
    sq = qualified[qualified["team"].astype(str) == scout_team]
    outlet = sq.sort_values("shots", ascending=False).iloc[0] if not sq.empty else None
    if "assists" in qualified.columns and qualified["assists"].notna().any():
        provq = qualified[(qualified["team"].astype(str) == scout_team) & (qualified["assists"] >= MIN_ASSISTS)]
        provider = provq.sort_values("assisted_xg", ascending=False).iloc[0] if not provq.empty else None
    else:
        provider = None

    style_read = (
        "quality-first shot selection" if float(srow["xg_per_shot_for"]) >= league_xg_ps + 0.005
        else "patient volume shooter" if float(srow["xg_per_shot_for"]) >= league_xg_ps - 0.005
        else "shot-happy from distance; volume over quality"
    )

    def md_line() -> str:
        lines = [
            f"## Scout card — {scout_team}",
            f"- Matches: {int(srow['matches'])} · Shots: {int(srow['shots_for'])} · xG: {float(srow['xg_for']):.1f} · xG/shot: {float(srow['xg_per_shot_for']):.3f}",
            f"- Shot diet: {float(srow['low_quality_share']):.0%} of attempts under 0.05 xG, {float(srow['long_shot_share']):.0%} from outside the box (league means: {league_lowq:.0%}, xG/shot {league_xg_ps:.3f})",
        ]
        if outlet is not None:
            lines.append(f"- Main outlet: {outlet.name} — {int(outlet['shots'])} shots, {int(outlet['goals'])} goals, {float(outlet['xg_shotworth']):.1f} xG, {float(outlet['goals_minus_xg']):+.2f} finishing over xG")
        if provider is not None:
            lines.append(f"- Main creator: {provider.name} — {int(provider['assists'])} assisted shots worth {float(provider['assisted_xg']):.1f} xG")
        lines.append(f"- Defensive frame: {float(srow['shots_against']) / smatches:.1f} shots and {float(srow['xg_against']) / smatches:.2f} xG faced per match; total differential {float(srow['xg_diff']):+.1f}")
        lines.append(f"- Style read: {style_read}")
        return "\n".join(lines)

    chips = "".join(
        f'<span class="piq-chip">{int(srow["matches"])} matches</span>'
        f'<span class="piq-chip">{int(srow["shots_for"])} shots</span>'
        f'<span class="piq-chip">{float(srow["xg_for"]):.1f} xG</span>'
        f'<span class="piq-chip">{float(srow["xg_per_shot_for"]):.3f} xG/shot</span>'
        f'<span class="piq-chip">{float(srow["xg_diff"]):+.1f} differential</span>'
    )
    bullets = [
        f"<div class='sub'>• <b>Shot diet</b> — {float(srow['low_quality_share']):.0%} of attempts under 0.05 xG, "
        f"{float(srow['long_shot_share']):.0%} from outside the box. League means: {league_lowq:.0%} and {league_xg_ps:.3f} xG/shot.</div>",
        f"<div class='sub'>• <b>Style read</b> — {style_read}.</div>",
        f"<div class='sub'>• <b>Back line</b> — {float(srow['shots_against']) / smatches:.1f} shots and "
        f"{float(srow['xg_against']) / smatches:.2f} xG faced per match; overall differential {float(srow['xg_diff']):+.1f} — "
        f"attack through them, not only over them.</div>",
    ]
    if outlet is not None:
        bullets.insert(1, f"<div class='sub'>• <b>Main outlet</b> — {outlet.name}: {int(outlet['shots'])} shots, "
                          f"{int(outlet['goals'])} goals, {float(outlet['xg_shotworth']):.1f} xG, finishing "
                          f"{float(outlet['goals_minus_xg']):+.2f} above expectation.</div>")
    if provider is not None:
        bullets.insert(2 if outlet is not None else 2, f"<div class='sub'>• <b>Main creator</b> — {provider.name}: "
                          f"{int(provider['assists'])} assisted shots worth {float(provider['assisted_xg']):.1f} xG.</div>")
    st.markdown(
        f"""<div class="piq-band" style="background:linear-gradient(102deg,{NAVY} 0%,{NAVY_2} 55%, rgba(28,94,54,0.5) 110%)">
<div class="lead">SCOUT CARD · {scout_team}</div>
<div class="piq-crumbs">{chips}</div>
<div style="height:4px"></div>
{''.join(bullets)}
</div>""",
        unsafe_allow_html=True,
    )
    st.download_button("Download card as Markdown", data=md_line().encode("utf-8"),
                       file_name=f"scout_card_{scout_team.lower().replace(' ', '_')}.md",
                       mime="text/markdown")

    with st.expander("Manager recommendations (from the model, every number labelled)", expanded=True):
        st.markdown(load_markdown("coach_recommendations.md"))


# ---------------------------------------------------------------------------
# Sidebar, artifact check, routing
# ---------------------------------------------------------------------------


def sidebar() -> str:
    """Branded navigation rail."""
    logo = ASSETS_DIR / "logo.svg"
    logo_html = (
        f'<span class="mark"><img src="data:image/svg+xml;base64,'
        f'{base64.b64encode(logo.read_bytes()).decode()}" width="34" height="34" alt=""></span>'
        if logo.exists() else '<span class="mark"></span>'
    )
    st.sidebar.markdown(
        f"""<div class="piq-brand">{logo_html}
<div><div class="name">SHOTWORTH</div><div class="tag">xG Intelligence</div></div>
</div>""",
        unsafe_allow_html=True,
    )
    page = st.sidebar.selectbox("Menu", list(page_titles.keys()), key="menu")
    st.sidebar.divider()
    st.sidebar.markdown(
        f"""<a class="piq-pill solid" href="{DEMO_URL}" target="_blank">{icon('arrow', 13)} LIVE DEMO</a>
<a class="piq-pill" href="{REPO_URL}" target="_blank">{icon('github', 14)} GITHUB REPO</a>""",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.caption(CREDIT)
    return page


def render_splash() -> None:
    """Full-screen poster splash for the first paint; CSS fades it away."""
    logo = ASSETS_DIR / "logo.svg"
    logo_html = (
        f'<img src="data:image/svg+xml;base64,'
        f'{base64.b64encode(logo.read_bytes()).decode()}" width="30" height="30" alt="">'
        if logo.exists() else ""
    )
    st.markdown(
        f"""<style>
.splash {{
    position: fixed; inset: 0; z-index: 100000; overflow: hidden;
    pointer-events: none;
    background:
        radial-gradient(1100px 520px at 82% -12%, {tint(BLUE_DEEP, 0.45)}, transparent 70%),
        radial-gradient(800px 420px at -8% 112%, {tint(GREEN, 0.20)}, transparent 70%),
        linear-gradient(150deg, {NAVY} 0%, {INK} 80%);
    display: flex; align-items: center; justify-content: center;
    animation: splashGone 4.6s cubic-bezier(.4,0,.2,1) .1s forwards;
}}
.splash .wedge {{
    position: absolute; top: -40%; right: -4%; width: 320px; height: 190%;
    background: linear-gradient(180deg, {tint(BLUE_DEEP, 0.5)}, {tint(BLUE_DEEP, 0)});
    transform: rotate(18deg); border-left: 1.4px dashed rgba(78,161,255,0.35);
}}
.splash .wedge2 {{
    position: absolute; bottom: -52%; right: 22%; width: 220px; height: 170%;
    background: linear-gradient(180deg, {tint(GREEN, 0.32)}, {tint(GREEN, 0)});
    transform: rotate(-14deg);
}}
.splash-inner {{ width: min(780px, 92vw); position: relative; }}
.splash-item {{ animation: splashUp .8s cubic-bezier(.2,.8,.2,1) both; }}
.splash-d1 {{ animation-delay: .05s; }}
.splash-d2 {{ animation-delay: .18s; }}
.splash-d3 {{ animation-delay: .32s; }}
.splash-d4 {{ animation-delay: .5s; }}
@keyframes splashUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes splashGone {{ 0%, 62% {{ opacity: 1; }} 100% {{ opacity: 0; visibility: hidden; }} }}
@keyframes sweep {{ from {{ width: 0; }} to {{ width: 130px; }} }}
.splash-brand {{ display: flex; align-items: center; gap: 10px; color: {TEXT_DIM};
    font: 600 11px/2.1 {BODY}; letter-spacing: .24em; text-transform: uppercase; }}
.splash-word {{ font-family: {DISPLAY}; font-weight: 700; font-size: clamp(88px, 11.5vw, 150px);
    line-height: .92; letter-spacing: 2px; color: {TEXT}; text-transform: uppercase; margin: 12px 0 0 0; }}
.splash-word .ghost {{ color: transparent; -webkit-text-stroke: 1.6px rgba(230,237,243,.52); }}
.splash-rule {{ height: 3px; width: 130px; background: {GREEN}; border-radius: 2px;
    margin: 16px 0 12px 0; animation: sweep .9s .55s cubic-bezier(.2,.8,.2,1) both; }}
.splash-sub {{ font: 400 15.5px/1.65 {BODY}; color: {TEXT_DIM}; max-width: 520px; }}
.splash-stats {{ display: flex; gap: 34px; margin: 16px 0 0 0; flex-wrap: wrap; }}
.splash-stats .num {{ font-family: {DISPLAY}; font-weight: 700; font-size: 38px; line-height: 1; color: {TEXT}; }}
.splash-stats .lbl {{ font: 600 9.5px/2.2 {BODY}; letter-spacing: .18em; text-transform: uppercase; color: {TEXT_DIM}; }}
.splash .flash {{ position: absolute; left: 0; right: 0; bottom: 26px; display: flex;
    justify-content: center; color: {TEXT_DIM}; font: 600 10px/2 {BODY}; letter-spacing: .2em;
    text-transform: uppercase; }}
</style>""",
        unsafe_allow_html=True,
    )
    # Fast identity read for the headline numbers - no model scoring, so the
    # splash paints immediately even on a cold session.
    shots = pd.read_parquet(DATA_DIR / "shots.parquet")
    shots_count = int((~shots["is_penalty"].fillna(False)).sum())
    matches_count = int(shots["match_id"].nunique())

    st.markdown(
        f"""<div class="splash">
  <span class="wedge"></span><span class="wedge2"></span>
  <div class="splash-inner">
    <div class="splash-brand splash-item splash-d1">{logo_html} SHOTWORTH · xG INTELLIGENCE</div>
    <div class="splash-word splash-item splash-d2">SHOT <span class="ghost">WORTH</span></div>
    <div class="splash-rule"></div>
    <div class="splash-sub splash-item splash-d3">Every shot has a price. This app prices
    {shots_count:,} real attempts from {matches_count} matches of StatsBomb Open Data —
    with a transparent model, honest metrics and coach-facing insights.</div>
    <div class="splash-stats splash-item splash-d4">
      <div><div class="num">{shots_count:,}</div><div class="lbl">Shots priced</div></div>
      <div><div class="num">{matches_count}</div><div class="lbl">Matches</div></div>
      <div><div class="num">3</div><div class="lbl">Competitions</div></div>
      <div><div class="num">0.796</div><div class="lbl">Test AUC</div></div>
    </div>
  </div>
  <div class="flash">From box scores to probabilities · one shot at a time</div>
</div>""",
        unsafe_allow_html=True,
    )


def artifacts_fresh(meta_path: Path) -> bool:
    """True when the model metadata matches the audited protocol and every
    downstream artifact is newer than the training timestamp."""
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if (
        metadata.get("audit_protocol_version") != "2026-10-29-v1"
        or metadata.get("split", {}).get("test_split_seed") != TEST_SPLIT_SEED
    ):
        return False
    try:
        built_at_value = metadata.get("pipeline_started_at_utc", metadata["trained_at_utc"])
        built_at = datetime.fromisoformat(built_at_value).timestamp()
        required_outputs = [
            MODELS_DIR / "best_model.joblib",
            DATA_DIR / "creators.parquet",
            REPORTS_DIR / "logreg_ablation.csv",
            REPORTS_DIR / "model_comparison.csv",
            REPORTS_DIR / "calibration_comparison.csv",
            REPORTS_DIR / "player_summary.csv",
            REPORTS_DIR / "team_summary.csv",
            REPORTS_DIR / "zone_summary.csv",
            REPORTS_DIR / "coach_recommendations.md",
            REPORTS_DIR / "insights_summary.md",
            REPORTS_DIR / "permutation_importance.csv",
            REPORTS_DIR / "shap_top_factors.csv",
            FIGURES_DIR / "fig5_calibration.png",
            FIGURES_DIR / "fig9_zone_map.png",
            FIGURES_DIR / "fig10_team_shot_selection.png",
            FIGURES_DIR / "fig11_player_finishing.png",
        ]
        return all(
            path.exists() and path.stat().st_mtime >= built_at
            for path in required_outputs
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def main() -> None:
    st.set_page_config(
        page_title="ShotWorth — Expected Goals Intelligence",
        page_icon=":material/sports_soccer:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    global_css()

    # Poster splash owns the very first paint: it is the FIRST element
    # streamed, so the page can never flash before it. It holds ~3s while the
    # page completes underneath, then fades itself away (CSS forwards) and
    # never renders again this session.
    if not st.session_state.get("splashed", False):
        render_splash()
        st.session_state["splashed"] = True

    page = sidebar()

    # Artifact integrity: warn if model/reports do not match the audited
    # protocol (regeneration missing, or files older than the model build).
    if not artifacts_fresh(MODELS_DIR / "metadata.json"):
        st.warning(
            "Model and/or report artifacts are out of sync with the post-audit "
            "pipeline. Run the full pipeline (`data`, `features`, `eda`, `train`, "
            "`creators`, `insights`) and restart the app — do not quote its "
            "numbers as final results."
        )

    if page == "Front Page":
        front_page()
    elif page == "Shot Map":
        shot_map_page()
    elif page == "Match Timeline":
        timeline_page()
    elif page == "xG Estimator":
        estimator_page()
    elif page == "Model Evaluation":
        evaluation_page()
    elif page == "Beat the Model":
        quiz_page()
    else:
        insights_page()

    footer_band()


if __name__ == "__main__":
    main()
