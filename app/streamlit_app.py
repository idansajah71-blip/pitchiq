"""PitchIQ — Expected Goals Intelligence (Streamlit UI).

Sportswear-poster design language: giant display typography, diagonal
navy-to-green bands, dotted stat cards, numbered sections and real inline-SVG
icons (no emoji). Pages: shot map, xG estimator, model evaluation, insights.
All numbers come from the saved pipeline (train.py + reports); StatsBomb xG is
a benchmark only, never an input feature.

Run from the project root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

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

REPO_URL = "https://github.com/idansajah71-blip/pitchiq"
DEMO_URL = "https://pitchiq-xg.streamlit.app"

CREDIT = (
    "Data: StatsBomb Open Data (via statsbombpy). PitchIQ adalah proyek "
    "hackathon independen dan tidak berafiliasi dengan StatsBomb."
)

page_titles: dict[str, str] = {
    "Shot Map": "Shot Map — kualitas setiap tembakan",
    "xG Estimator": "xG Estimator — prediksi probabilitas gol",
    "Evaluasi Model": "Evaluasi Model — seberapa jujur prediksinya",
    "Insight Pemain & Tim": "Insight Pemain & Tim — leaderboard & rekomendasi",
}

# Poster configuration per page: index letter, display words, subtitle.
page_posters: dict[str, dict[str, str]] = {
    "Shot Map": dict(
        index="A", word_1="SHOT", word_2="MAP",
        subtitle="Setiap titik = satu tembakan, serangan menghadap gawang di kanan. "
                 "Ukuran & warna titik = xG: makin besar dan makin hijau, makin besar peluang jadi gol.",
        asset="hero_shotmap.png",
    ),
    "xG Estimator": dict(
        index="B", word_1="xG", word_2="ESTIMATOR",
        subtitle="Geser posisi & situasi tembakan apa pun — probabilitas gol dihitung "
                 "real-time dengan model yang sama seperti di dataset nyata.",
        asset="hero_estimator.png",
    ),
    "Evaluasi Model": dict(
        index="C", word_1="MODEL", word_2="AUDIT",
        subtitle="Holdout berkelompok per match tanpa bocor, metrik probabilitas yang "
                 "jujur, dan benchmark StatsBomb xG diuji pada data yang identik.",
        asset="hero_evaluation.png",
    ),
    "Insight Pemain & Tim": dict(
        index="D", word_1="TEAM", word_2="INSIGHT",
        subtitle="Leaderboard scoped per pemain–tim–kompetisi, pola zona, dan "
                 "rekomendasi pelatih berlabel asumsi — bukan klaim kausal.",
        asset="hero_insights.png",
    ),
}

# --- Design tokens ------------------------------------------------------------
INK = "#0b0e14"
NAVY = "#101a2c"
NAVY_2 = "#152944"
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

XG_SCALE = [
    [0.0, "#7f1d1d"],
    [0.15, "#b91c1c"],
    [0.35, "#ea580c"],
    [0.55, "#d9a406"],
    [0.75, "#65a30d"],
    [1.0, "#16a34a"],
]
HOVERLABEL = dict(bgcolor="#0e1520", bordercolor="#26364a", font=dict(color=TEXT, size=13))

PITCH_LINE = "rgba(230, 237, 243, 0.5)"

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


@st.cache_resource(show_spinner="Memuat model...")
def load_model() -> dict[str, Any]:
    """Load the artifact: {model_name, pipeline, calibrator, feature_columns, metadata}."""
    return joblib.load(MODELS_DIR / "best_model.joblib")


@st.cache_data(show_spinner="Memuat data tembakan...")
def load_shots() -> pd.DataFrame:
    """Load features and score every shot with the saved pipeline."""
    features = pd.read_parquet(FEATURES_PARQUET)
    artifact = load_model()
    probabilities = artifact["pipeline"].predict_proba(features[FEATURE_COLUMNS])[:, 1]
    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        probabilities = calibrator.predict(probabilities)
    features["pitchiq_xg"] = probabilities
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

/* Poster hero band */
.piq-hero {{
    position: relative; overflow: hidden;
    background: linear-gradient(102deg, {NAVY} 0%, {NAVY_2} 52%, rgba(28,94,54,0.55) 100%);
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 24px 26px 20px 26px;
    margin-bottom: 4px;
}}
.piq-hero::before {{
    content: ""; position: absolute; top: -55%; right: 6%;
    width: 280px; height: 190%;
    background: linear-gradient(180deg, rgba(46,194,126,0.20), rgba(46,194,126,0));
    transform: rotate(16deg);
}}
.piq-hero::after {{
    content: ""; position: absolute; top: 0; bottom: 0; left: 0;
    width: 3px; background: linear-gradient(180deg, {GREEN}, transparent);
}}
.piq-hero-img {{
    position: absolute; right: 18px; bottom: -8px; height: 158px;
    filter: drop-shadow(0 12px 26px rgba(0,0,0,0.55));
    pointer-events: none;
}}
.piq-index {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    border: 1.5px solid {TEXT_DIM}; color: {TEXT_DIM};
    font: 600 12px/1 {BODY}; letter-spacing: 0;
    vertical-align: middle; margin-right: 10px;
}}
.piq-word {{
    display: block;
    font-family: {DISPLAY}; font-weight: 700;
    font-size: clamp(42px, 5.6vw, 74px);
    line-height: 0.95; letter-spacing: 2px;
    text-transform: uppercase; color: {TEXT};
    margin: 8px 0 10px 0; white-space: nowrap;
}}
.piq-word .ghost {{
    color: transparent;
    -webkit-text-stroke: 1.5px rgba(230,237,243,0.42);
    margin-left: 0.22em;
}}
.piq-word .amp {{ color: {GREEN}; }}
.piq-sub {{
    font: 400 13.5px/1.55 {BODY}; color: {TEXT_DIM};
    max-width: 640px; margin: 0 0 10px 0;
}}
.piq-crumbs {{ margin-top: 6px; }}
.piq-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 3px 11px; margin-right: 7px;
    border: 1px solid {LINE}; border-radius: 999px;
    background: rgba(255,255,255,0.03);
    color: {TEXT_DIM}; font: 600 10.5px/1.9 {BODY};
    letter-spacing: 0.14em; text-transform: uppercase;
}}
.piq-chip svg {{ color: {GREEN}; }}

/* Section headings: letter circle + spaced caps + rule */
.piq-section {{
    display: flex; align-items: center; gap: 12px;
    margin: 22px 0 12px 0;
}}
.piq-section .idx {{
    flex: 0 0 auto; width: 27px; height: 27px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    background: {GREEN_DIM}; color: {GREEN};
    font: 700 12px/1 {DISPLAY}; border: 1px solid rgba(46,194,126,0.4);
}}
.piq-section .t {{
    flex: 0 0 auto; white-space: nowrap;
    font: 700 12.5px/1.6 {BODY}; letter-spacing: 0.16em;
    text-transform: uppercase; color: {TEXT};
}}
.piq-section .rule {{ flex: 1 1 auto; height: 1px; background: {LINE}; }}

/* Metric tiles: dotted border, big numbers */
div[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {PANEL} 0%, #0f1622 100%);
    border: 1.5px dashed {LINE};
    border-radius: 12px;
    padding: 14px 16px 10px 16px;
}}
div[data-testid="stMetricLabel"] p {{ margin-bottom: 2px; }}

/* Dataframe panels */
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; }}

/* Pill buttons (sidebar) */
.piq-pill {{
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 14px; border-radius: 999px; margin: 3px 6px 3px 0;
    border: 1.4px solid {LINE}; background: {PANEL};
    color: {TEXT} !important; text-decoration: none !important;
    font: 600 11px/2 {BODY}; letter-spacing: 0.12em; text-transform: uppercase;
    transition: border-color .15s ease, color .15s ease;
}}
.piq-pill:hover {{ border-color: {GREEN}; color: {GREEN} !important; }}
.piq-pill svg {{ color: {GREEN}; }}

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
</style>
""",
        unsafe_allow_html=True,
    )


def hero(page: str) -> None:
    """Poster band: giant display words + diagonal band + numbered index."""
    cfg = page_posters[page]
    img = ASSETS_DIR / cfg["asset"]
    img_html = (
        f'<img class="piq-hero-img" src="data:image/png;base64,{__import__("base64").b64encode(img.read_bytes()).decode()}" alt="">'
        if img.exists() else ""
    )
    word = (
        f"<span>{cfg['word_1']}</span>"
        f"<span class='ghost'>{cfg['word_2']}</span>"
    )
    chips = "".join(
        f'<span class="piq-chip">{icon(ic, 13)} {txt}</span>'
        for ic, txt in (
            ("pitch", "12.241 tembakan"),
            ("zone", "479 match"),
            ("chart", "3 kompetisi"),
        )
    )
    st.markdown(
        f"""
<div class="piq-hero">
  {img_html}
  <span class="piq-index">{cfg['index']}</span>
  <span class="piq-chip" style="border-color:rgba(46,194,126,0.5);color:{GREEN};">{icon('users',13)} PITCHIQ xG INTELLIGENCE</span>
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
    # Keep real widget spacing tight after the custom rule.
    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


def footer() -> None:
    st.divider()
    st.caption(CREDIT)


# ---------------------------------------------------------------------------
# 1. Shot map
# ---------------------------------------------------------------------------


def filter_and_display(data: pd.DataFrame, show_penalties: bool) -> pd.DataFrame:
    """Return the visible shot frame plus the xG column used for color/size."""
    view = data.copy()
    if show_penalties:
        # Penalty rows are outside the model; show the StatsBomb benchmark there.
        view["xg_display"] = np.where(view["is_penalty"], view["xg_statsbomb"], view["pitchiq_xg"])
    else:
        view = view[~view["is_penalty"].fillna(False)]
        view["xg_display"] = view["pitchiq_xg"]
    return view


def shot_map_page() -> None:
    data = load_shots()
    hero("Shot Map")

    section("1", "Filter tembakan", "target")
    head = st.columns(6)
    team = head[0].selectbox("Tim", ["Semua"] + sorted(data["team"].astype(str).unique().tolist()))
    team_players = sorted(data.loc[data["team"] == team, "player"].astype(str).unique())
    all_players = sorted(data["player"].value_counts().loc[lambda c: c >= 15].index.astype(str).tolist())
    player = head[1].selectbox("Pemain", ["Semua"] + (team_players if team != "Semua" else all_players))
    body = head[2].selectbox("Bagian tubuh", ["Semua"] + sorted(data["body_part"].dropna().astype(str).unique().tolist()))
    shot_type = head[3].selectbox("Tipe tembakan", ["Semua"] + sorted(data["shot_type"].dropna().astype(str).unique().tolist()))
    goal_filter = head[4].radio("Hasil", ["Semua", "Gol", "Bukan gol"], horizontal=True)
    show_penalties = head[5].toggle("Sertakan penalti", value=False)

    view = filter_and_display(data, show_penalties)
    if team != "Semua":
        view = view[view["team"].astype(str) == team]
    if player != "Semua":
        view = view[view["player"].astype(str) == player]
    if body != "Semua":
        view = view[view["body_part"].astype(str) == body]
    if shot_type != "Semua":
        view = view[view["shot_type"].astype(str) == shot_type]
    if goal_filter == "Gol":
        view = view[view["is_goal"]]
    elif goal_filter == "Bukan gol":
        view = view[~view["is_goal"]]

    if view.empty:
        st.info("Tidak ada tembakan pada kombinasi filter ini — longgarkan filternya.")
        return

    section("2", "Ringkasan pilihan", "users")
    metrics = st.columns(3)
    conversion = view["is_goal"].mean()
    metrics[0].metric("Tembakan ditampilkan", f"{len(view):,}")
    metrics[1].metric("Gol", f"{int(view['is_goal'].sum()):,}")
    mean_label = "Rata-rata xG tampilan" if show_penalties else "Rata-rata PitchIQ xG"
    metrics[2].metric(mean_label, f"{view['xg_display'].mean():.3f}", delta=f"konversi {conversion:.1%}")
    if show_penalties:
        st.caption("xG penalti berasal dari benchmark StatsBomb; non-penalti memakai PitchIQ. Rata-rata gabungan mencampur dua sumber — bukan satu model homogen.")

    import plotly.graph_objects as go

    section("3", "Peta tembakan", "pitch")

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
            np.where(view["is_goal"], "GOL", view["outcome"].astype(str)).astype(object),
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
                colorbar=dict(title=dict(text="xG", side="right"), thickness=11, len=0.72,
                              outlinewidth=0, tickfont=dict(size=10)),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b> • menit %{customdata[1]}<br>"
                "Tim: %{customdata[2]}<br>"
                "Bagian: %{customdata[3]} • Tipe: %{customdata[4]}<br>"
                "xG: %{customdata[5]:.3f} • Hasil: %{customdata[6]}<extra></extra>"
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
        height=560,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-3, 126], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="y", scaleratio=1.0),
        yaxis=dict(range=[-3, 83], showgrid=False, zeroline=False, visible=False),
    )
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "**Cara membaca:** titik besar hijau = peluang bagus (xG tinggi); titik "
        "kecil merah = percobaan berisiko (xG rendah). Pakai filter di atas "
        "untuk menjelajah tim/pemain/konteks tembakan."
    )


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
        section("1", "Posisi & situasi", "target")
        x = st.slider("Kedalaman tembakan (x: 60 = garis tengah, 120 = gawang lawan)", 0, 119, 110)
        y = st.slider("Posisi samping (y: 40 = tengah gawang)", 0, 80, 40)
        body_part = st.selectbox("Bagian tubuh", BODY_PARTS)
        shot_type = st.selectbox("Tipe tembakan", SHOT_TYPES)
        under_pressure = st.checkbox("Tembakan di bawah tekanan")
        first_time = st.checkbox("Tembakan first-time (tanpa kontrol dulu)")

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

        section("2", "Hasil model", "bolt")
        prob_col, dist_col, ang_col = st.columns([1.3, 1, 1])
        with prob_col:
            st.markdown(
                f"""<div style="font-family:{DISPLAY};font-weight:700;letter-spacing:1px;
                font-size:56px;line-height:1.02;color:{prob_color};">{probability:.1%}</div>
                <div style="font:600 10.5px/1.6 {BODY};letter-spacing:0.16em;text-transform:uppercase;
                color:{TEXT_DIM};margin-top:2px;">Peluang jadi gol</div>""",
                unsafe_allow_html=True,
            )
            st.progress(min(probability, 1.0))
        dist_col.metric("Jarak ke gawang", f"{distance:.1f}")
        ang_col.metric("Sudut gawang terbuka", f"{angle:.1f}")

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
                name="Tembakan",
                hovertemplate=f"x={x:.0f}, y={y:.0f}, peluang {probability:.0%}<extra></extra>"
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
        st.plotly_chart(goal_fig, width="stretch")

    with st.expander("Pahami angkanya"):
        st.markdown(
            "- **Peluang** = estimasi bagian tembakan serupa yang menjadi gol pada "
            "situasi identik, dari puluhan ribu tembakan nyata.\n"
            "- Estimator **tidak membaca posisi bek/kiper** — fitur freeze-frame "
            "diimputasi median dataset, jadi ini situasi 'tenang'. Dengan tekanan "
            "nyata biasanya menurunkan peluangnya lagi.\n"
            f"- Pembanding: konversi penalti pada data {penalty_rate:.0%} — penalti "
            "dilaporkan terpisah karena 12 yard adalah tugas yang berbeda."
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
    hero("Evaluasi Model")
    artifact = load_model()
    metadata = artifact["metadata"]
    X_test, y_test, sb_xg = _test_split()
    probabilities = artifact["pipeline"].predict_proba(X_test)[:, 1]
    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        probabilities = calibrator.predict(probabilities)

    section("1", "Skor holdout", "chart")
    metrics_head = st.columns(4)
    metrics_head[0].metric("Model deployment", "Logistic Regression")
    metrics_head[1].metric("ROC-AUC (test)", f"{metadata['test_metrics']['roc_auc']:.3f}")
    metrics_head[2].metric("Log loss (test)", f"{metadata['test_metrics']['log_loss']:.3f}")
    metrics_head[3].metric("Brier (test)", f"{metadata['test_metrics']['brier']:.4f}")

    left, right = st.columns([1.15, 1], gap="large")

    with right:
        section("2", "Tabel perbandingan", "users")
        comparison = load_csv("model_comparison.csv")
        show = comparison[
            ["model", "test_roc_auc", "test_log_loss", "test_brier",
             "cv_roc_auc_mean", "cv_roc_auc_std"]
        ].copy()
        show.columns = ["Model", "AUC test", "Log loss", "Brier", "AUC CV", "± CV"]
        st.dataframe(
            show,
            width="stretch",
            hide_index=True,
            column_config={
                "AUC test": st.column_config.NumberColumn(format="%.3f"),
                "Log loss": st.column_config.NumberColumn(format="%.3f"),
                "Brier": st.column_config.NumberColumn(format="%.4f"),
                "AUC CV": st.column_config.NumberColumn(format="%.3f"),
                "± CV": st.column_config.NumberColumn(format="±%.3f"),
            },
        )
        st.caption(
            "Tuned-tree CV adalah eksploratif (belum nested CV). Metrik StatsBomb "
            "ikut diuji hanya untuk konteks, bukan lawan pelatihan."
        )

        left_ablation, right_ablation = st.columns(2, gap="small")
        with left_ablation:
            st.markdown("##### Ablasi LogReg")
            ablation = load_csv("logreg_ablation.csv")
            st.dataframe(
                ablation[["stage", "roc_auc", "brier"]].rename(
                    columns={"stage": "Tahap", "roc_auc": "AUC", "brier": "Brier"}
                ),
                width="stretch", hide_index=True,
                column_config={"AUC": st.column_config.NumberColumn(format="%.3f"),
                               "Brier": st.column_config.NumberColumn(format="%.4f")},
            )
        with right_ablation:
            st.markdown("##### Kalibrasi")
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

        section("3", "Kurva kualitas probabilitas", "chart")
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
            x=[0, 1], y=[0, 1], mode="lines", name="Acak",
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
            x=[0, 1], y=[0, 1], mode="lines", name="Kalibrasi sempurna",
            line=dict(color="#5b6b7e", width=1.2, dash="dash"),
        ))
        cal_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=HOVERLABEL, font=dict(family=BODY, color=TEXT_DIM),
            xaxis=dict(title="Rata-rata xG yang diprediksi", range=[0, 0.5], gridcolor="#1a2434", zeroline=False),
            yaxis=dict(title="Frekuensi gol yang terjadi", range=[0, 0.5], gridcolor="#1a2434", zeroline=False),
            height=400, margin=dict(l=44, r=16, t=28, b=44),
        )
        st.plotly_chart(cal_fig, width="stretch")

    with st.expander("Cara membaca metrik (bahasa awam)"):
        st.markdown(
            "- **ROC-AUC** — seberapa sering model mengurutkan dua tembakan acak "
            "dengan benar (gol vs bukan). 0.5 = menebak, 1.0 = sempurna.\n"
            "- **Log loss** — hukuman keyakinan yang salah; prediksi 99% pada shot "
            "gagal dihukum lebih keras dari prediksi 60%. Kecil = baik.\n"
            "- **Brier** — rata-rata selisih kuadrat antara probabilitas dan hasil "
            "nyata; metrik jujur untuk semua rentang xG. Kecil = baik.\n"
            "- **Kalibrasi** — shot bernilai 0.25 harus masuk ~seperempat waktu; "
            "kurva yang mengikuti diagonal = boleh dipercaya."
        )


# ---------------------------------------------------------------------------
# 4. Insights
# ---------------------------------------------------------------------------


def insights_page() -> None:
    hero("Insight Pemain & Tim")
    player_stats = load_csv("player_summary.csv").set_index("player")
    team_stats = load_csv("team_summary.csv")

    section("1", "Kontrol sampel", "target")
    control, _ = st.columns([1, 1.4])
    with control:
        min_shots = st.slider(
            "Minimum jumlah tembakan per pemain", 5, 100, MIN_SHOTS_DEFAULT, step=5
        )
    st.caption(
        "Gol − xG pada sampel kecil itu bising; slider ini menjaga sinyal "
        "tetap bisa dibaca tanpa mengubah arah kesimpulannya."
    )

    qualified = player_stats[player_stats["shots"] >= min_shots]

    section("2", "Pemain & penyedia peluang", "users")
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("##### Leaderboard pemain (tembakan sendiri, non-penalti)")
        sort_metric = st.selectbox(
            "Urutkan berdasarkan",
            ["xG/shot (kualitas)", "goals − xG (finishing)", "total xG (volume)"],
        )
        if sort_metric.startswith("xG/shot"):
            ranked = qualified.sort_values("xg_per_shot", ascending=False)
        elif sort_metric.startswith("goals"):
            ranked = qualified.sort_values("goals_minus_xg", ascending=False)
        else:
            ranked = qualified.sort_values("xg_pitchiq", ascending=False)
        table = ranked.head(15)[
            ["team", "competition", "shots", "goals", "xg_pitchiq", "xg_per_shot", "goals_minus_xg"]
        ].copy()
        table.insert(0, "Pemain", ranked.head(15).index.astype(str))
        table.columns = [
            "Pemain", "Tim", "Kompetisi", "Tembakan", "Gol",
            "Total xG", "xG/tembakan", "Gol − xG",
        ]
        st.dataframe(
            table,
            width="stretch", hide_index=True,
            column_config={
                "Tembakan": st.column_config.NumberColumn(format="int"),
                "Gol": st.column_config.NumberColumn(format="int"),
                "Total xG": st.column_config.NumberColumn(format="%.2f"),
                "xG/tembakan": st.column_config.NumberColumn(format="%.3f"),
                "Gol − xG": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

        st.markdown(f"##### Penyedia peluang — min {MIN_ASSISTS} assisted shots (key-pass linked)")
        if "assists" in player_stats.columns and player_stats["assists"].notna().any():
            creators = (
                player_stats[player_stats["assists"] >= MIN_ASSISTS]
                .sort_values("assisted_xg", ascending=False)
                .head(10)
            )
            creator_table = creators[["team", "competition", "assists", "assisted_goals", "assisted_xg"]].copy()
            creator_table.insert(0, "Pemain", creators.index.astype(str))
            creator_table.columns = [
                "Pemain", "Tim", "Kompetisi", "Assisted shots", "Gol dari peluang", "xG assist"
            ]
            st.dataframe(
                creator_table,
                width="stretch", hide_index=True,
                column_config={
                    "Assisted shots": st.column_config.NumberColumn(format="int"),
                    "Gol dari peluang": st.column_config.NumberColumn(format="int"),
                    "xG assist": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption("Assisted shots = key pass yang tertaut langsung ke tembakan, bukan total assist resmi.")
        else:
            st.info("Data creators belum dibangun. Jalankan `python -m src.creators`.")

    with right:
        st.markdown("##### Tim: pemilihan tembakan & selisih xG")
        team_view = team_stats[team_stats["shots_for"] >= 100].sort_values("xg_for", ascending=False).copy()
        team_show = team_view.head(15)[
            ["team", "shots_for", "xg_for", "xg_per_shot_for", "low_quality_share", "xg_diff"]
        ].copy()
        team_show.columns = [
            "Tim", "Tembakan", "xG untuk", "xG/tembakan",
            "Share tembakan < 0.05 xG", "Selisih xG (untuk−terhadap)",
        ]
        st.dataframe(
            team_show,
            width="stretch", hide_index=True,
            column_config={
                "Tembakan": st.column_config.NumberColumn(format="int"),
                "xG untuk": st.column_config.NumberColumn(format="%.1f"),
                "xG/tembakan": st.column_config.NumberColumn(format="%.3f"),
                "Share tembakan < 0.05 xG": st.column_config.NumberColumn(format="%.1%"),
                "Selisih xG (untuk−terhadap)": st.column_config.NumberColumn(format="%+.1f"),
            },
        )

        zone_fig = FIGURES_DIR / "fig10_team_shot_selection.png"
        if zone_fig.exists():
            st.markdown("##### Pemilihan tembakan per tim")
            st.image(str(zone_fig))

    section("3", "Zona paling produktif", "zone")
    zone_fig = FIGURES_DIR / "fig9_zone_map.png"
    left_zone, right_zone = st.columns([1.4, 1], gap="large")
    with left_zone:
        if zone_fig.exists():
            st.image(str(zone_fig))
    with right_zone:
        zone = load_csv("zone_summary.csv")
        zone_top = zone[zone["shots"] >= 60].sort_values("mean_pitchiq_xg", ascending=False).head(8)
        zone_top = zone_top[["x_band", "y_band", "shots", "conversion", "mean_pitchiq_xg"]]
        zone_top.columns = ["Band x", "Band y", "Tembakan", "Konversi", "Rata-rata xG"]
        st.dataframe(
            zone_top,
            width="stretch", hide_index=True,
            column_config={
                "Tembakan": st.column_config.NumberColumn(format="int"),
                "Konversi": st.column_config.NumberColumn(format="%.1%"),
                "Rata-rata xG": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("Zona band lateral yang mirip dipasangkan sejak audit; baris di atas adalah pasangan terbaik setelah digabung.")

    with st.expander("Rekomendasi untuk pelatih (dari model, seluruh label asumsi)", expanded=True):
        st.markdown(load_markdown("coach_recommendations.md"))


# ---------------------------------------------------------------------------
# Sidebar, artifact check, routing
# ---------------------------------------------------------------------------


def sidebar() -> None:
    """Branded navigation rail."""
    logo = ASSETS_DIR / "logo.svg"
    logo_html = (
        f'<span class="mark"><img src="data:image/svg+xml;base64,'
        f'{__import__("base64").b64encode(logo.read_bytes()).decode()}" width="34" height="34" alt=""></span>'
        if logo.exists() else '<span class="mark"></span>'
    )
    st.sidebar.markdown(
        f"""<div class="piq-brand">{logo_html}
<div><div class="name">PITCHIQ</div><div class="tag">xG Intelligence</div></div>
</div>""",
        unsafe_allow_html=True,
    )
    page = st.sidebar.selectbox("Menu", list(page_titles.keys()))
    st.sidebar.divider()
    st.sidebar.markdown(
        f"""<a class="piq-pill" href="{REPO_URL}" target="_blank">{icon('github', 14)} REPO GITHUB</a>
<a class="piq-pill" href="{DEMO_URL}" target="_blank">{icon('arrow', 13)} DEMO LIVE</a>""",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.caption(CREDIT)
    return page


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
        page_title="PitchIQ — Expected Goals Intelligence",
        page_icon=":material/sports_soccer:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    global_css()

    page = sidebar()

    # Poster hero is rendered inside each page function, right before content.

    # Artifact integrity: warn if model/reports do not match the audited
    # protocol (regeneration missing, or files older than the model build).
    if not artifacts_fresh(MODELS_DIR / "metadata.json"):
        st.warning(
            "Artefak model dan/atau laporan belum sinkron dengan pipeline "
            "post-audit. Jalankan pipeline lengkap (`data`, `features`, `eda`, "
            "`train`, `creators`, `insights`), lalu mulai ulang aplikasi — jangan "
            "kutip metriknya sebagai hasil final."
        )

    if page == "Shot Map":
        shot_map_page()
    elif page == "xG Estimator":
        estimator_page()
    elif page == "Evaluasi Model":
        evaluation_page()
    else:
        insights_page()

    footer()


if __name__ == "__main__":
    main()
