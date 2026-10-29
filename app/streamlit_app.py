"""PitchIQ - Expected Goals Intelligence (Streamlit UI).

Pages: shot map, xG estimator, model evaluation, player/team insights.
All numbers come from the saved pipeline (train.py + reports). The StatsBomb
xG value is used only as a benchmark, never as an input feature.

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
LOW_QUALITY_XG = 0.05

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

page_subtitles: dict[str, str] = {
    "Shot Map": "Setiap titik = satu tembakan (serangan ke gawang di kanan). Ukuran & warna = xG — makin besar dan makin hijau, makin besar peluang jadi gol.",
    "xG Estimator": "Geser posisi & situasi tembakan anggakan — probabilitas gol dihitung real-time dengan model yang sama seperti di dataset nyata.",
    "Evaluasi Model": "Holdout berkelompok per match (tanpa bocor), metrik probabilitas yang jujur, dan benchmark StatsBomb xG pada data yang identik.",
    "Insight Pemain & Tim": "Leaderboard scoped per pemain–tim–kompetisi, pola zona, dan rekomendasi pelatih yang berlabel asumsi — bukan klaim kausal.",
}

# --- Plotly design tokens ----------------------------------------------------
PITCH_LINE = "#3d4b5c"
GRID_DIM = "#22303f"
ACCENT = "#22d3a5"
ACCENT_SOFT = "rgba(34, 211, 165, 0.35)"
DANGER = "#ef4444"
BENCH = "#f59e0b"
MUTED = "#7c8b9b"

XG_SCALE = [
    [0.0, "#7f1d1d"],
    [0.15, "#b91c1c"],
    [0.35, "#ea580c"],
    [0.55, "#d9a406"],
    [0.75, "#65a30d"],
    [1.0, "#16a34a"],
]
HOVERLABEL = dict(
    bgcolor="#0f172a", bordercolor="#334155", font=dict(color="#e2e8f0", size=13)
)

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

LAYOUT_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=HOVERLABEL,
    font=dict(family="Inter, 'Source Sans Pro', system-ui, sans-serif", color="#c9d4e0"),
)


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


CSS = """
<style>
    div[data-testid="stMetric"] {
        background: #121a24;
        border: 1px solid #22303f;
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetricLabel"] p { margin-bottom: 2px; }
    .pitchiq-brand {
        background: linear-gradient(135deg, #101826 0%, #16283a 60%, #1c4d3f 130%);
        border: 1px solid #22303f;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .pitchiq-brand h2 { margin: 0; letter-spacing: 0.2px; }
    .pitchiq-brand p { margin: 6px 0 0 0; color: #93a4b5; font-size: 0.9em; }
    .pitchiq-headline {
        background: linear-gradient(90deg, rgba(34,211,165,0.16), rgba(34,211,165,0.0) 62%);
        border-left: 3px solid #22d3a5;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 4px 0 14px 0;
    }
    .pitchiq-headline p { margin: 0; color: #aebbc9; }
    .pitchiq-prob {
        font-size: 3.2rem; font-weight: 800; line-height: 1.05;
        letter-spacing: -1px;
    }
    .pitchiq-chip {
        display: inline-block; padding: 2px 10px; margin: 2px 4px 2px 0;
        border: 1px solid #334155; border-radius: 999px;
        background: #121a24; color: #aebbc9; font-size: 0.85em;
    }
    a.pitchiq-link { color: #22d3a5 !important; text-decoration: none; }
</style>
"""


def page_header(page: str) -> None:
    """Render the consistent page headline block."""
    st.title(page_titles[page].split(" — ")[0])
    st.markdown(
        f'<div class="pitchiq-headline"><p>{page_subtitles[page]}</p></div>',
        unsafe_allow_html=True,
    )


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
    page_header("Shot Map")

    with st.expander("🎛 Filter tembakan", expanded=True):
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

    metrics = st.columns(3)
    conversion = view["is_goal"].mean()
    metrics[0].metric("Tembakan ditampilkan", f"{len(view):,}")
    metrics[1].metric("Gol", f"{int(view['is_goal'].sum()):,}")
    mean_label = "Rata-rata xG tampilan" if show_penalties else "Rata-rata PitchIQ xG"
    metrics[2].metric(mean_label, f"{view['xg_display'].mean():.3f}", delta=f"konversi {conversion:.1%}")
    if show_penalties:
        st.caption("xG penalti berasal dari benchmark StatsBomb; non-penalti memakai PitchIQ. Rata-rata gabungan mencampur dua sumber — bukan satu model homogen.")

    import plotly.graph_objects as go

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
                line=dict(width=0.7, color="#0b1117"),
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
        **LAYOUT_BASE,
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
    page_header("xG Estimator")
    artifact = load_model()
    penalty_rate = load_shots().loc[lambda df: df["is_penalty"].fillna(False), "is_goal"].mean()

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("##### Posisi & situasi")
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
            prob_color = "#22c55e"
        elif probability >= 0.12:
            prob_color = "#d9a406"
        else:
            prob_color = "#ef4444"

        prob_col, dist_col, ang_col = st.columns([1.3, 1, 1])
        with prob_col:
            st.markdown(
                f'<div class="pitchiq-prob" style="color:{prob_color}">'
                f'{probability:.1%}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Peluang jadi gol")
            st.progress(min(probability, 1.0))
        dist_col.metric("Jarak ke gawang", f"{distance:.1f}")
        ang_col.metric("Sudut gawang terbuka", f"{angle:.1f}°")

        import plotly.graph_objects as go

        goal_fig = go.Figure()
        goal_fig.add_trace(
            go.Scatter(
                x=[x, 120.0, 120.0, x],
                y=[y, 36.0, 44.0, y],
                fill="toself",
                fillcolor=ACCENT_SOFT,
                line=dict(color=ACCENT, width=2),
                mode="lines",
                hoverinfo="skip",
            )
        )
        goal_fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode="markers",
                marker=dict(size=16, color=DANGER, symbol="x", line=dict(width=2)),
                name="Tembakan",
                hovertemplate=f"x={x:.0f}, y={y:.0f}, peluang {probability:.0%}<extra></extra>",
            )
        )
        goal_fig.update_layout(
            **LAYOUT_BASE,
            shapes=[
                dict(type="line", x0=120, y0=36, x1=120, y1=44, line=dict(color="#ffffff", width=5)),
                dict(type="rect", x0=120, y0=18, x1=114, y1=62, line=dict(color=PITCH_LINE, width=1.4)),
            ],
            height=360,
            margin=dict(l=10, r=10, t=0, b=10),
            title=dict(text=f"<span style='font-size:12.5px'>Sudut antara dua tiang dari posisi tembakan</span>", font=dict(color="#93a4b5")),
            xaxis=dict(range=[max(60, min(float(x), 118)) - 10, 123.0],
                       showgrid=False, zeroline=False, visible=False,
                       scaleanchor="y", scaleratio=1.0),
            yaxis=dict(range=[22, 58], showgrid=False, zeroline=False, visible=False),
        )
        st.plotly_chart(goal_fig, width="stretch")

    with st.expander("Pahami angkanya", expanded=False):
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
    page_header("Evaluasi Model")
    artifact = load_model()
    metadata = artifact["metadata"]
    X_test, y_test, sb_xg = _test_split()
    probabilities = artifact["pipeline"].predict_proba(X_test)[:, 1]
    calibrator = artifact.get("calibrator")
    if calibrator is not None:
        probabilities = calibrator.predict(probabilities)

    test_matches = metadata["split"]["test_matches"]
    st.markdown("".join([
        '<div class="pitchiq-headline"><p>',
        f'<span class="pitchiq-chip">holdout: {int(test_matches)} match</span>',
        '<span class="pitchiq-chip">seed berkelompok: 20261029</span>',
        '<span class="pitchiq-chip">tanpa overlap match</span>',
        "</p></div>",
    ]), unsafe_allow_html=True)

    metrics_head = st.columns(4)
    metrics_head[0].metric("Model deployment", "Logistic Regression")
    metrics_head[1].metric("ROC-AUC (test)", f"{metadata['test_metrics']['roc_auc']:.3f}")
    metrics_head[2].metric("Log loss (test)", f"{metadata['test_metrics']['log_loss']:.3f}")
    metrics_head[3].metric("Brier (test)", f"{metadata['test_metrics']['brier']:.4f}")

    left, right = st.columns([1.15, 1], gap="large")

    with right:
        st.markdown("##### Perbandingan model (test set yang sama)")
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

        fig = go.Figure()
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=artifact["model_name"],
            line=dict(color=ACCENT, width=2.6),
            hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>",
        ))
        fpr_sb, tpr_sb, _ = roc_curve(y_test, sb_xg)
        fig.add_trace(go.Scatter(
            x=fpr_sb, y=tpr_sb, mode="lines", name="StatsBomb xG",
            line=dict(color=BENCH, width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Acak",
            line=dict(color=MUTED, width=1.2, dash="dash"),
        ))
        fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text="<span style='font-size:13px'>Kurva ROC (holdout test)</span>",
                       font=dict(color="#93a4b5")),
            xaxis=dict(title="False positive rate", range=[0, 1], gridcolor=GRID_DIM, zeroline=False),
            yaxis=dict(title="True positive rate", range=[0, 1.02], gridcolor=GRID_DIM, zeroline=False),
            height=400, margin=dict(l=44, r=16, t=36, b=44),
            legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig, width="stretch")

        observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
        obs_sb, pred_sb = calibration_curve(y_test, sb_xg, n_bins=10, strategy="quantile")
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(
            x=predicted, y=observed, mode="lines+markers",
            name=artifact["model_name"], line=dict(color=ACCENT, width=2.2),
            marker=dict(size=7),
        ))
        cal_fig.add_trace(go.Scatter(
            x=pred_sb, y=obs_sb, mode="lines+markers", name="StatsBomb xG",
            line=dict(color=BENCH, width=2, dash="dot"), marker=dict(size=7),
        ))
        cal_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Kalibrasi sempurna",
            line=dict(color=MUTED, width=1.2, dash="dash"),
        ))
        cal_fig.update_layout(
            **LAYOUT_BASE,
            title=dict(text="<span style='font-size:13px'>Kurva kalibrasi (10 bin kuantil)</span>",
                       font=dict(color="#93a4b5")),
            xaxis=dict(title="Rata-rata xG yang diprediksi", range=[0, 0.5], gridcolor=GRID_DIM, zeroline=False),
            yaxis=dict(title="Frekuensi gol yang terjadi", range=[0, 0.5], gridcolor=GRID_DIM, zeroline=False),
            height=400, margin=dict(l=44, r=16, t=36, b=44),
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
    page_header("Insight Pemain & Tim")
    player_stats = load_csv("player_summary.csv").set_index("player")
    team_stats = load_csv("team_summary.csv")

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

    st.markdown("##### Zona paling produktif")
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

    with st.expander("📋 Rekomendasi untuk pelatih (dari model, seluruh label asumsi)", expanded=True):
        st.markdown(load_markdown("coach_recommendations.md"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        artifacts_current = all(
            path.exists() and path.stat().st_mtime >= built_at
            for path in required_outputs
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False
    return artifacts_current


def main() -> None:
    st.set_page_config(
        page_title="PitchIQ — Expected Goals Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.markdown(
        '<div class="pitchiq-brand"><h2>⚽ PitchIQ</h2>'
        "<p>Expected Goals Intelligence</p></div>",
        unsafe_allow_html=True,
    )
    menu = st.sidebar.selectbox("Menu", list(page_titles.keys()))
    st.sidebar.divider()
    st.sidebar.caption(page_subtitles[menu])
    st.sidebar.divider()
    st.sidebar.markdown(
        f'<span class="pitchiq-chip">12.241 tembakan</span>'
        f'<span class="pitchiq-chip">479 match</span>'
        f'<span class="pitchiq-chip">3 kompetisi</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"🔗 **[Repositori GitHub]({REPO_URL})**<br>"
        f"▶️ **[Demo live]({DEMO_URL})**",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.caption(CREDIT)

    page = menu
    page_header(page)

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
