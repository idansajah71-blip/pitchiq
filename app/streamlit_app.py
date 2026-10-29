"""PitchIQ - Expected Goals Intelligence (Streamlit app).

Pages: shot map, xG estimator, model evaluation, player/team insights.
All numbers come from the saved pipeline (train.py + reports). The StatsBomb
xG value is used only as a benchmark, never as an input feature.

Run from the project root:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURES_PARQUET, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SEED
from src.features import FEATURE_COLUMNS, distance_to_goal, shot_angle_deg

MIN_SHOTS_DEFAULT = 15
MIN_ASSISTS = 10
LOW_QUALITY_XG = 0.05

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

PITCH_SHAPES = [
    dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="#5a6673", width=1.5), layer="below"),
    dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="#5a6673", width=1.2), layer="below"),
    dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="#5a6673", width=1.2), layer="below"),
    dict(type="rect", x0=114, y0=34, x1=120, y1=46, line=dict(color="#5a6673", width=1.2), layer="below"),
    dict(type="circle", x0=52, y0=32, x1=68, y1=48, line=dict(color="#5a6673", width=1.2), layer="below"),
    dict(type="line", x0=120, y0=36, x1=120, y1=44, line=dict(color="#ffffff", width=3), layer="below"),
]

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


def footer() -> None:
    st.caption("Data: StatsBomb Open Data. PitchIQ adalah proyek hackathon independen dan tidak berafiliasi dengan StatsBomb.")


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
    st.markdown(
        "Setiap titik = satu tembakan (serangan ke arah gawang di kanan). "
        "Ukuran & warna = **xG**: makin besar dan makin hijau, makin besar "
        "peluang tembakan itu jadi gol."
    )

    head = st.columns(6)
    team = head[0].selectbox("Tim", ["Semua"] + sorted(data["team"].astype(str).unique().tolist()))
    team_players = sorted(data.loc[data["team"] == team, "player"].astype(str).unique())
    all_players = sorted(data["player"].value_counts().loc[lambda counts: counts >= 15].index.astype(str).tolist())
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
    metrics[0].metric("Tembakan ditampilkan", f"{len(view):,}")
    metrics[1].metric("Gol", f"{int(view['is_goal'].sum()):,}")
    metrics[2].metric("Rata-rata xG", f"{view['xg_display'].mean():.3f}")

    import plotly.graph_objects as go

    minutes = view["minute"].astype(int).astype(str)
    seconds = view["second"].astype(int).map(lambda s: f'{s:02d}')
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
                colorscale="RdYlGn",
                cmin=0.0,
                cmax=float(np.percentile(view["xg_display"], 98)),
                opacity=0.85,
                line=dict(width=0.5, color="#22303f"),
                colorbar=dict(title="PitchIQ xG", thickness=12, len=0.75),
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
        shapes=PITCH_SHAPES,
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-2, 124], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="y", scaleratio=1.0),
        yaxis=dict(range=[-2, 82], showgrid=False, zeroline=False, visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**Cara membaca:** titik besar hijau = peluang bagus (xG tinggi), titik "
        "kecil merah = percobaan berisiko (xG rendah). Filter di atas untuk "
        "mengeksplorasi tim/pemain/topologi tembakan."
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
    values: dict[str, Any] = {
        column: np.nan for column in FEATURE_COLUMNS
    }
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
    st.markdown(
        "Pilih posisi & situasi tembakan — PitchIQ menghitung probabilitas gol "
        "**real-time** dengan model yang sama seperti di dataset nyata."
    )
    artifact = load_model()

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        x = st.slider("Kedalaman tembakan (x: 0 = garis tengah, 120 = garis gawang)", 0, 119, 110)
        y = st.slider("Posisi samping (y: 40 = tengah gawang)", 0, 80, 40)
        body_part = st.selectbox("Bagian tubuh", BODY_PARTS)
        shot_type = st.selectbox("Tipe tembakan", SHOT_TYPES)
        under_pressure = st.checkbox("Tembakan di bawah tekanan")
        first_time = st.checkbox("Tembakan first-time (tanpa kontrol dulu)")
        penalty_rate = load_shots().loc[lambda df: df["is_penalty"].fillna(False), "is_goal"].mean()

    with right:
        row, distance, angle = build_estimator_row(x, y, body_part, shot_type, under_pressure, first_time)
        probability = float(artifact["pipeline"].predict_proba(row)[:, 1][0])
        calibrator = artifact.get("calibrator")
        if calibrator is not None:
            probability = float(calibrator.predict(np.array([probability]))[0])

        probs = st.columns(3)
        probs[0].metric("Probabilitas gol", f"{probability:.0%}")
        probs[1].metric("Jarak ke gawang", f"{distance:.1f}")
        probs[2].metric("Sudut gawang terbuka", f"{angle:.1f}°")

        import plotly.graph_objects as go

        goal_fig = go.Figure()
        goal_fig.add_trace(
            go.Scatter(
                x=[x, 120.0, 120.0, x],
                y=[y, 36.0, 44.0, y],
                fill="toself",
                fillcolor="rgba(42,157,143,0.28)",
                line=dict(color="rgba(42,157,143,0.9)", width=2),
                mode="lines",
                hoverinfo="skip",
            )
        )
        goal_fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode="markers",
                marker=dict(size=15, color="#e63946", symbol="x"),
                name="Tembakan",
                hovertemplate=f"x={x:.0f}, y={y:.0f}, prob {probability:.0%}<extra></extra>",
            )
        )
        goal_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            shapes=[dict(type="line", x0=120, y0=36, x1=120, y1=44,
                         line=dict(color="#ffffff", width=4))],
            height=340,
            margin=dict(l=10, r=10, t=24, b=10),
            title=dict(text="Sudut antara dua tiang dari posisi tembakan", font=dict(size=13)),
            xaxis=dict(range=[max(60, min(float(x), 118)) - 8, 123.0],
                       showgrid=False, zeroline=False, visible=False,
                       scaleanchor="y", scaleratio=1.0),
            yaxis=dict(range=[22, 58], showgrid=False, zeroline=False, visible=False),
        )
        st.plotly_chart(goal_fig, use_container_width=True)

    st.caption(
        "**Cara memakai:** probabilitas = peluang tembakan rata-rata pemain pada "
        "posisi & situasi itu menjadi gol. Estimator tidak memperhitungkan posisi "
        "bek/kiper (imputed rata-rata dataset), jadi hasilnya `tenang` - di "
        "pertandingan nyata angkanya bisa lebih kecil saat ada bek yang menghadang. "
        f"Pembanding: konversi penalti pada data ini {penalty_rate:.0%} (penalti di-laporkan terpisah, tidak lewat model)."
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
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_index, test_index = next(splitter.split(X, y, groups))
    return X.iloc[test_index], y.iloc[test_index], features.loc[X.index[test_index], "xg_statsbomb"]


def evaluation_page() -> None:
    artifact = load_model()
    metadata = artifact["metadata"]
    X_test, y_test, sb_xg = _test_split()
    probabilities = artifact["pipeline"].predict_proba(X_test)[:, 1]

    metrics_head = st.columns(4)
    metrics_head[0].metric("Model terbaik", artifact["model_name"].replace(" +", " (+)"))
    metrics_head[1].metric("ROC-AUC (test)", f"{metadata['test_metrics']['roc_auc']:.3f}")
    metrics_head[2].metric("Log loss (test)", f"{metadata['test_metrics']['log_loss']:.3f}")
    metrics_head[3].metric("Brier (test)", f"{metadata['test_metrics']['brier']:.4f}")

    left, right = st.columns([1.15, 1], gap="large")

    with right:
        st.markdown("#### Tabel perbandingan (test set yang sama)")
        comparison = load_csv("model_comparison.csv")
        show = comparison[
            ["model", "test_roc_auc", "test_log_loss", "test_brier",
             "cv_roc_auc_mean", "cv_roc_auc_std"]
        ].copy()
        show["test_roc_auc"] = show["test_roc_auc"].map("{:.3f}".format)
        show["test_log_loss"] = show["test_log_loss"].map("{:.3f}".format)
        show["test_brier"] = show["test_brier"].map("{:.4f}".format)
        show["cv_roc_auc_mean"] = show["cv_roc_auc_mean"].map("{:.3f}".format)
        show["cv_roc_auc_std"] = show["cv_roc_auc_std"].map("{:.3f}".format)
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            "Benchmark StatsBomb xG ikut diuji pada data yang sama untuk "
            "konteks, bukan sebagai lawan pelatihan."
        )

        left_ablation, right_ablation = st.columns(2)
        with left_ablation:
            st.markdown("#### Abalasi LogReg")
            ablation = load_csv("logreg_ablation.csv")
            ablation_show = ablation[["stage", "roc_auc", "brier"]].copy()
            ablation_show["roc_auc"] = ablation_show["roc_auc"].map("{:.3f}".format)
            ablation_show["brier"] = ablation_show["brier"].map("{:.4f}".format)
            st.dataframe(ablation_show, use_container_width=True, hide_index=True)
        with right_ablation:
            st.markdown("#### Kalibrasi")
            calib = load_csv("calibration_comparison.csv")
            st.dataframe(
                calib[["model", "variant", "brier", "calibration_slope"]].round(4),
                use_container_width=True, hide_index=True,
            )

    with left:
        import plotly.graph_objects as go
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import roc_curve

        fig = go.Figure()
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=artifact["model_name"],
                                 line=dict(color="#e63946", width=2.4),
                                 hovertemplate="FPR %{x:.2f} - TPR %{y:.2f}<extra></extra>"))
        fpr_sb, tpr_sb, _ = roc_curve(y_test, sb_xg)
        fig.add_trace(go.Scatter(x=fpr_sb, y=tpr_sb, mode="lines", name="StatsBomb xG",
                                 line=dict(color="#2a9d8f", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Acak",
                                 line=dict(color="#7c8b9b", width=1.2, dash="dash")))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Kurva ROC (holdout test)", font=dict(size=14)),
            xaxis=dict(title="False positive rate", range=[0, 1]),
            yaxis=dict(title="True positive rate", range=[0, 1.02]),
            height=380, margin=dict(l=40, r=20, t=40, b=40), legend=dict(font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

        observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
        obs_sb, pred_sb = calibration_curve(y_test, sb_xg, n_bins=10, strategy="quantile")
        cal_fig = go.Figure()
        cal_fig.add_trace(go.Scatter(x=predicted, y=observed, mode="lines+markers",
                                     name=artifact["model_name"], line=dict(color="#e63946", width=2.2)))
        cal_fig.add_trace(go.Scatter(x=pred_sb, y=obs_sb, mode="lines+markers",
                                     name="StatsBomb xG", line=dict(color="#2a9d8f", width=2, dash="dot")))
        cal_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Kalibrasi sempurna",
                                     line=dict(color="#7c8b9b", width=1.2, dash="dash")))
        cal_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text="Kurva kalibrasi (10 bin)", font=dict(size=14)),
            xaxis=dict(title="Rata-rata xG yang diprediksi", range=[0, 0.5]),
            yaxis=dict(title="Gol yang terjadi (sebenarnya)", range=[0, 0.5]),
            height=380, margin=dict(l=40, r=20, t=40, b=40),
        )
        st.plotly_chart(cal_fig, use_container_width=True)

    st.markdown("#### Cara membaca metrik (bahasa awam)")
    st.markdown(
        "- **ROC-AUC** — seberapa sering model mengurutkan dua tembakan acak "
        "dengan benar (tembakan yang digol vs yang tidak). 0.5 = menebak, 1.0 "
        "= sempurna. Nilai ~0.77 berarti model jauh lebih baik daripada menebak.\n"
        "- **Log loss** — hukuman untuk keyakinan yang salah: prediksi 99% pada "
        "tembakan yang gagal dihukum lebih keras daripada prediksi 60%. Angka "
        "lebih kecil = lebih baik.\n"
        "- **Brier** — rata-rata selisih kuadrat antara probabilitas dan hasil "
        "nyata. Lebih kecil = lebih baik. Itu metrik yang jujur untuk semua rentang xG.\n"
        "- **Kalibrasi** — cek apakah probabilitas bisa dipercaya: tembakan yang "
        "model nilai 0.25 harus benar-benar masuk gol sekitar 25% waktu dalam "
        "data nyata. Kurva mengikuti garis diagonal = dipercaya."
    )


# ---------------------------------------------------------------------------
# 4. Insights
# ---------------------------------------------------------------------------


def insights_page() -> None:
    player_stats = load_csv("player_summary.csv").set_index("player")
    team_stats = load_csv("team_summary.csv")

    min_shots = st.slider(
        "Minimum jumlah tembakan per pemain", 5, 100, MIN_SHOTS_DEFAULT, step=5
    )
    st.caption(
        "goals mengkonversi xG cukup bising di sampel kecil - angka ini lebih "
        "memadai sebagai sinyal daripada kepastian."
    )

    qualified = player_stats[player_stats["shots"] >= min_shots]

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Leaderboard pemain (tembakan sendiri, non-penalti)")
        sort_metric = st.selectbox(
            "Urutkan berdasarkan",
            ["xG/shot (kualitas)", "goals - xG (finishing)", "total xG (volume)"],
        )
        if sort_metric.startswith("xG/shot"):
            ranked = qualified.sort_values("xg_per_shot", ascending=False)
        elif sort_metric.startswith("goals"):
            ranked = qualified.sort_values("goals_minus_xg", ascending=False)
        else:
            ranked = qualified.sort_values("xg_pitchiq", ascending=False)
        table = ranked.head(15)[
            ["team", "shots", "goals", "xg_pitchiq", "xg_per_shot", "goals_minus_xg"]
        ].copy()
        table.columns = [
            "Tim", "Tembakan", "Gol", "Total xG", "xG/tembakan", "Gol - xG"
        ]
        st.dataframe(
            table.round(3),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(f"#### Penyedia peluang (assist) — penata peluang min {MIN_ASSISTS} assisted shots")
        if "assists" in player_stats.columns and player_stats["assists"].notna().any():
            creators = (
                player_stats[player_stats["assists"] >= MIN_ASSISTS]
                .sort_values("assisted_xg", ascending=False)
                .head(10)
            )
            creator_table = creators[["team", "assists", "assisted_goals", "assisted_xg"]].copy()
            creator_table.columns = ["Tim", "Assisted shots", "Gol dari initiatif", "Total xG assist"]
            st.dataframe(creator_table.round(2), use_container_width=True, hide_index=True)
        else:
            st.info("Data creators belum dibangun. Jalankan `python -m src.creators`.")

    with right:
        st.markdown("#### Tim: pemilihan tembakan & xG selisih")
        team_view = team_stats[team_stats["shots_for"] >= 100].sort_values("xg_for", ascending=False).copy()
        team_columns = [
            "Tim", "Tembakan", "xG untuk", "xG/tembakan", "Share tembakan murah",
            "Selisih xG (untuk-terhadap)",
        ]
        team_show = team_view.head(15)[
            ["team", "shots_for", "xg_for", "xg_per_shot_for", "low_quality_share", "xg_diff"]
        ].copy()
        team_show.columns = team_columns
        st.dataframe(team_show.round(3), use_container_width=True, hide_index=True)

        zone_fig = FIGURES_DIR / "fig10_team_shot_selection.png"
        if zone_fig.exists():
            st.image(str(zone_fig), caption="Pemilihan tembakan per tim (lebih tinggi share murah = harus diperbaiki)")

    st.markdown("#### Zona paling produktif")
    zone_fig = FIGURES_DIR / "fig9_zone_map.png"
    left_zone, right_zone = st.columns([1.4, 1], gap="large")
    with left_zone:
        if zone_fig.exists():
            st.image(str(zone_fig), caption="Dari mana tembakan datang, dan seberapa bernilai")
    with right_zone:
        zone = load_csv("zone_summary.csv")
        zone_top = zone[zone["shots"] >= 60].sort_values("mean_pitchiq_xg", ascending=False).head(8)
        zone_top = zone_top[["x_band", "y_band", "shots", "conversion", "mean_pitchiq_xg"]]
        zone_top.columns = ["Band x", "Band y", "Tembakan", "Konversi", "Rata2 xG"]
        st.dataframe(zone_top.round(3), use_container_width=True, hide_index=True)

    with st.expander("Rekomendasi untuk Pelatih (dari model, angka nyata)", expanded=True):
        st.markdown(load_markdown("coach_recommendations.md"))

    footer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="PitchIQ — Expected Goals Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    page = st.sidebar.selectbox("Halaman", list(page_titles.keys()))
    st.sidebar.markdown(f"### ⚽ PitchIQ\n**{page_titles[page]}**")
    st.sidebar.markdown(
        "Alat xG untuk pelatih & analis: dari StatsBomb Open Data "
        "(12 ribu tembakan, 479 match) ke model dan rekomendasi."
    )
    st.title(page_titles[page])

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
