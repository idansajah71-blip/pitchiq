"""Phase 3: pull actionable insight out of the fitted ShotWorth xG model.

Deliverables (coach / non-technical facing):

* A. Factor influence   - SHAP (grouped effects) + permutation importance.
* B. Player tables      - shots, ShotWorth xG, xG/shot, goals, goals - xG and
                          assisted-chance stats (real chance creators).
* C. Team & zone tables - zone efficiency, xG for/against and shot selection
                          (share of low-quality shots, xG < 0.05).

Outputs go to ``reports/`` (CSV + markdown + figures).

Run: ``python -m src.insights``.
"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit

from .config import DATA_DIR, FEATURES_PARQUET, FIGURES_DIR, REPORTS_DIR, SEED, TEST_SPLIT_SEED
from .features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_X_y,
)
from .train import TEST_SIZE, load_artifact, predict_probabilities

MIN_SHOTS = 15           # minimum sample for player rankings
MIN_ASSISTS = 10         # minimum sample for creator rankings
LOW_QUALITY_XG = 0.05    # threshold for "low-quality" shots
LONG_SHOT_UNITS = 30    # StatsBomb pitch-coordinate units; not labelled as metres
SHAP_DISPLAY_ROWS = 2000
CREATORS_PARQUET = DATA_DIR / "creators.parquet"

CREDIT_NOTE = "Data: StatsBomb Open Data (via statsbombpy)"


def estimate_replacement_xg_gain(
    shot_count: int,
    mean_current_xg: float,
    replacement_share: float = 1.0 / 3.0,
    replacement_xg: float = 0.10,
) -> float:
    """Estimate an assumption-based xG uplift when some low-xG shots are replaced.

    This is a scenario calculator, not a causal estimate. It assumes each
    replaced shot is substituted one-for-one by a chance with ``replacement_xg``.
    """
    if shot_count < 0:
        raise ValueError("shot_count must be non-negative")
    if not 0.0 <= replacement_share <= 1.0:
        raise ValueError("replacement_share must be between 0 and 1")
    if not 0.0 <= mean_current_xg <= 1.0 or not 0.0 <= replacement_xg <= 1.0:
        raise ValueError("xG values must lie between 0 and 1")
    return float(shot_count * replacement_share * max(0.0, replacement_xg - mean_current_xg))


FACTOR_EXPLAIN: dict[str, str] = {
    "distance": "Closer to goal = more of the goal is open and the keeper has less time. This is the single biggest driver.",
    "angle_deg": "A wide angle means the shooter sees more goal between the posts; tight angles are almost never worth taking.",
    "body_part": "Contact type matters mostly through where it happens - headers and volleys come from hard, crowded positions.",
    "n_defenders_cone": "Counts defenders between shooter and goal; interpret alongside the shot location and freeze-frame availability.",
    "nearest_defender_dist": "Distance to the nearest defender is a proxy for space, not a complete measure of pressure.",
    "gk_off_line": "Goalkeeper distance from the goal line changes the visible target, but its effect depends on angle and shot context.",
    "gk_y_offset": "Goalkeeper lateral position may open one side; the effect depends on shot angle and placement.",
    "first_time": "First-time shots arise in different contexts; the association is not an isolated causal effect.",
    "shot_type": "Free kicks and open play behave differently; penalties are handled separately.",
    "play_pattern": "Shot quality can differ by play pattern; location and game state may explain part of the association.",
    "under_pressure": "Pressure is a marker of shot context; the observed association alone does not establish a causal effect.",
    "open_goal": "An empty net is near-certain; the model flags it almost as a sure goal.",
    "one_on_one": "A clean one-on-one much more than doubles typical shot quality.",
    "minute": "Game state and fatigue add a small effect late in games.",
    "technique": "Volleys, lobs and half-volleys are rarer, harder contacts.",
    "aerial_won": "Winning an aerial duel right before shooting usually means a header from a crowded area.",
    "has_freeze_frame": "Data-availability flag; it also lets the model trust freeze-frame features less where they are missing.",
    "abs_y_from_center": "Central shots face a bigger goal than shots from the wings.",
    "distance_to_goal_line": "Depth relative to the goal line separates wide-close looks from central-far ones.",
    "log_distance": "A transformed distance that matches how fast quality decays.",
    "n_defenders_near": "Crowding near the shooter lowers quality.",
    "xg_statsbomb": "(never a feature - benchmark only)",
    "outcome": "(never a feature - target only)",
}


# ---------------------------------------------------------------------------
# Loading and splitting
# ---------------------------------------------------------------------------


def load_scored() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Load features and score every non-penalty shot with the saved model."""
    features = pd.read_parquet(FEATURES_PARQUET)
    X, y = get_X_y(features)  # penalties excluded by default
    artifact = load_artifact()
    probabilities = predict_probabilities(artifact, X)
    pxg = pd.Series(np.asarray(probabilities, dtype=float), index=X.index, name="shotworth_xg")
    return features, X, y, pxg, artifact


def grouped_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the grouped 80/20 split used in training (same seed)."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=TEST_SPLIT_SEED)
    train_index, test_index = next(splitter.split(X, y, groups))
    return train_index, test_index


def _framed(
    features: pd.DataFrame, X: pd.DataFrame, y: pd.Series, pxg: pd.Series
) -> pd.DataFrame:
    """One tidy frame with identity, target, model xG and key context."""
    meta = ["player", "team", "competition", "event_id", "match_id",
            "home_team", "away_team", "xg_statsbomb", "x", "y"]
    missing = [column for column in meta if column not in features.columns]
    if missing:
        raise KeyError(f"features table is missing columns: {missing}")
    frame = features.loc[X.index, meta].copy()
    frame["is_goal"] = y.to_numpy()
    frame["shotworth_xg"] = pxg.to_numpy()
    frame["distance"] = X["distance"].to_numpy()
    frame["under_pressure"] = X["under_pressure"].to_numpy()
    frame["first_time"] = X["first_time"].to_numpy()
    frame["body_part"] = X["body_part"].to_numpy()
    if "play_pattern" in features.columns:
        frame["play_pattern"] = features.loc[X.index, "play_pattern"].to_numpy()
    frame["opponent"] = np.where(
        frame["team"] == frame["home_team"],
        frame["away_team"],
        np.where(frame["team"] == frame["away_team"], frame["home_team"], None),
    )
    return frame


# ---------------------------------------------------------------------------
# A. Factor influence (SHAP + permutation importance)
# ---------------------------------------------------------------------------


def _feature_groups(
    names: list[str],
    numeric: list[str],
    boolean: list[str],
    categorical: list[str],
) -> list[str]:
    """Map transformed (one-hot expanded) column names back to feature groups."""
    scalar = set(numeric) | set(boolean)
    groups: list[str] = []
    for name in names:
        if name in scalar:
            groups.append(name)
            continue
        for category in categorical:
            if name == category or name.startswith(category + "_"):
                groups.append(category)
                break
        else:
            groups.append(name)
    return groups


def shap_top_factors(
    artifact: dict[str, Any],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame]:
    """Grouped SHAP importance for the (linear) chosen model.

    The model works on standardised, one-hot-expanded columns; effects for
    dummies are collapsed back to the original feature group so the ranking is
    readable for humans.
    """
    import shap  # deferred: slow import

    pipeline = artifact["pipeline"]
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    names = list(prep.get_feature_names_out())
    background = prep.transform(
        X_train.sample(min(3000, len(X_train)), random_state=SEED)
    )
    explainer = shap.LinearExplainer(clf, background)

    display = X_test.head(SHAP_DISPLAY_ROWS)
    shap_values = np.asarray(explainer.shap_values(prep.transform(display)))

    groups = _feature_groups(names, NUMERIC_FEATURES, BOOLEAN_FEATURES, CATEGORICAL_FEATURES)
    mean_abs = pd.Series(
        np.abs(shap_values).mean(axis=0), index=pd.Index(groups, name="feature_group")
    )
    grouped = (
        mean_abs.groupby(level=0, sort=False).mean().sort_values(ascending=False)
    )
    table = grouped.rename("mean_abs_shap").reset_index()
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(REPORTS_DIR / "shap_top_factors.csv", index=False)
    return table, shap_values, names, display


def plot_shap_summary(table: pd.DataFrame, top: int = 15) -> None:
    """Horizontal bar chart of grouped SHAP importance."""
    top_table = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(top_table["feature_group"], top_table["mean_abs_shap"], color="#1d3557")
    ax.set_xlabel("Mean |SHAP| effect on goal probability (log-odds)")
    ax.set_title("What drives ShotWorth xG most", loc="left")
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.99, 0.005, CREDIT_NOTE, ha="right", fontsize=7, color="#8a94a6")
    fig.savefig(FIGURES_DIR / "fig6_shap_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_shap_dependence(
    shap_values: np.ndarray,
    names: list[str],
    X_display: pd.DataFrame,
    y_display: pd.Series,
) -> None:
    """SHAP vs the raw feature value for distance and angle, colored by outcome."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    goals = y_display.to_numpy(dtype=bool)
    for ax, feature in zip(axes, ["distance", "angle_deg"]):
        column = names.index(feature)
        values = X_display[feature].to_numpy()
        ax.scatter(
            values[~goals], shap_values[~goals, column],
            s=7, alpha=0.35, color="#9aa8b2", label="No goal",
        )
        ax.scatter(
            values[goals], shap_values[goals, column],
            s=13, alpha=0.85, color="#e63946", label="Goal",
        )
        ax.axhline(0.0, color="#334", lw=0.8, ls="--")
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value (log-odds)")
        ax.grid(color="#d9e2ec", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("How distance and angle move ShotWorth xG", x=0.09, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.99, 0.01, CREDIT_NOTE, ha="right", fontsize=7, color="#8a94a6")
    fig.savefig(FIGURES_DIR / "fig7_shap_dependence.png", bbox_inches="tight")
    plt.close(fig)


def permutation_top_features(
    artifact: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    """Permutation importance on the held-out grouped test matches."""
    calibrator = artifact.get("calibrator")

    def calibrated_neg_brier(estimator: Any, matrix: pd.DataFrame, target: pd.Series) -> float:
        """Score the actual deployed probability, including optional calibration."""
        probabilities = estimator.predict_proba(matrix)[:, 1]
        if calibrator is not None:
            probabilities = calibrator.predict(probabilities)
        return -float(np.mean((np.asarray(target, dtype=float) - probabilities) ** 2))

    result = permutation_importance(
        artifact["pipeline"],
        X_test,
        y_test,
        scoring=calibrated_neg_brier,
        n_repeats=n_repeats,
        random_state=SEED,
        n_jobs=4,
    )
    table = pd.DataFrame(
        {
            "feature": X_test.columns,
            "brier_delta_mean": result.importances_mean,
            "brier_delta_std": result.importances_std,
        }
    ).sort_values("brier_delta_mean", ascending=False, ignore_index=True)
    table.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

    top = table.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.barh(top["feature"], top["brier_delta_mean"], color="#2a9d8f")
    ax.set_xlabel("Brier score increase when the feature is destroyed")
    ax.set_title("Permutation importance (held-out matches)", loc="left")
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.99, 0.005, CREDIT_NOTE, ha="right", fontsize=7, color="#8a94a6")
    fig.savefig(FIGURES_DIR / "fig8_permutation_importance.png", bbox_inches="tight")
    plt.close(fig)
    return table


# ---------------------------------------------------------------------------
# B. Players and creators
# ---------------------------------------------------------------------------


def player_table(
    frame: pd.DataFrame, creators: pd.DataFrame | None
) -> pd.DataFrame:
    """Aggregate player stats separately for each team and competition.

    A player who appears for a club and a national team receives separate rows;
    this avoids combining incompatible samples and then labelling the sum as one
    team's performance.
    """
    scope = ["player", "team", "competition"]
    stats = frame.groupby(scope, dropna=False).agg(
        shots=("event_id", "size"),
        goals=("is_goal", "sum"),
        xg_shotworth=("shotworth_xg", "sum"),
        xg_statsbomb=("xg_statsbomb", "sum"),
    )
    stats["xg_per_shot"] = stats["xg_shotworth"] / stats["shots"]
    stats["goals_minus_xg"] = stats["goals"] - stats["xg_shotworth"]
    stats["sample_note"] = np.where(
        stats["shots"] < 30,
        "small sample - goals vs xG is noisy",
        "",
    )

    if creators is not None and not creators.empty:
        assisted = frame.merge(
            creators[["match_id", "shot_event_id", "creator"]],
            how="inner",
            left_on=["match_id", "event_id"],
            right_on=["match_id", "shot_event_id"],
            validate="one_to_one",
        )
        # The assisting creator belongs to the same team/competition as the
        # shot. Aggregate within that scope so club and national-team totals do
        # not leak into one another through a name-only join.
        creator_stats = assisted.groupby(
            ["creator", "team", "competition"], dropna=False
        ).agg(
            assists=("event_id", "size"),
            assisted_goals=("is_goal", "sum"),
            assisted_xg=("shotworth_xg", "sum"),
        )
        creator_stats.index = creator_stats.index.set_names(scope)
        # Outer join keeps creators who have key passes linked to shots even if
        # they did not themselves take a shot in this dataset slice.
        stats = stats.join(creator_stats, how="outer")
        stats[["assists", "assisted_goals", "assisted_xg"]] = stats[
            ["assists", "assisted_goals", "assisted_xg"]
        ].fillna(0)
        stats[["shots", "goals", "xg_shotworth", "xg_statsbomb"]] = stats[
            ["shots", "goals", "xg_shotworth", "xg_statsbomb"]
        ].fillna(0)
        stats["xg_per_shot"] = stats["xg_shotworth"] / stats["shots"].replace(0, np.nan)
        stats["goals_minus_xg"] = stats["goals"] - stats["xg_shotworth"]
        stats["sample_note"] = np.where(
            stats["shots"] < 30,
            "small sample - goals vs xG is noisy",
            "",
        )
    else:
        stats["assists"] = np.nan
        stats["assisted_goals"] = np.nan
        stats["assisted_xg"] = np.nan

    return (
        stats.reset_index()
        .set_index("player")
        .sort_values(["shots", "xg_shotworth"], ascending=False)
    )

def top_players_markdown(player_stats: pd.DataFrame) -> str:
    """Markdown rankings used in reports/insights_summary.md."""
    qualified = player_stats[player_stats["shots"] >= MIN_SHOTS]
    creators = (
        player_stats[player_stats["assists"] >= MIN_ASSISTS]
        .sort_values("assisted_xg", ascending=False)
        .head(10)
    )
    finishers = qualified.sort_values("goals_minus_xg", ascending=False).head(10)
    quality = qualified.sort_values("xg_per_shot", ascending=False).head(10)

    def _players(table: pd.DataFrame) -> str:
        header = "| player | team | competition | shots | xG | xG/shot | goals | goals-xG |"
        line = "|---|---|---|---|---|---|---|---|"
        rows = [
            f"| {player} | {row['team']} | {row['competition']} | {int(row['shots'])} | "
            f"{row['xg_shotworth']:.2f} | {row['xg_per_shot']:.3f} | "
            f"{int(row['goals'])} | {row['goals_minus_xg']:+.2f} |"
            for player, row in table.iterrows()
        ]
        return "\n".join([header, line, *rows])

    def _creators(table: pd.DataFrame) -> str:
        header = "| creator | team | competition | assisted shots | assisted goals | assisted xG |"
        line = "|---|---|---|---|---|---|"
        rows = [
            f"| {player} | {row['team']} | {row['competition']} | "
            f"{int(row['assists'])} | {int(row['assisted_goals'])} | "
            f"{row['assisted_xg']:.2f} |"
            for player, row in table.iterrows()
        ]
        return "\n".join([header, line, *rows])

    parts = [
        f"### Top chance creators (assists received as shooter, min {MIN_ASSISTS})",
        "",
        _creators(creators),
        "",
        f"### Top finishers (min {MIN_SHOTS} shots, ranked by goals - ShotWorth xG)",
        "",
        _players(finishers),
        "",
        f"### Best shot quality (min {MIN_SHOTS} shots, ranked by ShotWorth xG/shot)",
        "",
        _players(quality),
        "",
        "Goals - xG is noisy on small samples; treat single-season finishes "
        "(especially under ~30 shots) as luck-flavoured evidence.",
    ]
    return "\n".join(parts)


def plot_player_finishing(player_stats: pd.DataFrame) -> None:
    """Diverging bar chart: who beats their xG (and who does not)."""
    qualified = player_stats[player_stats["shots"] >= MIN_SHOTS]
    top = qualified.sort_values("goals_minus_xg", ascending=False).head(10)
    bottom = qualified.sort_values("goals_minus_xg").head(5)
    plot_frame = pd.concat([top, bottom]).sort_values("goals_minus_xg")

    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    colors = ["#e76f51" if value < 0 else "#2a9d8f" for value in plot_frame["goals_minus_xg"]]
    labels = [
        f"{player} — {row['team']} / {row['competition']} ({int(row['shots'])} shots)"
        for player, row in plot_frame.iterrows()
    ]
    ax.barh(labels, plot_frame["goals_minus_xg"], color=colors)
    ax.axvline(0, color="#334", lw=0.8)
    ax.set_xlabel("Goals - ShotWorth xG (amount of goals above expectation)")
    ax.set_title(
        f"Finishing: best and worst vs expectation (min {MIN_SHOTS} shots)",
        loc="left",
    )
    ax.grid(axis="x", color="#d9e2ec", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.99, 0.005, CREDIT_NOTE, ha="right", fontsize=7, color="#8a94a6")
    fig.savefig(FIGURES_DIR / "fig11_player_finishing.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# C. Teams and zones
# ---------------------------------------------------------------------------


def team_table(frame: pd.DataFrame, league_xg_per_shot: float) -> pd.DataFrame:
    """xG for/against and shot-selection quality per team."""
    low_quality = frame.assign(
        low=(frame["shotworth_xg"] < LOW_QUALITY_XG).astype(float),
        long=(frame["distance"] > LONG_SHOT_UNITS).astype(float),
    )

    for_side = low_quality.groupby("team").agg(
        matches=("match_id", "nunique"),
        shots_for=("event_id", "size"),
        goals_for=("is_goal", "sum"),
        xg_for=("shotworth_xg", "sum"),
        xg_statsbomb_for=("xg_statsbomb", "sum"),
        low_quality_share=("low", "mean"),
        long_shot_share=("long", "mean"),
    )
    against = low_quality.groupby("opponent").agg(
        shots_against=("event_id", "size"),
        goals_against=("is_goal", "sum"),
        xg_against=("shotworth_xg", "sum"),
    )

    table = for_side.join(against, how="left")
    table = table.fillna({column: 0.0 for column in against.columns})
    table["xg_per_shot_for"] = table["xg_for"] / table["shots_for"]
    table["xg_diff"] = table["xg_for"] - table["xg_against"]
    table["xg_for_per_game"] = table["xg_for"] / table["matches"]
    table["vs_league_xg_per_shot"] = table["xg_per_shot_for"] - league_xg_per_shot
    return table.sort_values("xg_for", ascending=False)


def plot_team_shot_selection(team_stats: pd.DataFrame, league_xg_per_shot: float) -> None:
    """Shot-selection map: low-quality share vs xG per shot."""
    plot_frame = team_stats[team_stats["shots_for"] >= 40]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    sizes = 14.0 + 1.4 * np.sqrt(plot_frame["shots_for"])
    ax.scatter(
        plot_frame["xg_per_shot_for"],
        plot_frame["low_quality_share"],
        s=sizes,
        alpha=0.75,
        color="#457b9d",
        edgecolor="white",
        linewidth=0.6,
    )
    ax.axvline(league_xg_per_shot, color="#334", lw=0.9, ls="--", label="League xG/shot")
    labeled = pd.concat(
        [
            plot_frame.nlargest(6, "low_quality_share"),
            plot_frame.nlargest(6, "xg_per_shot_for"),
            plot_frame.nsmallest(4, "low_quality_share"),
        ]
    )
    for team, row in labeled.drop_duplicates().iterrows():
        ax.annotate(
            team,
            (row["xg_per_shot_for"], row["low_quality_share"]),
            fontsize=7,
            color="#22333b",
            xytext=(3, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("ShotWorth xG per shot taken")
    ax.set_ylabel(f"Share of shots with xG < {LOW_QUALITY_XG}")
    ax.set_title("Shot selection by team (bubble size = shot volume)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#d9e2ec", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.99, 0.005, CREDIT_NOTE, ha="right", fontsize=7, color="#8a94a6")
    fig.savefig(FIGURES_DIR / "fig10_team_shot_selection.png", bbox_inches="tight")
    plt.close(fig)


def zone_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Conversion and ShotWorth xG per pitch zone (x/y bands)."""
    x_bands = [0, 60, 70, 80, 90, 100, 110, 120.01]
    y_bands = [0, 20, 40, 60, 80.01]
    x_labels = ["x<60", "60-70", "70-80", "80-90", "90-100", "100-110", "110+"]
    y_labels = ["y0-20", "20-40", "40-60", "60-80"]

    zones = frame.copy()
    zones["x_band"] = pd.cut(zones["x"].astype(float), x_bands, labels=x_labels)
    zones["y_band"] = pd.cut(zones["y"].astype(float), y_bands, labels=y_labels)
    table = (
        zones.groupby(["x_band", "y_band"], observed=True)
        .agg(
            shots=("event_id", "size"),
            goals=("is_goal", "sum"),
            conversion=("is_goal", "mean"),
            mean_shotworth_xg=("shotworth_xg", "mean"),
        )
        .reset_index()
    )
    table.to_csv(REPORTS_DIR / "zone_summary.csv", index=False)
    return table


def zone_shot_share_matrix(matrix_counts: pd.DataFrame) -> pd.DataFrame:
    """Normalize sparse zone counts while safely ignoring empty-zone NaNs."""
    total_shots = float(np.nansum(matrix_counts.to_numpy(dtype=float)))
    if not np.isfinite(total_shots) or total_shots <= 0:
        return matrix_counts.astype(float) * np.nan
    return matrix_counts.astype(float) / total_shots


def plot_zone_map(zone_stats: pd.DataFrame, league_xg_per_shot: float) -> None:
    """Two-panel zone view: where shots come from and what they convert at."""
    matrix_n = zone_stats.pivot(index="y_band", columns="x_band", values="shots")
    matrix_c = zone_stats.pivot(index="y_band", columns="x_band", values="mean_shotworth_xg")
    # A numpy sum of a sparse pivot propagates any NaN and made the whole panel blank.
    matrix_share = zone_shot_share_matrix(matrix_n)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, matrix, cmap, title in (
        (axes[0], matrix_share, "YlGnBu", "Share of all shots taken"),
        (axes[1], matrix_c, "RdYlGn", "Mean ShotWorth xG per shot"),
    ):
        data = np.asarray(matrix, dtype=float)
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=0, fontsize=8)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index, fontsize=8)
        ax.set_title(title, loc="left", fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    value = data[i, j]
                    ax.text(
                        j, i,
                        f"{value:.1%}" if title.startswith("Share") else f"{value:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="#1b1b1b",
                    )
    fig.suptitle(
        f"Shot zones (non-penalty). League mean ShotWorth xG/shot = "
        f"{league_xg_per_shot:.3f}",
        x=0.09,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.99, 0.005, CREDIT_NOTE, ha="right", fontsize=7, color="#8a94a6")
    fig.savefig(FIGURES_DIR / "fig9_zone_map.png", bbox_inches="tight")
    plt.close(fig)


def context_quality(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Conversion and xG per context dimension (for coach recommendations)."""
    rates: dict[str, pd.DataFrame] = {}
    for dimension in ("play_pattern", "under_pressure", "first_time", "body_part"):
        table = frame.groupby(dimension, observed=True).agg(
            shots=("event_id", "size"),
            goals=("is_goal", "sum"),
            conversion=("is_goal", "mean"),
            mean_shotworth_xg=("shotworth_xg", "mean"),
        ).reset_index()
        table.insert(0, "dimension", dimension)
        rates[dimension] = table
    pd.concat(rates.values(), ignore_index=True).to_csv(
        REPORTS_DIR / "context_quality.csv", index=False
    )
    return rates


# ---------------------------------------------------------------------------
# Coach recommendations and final summary
# ---------------------------------------------------------------------------


def build_recommendations(
    team_stats: pd.DataFrame,
    zone_stats: pd.DataFrame,
    context: dict[str, pd.DataFrame],
    player_stats: pd.DataFrame,
    league_xg_per_shot: float,
    frame: pd.DataFrame,
) -> list[str]:
    """Five concrete, number-backed recommendations for coaches."""
    recommend: list[str] = []
    eligible = team_stats[team_stats["shots_for"] >= 100]

    low = eligible.sort_values("low_quality_share", ascending=False).iloc[0]
    low_attempts = frame[
        (frame["team"] == low.name) & (frame["shotworth_xg"] < LOW_QUALITY_XG)
    ]
    low_attempt_count = len(low_attempts)
    mean_low_xg = (
        float(low_attempts["shotworth_xg"].mean()) if low_attempt_count else 0.0
    )
    replacement_share = 1.0 / 3.0
    replacement_target_xg = 0.10
    estimated_gain = estimate_replacement_xg_gain(
        low_attempt_count,
        mean_low_xg,
        replacement_share=replacement_share,
        replacement_xg=replacement_target_xg,
    )
    recommend.append(
        f"{low.name} take {low['low_quality_share']:.0%} of their shots below "
        f"{LOW_QUALITY_XG} xG (ShotWorth xG/shot {low['xg_per_shot_for']:.3f} "
        f"vs dataset mean {league_xg_per_shot:.3f}). Scenario, not a causal "
        f"forecast: replacing one-third of these {low_attempt_count} shots "
        f"(mean current xG {mean_low_xg:.3f}) with chances rated {replacement_target_xg:.2f} "
        f"xG would add about +{estimated_gain:.1f} expected goals if other factors "
        "stayed constant. Coaching hypothesis: look for better progression into "
        "the box rather than treating low xG alone as proof a shot was a mistake."
    )

    long_shots = eligible.drop(index=low.name)
    if not long_shots.empty:
        shooter = long_shots.sort_values("long_shot_share", ascending=False).iloc[0]
        long_attempt = frame[
            (frame["team"] == shooter.name) & (frame["distance"] > LONG_SHOT_UNITS)
        ]
        n_long = len(long_attempt)
        long_conversion = (
            float(long_attempt["is_goal"].mean()) if n_long else float("nan")
        )
        long_xg = (
            float(long_attempt["shotworth_xg"].mean()) if n_long else float("nan")
        )
        recommend.append(
            f"{shooter.name} attempt shots from {LONG_SHOT_UNITS}+ StatsBomb "
            f"pitch-coordinate units on {shooter['long_shot_share']:.0%} of "
            f"their attempts ({int(n_long)} shots: xG/shot {long_xg:.3f}, "
            f"conversion {long_conversion:.1%}). Their xG difference for vs "
            f"against is {shooter['xg_diff']:+.2f}. Coaching hypothesis: review "
            "whether progression or a pass creates a better option in these "
            "situations; low xG alone does not prove a shot was a mistake."
        )

    patterns = context["play_pattern"]
    patterns = patterns[patterns["shots"] >= 200]
    special = patterns[~patterns["play_pattern"].isin(["Regular Play"])]
    if not special.empty:
        best = special.sort_values("mean_shotworth_xg", ascending=False).iloc[0]
        baseline = patterns[patterns["play_pattern"] == "Regular Play"]
        base_xg = baseline["mean_shotworth_xg"].iloc[0] if not baseline.empty else league_xg_per_shot
        recommend.append(
            f"'{best['play_pattern']}' shots average {best['mean_shotworth_xg']:.3f} xG "
            f"(conversion {best['conversion']:.1%}) vs {base_xg:.3f} for regular "
            f"play, across {int(best['shots'])} observed shots. This association "
            "is descriptive—not proof that the pattern or tempo causes better chances; "
            "shot location and game state may explain part of the gap. Review the "
            "transition clips before prescribing a tactical change."
        )

    in_box = zone_stats[
        zone_stats["x_band"].isin(["100-110", "110+"])
    ].copy()
    if not in_box.empty:
        # Pair mirrored y-bands around the goal centre. Reporting only y=20–40
        # makes one side look under-used even though y=40–60 is its mirror.
        in_box["lateral_band"] = in_box["y_band"].astype(str).map(
            {"y0-20": "wide", "60-80": "wide", "20-40": "inside channel", "40-60": "inside channel"}
        )
        in_box = in_box.dropna(subset=["lateral_band"]).copy()
        in_box["xg_sum"] = in_box["mean_shotworth_xg"] * in_box["shots"]
        paired = in_box.groupby(["x_band", "lateral_band"], observed=True).agg(
            shots=("shots", "sum"),
            goals=("goals", "sum"),
            xg_sum=("xg_sum", "sum"),
        ).reset_index()
        paired["mean_shotworth_xg"] = paired["xg_sum"] / paired["shots"]
        paired["conversion"] = paired["goals"] / paired["shots"]
        paired = paired[paired["shots"] >= 40]
        if not paired.empty:
            best_zone = paired.sort_values("mean_shotworth_xg", ascending=False).iloc[0]
            share = best_zone["shots"] / zone_stats["shots"].sum()
            recommend.append(
                f"The attacking band {best_zone['x_band']} / {best_zone['lateral_band']} "
                f"(mirrored y-bands grouped together) has the highest mean xG among "
                f"qualified paired zones: {best_zone['mean_shotworth_xg']:.3f} "
                f"(conversion {best_zone['conversion']:.1%}; {int(best_zone['shots'])} "
                f"shots, {share:.1%} of all shots). Treat this as a location pattern "
                "to review on video, not proof that forcing shots into that band will "
                "improve outcomes."
            )

    qualified = player_stats[player_stats["shots"] >= MIN_SHOTS]
    finisher = qualified.sort_values("goals_minus_xg", ascending=False).iloc[0]
    recommend.append(
        f"{finisher.name} ({finisher['team']}, {finisher['competition']}) "
        f"recorded {int(finisher['goals'])} goals on "
        f"{finisher['xg_shotworth']:.2f} ShotWorth xG ({int(finisher['shots'])} shots, "
        f"{finisher['goals_minus_xg']:+.2f} goals minus xG, "
        f"{finisher['xg_per_shot']:.3f} xG/shot). This is an observed residual, "
        "not proof it will persist; validate on a larger or later sample before "
        "treating it as repeatable finishing skill."
    )

    pressure = context["under_pressure"]
    if not pressure.empty:
        pressed = pressure[pressure["under_pressure"] == 1]
        free = pressure[pressure["under_pressure"] == 0]
        if not pressed.empty and not free.empty:
            recommend.append(
                f"Shots under pressure average {pressed['mean_shotworth_xg'].iloc[0]:.3f} "
                f"xG/shot vs {free['mean_shotworth_xg'].iloc[0]:.3f} when not "
                f"(across {int(pressed['shots'].iloc[0]):,} vs "
                f"{int(free['shots'].iloc[0]):,} shots). Review clips for "
                "situations where an extra pass was available, but do not infer "
                "that passing would always have improved the chance."
            )
    return recommend[:5]


def write_coach_markdown(recommend: list[str]) -> None:
    """Persist the five recommendations with their numbers."""
    lines = [
        "# Recommendations for coaches",
        "",
        "Derived from ShotWorth xG (non-penalty shots). Numbers are "
        "descriptive averages from the dataset, not guarantees.",
        "",
    ]
    lines += [f"{index}. {text}" for index, text in enumerate(recommend, start=1)]
    lines += [
        "",
        "---",
        "Data: StatsBomb Open Data (via statsbombpy). ShotWorth is an "
        "independent hackathon project.",
    ]
    path = REPORTS_DIR / "coach_recommendations.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved recommendations -> {path}")


def write_insights_summary(
    factors: pd.DataFrame,
    permutation: pd.DataFrame,
    players_markdown: str,
    team_stats: pd.DataFrame,
    recommend: list[str],
) -> None:
    """Assemble the Phase 3 summary markdown."""
    factor_lines = []
    for _, row in factors.head(5).iterrows():
        explanation = FACTOR_EXPLAIN.get(
            row["feature_group"],
            "Moves shot quality, see table for size of the effect.",
        )
        factor_lines.append(
            f"{int(row['rank'])}. **{row['feature_group']}** "
            f"(mean |SHAP| {row['mean_abs_shap']:.3f}) - {explanation}"
        )

    low_teams = (
        team_stats[team_stats["shots_for"] >= 100]
        .sort_values("low_quality_share", ascending=False)
        .head(5)
    )
    low_lines = [
        f"- {team}: {row['low_quality_share']:.1%} of shots below xG "
        f"{LOW_QUALITY_XG} ({int(row['shots_for'])} shots)"
        for team, row in low_teams.iterrows()
    ]

    rec_lines = [f"- {text}" for text in recommend]

    body = "\n".join(
        [
            "# Insights summary (Phase 3)",
            "",
            "Model output: ShotWorth xG = saved pipeline + optional calibration.",
            "Penalties excluded everywhere (reported separately). Player and "
            "team tables use **descriptive averages**, not guarantees - "
            "goals - xG is noisy below ~30 shots.",
            "",
            "## Top 5 factors (SHAP, grouped to original features)",
            "",
            *factor_lines,
            "",
            "Permutation importance agrees on order: "
            + ", ".join(permutation.head(5)["feature"].tolist())
            + " (see `permutation_importance.csv`).",
            "",
            players_markdown,
            "",
            "## Teams with the most low-quality shooting",
            "",
            *low_lines,
            "",
            "## Recommendations for coaches (short version)",
            "",
            *rec_lines,
            "",
            "Files: `shap_top_factors.csv`, `permutation_importance.csv`, "
            "`player_summary.csv`, `team_summary.csv`, `zone_summary.csv`, "
            "`context_quality.csv`, `coach_recommendations.md`.",
            "",
            "Data: StatsBomb Open Data. ShotWorth is an independent hackathon "
            "project and is not affiliated with StatsBomb.",
        ]
    )
    path = REPORTS_DIR / "insights_summary.md"
    path.write_text(body, encoding="utf-8")
    print(f"Saved summary -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_insights(min_shots: int = MIN_SHOTS) -> None:
    """Run the full Phase 3 insight workflow."""
    global MIN_SHOTS
    MIN_SHOTS = min_shots

    features, X, y, pxg, artifact = load_scored()
    frame = _framed(features, X, y, pxg)
    league_xg_per_shot = float(frame["shotworth_xg"].mean())

    print("=== A. FACTOR INFLUENCE ===")
    train_index, test_index = grouped_split(
        X, y, features.loc[X.index, "match_id"]
    )
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_test = y.iloc[test_index]
    factors, shap_values, names, X_display = shap_top_factors(
        artifact, X_train, X_test
    )
    y_display = y_test.head(SHAP_DISPLAY_ROWS)
    plot_shap_summary(factors)
    plot_shap_dependence(shap_values, names, X_display, y_display)
    permutation = permutation_top_features(artifact, X_test, y_test)
    print(factors.head(5).to_string(index=False))

    print("\n=== B. PLAYERS & CREATORS ===")
    creators: pd.DataFrame | None = None
    if CREATORS_PARQUET.exists():
        creators = pd.read_parquet(CREATORS_PARQUET)
        print(f"Creator mappings loaded: {len(creators):,} assisted shots")
    else:
        print("creators.parquet not found - creator stats skipped "
              "(run `python -m src.creators`).")
    player_stats = player_table(frame, creators)
    player_stats.to_csv(REPORTS_DIR / "player_summary.csv")
    plot_player_finishing(player_stats)
    print("- Top 5 creators by assisted xG:")
    if creators is not None:
        creators_view = player_stats[player_stats["assists"] >= MIN_ASSISTS]
        print(
            creators_view.sort_values("assisted_xg", ascending=False)
            .head(5)[["team", "assists", "assisted_xg"]]
            .to_string()
        )
    print("- Top 5 finishers (min shots):")
    qualified = player_stats[player_stats["shots"] >= MIN_SHOTS]
    print(
        qualified.sort_values("goals_minus_xg", ascending=False)
        .head(5)[["team", "shots", "goals", "xg_shotworth", "goals_minus_xg"]]
        .to_string()
    )

    print("\n=== C. TEAMS & ZONES ===")
    team_stats = team_table(frame, league_xg_per_shot)
    team_stats.to_csv(REPORTS_DIR / "team_summary.csv")
    plot_team_shot_selection(team_stats, league_xg_per_shot)
    zone_stats = zone_table(frame)
    plot_zone_map(zone_stats, league_xg_per_shot)
    context = context_quality(frame)
    print(f"- League xG/shot (ShotWorth): {league_xg_per_shot:.3f}")
    print(team_stats.head(5)[["shots_for", "xg_for", "xg_per_shot_for", "low_quality_share"]].to_string())

    print("\n=== D. RECOMMENDATIONS ===")
    recommend = build_recommendations(
        team_stats, zone_stats, context, player_stats, league_xg_per_shot, frame
    )
    write_coach_markdown(recommend)
    write_insights_summary(
        factors, permutation, top_players_markdown(player_stats),
        team_stats, recommend,
    )
    for index, text in enumerate(recommend, start=1):
        print(f"{index}. {text}")
    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ShotWorth insights.")
    parser.add_argument("--min-shots", type=int, default=MIN_SHOTS)
    args = parser.parse_args()
    run_insights(min_shots=args.min_shots)


if __name__ == "__main__":
    main()
