"""Exploratory data analysis for ShotWorth.

Produces four figures in ``reports/figures`` and a compact markdown summary.
Run with ``python -m src.eda``.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import FEATURES_PARQUET, FIGURES_DIR, REPORTS_DIR, SHOTS_PARQUET
from .features import prepare_features

GOAL_COLOR = "#e63946"
NO_GOAL_COLOR = "#8d99ae"
ACCENT = "#1d3557"
GRID_COLOR = "#d9e2ec"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#9fb3c8",
        "axes.labelcolor": "#243b53",
        "xtick.color": "#486581",
        "ytick.color": "#486581",
    }
)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a dataframe as a markdown table without extra dependencies."""
    headers = list(frame.columns)
    lines = ["| " + " | ".join(str(header) for header in headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in frame.iterrows():
        lines.append(
            "| " + " | ".join(str(row[header]) for header in headers) + " |"
        )
    return "\n".join(lines)


def _conversion_table(
    shots: pd.DataFrame, column: str, bins: list[float], labels: list[str]
) -> pd.DataFrame:
    """Count shots and goals per bin, with conversion rate."""
    binned = pd.cut(shots[column], bins=bins, labels=labels, right=False)
    table = (
        shots.groupby(binned, observed=True)
        .agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
        .reset_index(names=column)
    )
    table["conversion"] = table["goals"] / table["shots"]
    return table


def plot_shot_map(shots: pd.DataFrame) -> None:
    """Figure 1: shot locations coloured by outcome."""
    data = shots.loc[~shots["is_penalty"]]
    no_goal = data.loc[~data["is_goal"]]
    goals = data.loc[data["is_goal"]]

    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="#101820",
        line_color="#b8c4d0",
        linewidth=1.1,
    )
    fig, ax = pitch.draw(figsize=(11, 7.2))
    fig.patch.set_facecolor("#101820")
    ax.scatter(
        no_goal["x"],
        no_goal["y"],
        s=8,
        color="#7f8c9b",
        alpha=0.35,
        linewidths=0,
        label=f"No goal (n={len(no_goal):,})",
    )
    ax.scatter(
        goals["x"],
        goals["y"],
        s=24,
        color=GOAL_COLOR,
        edgecolor="white",
        linewidth=0.4,
        label=f"Goal (n={len(goals):,})",
        zorder=3,
    )
    ax.legend(
        loc="upper left",
        facecolor="#101820",
        edgecolor="#3a4a5a",
        labelcolor="white",
        framealpha=0.9,
    )
    ax.set_title(
        "Shot locations: goals cluster close to the goal, central angles",
        color="white",
        fontsize=14,
        loc="left",
        pad=12,
    )
    fig.text(
        0.012,
        0.012,
        "StatsBomb Open Data | non-penalty shots | 120x80 pitch",
        color="#8fa3b8",
        fontsize=8,
    )
    fig.savefig(
        FIGURES_DIR / "fig1_shot_map.png",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def plot_distance_angle(shots: pd.DataFrame) -> pd.DataFrame:
    """Figure 2: conversion by shot distance and goal angle."""
    data = shots.loc[~shots["is_penalty"]]
    distance = _conversion_table(
        data,
        "distance",
        [0, 6, 12, 18, 24, 30, 200],
        ["0-6", "6-12", "12-18", "18-24", "24-30", "30+"],
    )
    angle = _conversion_table(
        data,
        "angle_deg",
        [0, 5, 10, 15, 20, 25, 35, 180],
        ["0-5", "5-10", "10-15", "15-20", "20-25", "25-35", "35+"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, table, column, xlabel, title in (
        (
            axes[0],
            distance,
            "distance",
            "Distance to goal (units)",
            "Distance: closer is better",
        ),
        (
            axes[1],
            angle,
            "angle_deg",
            "Goal angle (degrees)",
            "Angle: wide-open looks finish best",
        ),
    ):
        bars = ax.bar(
            np.arange(len(table)),
            table["conversion"],
            color=ACCENT,
            width=0.72,
        )
        bars[0].set_color(GOAL_COLOR)
        ax.set_xticks(np.arange(len(table)), table[column], rotation=0)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, max(table["conversion"]) * 1.38)
        ax.set_ylabel("Conversion rate")
        ax.set_title(title, loc="left", fontsize=12, color="#102a43")
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for index, row in table.iterrows():
            ax.text(
                index,
                row["conversion"] + max(table["conversion"]) * 0.035,
                f"{row['conversion']:.1%}\n(n={int(row['shots']):,})",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334e68",
            )
    fig.suptitle(
        "Distance dominates conversion; angle matters most for open looks",
        x=0.01,
        ha="left",
        fontsize=14,
        color="#102a43",
    )
    fig.text(
        0.01,
        0.01,
        "StatsBomb Open Data | non-penalty shots",
        fontsize=8,
        color="#829ab1",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(FIGURES_DIR / "fig2_distance_angle.png", bbox_inches="tight")
    plt.close(fig)
    return distance, angle


def plot_body_part_type(shots: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Figure 3: conversion by body part and shot type."""
    data = shots.loc[~shots["is_penalty"]]
    body = (
        data.groupby("body_part", observed=True)
        .agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
        .reset_index()
    )
    body["conversion"] = body["goals"] / body["shots"]
    body = (
        body.loc[body["body_part"] != "Other"]
        .sort_values("conversion")
        .reset_index(drop=True)
    )

    shot_type = (
        data.groupby("shot_type", observed=True)
        .agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
        .reset_index()
    )
    shot_type["conversion"] = shot_type["goals"] / shot_type["shots"]
    shot_type = (
        shot_type.loc[shot_type["shots"] >= 20]
        .sort_values("conversion")
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, table, label, title in (
        (axes[0], body, "body_part", "Body part"),
        (axes[1], shot_type, "shot_type", "Shot type"),
    ):
        colors = [GOAL_COLOR if index == len(table) - 1 else ACCENT for index in range(len(table))]
        ax.barh(np.arange(len(table)), table["conversion"], color=colors, height=0.65)
        ax.set_yticks(np.arange(len(table)), table[label])
        ax.set_xlim(0, max(table["conversion"]) * 1.35)
        ax.set_xlabel("Conversion rate")
        ax.set_title(title, loc="left", fontsize=12, color="#102a43")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        for index, row in table.iterrows():
            ax.text(
                row["conversion"] + max(table["conversion"]) * 0.03,
                index,
                f"{row['conversion']:.1%} (n={int(row['shots']):,})",
                va="center",
                fontsize=8,
                color="#334e68",
            )
    fig.suptitle(
        "Conversion by body part and shot type",
        x=0.01,
        ha="left",
        fontsize=14,
        color="#102a43",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES_DIR / "fig3_body_part_type.png", bbox_inches="tight")
    plt.close(fig)
    return body, shot_type


def plot_xg_calibration(shots: pd.DataFrame) -> dict[str, float]:
    """Figure 4: how well StatsBomb xG is calibrated (the benchmark)."""
    data = shots.loc[~shots["is_penalty"]].dropna(
        subset=["xg_statsbomb", "is_goal"]
    ).copy()
    data["xg_bin"] = pd.qcut(data["xg_statsbomb"], 10, duplicates="drop")
    calibration = (
        data.groupby("xg_bin", observed=True)
        .agg(
            predicted=("xg_statsbomb", "mean"),
            observed=("is_goal", "mean"),
            n=("is_goal", "size"),
        )
        .reset_index()
    )

    metrics = {
        "roc_auc": float(roc_auc_score(data["is_goal"], data["xg_statsbomb"])),
        "brier": float(brier_score_loss(data["is_goal"], data["xg_statsbomb"])),
        "log_loss": float(log_loss(data["is_goal"], data["xg_statsbomb"])),
    }

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9fb3c8", label="Perfect calibration")
    ax.plot(
        calibration["predicted"],
        calibration["observed"],
        marker="o",
        markersize=6,
        color=ACCENT,
        linewidth=2,
        label="StatsBomb xG (deciles)",
    )
    ax.fill_between(
        calibration["predicted"],
        calibration["observed"],
        calibration["predicted"],
        color=GOAL_COLOR,
        alpha=0.08,
    )
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Predicted xG (mean per decile)")
    ax.set_ylabel("Observed conversion")
    ax.set_title(
        "StatsBomb xG is well calibrated - the bar to match",
        loc="left",
        fontsize=12,
        color="#102a43",
    )
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.04,
        f"AUC {metrics['roc_auc']:.3f}\nBrier {metrics['brier']:.4f}\n"
        f"Log loss {metrics['log_loss']:.4f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        color="#334e68",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", edgecolor="#d9e2ec"),
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_xg_calibration.png", bbox_inches="tight")
    plt.close(fig)
    return metrics


def freeze_frame_effect(shots: pd.DataFrame) -> pd.DataFrame | None:
    """Compare conversion with and without a defender in the shot cone."""
    if not FEATURES_PARQUET.exists():
        return None
    features = pd.read_parquet(FEATURES_PARQUET)
    data = features.loc[
        (~features["is_penalty"]) & (features["has_freeze_frame"] == 1)
    ].copy()
    data["defender_in_cone"] = np.where(
        data["n_defenders_cone"].fillna(0) > 0, "1+ defender in cone", "No defender in cone"
    )
    table = (
        data.groupby("defender_in_cone", observed=True)
        .agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
        .reset_index()
    )
    table["conversion"] = table["goals"] / table["shots"]
    return table


def run_eda() -> None:
    """Run the full EDA and write figures plus a markdown summary."""
    if FEATURES_PARQUET.exists():
        shots = pd.read_parquet(FEATURES_PARQUET)
    else:
        shots = prepare_features(pd.read_parquet(SHOTS_PARQUET))
    non_penalty = shots.loc[~shots["is_penalty"]]
    penalties = shots.loc[shots["is_penalty"]]

    plot_shot_map(shots)
    distance, angle = plot_distance_angle(shots)
    body, shot_type = plot_body_part_type(shots)
    benchmark = plot_xg_calibration(shots)
    freeze = freeze_frame_effect(shots)

    print("\n=== EDA ===")
    print(
        f"Shots {len(shots):,} | goals {int(shots['is_goal'].sum()):,} | "
        f"base rate {shots['is_goal'].mean():.2%}"
    )
    print(
        f"Non-penalty base rate {non_penalty['is_goal'].mean():.2%} | "
        f"penalties {len(penalties):,} ({len(penalties) / len(shots):.2%}), "
        f"conversion {penalties['is_goal'].mean():.1%}"
    )
    print("\nDistance conversion:")
    print(distance.to_string(index=False, formatters={"conversion": "{:.2%}".format}))
    print("\nAngle conversion:")
    print(angle.to_string(index=False, formatters={"conversion": "{:.2%}".format}))
    print("\nBody part conversion:")
    print(body.to_string(index=False, formatters={"conversion": "{:.2%}".format}))
    print("\nShot type conversion:")
    print(shot_type.to_string(index=False, formatters={"conversion": "{:.2%}".format}))
    print("\nStatsBomb xG benchmark:")
    print(
        f"  AUC {benchmark['roc_auc']:.3f} | Brier {benchmark['brier']:.4f} | "
        f"Log loss {benchmark['log_loss']:.4f}"
    )
    if freeze is not None:
        print("\nFreeze-frame effect:")
        print(freeze.to_string(index=False, formatters={"conversion": "{:.2%}".format}))

    def rate_table(table: pd.DataFrame) -> pd.DataFrame:
        return table.assign(conversion=table["conversion"].map("{:.2%}".format))

    report: list[str] = [
        "# EDA summary",
        "",
        "Source: StatsBomb Open Data via `statsbombpy`. Non-penalty shots unless stated.",
        "",
        f"- Matches: {shots['match_id'].nunique():,}",
        f"- Shots: {len(shots):,}",
        f"- Base rate: {shots['is_goal'].mean():.2%}",
        f"- Non-penalty base rate: {non_penalty['is_goal'].mean():.2%}",
        f"- Penalties: {len(penalties):,} ({len(penalties) / len(shots):.2%}), "
        f"conversion {penalties['is_goal'].mean():.1%}",
        "",
        "## Conversion by distance",
        "",
        _markdown_table(rate_table(distance)),
        "",
        "## Conversion by angle",
        "",
        _markdown_table(rate_table(angle)),
        "",
        "## Conversion by body part",
        "",
        _markdown_table(rate_table(body)),
        "",
        "## Conversion by shot type",
        "",
        _markdown_table(rate_table(shot_type)),
        "",
        "## StatsBomb xG benchmark",
        "",
        f"- ROC-AUC: {benchmark['roc_auc']:.3f}",
        f"- Brier: {benchmark['brier']:.4f}",
        f"- Log loss: {benchmark['log_loss']:.4f}",
        "",
    ]
    if freeze is not None:
        report += [
            "## Freeze-frame effect",
            "",
            _markdown_table(rate_table(freeze)),
            "",
        ]
    report += [
        "## Figures",
        "",
        "- `fig1_shot_map.png`",
        "- `fig2_distance_angle.png`",
        "- `fig3_body_part_type.png`",
        "- `fig4_xg_calibration.png`",
        "",
    ]
    path = REPORTS_DIR / "eda_summary.md"
    path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved EDA summary -> {path}")
    print(f"Saved figures -> {FIGURES_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShotWorth EDA.")
    parser.parse_args()
    run_eda()


if __name__ == "__main__":
    main()
