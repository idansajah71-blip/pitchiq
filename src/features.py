"""Feature engineering for ShotWorth shot-level expected goals.

All features are known at the moment of the shot. The StatsBomb xG value,
shot outcome and shot end location are never used as inputs.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import FEATURES_PARQUET, REPORTS_DIR, SHOTS_PARQUET

PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0
GOAL_Y_MIN: float = 36.0
GOAL_Y_MAX: float = 44.0
GOAL_CENTER_Y: float = 40.0

NUMERIC_FEATURES: list[str] = [
    "distance",
    "angle_deg",
    "log_distance",
    "distance_to_goal_line",
    "abs_y_from_center",
    "minute",
    "n_defenders_cone",
    "n_defenders_near",
    "nearest_defender_dist",
    "gk_off_line",
    "gk_y_offset",
    "has_freeze_frame",
]

BOOLEAN_FEATURES: list[str] = [
    "under_pressure",
    "first_time",
    "open_goal",
    "one_on_one",
    "aerial_won",
]

CATEGORICAL_FEATURES: list[str] = [
    "body_part",
    "shot_type",
    "technique",
    "play_pattern",
]

FEATURE_COLUMNS: list[str] = (
    NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
)

ENGINEERED_COLUMNS: list[str] = [
    "distance",
    "angle_deg",
    "log_distance",
    "distance_to_goal_line",
    "abs_y_from_center",
    "n_defenders_cone",
    "n_defenders_near",
    "nearest_defender_dist",
    "gk_off_line",
    "gk_y_offset",
    "has_freeze_frame",
]

FORBIDDEN_FEATURES: set[str] = {
    "outcome",
    "is_goal",
    "xg_statsbomb",
    "shot_end_location",
    "end_location",
}

assert not FORBIDDEN_FEATURES.intersection(FEATURE_COLUMNS), "Leakage in features!"


def distance_to_goal(x: float | np.ndarray, y: float | np.ndarray) -> Any:
    """Euclidean distance from a shot location to the goal centre (120, 40)."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    return np.hypot(PITCH_LENGTH - x_array, GOAL_CENTER_Y - y_array)


def shot_angle_deg(x: float | np.ndarray, y: float | np.ndarray) -> Any:
    """Angle (degrees) subtended by the goal posts from the shot location."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    to_near_post = np.arctan2(GOAL_Y_MIN - y_array, PITCH_LENGTH - x_array)
    to_far_post = np.arctan2(GOAL_Y_MAX - y_array, PITCH_LENGTH - x_array)
    difference = np.abs(to_near_post - to_far_post)
    difference = np.where(difference > np.pi, 2.0 * np.pi - difference, difference)
    return np.degrees(difference)


def point_in_shot_cone(
    px: float,
    py: float,
    shooter_x: float,
    shooter_y: float,
    goal_y_min: float = GOAL_Y_MIN,
    goal_y_max: float = GOAL_Y_MAX,
) -> bool:
    """Return True when a point lies inside the triangle shooter-to-posts."""
    ax, ay = shooter_x, shooter_y
    bx, by = PITCH_LENGTH, goal_y_min
    cx, cy = PITCH_LENGTH, goal_y_max

    denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denominator) < 1e-12:
        return False

    u = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
    v = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
    w = 1.0 - u - v
    return u >= 0.0 and v >= 0.0 and w >= 0.0


def _parse_freeze_frame(value: Any) -> list[dict[str, Any]] | None:
    """Parse a freeze frame from a list or a JSON string."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return [entry for entry in value if isinstance(entry, dict)]
    return None


def _freeze_frame_features(
    shooter_x: float, shooter_y: float, value: Any
) -> dict[str, float]:
    """Summarise defender and goalkeeper positioning from a freeze frame."""
    features: dict[str, float] = {
        "n_defenders_cone": float("nan"),
        "n_defenders_near": float("nan"),
        "nearest_defender_dist": float("nan"),
        "gk_off_line": float("nan"),
        "gk_y_offset": float("nan"),
        "has_freeze_frame": 0.0,
    }
    frame = _parse_freeze_frame(value)
    if frame is None or not np.isfinite(shooter_x) or not np.isfinite(shooter_y):
        return features

    features["has_freeze_frame"] = 1.0
    opponents: list[dict[str, Any]] = []
    goalkeeper: dict[str, Any] | None = None

    for player in frame:
        if player.get("teammate", True):
            continue
        location = player.get("location")
        if not isinstance(location, (list, tuple)) or len(location) < 2:
            continue
        position = (player.get("position") or {}).get("name", "")
        if position == "Goalkeeper":
            goalkeeper = player
        else:
            opponents.append(player)

    if opponents:
        distances = [
            math.hypot(px - shooter_x, py - shooter_y)
            for px, py in (opponent["location"][:2] for opponent in opponents)
        ]
        features["nearest_defender_dist"] = float(min(distances))
        features["n_defenders_near"] = float(sum(distance <= 5.0 for distance in distances))
        features["n_defenders_cone"] = float(
            sum(
                point_in_shot_cone(px, py, shooter_x, shooter_y)
                for px, py in (opponent["location"][:2] for opponent in opponents)
            )
        )

    if goalkeeper is not None:
        gk_x, gk_y = goalkeeper["location"][:2]
        features["gk_off_line"] = float(max(0.0, PITCH_LENGTH - gk_x))
        features["gk_y_offset"] = float(abs(gk_y - GOAL_CENTER_Y))

    return features


def prepare_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Add model features to a shot-level dataframe.

    ``shots`` must contain the raw StatsBomb-derived columns produced by
    :mod:`src.data`. The function is intentionally stateless so the same code
    can be reused at inference time.
    """
    features = shots.copy()
    features = features.drop(
        columns=[column for column in ENGINEERED_COLUMNS if column in features.columns]
    )
    features["x"] = pd.to_numeric(features["x"], errors="coerce")
    features["y"] = pd.to_numeric(features["y"], errors="coerce")

    features["distance"] = distance_to_goal(features["x"], features["y"])
    features["angle_deg"] = shot_angle_deg(features["x"], features["y"])
    features["log_distance"] = np.log1p(features["distance"])
    features["distance_to_goal_line"] = PITCH_LENGTH - features["x"]
    features["abs_y_from_center"] = (features["y"] - GOAL_CENTER_Y).abs()
    features["minute"] = pd.to_numeric(features["minute"], errors="coerce")

    freeze_frame = (
        features["freeze_frame"]
        if "freeze_frame" in features
        else pd.Series([None] * len(features), index=features.index)
    )
    freeze_rows = [
        _freeze_frame_features(x, y, frame)
        for x, y, frame in zip(features["x"], features["y"], freeze_frame)
    ]
    freeze_features = pd.DataFrame(freeze_rows, index=features.index)
    features = pd.concat([features, freeze_features], axis=1)

    for column in BOOLEAN_FEATURES:
        if column not in features:
            features[column] = False
        features[column] = (
            features[column].fillna(False).astype(bool).astype("int8")
        )

    features["has_freeze_frame"] = (
        features["has_freeze_frame"].fillna(0).astype("int8")
    )
    return features


def get_X_y(
    features: pd.DataFrame, include_penalties: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model matrix and binary target.

    Penalties are excluded by default: they are a different task (a fixed
    one-on-one from 12 yards) and would distort shot-quality learning.
    """
    mask = pd.Series(True, index=features.index)
    if not include_penalties:
        mask &= ~features["is_penalty"].fillna(False).astype(bool)

    X = features.loc[mask, FEATURE_COLUMNS].copy()
    y = features.loc[mask, "is_goal"].astype(int).copy()
    return X, y


def build_preprocessor(
    numeric_features: Sequence[str] | None = None,
    boolean_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> ColumnTransformer:
    """Build a reusable preprocessing pipeline for training and inference.

    Feature lists can be overridden for ablation studies; the defaults use all
    engineered features.
    """
    numeric_columns = list(
        NUMERIC_FEATURES if numeric_features is None else numeric_features
    )
    boolean_columns = list(
        BOOLEAN_FEATURES if boolean_features is None else boolean_features
    )
    categorical_columns = list(
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns or boolean_columns:
        transformers.append(
            ("num", numeric_pipeline, numeric_columns + boolean_columns)
        )
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


FEATURE_DOCS: list[dict[str, str]] = [
    {
        "feature": "distance",
        "source": "location [x, y]",
        "football_reason": "Closer shots beat the keeper more often.",
        "leakage_risk": "None - known before the shot.",
    },
    {
        "feature": "angle_deg",
        "source": "location [x, y]",
        "football_reason": "Wider angle to the posts means more goal to aim at.",
        "leakage_risk": "None - known before the shot.",
    },
    {
        "feature": "log_distance",
        "source": "derived from distance",
        "football_reason": "Linearises the strong distance decay for the LR baseline.",
        "leakage_risk": "None - deterministic transform.",
    },
    {
        "feature": "distance_to_goal_line",
        "source": "120 - location x",
        "football_reason": "How deep the shooter is; separates wide-close from central-far.",
        "leakage_risk": "None - known before the shot.",
    },
    {
        "feature": "abs_y_from_center",
        "source": "location y vs goal centre 40",
        "football_reason": "Central shots have a better angle than wide ones.",
        "leakage_risk": "None - known before the shot.",
    },
    {
        "feature": "minute",
        "source": "event minute",
        "football_reason": "Late-game fatigue and game-state effects.",
        "leakage_risk": "None - timestamp of the shot.",
    },
    {
        "feature": "n_defenders_cone",
        "source": "shot_freeze_frame",
        "football_reason": "Defenders between shooter and goal block the shot.",
        "leakage_risk": "None - snapshot at shot time, not after.",
    },
    {
        "feature": "n_defenders_near",
        "source": "shot_freeze_frame",
        "football_reason": "Pressure from nearby defenders lowers shot quality.",
        "leakage_risk": "None - snapshot at shot time.",
    },
    {
        "feature": "nearest_defender_dist",
        "source": "shot_freeze_frame",
        "football_reason": "Time and space on the ball.",
        "leakage_risk": "None - snapshot at shot time.",
    },
    {
        "feature": "gk_off_line",
        "source": "shot_freeze_frame (Goalkeeper)",
        "football_reason": "A keeper caught off the line concedes easier finishes.",
        "leakage_risk": "None - position at shot time.",
    },
    {
        "feature": "gk_y_offset",
        "source": "shot_freeze_frame (Goalkeeper)",
        "football_reason": "A keeper shaded to one side leaves the other side open.",
        "leakage_risk": "None - position at shot time.",
    },
    {
        "feature": "has_freeze_frame",
        "source": "shot_freeze_frame presence",
        "football_reason": "Makes missing defensive context explicit rather than silent.",
        "leakage_risk": "None - data availability indicator.",
    },
    {
        "feature": "under_pressure",
        "source": "under_pressure flag",
        "football_reason": "Pressure reduces technique and placement.",
        "leakage_risk": "None - recorded during the event.",
    },
    {
        "feature": "first_time",
        "source": "shot_first_time flag",
        "football_reason": "First-time finishes are harder to control.",
        "leakage_risk": "None - known before contact.",
    },
    {
        "feature": "open_goal",
        "source": "shot_open_goal flag",
        "football_reason": "Empty-net situations are near-certain goals.",
        "leakage_risk": "None - known before the shot.",
    },
    {
        "feature": "one_on_one",
        "source": "shot_one_on_one flag",
        "football_reason": "A clean look against the keeper is a different shot type.",
        "leakage_risk": "None - known before the shot.",
    },
    {
        "feature": "aerial_won",
        "source": "shot_aerial_won flag",
        "football_reason": "Aerial duels won before the shot signal a cleaner contact.",
        "leakage_risk": "None - occurs before the shot.",
    },
    {
        "feature": "body_part",
        "source": "shot_body_part",
        "football_reason": "Contact type changes control; headers cluster at close range, so read it with location.",
        "leakage_risk": "None - known at contact.",
    },
    {
        "feature": "shot_type",
        "source": "shot_type",
        "football_reason": "Open play and free kicks have different geometries; penalties are removed.",
        "leakage_risk": "None; penalty rows are removed before training.",
    },
    {
        "feature": "technique",
        "source": "shot_technique",
        "football_reason": "Volleys, lobs and backheels are distinct contact types selected by situation.",
        "leakage_risk": "None - known at contact.",
    },
    {
        "feature": "play_pattern",
        "source": "play_pattern",
        "football_reason": "Counters and set pieces create different shot contexts.",
        "leakage_risk": "None - known before the shot.",
    },
]


def write_feature_docs() -> None:
    """Write the feature justification table to reports/."""
    lines = [
        "# Feature justification",
        "",
        "Every feature is recorded before or at the moment of the shot.",
        "",
        "| Feature | Source column | Football reason | Leakage risk |",
        "|---|---|---|---|",
    ]
    for row in FEATURE_DOCS:
        lines.append(
            f"| `{row['feature']}` | {row['source']} | "
            f"{row['football_reason']} | {row['leakage_risk']} |"
        )
    lines += [
        "",
        "## Excluded from the model",
        "",
        "| Column | Why it is excluded |",
        "|---|---|",
        "| `outcome` / `is_goal` | Target only. |",
        "| `xg_statsbomb` | StatsBomb benchmark; using it as a feature would "
        "copy the answer. |",
        "| `shot_end_location` | Only exists after the shot; never stored. |",
        "| `player`, `team` | Identity shortcuts that do not generalise to new "
        "matches. |",
        "| penalty shots | Fixed 12-yard one-on-one; excluded from the main "
        "model and reported separately. |",
        "",
        "## Missing values",
        "",
        "Freeze-frame features are missing when StatsBomb has no freeze frame "
        "for a shot (1.2% of shots). They are median-imputed inside the sklearn "
        "pipeline and flagged with `has_freeze_frame`, so the model can learn "
        "how much to trust them. Goalkeeper features are missing in 1.3% of "
        "shots. The same preprocessor is used at inference time.",
        "",
    ]
    path = REPORTS_DIR / "feature_justification.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved feature docs -> {path}")


def build_features(shots: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build and persist the engineered feature table."""
    if shots is None:
        shots = pd.read_parquet(SHOTS_PARQUET)

    features = prepare_features(shots)
    features.to_parquet(FEATURES_PARQUET, index=False)

    print("\n=== FEATURES ===")
    print(f"Shots rows          : {len(features):,}")
    print(f"Penalties (excluded): {int(features['is_penalty'].sum()):,}")
    print(f"Model rows          : {int((~features['is_penalty']).sum()):,}")
    print(
        "Freeze frame present: "
        f"{features['has_freeze_frame'].mean():.1%} of shots"
    )
    missing = (
        features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        .isna()
        .mean()
        .sort_values(ascending=False)
    )
    print("\nMissing rate for engineered features:")
    print(missing.to_string(float_format=lambda value: f"{value:.2%}"))

    by_competition = (
        features.groupby(["competition", "season"])["has_freeze_frame"]
        .mean()
        .sort_values()
    )
    print("\nFreeze-frame coverage by competition:")
    print(by_competition.to_string(float_format=lambda value: f"{value:.1%}"))

    write_feature_docs()
    print(f"\nSaved features -> {FEATURES_PARQUET}")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ShotWorth features.")
    parser.parse_args()
    build_features()


if __name__ == "__main__":
    main()
