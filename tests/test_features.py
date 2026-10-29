"""Unit tests for shot geometry and freeze-frame features."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    distance_to_goal,
    get_X_y,
    point_in_shot_cone,
    prepare_features,
    shot_angle_deg,
)


def test_distance_from_penalty_spot() -> None:
    assert distance_to_goal(108.0, 40.0) == pytest.approx(12.0)


def test_distance_from_goal_post() -> None:
    assert distance_to_goal(120.0, 36.0) == pytest.approx(4.0)


def test_angle_on_goal_line_center() -> None:
    assert shot_angle_deg(120.0, 40.0) == pytest.approx(180.0)


def test_angle_from_twelve_yards() -> None:
    expected = math.degrees(2.0 * math.atan2(4.0, 12.0))
    assert shot_angle_deg(108.0, 40.0) == pytest.approx(expected)


def test_angle_is_symmetric_around_goal_center() -> None:
    assert shot_angle_deg(108.0, 36.0) == pytest.approx(
        shot_angle_deg(108.0, 44.0)
    )


def test_angle_shrinks_with_distance() -> None:
    assert shot_angle_deg(102.0, 40.0) < shot_angle_deg(114.0, 40.0)


def test_point_in_cone() -> None:
    assert point_in_shot_cone(114.0, 40.0, 108.0, 40.0)
    assert not point_in_shot_cone(114.0, 55.0, 108.0, 40.0)
    assert not point_in_shot_cone(100.0, 40.0, 108.0, 40.0)


def _raw_shot(**overrides: object) -> dict[str, object]:
    shot: dict[str, object] = {
        "x": 114.0,
        "y": 40.0,
        "minute": 30,
        "body_part": "Right Foot",
        "shot_type": "Open Play",
        "technique": "Normal",
        "play_pattern": "Regular Play",
        "under_pressure": False,
        "first_time": False,
        "open_goal": False,
        "one_on_one": False,
        "aerial_won": False,
        "is_penalty": False,
        "is_goal": True,
        "freeze_frame": None,
    }
    shot.update(overrides)
    return shot


def test_freeze_frame_features() -> None:
    frame = [
        {"location": [117.0, 38.0], "teammate": False, "position": {"name": "Center Back"}},
        {"location": [110.0, 60.0], "teammate": False, "position": {"name": "Left Back"}},
        {"location": [116.0, 40.0], "teammate": False, "position": {"name": "Goalkeeper"}},
        {"location": [117.0, 39.0], "teammate": True, "position": {"name": "Center Forward"}},
    ]
    features = prepare_features(pd.DataFrame([_raw_shot(freeze_frame=frame)]))

    row = features.iloc[0]
    assert row["n_defenders_cone"] == 1
    assert row["n_defenders_near"] == 1
    assert row["nearest_defender_dist"] == pytest.approx(math.hypot(3.0, 2.0))
    assert row["gk_off_line"] == pytest.approx(4.0)
    assert row["gk_y_offset"] == pytest.approx(0.0)
    assert row["has_freeze_frame"] == 1


def test_freeze_frame_missing_is_flagged() -> None:
    features = prepare_features(pd.DataFrame([_raw_shot()]))
    row = features.iloc[0]

    assert row["has_freeze_frame"] == 0
    assert math.isnan(row["nearest_defender_dist"])
    assert row["distance"] == pytest.approx(6.0)
    assert row["angle_deg"] == pytest.approx(
        math.degrees(2.0 * math.atan2(4.0, 6.0))
    )


def test_penalties_are_excluded_by_default() -> None:
    features = prepare_features(
        pd.DataFrame(
            [_raw_shot(), _raw_shot(is_penalty=True, shot_type="Penalty")]
        )
    )
    X, y = get_X_y(features)

    assert len(X) == 1
    assert len(y) == 1
    assert list(X.columns) == FEATURE_COLUMNS


def test_no_forbidden_features() -> None:
    assert not FORBIDDEN_FEATURES.intersection(FEATURE_COLUMNS)
