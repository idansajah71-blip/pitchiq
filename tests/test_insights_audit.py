"""Regression tests for scope labeling and counterfactual arithmetic."""

from __future__ import annotations

import pandas as pd
import pytest

from src.insights import (
    estimate_replacement_xg_gain,
    player_table,
    top_players_markdown,
    zone_shot_share_matrix,
)


def test_player_stats_are_separated_by_team_and_competition() -> None:
    frame = pd.DataFrame(
        [
            {
                "player": "Alex Example",
                "team": "Club A",
                "competition": "League A",
                "event_id": "shot-1",
                "match_id": 1,
                "is_goal": 1,
                "pitchiq_xg": 0.3,
                "xg_statsbomb": 0.25,
            },
            {
                "player": "Alex Example",
                "team": "Country B",
                "competition": "World Cup",
                "event_id": "shot-2",
                "match_id": 2,
                "is_goal": 0,
                "pitchiq_xg": 0.1,
                "xg_statsbomb": 0.08,
            },
        ]
    )

    result = player_table(frame, creators=None)
    assert len(result.loc["Alex Example"]) == 2
    assert set(result.loc["Alex Example", "team"]) == {"Club A", "Country B"}
    assert set(result.loc["Alex Example", "competition"]) == {"League A", "World Cup"}
    assert sorted(result.loc["Alex Example", "shots"].tolist()) == [1, 1]


def test_replacement_xg_gain_uses_actual_baseline_probability() -> None:
    # One-third of 30 attempts replaced; each replacement moves from 0.02 to 0.10 xG.
    assert estimate_replacement_xg_gain(30, 0.02) == pytest.approx(0.8)


def test_replacement_xg_gain_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        estimate_replacement_xg_gain(-1, 0.02)
    with pytest.raises(ValueError):
        estimate_replacement_xg_gain(30, 0.02, replacement_share=1.2)


def test_zone_shares_ignore_empty_cells_and_sum_to_one() -> None:
    counts = pd.DataFrame([[10.0, float("nan")], [30.0, 0.0]])
    shares = zone_shot_share_matrix(counts)
    assert shares.iloc[0, 0] == pytest.approx(0.25)
    assert shares.iloc[1, 0] == pytest.approx(0.75)
    assert pd.isna(shares.iloc[0, 1])
    assert float(shares.sum().sum()) == pytest.approx(1.0)


def test_zone_shares_for_empty_total_remain_nan() -> None:
    counts = pd.DataFrame([[float("nan"), float("nan")]])
    shares = zone_shot_share_matrix(counts)
    assert shares.isna().all().all()


def test_player_markdown_keeps_team_and_competition_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.insights.MIN_SHOTS", 1)
    frame = pd.DataFrame(
        [
            {
                "player": "Alex Example",
                "team": "Club A",
                "competition": "League A",
                "event_id": "shot-1",
                "match_id": 1,
                "is_goal": 1,
                "pitchiq_xg": 0.3,
                "xg_statsbomb": 0.25,
            },
            {
                "player": "Alex Example",
                "team": "Country B",
                "competition": "World Cup",
                "event_id": "shot-2",
                "match_id": 2,
                "is_goal": 0,
                "pitchiq_xg": 0.1,
                "xg_statsbomb": 0.08,
            },
        ]
    )
    stats = player_table(frame, creators=None)
    rendered = top_players_markdown(stats)
    assert "Alex Example | Club A | League A" in rendered
    assert "Alex Example | Country B | World Cup" in rendered


def test_creator_without_own_shot_remains_in_creator_leaderboard() -> None:
    frame = pd.DataFrame(
        [
            {
                "player": "Shooter",
                "team": "Club A",
                "competition": "League A",
                "event_id": "shot-1",
                "match_id": 1,
                "is_goal": 1,
                "pitchiq_xg": 0.3,
                "xg_statsbomb": 0.25,
            }
        ]
    )
    creators = pd.DataFrame(
        [
            {
                "match_id": 1,
                "shot_event_id": "shot-1",
                "creator": "Creator Without Shot",
            }
        ]
    )
    stats = player_table(frame, creators=creators)
    creator = stats.loc["Creator Without Shot"]
    assert int(creator["shots"]) == 0
    assert int(creator["assists"]) == 1
    assert creator["team"] == "Club A"
    assert creator["competition"] == "League A"
