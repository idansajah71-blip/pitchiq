"""Download StatsBomb open-data shots and cache them as parquet files.

Data source: StatsBomb Open Data (https://github.com/statsbomb/open-data),
accessed through ``statsbombpy``.

Design notes
------------
* Only pre-shot information is used later as model input.
* ``shot_outcome`` is kept solely to build the binary target ``is_goal``.
* ``shot_statsbomb_xg`` is kept solely as an external benchmark.
* ``shot_end_location`` is intentionally not stored.

Run ``python -m src.data`` to fetch (or resume) the cached dataset.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from statsbombpy import sb

from .config import MATCHES_PARQUET, RAW_SHOTS_DIR, REPORTS_DIR, SHOTS_PARQUET

warnings.filterwarnings("ignore", message=".*credentials were not supplied.*")


@dataclass(frozen=True)
class Competition:
    """A StatsBomb competition/season pair."""

    competition_id: int
    season_id: int
    name: str
    season: str


# StatsBomb open data does not contain every match of every season.
# These three competitions give a good mix of league and tournament football.
COMPETITIONS: tuple[Competition, ...] = (
    Competition(2, 27, "Premier League", "2015/2016"),
    Competition(11, 90, "La Liga", "2019/2020"),
    Competition(43, 106, "FIFA World Cup", "2022"),
)

SHOT_COLUMNS: list[str] = [
    "event_id",
    "match_id",
    "minute",
    "second",
    "period",
    "player_id",
    "player",
    "team_id",
    "team",
    "possession_team",
    "x",
    "y",
    "xg_statsbomb",
    "outcome",
    "is_goal",
    "is_penalty",
    "body_part",
    "shot_type",
    "technique",
    "play_pattern",
    "under_pressure",
    "first_time",
    "open_goal",
    "one_on_one",
    "aerial_won",
    "freeze_frame",
]

COLUMN_ROLES: dict[str, str] = {
    "event_id": "meta",
    "match_id": "meta (group split)",
    "minute": "safe (pre-shot)",
    "second": "safe (pre-shot)",
    "period": "safe (pre-shot)",
    "player": "meta",
    "player_id": "meta",
    "team": "meta",
    "team_id": "meta",
    "possession_team": "meta",
    "x": "safe (pre-shot location)",
    "y": "safe (pre-shot location)",
    "body_part": "safe (pre-shot)",
    "shot_type": "safe (pre-shot)",
    "technique": "safe (pre-shot)",
    "play_pattern": "safe (pre-shot)",
    "under_pressure": "safe (pre-shot)",
    "first_time": "safe (pre-shot)",
    "open_goal": "safe (pre-shot)",
    "one_on_one": "safe (pre-shot)",
    "aerial_won": "safe (pre-shot)",
    "freeze_frame": "safe (snapshot at shot time)",
    "is_penalty": "flag (penalties excluded from model)",
    "is_goal": "TARGET (derived from outcome)",
    "outcome": "LEAKAGE / target source",
    "xg_statsbomb": "BENCHMARK only - never a feature",
}


def _col(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return a column, or an all-None series when StatsBomb omitted it."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def _xy(location: Any) -> tuple[float, float]:
    """Convert a StatsBomb ``[x, y]`` location to two floats."""
    if isinstance(location, (list, tuple, np.ndarray)) and len(location) >= 2:
        try:
            return float(location[0]), float(location[1])
        except (TypeError, ValueError):
            pass
    return float("nan"), float("nan")


def _freeze_frame_json(value: Any) -> str | None:
    """Serialise the shot freeze frame for portable parquet storage."""
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return json.dumps(list(value))
    return None


def _extract_shots(events: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Extract one row per shot from a StatsBomb events dataframe."""
    shots = events.loc[events["type"] == "Shot"].copy()
    if shots.empty:
        return pd.DataFrame(columns=SHOT_COLUMNS)

    xs: list[float] = []
    ys: list[float] = []
    for location in _col(shots, "location"):
        x, y = _xy(location)
        xs.append(x)
        ys.append(y)

    frame = pd.DataFrame(
        {
            "event_id": _col(shots, "id").to_numpy(),
            "match_id": int(match_id),
            "minute": pd.to_numeric(_col(shots, "minute"), errors="coerce"),
            "second": pd.to_numeric(_col(shots, "second"), errors="coerce"),
            "period": pd.to_numeric(_col(shots, "period"), errors="coerce"),
            "player_id": pd.to_numeric(_col(shots, "player_id"), errors="coerce"),
            "player": _col(shots, "player").to_numpy(),
            "team_id": pd.to_numeric(_col(shots, "team_id"), errors="coerce"),
            "team": _col(shots, "team").to_numpy(),
            "possession_team": _col(shots, "possession_team").to_numpy(),
            "x": xs,
            "y": ys,
            "xg_statsbomb": pd.to_numeric(
                _col(shots, "shot_statsbomb_xg"), errors="coerce"
            ).to_numpy(),
            "outcome": _col(shots, "shot_outcome").to_numpy(),
            "body_part": _col(shots, "shot_body_part").to_numpy(),
            "shot_type": _col(shots, "shot_type").to_numpy(),
            "technique": _col(shots, "shot_technique").to_numpy(),
            "play_pattern": _col(shots, "play_pattern").to_numpy(),
            "under_pressure": _col(shots, "under_pressure").eq(True).to_numpy(),
            "first_time": _col(shots, "shot_first_time").eq(True).to_numpy(),
            "open_goal": _col(shots, "shot_open_goal").eq(True).to_numpy(),
            "one_on_one": _col(shots, "shot_one_on_one").eq(True).to_numpy(),
            "aerial_won": _col(shots, "shot_aerial_won").eq(True).to_numpy(),
            "freeze_frame": [
                _freeze_frame_json(value)
                for value in _col(shots, "shot_freeze_frame")
            ],
        }
    )

    frame["is_goal"] = frame["outcome"].eq("Goal")
    frame["is_penalty"] = frame["shot_type"].eq("Penalty")
    return frame[SHOT_COLUMNS]


def fetch_matches(
    competitions: Sequence[Competition] = COMPETITIONS, refresh: bool = False
) -> pd.DataFrame:
    """Fetch match metadata for the configured competitions."""
    if MATCHES_PARQUET.exists() and not refresh:
        return pd.read_parquet(MATCHES_PARQUET)

    frames: list[pd.DataFrame] = []
    for competition in competitions:
        matches = sb.matches(
            competition_id=competition.competition_id,
            season_id=competition.season_id,
        )
        frame = pd.DataFrame(
            {
                "match_id": matches["match_id"].astype(int),
                "match_date": matches["match_date"],
                "home_team": matches["home_team"],
                "away_team": matches["away_team"],
                "home_score": matches["home_score"],
                "away_score": matches["away_score"],
            }
        )
        frame["competition"] = competition.name
        frame["season"] = competition.season
        frames.append(frame)

    all_matches = pd.concat(frames, ignore_index=True)
    all_matches["match_date"] = pd.to_datetime(all_matches["match_date"])
    all_matches.to_parquet(MATCHES_PARQUET, index=False)
    return all_matches


def _fetch_match_shots(match_id: int, refresh: bool = False) -> pd.DataFrame:
    """Fetch one match and cache its shot rows; retries transient failures."""
    cache_path = RAW_SHOTS_DIR / f"{match_id}.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            events = sb.events(match_id=match_id)
            shots = _extract_shots(events, match_id)
            shots.to_parquet(cache_path, index=False)
            return shots
        except Exception as exc:  # network / rate-limit / malformed response
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"match {match_id} failed: {last_error}")


def fetch_all_shots(
    matches: pd.DataFrame, refresh: bool = False, max_workers: int = 16
) -> pd.DataFrame:
    """Fetch and concatenate shots for every match, in parallel."""
    match_ids = sorted({int(value) for value in matches["match_id"]})
    results: list[pd.DataFrame] = []
    failures: list[tuple[int, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_match_shots, match_id, refresh): match_id
            for match_id in match_ids
        }
        for done, future in enumerate(as_completed(futures), start=1):
            match_id = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append((match_id, str(exc)))
            if done % 25 == 0 or done == len(match_ids):
                print(f"  fetched {done}/{len(match_ids)} matches", flush=True)

    if failures:
        print(f"ERROR: {len(failures)} matches failed; refusing to build a partial dataset.")
        for match_id, message in failures[:10]:
            print(f"  {match_id}: {message}")
        remaining = len(failures) - min(len(failures), 10)
        if remaining:
            print(f"  ... and {remaining} more failed matches")
        raise RuntimeError(
            f"StatsBomb download incomplete: {len(failures)}/{len(match_ids)} matches failed. "
            "Retry with fewer workers or rerun after the network/rate limit clears."
        )
    if not results:
        raise RuntimeError("No shot data were fetched; refusing to create an empty dataset.")

    shots = pd.concat(results, ignore_index=True)
    fetched_ids = set(shots["match_id"].dropna().astype(int).unique())
    missing_ids = set(match_ids) - fetched_ids
    if missing_ids:
        raise RuntimeError(
            f"No shot rows were returned for {len(missing_ids)} expected matches; "
            "check data coverage before training."
        )
    return shots


def build_shots(
    refresh: bool = False, max_workers: int = 16
) -> pd.DataFrame:
    """Build ``data/shots.parquet`` from StatsBomb open data."""
    matches = fetch_matches(refresh=refresh)
    shots = fetch_all_shots(matches, refresh=refresh, max_workers=max_workers)

    metadata = matches[
        [
            "match_id",
            "match_date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "competition",
            "season",
        ]
    ]
    shots = shots.merge(metadata, on="match_id", how="left", validate="many_to_one")
    shots = shots.sort_values(
        ["match_id", "minute", "second"], kind="stable"
    ).reset_index(drop=True)
    shots.to_parquet(SHOTS_PARQUET, index=False)
    return shots


def _markdown_table(rows: Iterable[dict[str, Any]]) -> str:
    """Render simple dictionaries as a markdown table."""
    rows = list(rows)
    if not rows:
        return ""
    headers = list(rows[0])
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    return "\n".join(lines)


def summarize_shots(shots: pd.DataFrame) -> None:
    """Print dataset headline numbers and a column safety audit."""
    total = len(shots)
    goals = int(shots["is_goal"].sum())
    penalties = shots["is_penalty"]
    non_penalty = shots.loc[~penalties]

    print("\n=== SHOTWORTH DATASET ===")
    print(f"Competitions : {shots['competition'].nunique()}")
    print(f"Matches      : {shots['match_id'].nunique():,}")
    print(f"Shots        : {total:,}")
    print(f"Goals        : {goals:,}")
    print(f"Base rate    : {goals / total:.3%}")
    print(
        f"Penalties    : {int(penalties.sum()):,} "
        f"({penalties.mean():.2%} of shots), "
        f"conversion {shots.loc[penalties, 'is_goal'].mean():.1%}"
    )
    print(
        f"Non-penalty  : {len(non_penalty):,} shots, "
        f"base rate {non_penalty['is_goal'].mean():.3%}"
    )

    by_competition = (
        shots.groupby(["competition", "season"], as_index=False)
        .agg(
            matches=("match_id", "nunique"),
            shots=("event_id", "count"),
            goals=("is_goal", "sum"),
        )
        .assign(conversion=lambda frame: frame["goals"] / frame["shots"])
    )
    print("\nBy competition:")
    print(by_competition.to_string(index=False, formatters={"conversion": "{:.2%}".format}))

    rows = []
    for column in SHOT_COLUMNS:
        missing = shots[column].isna().mean()
        rows.append(
            {
                "column": column,
                "role": COLUMN_ROLES.get(column, "safe (pre-shot)"),
                "dtype": str(shots[column].dtype),
                "missing_pct": f"{missing:.1%}",
            }
        )

    print("\nColumn audit (missing %):")
    print(pd.DataFrame(rows).to_string(index=False))

    report = [
        "# Data summary",
        "",
        "Source: StatsBomb Open Data via `statsbombpy`.",
        "",
        f"- Competitions: {shots['competition'].nunique()}",
        f"- Matches: {shots['match_id'].nunique():,}",
        f"- Shots: {total:,}",
        f"- Goals: {goals:,}",
        f"- Base rate: {goals / total:.2%}",
        f"- Penalty share: {penalties.mean():.2%} "
        f"(conversion {shots.loc[penalties, 'is_goal'].mean():.1%})",
        f"- Non-penalty base rate: {non_penalty['is_goal'].mean():.2%}",
        "",
        "## Competitions",
        "",
        "```text",
        by_competition.to_string(index=False),
        "```",
        "",
        "## Column audit",
        "",
        _markdown_table(rows),
        "",
        "`outcome` is only used to derive the target. `xg_statsbomb` is a "
        "benchmark and is never passed to a model. `shot_end_location` is not "
        "stored at all.",
        "",
    ]
    report_path = REPORTS_DIR / "data_summary.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved report -> {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ShotWorth shot dataset.")
    parser.add_argument(
        "--refresh", action="store_true", help="ignore caches and re-download"
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    shots = build_shots(refresh=args.refresh, max_workers=args.workers)
    summarize_shots(shots)
    print(f"\nSaved dataset -> {SHOTS_PARQUET}")


if __name__ == "__main__":
    main()
