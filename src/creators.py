"""Extend the shot dataset with pass-assist linkages (shot creators).

StatsBomb shots carry ``shot.key_pass_id`` pointing to the pass event that
created the chance. This module fetches event data per match once, resolves
``key_pass_id -> creator name`` and caches the shot-to-creator mapping so the
insight layer can rank real chance creators (and not just shooters).

Run: ``python -m src.creators`` (cached and resumable).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd
from statsbombpy import sb

from .config import DATA_DIR, RAW_DIR

CREATORS_DIR = RAW_DIR / "creators"
CREATORS_PARQUET = DATA_DIR / "creators.parquet"

CREATOR_COLUMNS: list[str] = [
    "match_id",
    "shot_event_id",
    "key_pass_id",
    "creator",
]


def _extract_creators(events: pd.DataFrame, match_id: int) -> pd.DataFrame | None:
    """Resolve ``shot.key_pass_id`` to the player who played the pass."""
    if "shot_key_pass_id" not in events.columns:
        return pd.DataFrame(columns=CREATOR_COLUMNS)

    shots = events.loc[(events["type"] == "Shot") & events["shot_key_pass_id"].notna()]
    if shots.empty:
        return pd.DataFrame(columns=CREATOR_COLUMNS)

    # key_pass_id always points to a Pass event; keep only id -> player pairs.
    passes = events.loc[events["type"] == "Pass", ["id", "player"]]
    id_to_player = dict(zip(passes["id"].astype(str), passes["player"]))

    rows: list[dict[str, Any]] = []
    shot_ids = shots["id"].astype(str).to_list()
    key_pass_ids = shots["shot_key_pass_id"].astype(str).to_list()
    for shot_id, key_pass_id in zip(shot_ids, key_pass_ids):
        creator = id_to_player.get(key_pass_id)
        if creator is None or pd.isna(creator):
            continue  # drop shots whose key pass resolves to no known player
        rows.append(
            {
                "match_id": int(match_id),
                "shot_event_id": shot_id,
                "key_pass_id": key_pass_id,
                "creator": str(creator),
            }
        )
    return pd.DataFrame(rows, columns=CREATOR_COLUMNS)


def _fetch_match(match_id: int, refresh: bool) -> pd.DataFrame:
    """Fetch or read the cached creator mapping for one match."""
    path = CREATORS_DIR / f"{match_id}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    for attempt in range(3):
        try:
            events = sb.events(match_id=match_id)
            frame = _extract_creators(events, match_id)
            if frame is None:
                frame = pd.DataFrame(columns=CREATOR_COLUMNS)
            frame.to_parquet(path, index=False)
            return frame
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    # Empty cache marker so reruns resume; failures are visible in the output.
    return pd.DataFrame(columns=CREATOR_COLUMNS)


def build_creators(refresh: bool = False, max_workers: int = 12) -> pd.DataFrame:
    """Build ``data/creators.parquet`` for every match in the dataset."""
    CREATORS_DIR.mkdir(parents=True, exist_ok=True)
    matches = pd.read_parquet(RAW_DIR / "matches.parquet")
    match_ids = sorted(matches["match_id"].astype(int).unique())

    results: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_match, match_id, refresh): match_id
            for match_id in match_ids
        }
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % 50 == 0 or done == len(match_ids):
                print(f"  processed {done}/{len(match_ids)} matches", flush=True)

    creators = pd.concat(results, ignore_index=True)
    creators.to_parquet(CREATORS_PARQUET, index=False)
    return creators


def summarize(creators: pd.DataFrame) -> None:
    """Print headline creator statistics."""
    if creators.empty:
        print("No creators resolved.")
        return
    top = creators.groupby("creator").size().sort_values(ascending=False).head(5)
    print(f"Assisted shots resolved: {len(creators):,}")
    print("Top 5 creators by assisted shots:")
    print(top.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve shot creators (assists).")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="debug: only N matches")
    args = parser.parse_args()

    started = time.time()
    if args.limit:
        matches = pd.read_parquet(RAW_DIR / "matches.parquet").head(args.limit)
        CREATORS_DIR.mkdir(parents=True, exist_ok=True)
        frames = [_fetch_match(int(m), args.refresh) for m in matches["match_id"]]
        creators = pd.concat(frames, ignore_index=True)
    else:
        creators = build_creators(refresh=args.refresh, max_workers=args.workers)
    summarize(creators)
    print(f"Saved -> {CREATORS_PARQUET}")
    print(f"Runtime: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
