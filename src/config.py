"""Shared paths and constants for PitchIQ."""

from __future__ import annotations

from pathlib import Path

SEED: int = 42

ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
RAW_SHOTS_DIR: Path = RAW_DIR / "shots"
REPORTS_DIR: Path = ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
MODELS_DIR: Path = ROOT / "models"

SHOTS_PARQUET: Path = DATA_DIR / "shots.parquet"
FEATURES_PARQUET: Path = DATA_DIR / "features.parquet"
MATCHES_PARQUET: Path = RAW_DIR / "matches.parquet"

for _directory in (DATA_DIR, RAW_DIR, RAW_SHOTS_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
