"""
Configuration management for BSD-Analytics.
Provides centralized path resolution, schema definitions, and experiment constants.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

# Project Root Resolution
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

# Data Directories
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Final[Path] = DATA_DIR / "external"

# Artifact & Output Directories
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
METRICS_DIR: Final[Path] = REPORTS_DIR / "metrics"
DATA_QUALITY_DIR: Final[Path] = REPORTS_DIR / "data_quality"

# Database Configuration
DEFAULT_DB_PATH: Final[Path] = DATA_DIR / "analytics.db"
DATABASE_URI: Final[str] = os.getenv("BSD_DATABASE_URI", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

# Random Seed for Reproducibility
RANDOM_SEED: Final[int] = 42

# Target & Prohibited Leakage Features
TARGET_COLUMN: Final[str] = "count"
LEAKAGE_COLUMNS: Final[list[str]] = ["casual", "registered"]

# Canonical Expected Schema
EXPECTED_COLUMNS: Final[list[str]] = [
    "datetime",
    "season",
    "holiday",
    "workingday",
    "weather",
    "temp",
    "atemp",
    "humidity",
    "windspeed",
    "casual",
    "registered",
    "count",
]

# Primary Modeling Features (strictly excluding target & leakage columns)
BASE_NUMERIC_FEATURES: Final[list[str]] = [
    "temp",
    "atemp",
    "humidity",
    "windspeed",
]

BASE_CATEGORICAL_FEATURES: Final[list[str]] = [
    "season",
    "holiday",
    "workingday",
    "weather",
]

ENGINEERED_TEMPORAL_FEATURES: Final[list[str]] = [
    "hour",
    "month",
    "year",
    "weekday",
    "is_weekend",
    "peak_commute_hour",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

ENGINEERED_WEATHER_FEATURES: Final[list[str]] = [
    "temp_humidity_ratio",
    "temp_feels_diff",
    "wind_chill_approx",
    "bad_weather_flag",
]

ALL_FEATURE_COLUMNS: Final[list[str]] = (
    BASE_NUMERIC_FEATURES
    + BASE_CATEGORICAL_FEATURES
    + ENGINEERED_TEMPORAL_FEATURES
    + ENGINEERED_WEATHER_FEATURES
)

# Dataset Download URL (UCI Machine Learning Repository Capital Bikeshare archive)
UCI_DATASET_URL: Final[str] = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
)


def ensure_directories() -> None:
    """Ensure all required project directories exist on the filesystem."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        DATA_QUALITY_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
