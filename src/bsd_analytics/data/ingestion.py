"""
Data ingestion module for BSD-Analytics.
Handles automated downloading, extraction, schema mapping, and raw data persistence.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from bsd_analytics.config import (
    RAW_DATA_DIR,
    UCI_DATASET_URL,
    ensure_directories,
)

logger = logging.getLogger(__name__)


def download_dataset(
    url: str = UCI_DATASET_URL,
    target_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> Path:
    """
    Download the official Capital Bikeshare archive from UCI repository
    and extract hour.csv and day.csv.

    Args:
        url: Source archive URL.
        target_dir: Destination directory.
        force: If True, re-download even if already present.

    Returns:
        Path to the extracted hour.csv file.
    """
    ensure_directories()
    hour_csv_path = target_dir / "hour.csv"
    day_csv_path = target_dir / "day.csv"

    if hour_csv_path.exists() and not force:
        logger.info("Raw dataset already exists at %s", hour_csv_path)
        return hour_csv_path

    logger.info("Downloading dataset from %s ...", url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BSD-Analytics/1.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req, timeout=30) as response:
        content = response.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extract("hour.csv", path=target_dir)
        zf.extract("day.csv", path=target_dir)

    logger.info(
        "Successfully extracted hour.csv (%s) and day.csv (%s)", hour_csv_path, day_csv_path
    )
    return hour_csv_path


def load_raw_data(
    source_path: Path | None = None,
    denormalize_weather: bool = True,
) -> pd.DataFrame:
    """
    Load raw hourly bike sharing dataset and harmonize with standard competition schema.

    The original UCI repository stores continuous meteorological values normalized:
    - temp: normalized by 41 deg C
    - atemp: normalized by 50 deg C
    - hum: normalized by 100
    - windspeed: normalized by 67

    Harmonization restores original engineering units for intuitive interpretability:
    temp (deg C), atemp (deg C), humidity (%), windspeed (km/h).

    Args:
        source_path: Optional explicit path to raw CSV file.
        denormalize_weather: Whether to convert 0-1 metrics back to physical units.

    Returns:
        Standardized raw DataFrame with datetime, weather metrics, and rental counts.
    """
    if source_path is None or not Path(source_path).exists():
        source_path = download_dataset()

    df_raw = pd.read_csv(source_path)

    # Standardize schema from UCI naming to Canonical / Kaggle Competition format
    # UCI: dteday, hr, season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered, cnt
    datetime_series = pd.to_datetime(
        df_raw["dteday"] + " " + df_raw["hr"].astype(str).str.zfill(2) + ":00:00"
    )

    df_standard = pd.DataFrame(
        {
            "datetime": datetime_series,
            "season": df_raw["season"].astype(int),
            "holiday": df_raw["holiday"].astype(int),
            "workingday": df_raw["workingday"].astype(int),
            "weather": df_raw["weathersit"].astype(int),
            "temp": df_raw["temp"] * 41.0 if denormalize_weather else df_raw["temp"],
            "atemp": df_raw["atemp"] * 50.0 if denormalize_weather else df_raw["atemp"],
            "humidity": df_raw["hum"] * 100.0 if denormalize_weather else df_raw["hum"],
            "windspeed": df_raw["windspeed"] * 67.0 if denormalize_weather else df_raw["windspeed"],
            "casual": df_raw["casual"].astype(int),
            "registered": df_raw["registered"].astype(int),
            "count": df_raw["cnt"].astype(int),
        }
    )

    # Sort chronologically and drop any index artifacts
    df_standard = df_standard.sort_values("datetime").reset_index(drop=True)

    # Save standardized raw copy to raw data dir for auditability
    standard_raw_path = RAW_DATA_DIR / "bike_sharing_raw.csv"
    if not standard_raw_path.exists():
        df_standard.to_csv(standard_raw_path, index=False)
        logger.info("Saved harmonized raw dataset to %s", standard_raw_path)

    return df_standard


def get_dataset_metadata(df: pd.DataFrame) -> dict[str, Any]:
    """
    Extract summary metadata for data cataloging and reporting.
    """
    return {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "start_time": str(df["datetime"].min()),
        "end_time": str(df["datetime"].max()),
        "total_rentals": int(df["count"].sum()),
        "casual_rentals": int(df["casual"].sum()),
        "registered_rentals": int(df["registered"].sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }
