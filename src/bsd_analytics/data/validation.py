"""
Data validation and data quality auditing module for BSD-Analytics.
Ensures strict schema conformance, value sanity bounds, and audit reporting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bsd_analytics.config import (
    DATA_QUALITY_DIR,
    EXPECTED_COLUMNS,
    ensure_directories,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Detailed results of schema validation and quality assessment."""

    is_valid: bool = True
    total_records: int = 0
    missing_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    leakage_mismatches: int = 0  # Where casual + registered != count
    invalid_temperature_records: int = 0  # temp outside [-20, 55] C
    invalid_humidity_records: int = 0  # humidity outside [0, 100]%
    invalid_windspeed_records: int = 0  # windspeed < 0 or > 120 km/h
    negative_counts: int = 0
    anomalies_detected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check if DataFrame adheres to the required column specification.
    """
    present_cols = set(df.columns)
    required_cols = set(EXPECTED_COLUMNS)

    missing = sorted(list(required_cols - present_cols))
    if missing:
        return False, [f"Missing required columns: {missing}"]
    return True, []


def run_data_quality_audit(df: pd.DataFrame) -> ValidationResult:
    """
    Perform a comprehensive data quality audit on the raw dataset.

    Checks:
    - Missing columns
    - Missing / null values
    - Duplicate timestamps
    - Weather parameter physical bounds
    - Target count sanity
    - Target identity check: casual + registered == count

    Returns:
        ValidationResult dataclass with exhaustive diagnostic counters.
    """
    result = ValidationResult(total_records=len(df))

    # 1. Schema check
    present_cols = set(df.columns)
    required_cols = set(EXPECTED_COLUMNS)
    result.missing_columns = sorted(list(required_cols - present_cols))
    result.unexpected_columns = sorted(list(present_cols - required_cols))

    if result.missing_columns:
        result.is_valid = False
        result.anomalies_detected.append(f"Missing columns: {result.missing_columns}")

    # 2. Missing values
    missing = df.isnull().sum().to_dict()
    result.missing_values = {k: int(v) for k, v in missing.items() if v > 0}
    if result.missing_values:
        result.is_valid = False
        result.anomalies_detected.append(f"Null values present: {result.missing_values}")

    # 3. Duplicate timestamps
    if "datetime" in df.columns:
        duplicates = int(df.duplicated(subset=["datetime"]).sum())
        result.duplicate_rows = duplicates
        if duplicates > 0:
            result.is_valid = False
            result.anomalies_detected.append(f"Found {duplicates} duplicate datetime records.")

    # 4. Target non-negativity and identity validation
    if "count" in df.columns:
        negative_count = int((df["count"] < 0).sum())
        result.negative_counts = negative_count
        if negative_count > 0:
            result.is_valid = False
            result.anomalies_detected.append(f"Detected {negative_count} negative rental counts.")

    if all(col in df.columns for col in ["casual", "registered", "count"]):
        mismatches = int(((df["casual"] + df["registered"]) != df["count"]).sum())
        result.leakage_mismatches = mismatches
        if mismatches > 0:
            result.anomalies_detected.append(
                f"Detected {mismatches} records where casual + registered != count."
            )

    # 5. Meteorological sanity bounds
    if "temp" in df.columns:
        invalid_temp = int(((df["temp"] < -20.0) | (df["temp"] > 55.0)).sum())
        result.invalid_temperature_records = invalid_temp
        if invalid_temp > 0:
            result.anomalies_detected.append(f"Detected {invalid_temp} unrealistic temperatures.")

    if "humidity" in df.columns:
        invalid_hum = int(((df["humidity"] < 0.0) | (df["humidity"] > 100.0)).sum())
        result.invalid_humidity_records = invalid_hum
        if invalid_hum > 0:
            result.anomalies_detected.append(f"Detected {invalid_hum} unrealistic humidity levels.")

    if "windspeed" in df.columns:
        invalid_wind = int(((df["windspeed"] < 0.0) | (df["windspeed"] > 120.0)).sum())
        result.invalid_windspeed_records = invalid_wind
        if invalid_wind > 0:
            result.anomalies_detected.append(f"Detected {invalid_wind} unrealistic windspeeds.")

    return result


def generate_quality_report(
    df: pd.DataFrame,
    output_dir: Path = DATA_QUALITY_DIR,
) -> Path:
    """
    Generate and save a formal data quality report in Markdown and JSON formats.

    Args:
        df: Dataset to evaluate.
        output_dir: Directory where reports will be saved.

    Returns:
        Path to generated Markdown report.
    """
    ensure_directories()
    result = run_data_quality_audit(df)

    json_path = output_dir / "data_quality_report.json"
    md_path = output_dir / "data_quality_report.md"

    # Save JSON summary
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    # Save formatted Markdown summary
    status_str = "PASSED" if result.is_valid else "FAILED"
    markdown_content = f"""# BSD-Analytics Data Quality Audit Report

**Status:** {status_str}
**Total Records Analyzed:** {result.total_records:,}

## 1. Schema Validation
- Missing Required Columns: {len(result.missing_columns)} ({result.missing_columns})
- Unexpected Extra Columns: {len(result.unexpected_columns)} ({result.unexpected_columns})

## 2. Completeness & Integrity
- Missing Values (Nulls): {len(result.missing_values)}
- Duplicate Timestamps: {result.duplicate_rows}
- Target (count) Negative Records: {result.negative_counts}
- Target Identity Mismatches (casual + registered != count): {result.leakage_mismatches}

## 3. Physical Value Sanity Checks
- Out-of-bounds Temperatures ([-20C, 55C]): {result.invalid_temperature_records}
- Out-of-bounds Humidity ([0%, 100%]): {result.invalid_humidity_records}
- Out-of-bounds Windspeed ([0, 120 km/h]): {result.invalid_windspeed_records}

## 4. Anomaly Summary
{chr(10).join(f"- {a}" for a in result.anomalies_detected) if result.anomalies_detected else "- No critical anomalies detected. Dataset meets production quality standards."}
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    logger.info("Saved Data Quality Report to %s and %s", md_path, json_path)
    return md_path
