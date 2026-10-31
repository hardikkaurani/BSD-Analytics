"""Tests for schema validation and data quality auditing."""

import pytest
import pandas as pd
from bsd_analytics.data.validation import validate_schema, run_data_quality_audit

def test_validate_schema_valid(sample_raw_df):
    is_valid, missing = validate_schema(sample_raw_df)
    assert is_valid is True
    assert len(missing) == 0

def test_validate_schema_missing_column(sample_raw_df):
    corrupted = sample_raw_df.drop(columns=["temp"])
    is_valid, missing = validate_schema(corrupted)
    assert is_valid is False
    assert "Missing required columns" in missing[0]

def test_data_quality_audit_clean(sample_raw_df):
    result = run_data_quality_audit(sample_raw_df)
    assert result.is_valid is True
    assert result.total_records == 100
    assert result.duplicate_rows == 0
    assert result.leakage_mismatches == 0
    assert result.negative_counts == 0

def test_data_quality_audit_detects_negatives_and_bounds(sample_raw_df):
    corrupted = sample_raw_df.copy()
    corrupted.loc[0, "count"] = -5
    corrupted.loc[1, "temp"] = 99.0
    corrupted.loc[2, "humidity"] = 150.0

    result = run_data_quality_audit(corrupted)
    assert result.is_valid is False
    assert result.negative_counts == 1
    assert result.invalid_temperature_records == 1
    assert result.invalid_humidity_records == 1
