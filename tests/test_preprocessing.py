"""Tests for data preprocessing and leakage prevention."""

import pytest
from bsd_analytics.data.preprocessing import clean_and_preprocess_data, inspect_outliers

def test_preprocessing_drops_target_leakage(sample_raw_df):
    cleaned = clean_and_preprocess_data(sample_raw_df, save_processed=False, drop_leakage=True)
    assert "casual" not in cleaned.columns
    assert "registered" not in cleaned.columns
    assert "count" in cleaned.columns

def test_inspect_outliers(sample_raw_df):
    outliers = inspect_outliers(sample_raw_df, "count")
    assert "q25" in outliers
    assert "q75" in outliers
    assert "iqr" in outliers
    assert "outlier_count" in outliers
