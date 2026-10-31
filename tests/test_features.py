"""Tests for temporal, cyclical, and interaction feature engineering."""

import pytest
import numpy as np
from bsd_analytics.features.engineering import (
    FeatureEngineer,
    add_temporal_features,
    add_cyclical_features,
    add_weather_features,
)

def test_add_temporal_features(sample_raw_df):
    df_temp = add_temporal_features(sample_raw_df)
    assert "hour" in df_temp.columns
    assert "day" in df_temp.columns
    assert "weekday" in df_temp.columns
    assert "month" in df_temp.columns
    assert "year" in df_temp.columns
    assert "is_weekend" in df_temp.columns
    assert "peak_commute_hour" in df_temp.columns
    assert df_temp["hour"].min() >= 0
    assert df_temp["hour"].max() <= 23

def test_add_cyclical_features(sample_raw_df):
    df_temp = add_temporal_features(sample_raw_df)
    df_cyc = add_cyclical_features(df_temp)
    assert "hour_sin" in df_cyc.columns
    assert "hour_cos" in df_cyc.columns
    assert np.all(df_cyc["hour_sin"] >= -1.0) and np.all(df_cyc["hour_sin"] <= 1.0)
    assert np.all(df_cyc["hour_cos"] >= -1.0) and np.all(df_cyc["hour_cos"] <= 1.0)

def test_add_weather_features(sample_raw_df):
    df_w = add_weather_features(sample_raw_df)
    assert "temp_feels_diff" in df_w.columns
    assert "temp_humidity_ratio" in df_w.columns
    assert "wind_chill_approx" in df_w.columns
    assert "bad_weather_flag" in df_w.columns

def test_feature_engineer_transformer(sample_raw_df):
    fe = FeatureEngineer()
    transformed = fe.fit_transform(sample_raw_df)
    assert transformed.shape[1] > sample_raw_df.shape[1]
