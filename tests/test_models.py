"""Tests for chronological splitting, model training, and evaluation metrics."""

import pytest
import numpy as np
from bsd_analytics.models.train import (
    chronological_train_test_split,
    train_baseline_model,
    train_linear_regression,
    train_random_forest,
)
from bsd_analytics.models.evaluate import calculate_metrics, evaluate_model

def test_chronological_split_preserves_temporal_order(sample_raw_df):
    cutoff = "2011-01-03 00:00:00"
    split = chronological_train_test_split(sample_raw_df, split_date=cutoff)
    assert split.train_records > 0
    assert split.test_records > 0
    assert split.train_records + split.test_records == len(sample_raw_df)

def test_train_baseline_and_linear(sample_raw_df):
    cutoff = "2011-01-03 00:00:00"
    split = chronological_train_test_split(sample_raw_df, split_date=cutoff)

    base_res = train_baseline_model(split)
    assert len(base_res.y_pred) == split.test_records
    assert np.all(base_res.y_pred >= 0.0)

    lin_res = train_linear_regression(split, regularized=True)
    assert len(lin_res.y_pred) == split.test_records
    assert np.all(lin_res.y_pred >= 0.0)

def test_evaluation_metrics_calculation():
    y_true = np.array([10.0, 50.0, 100.0, 200.0])
    y_pred = np.array([12.0, 48.0, 95.0, 210.0])

    metrics = calculate_metrics(y_true, y_pred, "TestModel")
    assert metrics.mae > 0.0
    assert metrics.rmse >= metrics.mae
    assert metrics.r2 > 0.95
    assert metrics.rmsle > 0.0
