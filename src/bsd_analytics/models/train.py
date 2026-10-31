"""
Model training and chronological data splitting module for BSD-Analytics.
Enforces strict time-aware splits to prevent temporal look-ahead leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from bsd_analytics.config import (
    ALL_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)
from bsd_analytics.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_DATE: Final[str] = "2012-07-01 00:00:00"


@dataclass
class SplitData:
    """Chronologically partitioned training and evaluation matrices."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    split_timestamp: str
    train_records: int
    test_records: int


def get_feature_target_matrices(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Apply feature engineering and extract sanitized feature and target matrices,
    strictly verifying that leakage columns are excluded.
    """
    df_copy = df.copy()

    # Apply Feature Engineering
    fe = FeatureEngineer()
    df_transformed = fe.fit_transform(df_copy)

    # Strictly verify no leakage columns exist in feature space
    for leak in LEAKAGE_COLUMNS:
        if leak in df_transformed.columns and leak != target:
            df_transformed = df_transformed.drop(columns=[leak])

    if features is None:
        features = [col for col in ALL_FEATURE_COLUMNS if col in df_transformed.columns]

    X = df_transformed[features].copy()
    y = df_transformed[target].copy()

    return X, y, features


def chronological_train_test_split(
    df: pd.DataFrame,
    split_date: str = DEFAULT_SPLIT_DATE,
    features: list[str] | None = None,
    target: str = TARGET_COLUMN,
) -> SplitData:
    """
    Split the dataset chronologically based on a cutoff timestamp.
    Ensures observations in the test set strictly occur after the training set.

    Args:
        df: Input DataFrame with 'datetime'.
        split_date: Timestamp cutoff separating train and holdout.
        features: Optional subset of feature column names.
        target: Target column name.

    Returns:
        SplitData dataclass holding train and holdout sets.
    """
    df_sorted = df.sort_values("datetime").reset_index(drop=True)
    cutoff = pd.to_datetime(split_date)

    train_mask = df_sorted["datetime"] < cutoff
    test_mask = df_sorted["datetime"] >= cutoff

    df_train = df_sorted[train_mask].copy()
    df_test = df_sorted[test_mask].copy()

    X_train, y_train, feature_names = get_feature_target_matrices(df_train, features, target)
    X_test, y_test, _ = get_feature_target_matrices(df_test, features, target)

    logger.info(
        "Chronological split completed at %s: Train=%d, Test=%d, Features=%d",
        split_date,
        len(X_train),
        len(X_test),
        len(feature_names),
    )

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        split_timestamp=split_date,
        train_records=len(X_train),
        test_records=len(X_test),
    )
