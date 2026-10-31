"""
Model training and chronological data splitting module for BSD-Analytics.
Enforces strict time-aware splits to prevent temporal look-ahead leakage,
and provides training pipelines for baseline, linear, and ensemble models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bsd_analytics.config import (
    ALL_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    RANDOM_SEED,
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


@dataclass
class ModelResult:
    """Encapsulates a fitted estimator, its identity, and predictions."""

    model_name: str
    estimator: Any
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    feature_names: list[str]
    hyperparameters: dict[str, Any]


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


def train_baseline_model(split: SplitData, strategy: str = "mean") -> ModelResult:
    """
    Train a DummyRegressor baseline model.
    Serves as the empirical minimum threshold that every ML model must beat.
    """
    logger.info("Training DummyRegressor baseline (strategy=%s)...", strategy)
    dummy = DummyRegressor(strategy=strategy)
    dummy.fit(split.X_train, split.y_train)

    y_pred = np.maximum(0.0, dummy.predict(split.X_test))

    return ModelResult(
        model_name="Baseline (DummyRegressor)",
        estimator=dummy,
        X_test=split.X_test,
        y_test=split.y_test,
        y_pred=y_pred,
        feature_names=split.feature_names,
        hyperparameters={"strategy": strategy},
    )


def train_linear_regression(
    split: SplitData,
    regularized: bool = True,
    alpha: float = 1.0,
) -> ModelResult:
    """
    Train a Linear Regression or Ridge pipeline with StandardScaler
    and log1p target transformation to enforce non-negative predictions.
    """
    name = f"Ridge Regression (alpha={alpha})" if regularized else "Linear Regression (OLS)"
    logger.info("Training %s...", name)

    base_regressor = (
        Ridge(alpha=alpha, random_state=RANDOM_SEED)
        if regularized
        else LinearRegression()
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", base_regressor),
    ])

    # Target transform: log1p(y) on fit, expm1 on predict, ensuring outputs >= 0
    ttr = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )

    ttr.fit(split.X_train, split.y_train)
    raw_pred = ttr.predict(split.X_test)
    y_pred = np.maximum(0.0, raw_pred)

    return ModelResult(
        model_name=name,
        estimator=ttr,
        X_test=split.X_test,
        y_test=split.y_test,
        y_pred=y_pred,
        feature_names=split.feature_names,
        hyperparameters={"regularized": regularized, "alpha": alpha, "target_transform": "log1p"},
    )
