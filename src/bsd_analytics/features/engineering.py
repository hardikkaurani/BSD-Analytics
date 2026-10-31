"""
Feature engineering module for BSD-Analytics.
Extracts calendar, temporal, cyclical, and interaction features for demand forecasting.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar, hourly, and workday commute features from datetime.

    Args:
        df: Input DataFrame containing 'datetime' column.

    Returns:
        DataFrame augmented with temporal attributes.
    """
    df = df.copy()
    if "datetime" not in df.columns:
        raise ValueError("Cannot extract temporal features: 'datetime' column missing.")

    dt = pd.to_datetime(df["datetime"])

    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["weekday"] = dt.dt.weekday
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    # Rush-hour commute indicator: on working days, hours 7-9 AM and 5-7 PM experience surge
    if "workingday" in df.columns:
        df["peak_commute_hour"] = (
            (df["workingday"] == 1) & (df["hour"].isin([7, 8, 9, 17, 18, 19]))
        ).astype(int)
    else:
        df["peak_commute_hour"] = (
            (df["is_weekend"] == 0) & (df["hour"].isin([7, 8, 9, 17, 18, 19]))
        ).astype(int)

    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode periodic features (hour of day, month of year) into sine and cosine components.
    Ensures that hour 23 and hour 0 are mathematically continuous.

    Args:
        df: DataFrame containing 'hour' and 'month'.

    Returns:
        DataFrame augmented with cyclical sine/cosine pairs.
    """
    df = df.copy()

    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)

    if "month" in df.columns:
        df["month_sin"] = np.sin(2.0 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2.0 * np.pi * df["month"] / 12.0)

    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that automates feature extraction
    across training and prediction pipelines.
    """

    def __init__(self, include_cyclical: bool = True):
        self.include_cyclical = include_cyclical
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = add_temporal_features(df)
        if self.include_cyclical:
            df = add_cyclical_features(df)
        return df
