"""
Feature engineering module for BSD-Analytics.
Extracts calendar, temporal, cyclical, meteorological, and interaction features
for bike sharing demand forecasting.
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

    # Rush-hour commute indicator: on working days, hours 7-9 AM and 5-7 PM experience peak demand
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


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct non-linear meteorological indices, perceptual comfort differences,
    and adverse weather flags.

    Args:
        df: DataFrame containing meteorological features.

    Returns:
        DataFrame augmented with weather interaction metrics.
    """
    df = df.copy()

    # 1. Perceptual Temperature Delta
    if "atemp" in df.columns and "temp" in df.columns:
        df["temp_feels_diff"] = df["atemp"] - df["temp"]
    else:
        df["temp_feels_diff"] = 0.0

    # 2. Temperature to Humidity Ratio (perceived thermal comfort proxy)
    if "temp" in df.columns and "humidity" in df.columns:
        df["temp_humidity_ratio"] = df["temp"] / (df["humidity"] + 1.0)
    else:
        df["temp_humidity_ratio"] = 0.0

    # 3. Wind Chill Approximation (Index for cold/windy ridership impedance)
    if "temp" in df.columns and "windspeed" in df.columns:
        safe_wind = np.maximum(df["windspeed"].astype(float), 0.1)
        df["wind_chill_approx"] = (
            13.12
            + 0.6215 * df["temp"]
            - 11.37 * (safe_wind**0.16)
            + 0.3965 * df["temp"] * (safe_wind**0.16)
        )
    else:
        df["wind_chill_approx"] = 0.0

    # 4. Adverse Weather Indicator (weather category 3 = Light Snow/Rain, 4 = Heavy Storm)
    if "weather" in df.columns:
        df["bad_weather_flag"] = (df["weather"] >= 3).astype(int)
    else:
        df["bad_weather_flag"] = 0

    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that automates comprehensive feature engineering
    across both offline model training and real-time inference pipelines.
    """

    def __init__(
        self,
        include_cyclical: bool = True,
        include_weather_interactions: bool = True,
    ):
        self.include_cyclical = include_cyclical
        self.include_weather_interactions = include_weather_interactions
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> FeatureEngineer:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = add_temporal_features(df)
        if self.include_cyclical:
            df = add_cyclical_features(df)
        if self.include_weather_interactions:
            df = add_weather_features(df)
        return df
