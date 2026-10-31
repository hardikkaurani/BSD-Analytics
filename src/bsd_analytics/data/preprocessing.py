"""
Data preprocessing and cleaning module for BSD-Analytics.
Handles timestamp normalization, outlier analysis, categorical encoding,
and strict target leakage elimination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from bsd_analytics.config import (
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    ensure_directories,
)

logger = logging.getLogger(__name__)


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for dataset cleaning and normalization.
    Ensures identical transformations between training and inference phases.
    """

    def __init__(self, drop_leakage: bool = True):
        self.drop_leakage = drop_leakage
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> DataPreprocessor:
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning and data integrity rules."""
        df = X.copy()

        # 1. Parse timestamps
        if "datetime" in df.columns and not np.issubdtype(df["datetime"].dtype, np.datetime64):
            df["datetime"] = pd.to_datetime(df["datetime"])

        # 2. Strict Target Leakage Elimination
        # casual and registered sum up to count in training data.
        # They must NEVER be used as prediction features.
        if self.drop_leakage:
            for col in LEAKAGE_COLUMNS:
                if col in df.columns:
                    df = df.drop(columns=[col])

        # 3. Categorical types normalization
        categorical_cols = ["season", "holiday", "workingday", "weather"]
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # 4. Numeric types normalization
        numeric_cols = ["temp", "atemp", "humidity", "windspeed"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df


def clean_and_preprocess_data(
    df: pd.DataFrame,
    save_processed: bool = True,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """
    Standardize, clean, and validate full dataset, persisting processed artifacts.

    Args:
        df: Raw DataFrame.
        save_processed: Whether to persist parquet & csv copies.
        drop_leakage: Whether to strip casual & registered features.

    Returns:
        Cleaned DataFrame.
    """
    ensure_directories()
    preprocessor = DataPreprocessor(drop_leakage=drop_leakage)
    cleaned_df = preprocessor.fit_transform(df)

    if save_processed:
        parquet_path = PROCESSED_DATA_DIR / "bike_sharing_cleaned.parquet"
        csv_path = PROCESSED_DATA_DIR / "bike_sharing_cleaned.csv"

        cleaned_df.to_parquet(parquet_path, index=False)
        cleaned_df.to_csv(csv_path, index=False)
        logger.info("Saved processed dataset to %s and %s", parquet_path, csv_path)

    return cleaned_df


def inspect_outliers(df: pd.DataFrame, column: str = "count") -> dict[str, Any]:
    """
    Statistical outlier detection using the Interquartile Range (IQR) technique.
    Demand peaks are documented rather than removed, as they reflect legitimate peak ridership.
    """
    if column not in df.columns:
        return {"error": f"Column {column} not in dataframe"}

    values = df[column].dropna()
    q25, q75 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q75 - q25
    lower_bound = max(0.0, q25 - 1.5 * iqr)
    upper_bound = q75 + 1.5 * iqr

    outliers = values[(values < lower_bound) | (values > upper_bound)]
    outlier_percentage = float(len(outliers) / len(values) * 100.0)

    return {
        "column": column,
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(len(outliers)),
        "outlier_percentage": round(outlier_percentage, 2),
        "max_value": float(values.max()),
        "min_value": float(values.min()),
    }
