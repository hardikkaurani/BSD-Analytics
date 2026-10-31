"""
Model interpretability and feature importance analysis module for BSD-Analytics.
Extracts native tree importances, permutation importances, and business explanations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from bsd_analytics.config import RANDOM_SEED
from bsd_analytics.models.train import ModelResult

logger = logging.getLogger(__name__)


def extract_native_importance(estimator: Any, feature_names: list[str]) -> pd.DataFrame:
    """
    Extract tree-based feature importances from a raw or transformed estimator.
    """
    underlying = estimator
    # Handle TransformedTargetRegressor wrapper
    if hasattr(estimator, "regressor_"):
        underlying = estimator.regressor_

    # Handle Pipeline wrapper
    if hasattr(underlying, "named_steps") and "regressor" in underlying.named_steps:
        underlying = underlying.named_steps["regressor"]

    if hasattr(underlying, "feature_importances_"):
        importances = underlying.feature_importances_
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        df_imp["relative_pct"] = (df_imp["importance"] / df_imp["importance"].sum()) * 100.0
        return df_imp

    # Linear regression coefficients fallback
    if hasattr(underlying, "coef_"):
        coefs = np.abs(underlying.coef_)
        df_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": coefs,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        df_imp["relative_pct"] = (df_imp["importance"] / df_imp["importance"].sum()) * 100.0
        return df_imp

    return pd.DataFrame(columns=["feature", "importance", "relative_pct"])


def compute_permutation_importance(
    estimator: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    n_repeats: int = 5,
) -> pd.DataFrame:
    """
    Compute holdout permutation importance.
    Evaluates the decrease in model score when a feature's values are randomly shuffled.
    """
    logger.info("Computing permutation importance over %d holdout records...", len(X_val))
    r = permutation_importance(
        estimator,
        X_val,
        y_val,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

    return df_perm


def get_interpretability_summary(result: ModelResult) -> dict[str, Any]:
    """
    Generate an executive interpretability narrative linking statistical features
    to real-world bike-share rider dynamics.
    """
    native_df = extract_native_importance(result.estimator, result.feature_names)
    top_features = native_df.head(5)["feature"].tolist() if not native_df.empty else []

    explanations = {
        "hour": "Hour of the day dictates daily commuter and leisure circadian cycles.",
        "temp": "Air temperature is the primary physiological driver of outdoor cycling demand.",
        "peak_commute_hour": "Captures surging rush-hour ridership during 7-9 AM and 5-7 PM.",
        "humidity": "High humidity combined with heat drastically degrades riding comfort.",
        "workingday": "Differentiates utility commuter trips from recreational weekend trips.",
        "temp_feels_diff": "Apparent temperature index accounts for humidity and wind cooling.",
        "year": "Reflects macro ridership growth and fleet expansion between 2011 and 2012.",
    }

    insights = [explanations.get(f, f"Significant demand predictor ({f}).") for f in top_features]

    return {
        "model_name": result.model_name,
        "top_features": top_features,
        "feature_importances": native_df.to_dict(orient="records"),
        "business_insights": insights,
    }
