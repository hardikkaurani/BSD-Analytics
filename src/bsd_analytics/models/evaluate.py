"""
Model evaluation and diagnostic error analysis module for BSD-Analytics.
Computes standard regression metrics (MAE, RMSE, R2, RMSLE) alongside
deep residual diagnostics and domain-specific subgroup breakdowns.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bsd_analytics.models.train import ModelResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Consolidated performance metrics for a regression model."""

    model_name: str
    mae: float
    rmse: float
    r2: float
    rmsle: float
    mean_bias: float
    std_residual: float
    underprediction_pct: float
    overprediction_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> EvaluationMetrics:
    """
    Compute full suite of regression and residual diagnostic metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.maximum(0.0, np.asarray(y_pred, dtype=float))

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))

    # Kaggle official evaluation metric: Root Mean Squared Logarithmic Error
    log_true = np.log1p(y_true_arr)
    log_pred = np.log1p(y_pred_arr)
    rmsle = float(np.sqrt(mean_squared_error(log_true, log_pred)))

    # Residuals: actual - predicted
    residuals = y_true_arr - y_pred_arr
    mean_bias = float(np.mean(residuals))
    std_residual = float(np.std(residuals))

    # Over/under-prediction
    under = float(np.mean(y_pred_arr < y_true_arr) * 100.0)
    over = float(np.mean(y_pred_arr > y_true_arr) * 100.0)

    return EvaluationMetrics(
        model_name=model_name,
        mae=round(mae, 3),
        rmse=round(rmse, 3),
        r2=round(r2, 4),
        rmsle=round(rmsle, 4),
        mean_bias=round(mean_bias, 3),
        std_residual=round(std_residual, 3),
        underprediction_pct=round(under, 2),
        overprediction_pct=round(over, 2),
    )


def evaluate_model(result: ModelResult) -> EvaluationMetrics:
    """Convenience wrapper to evaluate a ModelResult."""
    return calculate_metrics(result.y_test.values, result.y_pred, result.model_name)


def compare_models(results: list[ModelResult]) -> pd.DataFrame:
    """
    Compile a benchmark leaderboard comparing all candidate models.

    Args:
        results: List of fitted ModelResult objects.

    Returns:
        DataFrame sorted by ascending MAE / descending R2.
    """
    rows = [evaluate_model(res).to_dict() for res in results]
    leaderboard = pd.DataFrame(rows)
    return leaderboard.sort_values(by="mae").reset_index(drop=True)


def subgroup_error_analysis(result: ModelResult) -> dict[str, Any]:
    """
    Examine model performance across specific operational segments:
    - Peak commute hours vs off-peak hours
    - Seasons (Spring, Summer, Fall, Winter)
    - Weather conditions
    """
    df_eval = result.X_test.copy()
    df_eval["actual"] = result.y_test.values
    df_eval["predicted"] = result.y_pred
    df_eval["absolute_error"] = np.abs(df_eval["actual"] - df_eval["predicted"])

    # 1. Peak Commute Performance
    if "peak_commute_hour" in df_eval.columns:
        commute_mae = df_eval.groupby("peak_commute_hour")["absolute_error"].mean().to_dict()
    else:
        commute_mae = {}

    # 2. Seasonal Performance
    if "season" in df_eval.columns:
        season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
        seasonal_mae = (
            df_eval.groupby("season")["absolute_error"]
            .mean()
            .rename(index=season_map)
            .to_dict()
        )
    else:
        seasonal_mae = {}

    # 3. Weather Condition Performance
    if "weather" in df_eval.columns:
        weather_mae = df_eval.groupby("weather")["absolute_error"].mean().to_dict()
    else:
        weather_mae = {}

    return {
        "model_name": result.model_name,
        "overall_mae": round(float(df_eval["absolute_error"].mean()), 2),
        "commute_hour_mae": commute_mae,
        "seasonal_mae": seasonal_mae,
        "weather_mae": weather_mae,
    }
