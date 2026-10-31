"""
Real-time prediction and inference engine for BSD-Analytics.
Validates input schemas, prevents negative outputs, applies feature transformations,
and tracks model provenance.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from bsd_analytics.features.engineering import FeatureEngineer
from bsd_analytics.models.registry import get_production_model

logger = logging.getLogger(__name__)


@dataclass
class PredictionOutput:
    """Standardized output container for demand forecasts."""

    predicted_count: int
    predicted_count_float: float
    model_name: str
    input_features_used: list[str]
    input_summary: dict[str, Any]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PredictionEngine:
    """
    Production inference engine. Loads saved model and transforms raw inputs.
    """

    def __init__(self):
        self.estimator, self.feature_names, self.metadata = get_production_model()
        self.feature_engineer = FeatureEngineer()

    def validate_inputs(self, df: pd.DataFrame) -> None:
        """Enforce physical bounds on inference inputs."""
        if "temp" in df.columns:
            invalid_temp = (df["temp"] < -30.0) | (df["temp"] > 60.0)
            if invalid_temp.any():
                raise ValueError("Temperature must be within [-30, 60] Celsius.")

        if "humidity" in df.columns:
            invalid_hum = (df["humidity"] < 0.0) | (df["humidity"] > 100.0)
            if invalid_hum.any():
                raise ValueError("Humidity must be within [0, 100] percent.")

        if "windspeed" in df.columns:
            invalid_wind = (df["windspeed"] < 0.0) | (df["windspeed"] > 150.0)
            if invalid_wind.any():
                raise ValueError("Windspeed must be within [0, 150] km/h.")

    def predict(self, input_df: pd.DataFrame) -> list[PredictionOutput]:
        """
        Generate bike sharing demand forecasts for one or multiple observations.

        Args:
            input_df: Raw input DataFrame with datetime/calendar and meteorological features.

        Returns:
            List of PredictionOutput records.
        """
        self.validate_inputs(input_df)

        # 1. Transform raw inputs through identical feature engineering pipeline
        X_transformed = self.feature_engineer.transform(input_df)

        # 2. Select exact required features in identical order
        missing = [f for f in self.feature_names if f not in X_transformed.columns]
        if missing:
            raise ValueError(f"Missing required model features after engineering: {missing}")

        X_model = X_transformed[self.feature_names]

        # 3. Generate predictions & enforce strict non-negativity
        raw_preds = self.estimator.predict(X_model)
        y_preds = np.maximum(0.0, np.round(raw_preds, 2))

        outputs: list[PredictionOutput] = []
        for idx, pred in enumerate(y_preds):
            pred_int = int(np.round(pred))
            sample_summary = input_df.iloc[idx].to_dict()
            # Clean non-serializable objects
            clean_summary = {
                k: (str(v) if isinstance(v, (pd.Timestamp, np.datetime64)) else v)
                for k, v in sample_summary.items()
                if k not in ["casual", "registered"]
            }

            outputs.append(
                PredictionOutput(
                    predicted_count=pred_int,
                    predicted_count_float=float(pred),
                    model_name=self.metadata.get("model_name", "Production Model"),
                    input_features_used=self.feature_names,
                    input_summary=clean_summary,
                    limitations=[
                        "Forecast represents estimated aggregate hourly demand across all stations.",
                        "Extreme weather events (e.g. blizzards, hurricanes) may cause unmodeled ridership drops.",
                        "Predictions do not account for station-level dock shortages or localized maintenance.",
                    ],
                )
            )

        return outputs


def predict_single_observation(
    datetime_str: str,
    season: int,
    holiday: int,
    workingday: int,
    weather: int,
    temp: float,
    atemp: float,
    humidity: float,
    windspeed: float,
) -> PredictionOutput:
    """
    Convenience function for single-sample demand forecasting (e.g., Streamlit forms).
    """
    df_single = pd.DataFrame(
        [
            {
                "datetime": pd.to_datetime(datetime_str),
                "season": int(season),
                "holiday": int(holiday),
                "workingday": int(workingday),
                "weather": int(weather),
                "temp": float(temp),
                "atemp": float(atemp),
                "humidity": float(humidity),
                "windspeed": float(windspeed),
            }
        ]
    )

    engine = PredictionEngine()
    results = engine.predict(df_single)
    return results[0]
