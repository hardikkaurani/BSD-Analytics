"""Tests for real-time prediction and inference engine."""

import pytest
from bsd_analytics.models.predict import predict_single_observation

def test_predict_single_observation():
    out = predict_single_observation(
        datetime_str="2026-06-15 08:00:00",
        season=2,
        holiday=0,
        workingday=1,
        weather=1,
        temp=22.0,
        atemp=24.0,
        humidity=60.0,
        windspeed=10.0,
    )
    assert out.predicted_count >= 0
    assert isinstance(out.predicted_count, int)
    assert len(out.limitations) > 0

def test_prediction_rejects_out_of_bounds_temperature():
    with pytest.raises(ValueError, match="Temperature must be within"):
        predict_single_observation(
            datetime_str="2026-06-15 08:00:00",
            season=2,
            holiday=0,
            workingday=1,
            weather=1,
            temp=999.0,  # Invalid
            atemp=24.0,
            humidity=60.0,
            windspeed=10.0,
        )
