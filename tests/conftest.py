"""Shared test fixtures for BSD-Analytics test suite."""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Generate a realistic synthetic sample of hourly bike sharing data."""
    dates = pd.date_range("2011-01-01 00:00:00", periods=100, freq="h")
    np.random.seed(42)

    temp = np.random.uniform(5.0, 32.0, size=100)
    atemp = temp + np.random.uniform(0.0, 3.0, size=100)
    humidity = np.random.uniform(30.0, 85.0, size=100)
    windspeed = np.random.uniform(0.0, 25.0, size=100)
    casual = np.random.randint(1, 50, size=100)
    registered = np.random.randint(10, 200, size=100)
    total_count = casual + registered

    return pd.DataFrame({
        "datetime": dates,
        "season": np.random.choice([1, 2, 3, 4], size=100),
        "holiday": np.zeros(100, dtype=int),
        "workingday": np.random.choice([0, 1], size=100),
        "weather": np.random.choice([1, 2, 3], size=100),
        "temp": temp,
        "atemp": atemp,
        "humidity": humidity,
        "windspeed": windspeed,
        "casual": casual,
        "registered": registered,
        "count": total_count,
    })
