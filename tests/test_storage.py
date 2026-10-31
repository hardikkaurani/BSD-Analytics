"""Tests for SQLite database persistence and experiment logging."""

from bsd_analytics.storage.db import DatabaseManager


def test_database_manager_in_memory():
    db = DatabaseManager(uri="sqlite:///:memory:")
    exp_id = db.log_experiment_run(
        model_name="Test Ridge",
        metrics={"mae": 110.0, "rmse": 180.0, "r2": 0.35, "rmsle": 0.65, "mean_bias": 1.2},
        hyperparameters={"alpha": 1.0},
    )
    assert exp_id == 1

    history = db.get_experiment_history()
    assert len(history) == 1
    assert history.iloc[0]["model_name"] == "Test Ridge"

    pred_id = db.log_inference_prediction(
        predicted_count=245,
        model_name="Test Ridge",
        input_summary={"hour": 8, "temp": 20.0},
    )
    assert pred_id == 1
