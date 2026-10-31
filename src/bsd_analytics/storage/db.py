"""
Database manager and session lifecycle handler for BSD-Analytics SQLite warehouse.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from bsd_analytics.config import DATABASE_URI, ensure_directories
from bsd_analytics.storage.schema import Base, ModelExperiment, PredictionLog

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Encapsulates SQLAlchemy database engine, table creation, and queries."""

    def __init__(self, uri: str = DATABASE_URI):
        ensure_directories()
        self.uri = uri
        self.engine = create_engine(self.uri, echo=False)
        self.SessionFactory = sessionmaker(bind=self.engine)
        self.create_tables()

    def create_tables(self) -> None:
        """Create all declared tables if they do not exist."""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        return self.SessionFactory()

    def log_experiment_run(
        self,
        model_name: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any] | None = None,
    ) -> int:
        """Persist a single model training evaluation record."""
        with self.get_session() as session:
            record = ModelExperiment(
                model_name=model_name,
                mae=float(metrics.get("mae", 0.0)),
                rmse=float(metrics.get("rmse", 0.0)),
                r2=float(metrics.get("r2", 0.0)),
                rmsle=float(metrics.get("rmsle", 0.0)),
                mean_bias=float(metrics.get("mean_bias", 0.0)),
                hyperparameters_json=json.dumps(hyperparameters or {}),
            )
            session.add(record)
            session.commit()
            return record.id

    def log_inference_prediction(
        self,
        predicted_count: int,
        model_name: str,
        input_summary: dict[str, Any],
    ) -> int:
        """Record real-time forecast telemetry."""
        with self.get_session() as session:
            record = PredictionLog(
                predicted_count=predicted_count,
                model_name=model_name,
                input_summary_json=json.dumps(input_summary),
            )
            session.add(record)
            session.commit()
            return record.id

    def get_experiment_history(self) -> pd.DataFrame:
        """Retrieve historical model leaderboard as a pandas DataFrame."""
        with self.get_session() as session:
            stmt = select(ModelExperiment).order_by(ModelExperiment.id.desc())
            records = session.scalars(stmt).all()
            if not records:
                return pd.DataFrame()
            return pd.DataFrame(
                [
                    {
                        "id": r.id,
                        "model_name": r.model_name,
                        "timestamp": r.timestamp,
                        "mae": r.mae,
                        "rmse": r.rmse,
                        "r2": r.r2,
                        "rmsle": r.rmsle,
                        "mean_bias": r.mean_bias,
                    }
                    for r in records
                ]
            )


# Global default database instance
_GLOBAL_DB_MANAGER: DatabaseManager | None = None


def get_db_manager(uri: str = DATABASE_URI) -> DatabaseManager:
    global _GLOBAL_DB_MANAGER
    if _GLOBAL_DB_MANAGER is None or _GLOBAL_DB_MANAGER.uri != uri:
        _GLOBAL_DB_MANAGER = DatabaseManager(uri=uri)
    return _GLOBAL_DB_MANAGER


def init_db(uri: str = DATABASE_URI) -> None:
    db = get_db_manager(uri)
    db.create_tables()


def log_experiment(
    model_name: str,
    metrics: dict[str, float],
    hyperparameters: dict[str, Any] | None = None,
) -> int:
    return get_db_manager().log_experiment_run(model_name, metrics, hyperparameters)


def log_prediction(
    predicted_count: int,
    model_name: str,
    input_summary: dict[str, Any],
) -> int:
    return get_db_manager().log_inference_prediction(predicted_count, model_name, input_summary)
