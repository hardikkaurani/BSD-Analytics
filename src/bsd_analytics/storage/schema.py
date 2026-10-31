"""
SQLAlchemy ORM models for BSD-Analytics analytical warehouse.
Defines schemas for experiment tracking, data quality audits, and prediction telemetry.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ModelExperiment(Base):
    """Tracks training runs, model hyperparameter configurations, and evaluation metrics."""

    __tablename__ = "model_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    timestamp: Mapped[str] = mapped_column(
        String(50), default=lambda: datetime.now(timezone.utc).isoformat()
    )
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    r2: Mapped[float] = mapped_column(Float, nullable=False)
    rmsle: Mapped[float] = mapped_column(Float, nullable=False)
    mean_bias: Mapped[float] = mapped_column(Float, nullable=False)
    hyperparameters_json: Mapped[str] = mapped_column(Text, nullable=True)


class PredictionLog(Base):
    """Telemetry logging for real-time inference requests."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(
        String(50), default=lambda: datetime.now(timezone.utc).isoformat()
    )
    predicted_count: Mapped[int] = mapped_column(Integer, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    input_summary_json: Mapped[str] = mapped_column(Text, nullable=False)


class DataQualityAudit(Base):
    """Historical audit records of data validation and schema conformance."""

    __tablename__ = "data_quality_audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[str] = mapped_column(
        String(50), default=lambda: datetime.now(timezone.utc).isoformat()
    )
    total_records: Mapped[int] = mapped_column(Integer, nullable=False)
    is_valid: Mapped[int] = mapped_column(Integer, nullable=False)  # 1=Valid, 0=Invalid
    missing_values_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_rows_count: Mapped[int] = mapped_column(Integer, default=0)
    audit_summary_json: Mapped[str] = mapped_column(Text, nullable=True)
