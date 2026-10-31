"""
Model registry, artifact serialization, and metadata persistence module for BSD-Analytics.
Provides atomic model saving, loading, versioning, and provenance tracking.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from bsd_analytics.config import MODELS_DIR, ensure_directories
from bsd_analytics.models.evaluate import EvaluationMetrics, evaluate_model
from bsd_analytics.models.train import ModelResult

logger = logging.getLogger(__name__)


def save_model(
    result: ModelResult,
    metrics: EvaluationMetrics | None = None,
    target_dir: Path = MODELS_DIR,
    filename: str = "best_model.joblib",
) -> tuple[Path, Path]:
    """
    Serialize fitted estimator and persist complete provenance metadata.

    Args:
        result: Fitted ModelResult object.
        metrics: Optional pre-calculated EvaluationMetrics.
        target_dir: Destination folder.
        filename: Destination binary filename.

    Returns:
        Tuple of (model_path, metadata_path).
    """
    ensure_directories()
    if metrics is None:
        metrics = evaluate_model(result)

    model_path = target_dir / filename
    meta_path = target_dir / f"{Path(filename).stem}_metadata.json"

    # 1. Save binary estimator with joblib
    joblib.dump(result.estimator, model_path)

    # 2. Save comprehensive JSON metadata
    metadata: dict[str, Any] = {
        "model_name": result.model_name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "binary_filename": filename,
        "feature_names": result.feature_names,
        "feature_count": len(result.feature_names),
        "hyperparameters": result.hyperparameters,
        "metrics": metrics.to_dict(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Successfully serialized model to %s and metadata to %s", model_path, meta_path)
    return model_path, meta_path


def load_model(
    model_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """
    Load saved model artifact and corresponding metadata.

    Args:
        model_path: Optional explicit model path (defaults to models/best_model.joblib).
        meta_path: Optional explicit metadata path.

    Returns:
        Tuple of (estimator, metadata_dict).
    """
    if model_path is None:
        model_path = MODELS_DIR / "best_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train models first.")

    estimator = joblib.load(model_path)

    if meta_path is None:
        meta_path = model_path.parent / f"{model_path.stem}_metadata.json"

    metadata: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return estimator, metadata


def get_production_model() -> tuple[Any, list[str], dict[str, Any]]:
    """
    High-level helper for inference pipelines to retrieve current production model.
    """
    estimator, metadata = load_model()
    feature_names = metadata.get("feature_names", [])
    return estimator, feature_names, metadata
