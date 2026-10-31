"""Machine Learning modeling subpackage for BSD-Analytics."""

from bsd_analytics.models.evaluate import (
    EvaluationMetrics,
    calculate_metrics,
    compare_models,
    evaluate_model,
    subgroup_error_analysis,
)
from bsd_analytics.models.interpretability import (
    compute_permutation_importance,
    extract_native_importance,
    get_interpretability_summary,
)
from bsd_analytics.models.predict import (
    PredictionEngine,
    PredictionOutput,
    predict_single_observation,
)
from bsd_analytics.models.registry import (
    get_production_model,
    load_model,
    save_model,
)
from bsd_analytics.models.train import (
    ModelResult,
    SplitData,
    chronological_train_test_split,
    get_feature_target_matrices,
    train_all_models,
    train_baseline_model,
    train_gradient_boosting,
    train_hist_gradient_boosting,
    train_linear_regression,
    train_random_forest,
)

__all__ = [
    "SplitData",
    "ModelResult",
    "chronological_train_test_split",
    "get_feature_target_matrices",
    "train_baseline_model",
    "train_linear_regression",
    "train_random_forest",
    "train_gradient_boosting",
    "train_hist_gradient_boosting",
    "train_all_models",
    "EvaluationMetrics",
    "calculate_metrics",
    "evaluate_model",
    "compare_models",
    "subgroup_error_analysis",
    "save_model",
    "load_model",
    "get_production_model",
    "extract_native_importance",
    "compute_permutation_importance",
    "get_interpretability_summary",
    "PredictionEngine",
    "PredictionOutput",
    "predict_single_observation",
]
