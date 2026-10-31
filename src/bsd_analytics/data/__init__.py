"""Data ingestion, validation, and preprocessing subpackage."""

from bsd_analytics.data.ingestion import (
    download_dataset,
    get_dataset_metadata,
    load_raw_data,
)
from bsd_analytics.data.preprocessing import (
    DataPreprocessor,
    clean_and_preprocess_data,
    inspect_outliers,
)
from bsd_analytics.data.validation import (
    ValidationResult,
    generate_quality_report,
    run_data_quality_audit,
    validate_schema,
)

__all__ = [
    "download_dataset",
    "load_raw_data",
    "get_dataset_metadata",
    "validate_schema",
    "run_data_quality_audit",
    "generate_quality_report",
    "ValidationResult",
    "DataPreprocessor",
    "clean_and_preprocess_data",
    "inspect_outliers",
]
