"""Script to execute full data ingestion, validation, and preprocessing pipeline."""

import logging
from bsd_analytics.data.ingestion import load_raw_data
from bsd_analytics.data.validation import generate_quality_report, run_data_quality_audit
from bsd_analytics.data.preprocessing import clean_and_preprocess_data, inspect_outliers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    print("Running end-to-end data pipeline...")
    # 1. Ingestion
    df_raw = load_raw_data()
    print(f"Step 1: Loaded raw dataset with shape {df_raw.shape}")

    # 2. Validation & Quality Audit
    audit = run_data_quality_audit(df_raw)
    report_path = generate_quality_report(df_raw)
    print(f"Step 2: Data quality audit status: {'PASSED' if audit.is_valid else 'FAILED'}. Report saved to {report_path}")

    # 3. Preprocessing & Leakage Elimination
    df_clean = clean_and_preprocess_data(df_raw, save_processed=True, drop_leakage=True)
    outlier_info = inspect_outliers(df_clean, "count")
    print(f"Step 3: Preprocessed dataset shape: {df_clean.shape}. Saved parquet and csv artifacts.")
    print(f"Outlier count in target: {outlier_info['outlier_count']} ({outlier_info['outlier_percentage']}%)")
    print("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()
