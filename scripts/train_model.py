"""Script to execute model benchmark training, evaluation, and production model serialization."""

import logging
from bsd_analytics.data.ingestion import load_raw_data
from bsd_analytics.models.train import chronological_train_test_split, train_all_models
from bsd_analytics.models.evaluate import compare_models, evaluate_model
from bsd_analytics.models.registry import save_model
from bsd_analytics.storage.db import log_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def main():
    print("Executing model training experiments...")
    df = load_raw_data()
    split = chronological_train_test_split(df)
    print(f"Dataset split chronologically: Train={split.train_records:,}, Holdout Test={split.test_records:,}")

    results = train_all_models(split)
    leaderboard = compare_models(results)
    print("\nBENCHMARK LEADERBOARD (Chronological Holdout):")
    print(leaderboard[["model_name", "mae", "rmse", "r2", "rmsle"]].to_string(index=False))

    # Identify best model (lowest MAE)
    best_res = results[0]
    best_metrics = evaluate_model(best_res)
    for res in results:
        m = evaluate_model(res)
        # Log to SQLite
        log_experiment(res.model_name, m.to_dict(), res.hyperparameters)
        if m.mae < best_metrics.mae:
            best_metrics = m
            best_res = res

    # Save best model to registry
    model_path, meta_path = save_model(best_res, best_metrics)
    print(f"\nSaved Best Model ({best_res.model_name}) to {model_path}")
    print(f"Model Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
