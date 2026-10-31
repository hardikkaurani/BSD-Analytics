"""Command line interface for BSD-Analytics."""

import click

from bsd_analytics import __version__
from bsd_analytics.data.ingestion import download_dataset, load_raw_data
from bsd_analytics.data.preprocessing import clean_and_preprocess_data
from bsd_analytics.data.validation import generate_quality_report
from bsd_analytics.models.evaluate import compare_models, evaluate_model
from bsd_analytics.models.predict import predict_single_observation
from bsd_analytics.models.registry import save_model
from bsd_analytics.models.train import chronological_train_test_split, train_all_models


@click.group()
@click.version_option(version=__version__)
def main():
    """BSD-Analytics: Bike Sharing Demand Analytics System."""
    pass


@main.command()
def download():
    """Download and verify raw bike sharing dataset."""
    path = download_dataset()
    click.echo(f"Dataset downloaded to {path}")


@main.command()
def pipeline():
    """Run data ingestion, validation, and preprocessing pipeline."""
    df = load_raw_data()
    generate_quality_report(df)
    df_clean = clean_and_preprocess_data(df)
    click.echo(f"Pipeline executed successfully. Processed shape: {df_clean.shape}")


@main.command()
def train():
    """Train regression models, print benchmark leaderboard, and save best model."""
    df = load_raw_data()
    split = chronological_train_test_split(df)
    results = train_all_models(split)
    lb = compare_models(results)
    click.echo("\nBenchmark Leaderboard:")
    click.echo(lb[["model_name", "mae", "rmse", "r2", "rmsle"]].to_string(index=False))

    best_res = min(results, key=lambda r: evaluate_model(r).mae)
    m_path, _ = save_model(best_res)
    click.echo(f"\nBest model ({best_res.model_name}) saved to {m_path}")


@main.command()
@click.option("--datetime", "dt", default="2026-06-15 08:00:00", help="Observation timestamp")
@click.option("--temp", default=22.0, type=float, help="Temperature in Celsius")
@click.option("--humidity", default=60.0, type=float, help="Humidity percentage (0-100)")
@click.option("--windspeed", default=10.0, type=float, help="Windspeed in km/h")
def predict(dt, temp, humidity, windspeed):
    """Forecast bike rental demand for a given weather condition."""
    out = predict_single_observation(
        datetime_str=dt,
        season=2,
        holiday=0,
        workingday=1,
        weather=1,
        temp=temp,
        atemp=temp + 2.0,
        humidity=humidity,
        windspeed=windspeed,
    )
    click.echo(f"Predicted Demand: {out.predicted_count} rentals/hour (Model: {out.model_name})")


if __name__ == "__main__":
    main()
