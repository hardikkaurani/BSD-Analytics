"""
Exploratory Data Analysis (EDA) module for BSD-Analytics.
Computes descriptive statistics, temporal aggregates, seasonal breakdowns,
and user-behavior metrics across ridership profiles.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from bsd_analytics.features.engineering import add_temporal_features

logger = logging.getLogger(__name__)


def ensure_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to ensure hour, weekday, month exist in DataFrame."""
    if "hour" not in df.columns or "weekday" not in df.columns:
        return add_temporal_features(df)
    return df


def compute_hourly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregate demand statistics grouped by hour of the day (0-23).
    Identifies commute peaks (morning rush 8 AM, evening rush 5-6 PM).
    """
    data = ensure_temporal(df)
    agg_dict: dict[str, list[str]] = {"count": ["mean", "median", "std", "min", "max", "count"]}

    if "casual" in data.columns and "registered" in data.columns:
        agg_dict["casual"] = ["mean", "sum"]
        agg_dict["registered"] = ["mean", "sum"]

    summary = data.groupby("hour").agg(agg_dict)
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    return summary.reset_index()


def compute_day_of_week_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute demand statistics grouped by day of week (Monday=0 to Sunday=6).
    """
    data = ensure_temporal(df)
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    summary = (
        data.groupby("weekday")["count"]
        .agg(["mean", "median", "std", "sum", "count"])
        .reset_index()
    )
    summary["day_name"] = summary["weekday"].apply(lambda d: days[d])
    return summary


def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute demand statistics grouped by calendar month (1-12).
    """
    data = ensure_temporal(df)
    months = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    summary = (
        data.groupby("month")["count"]
        .agg(["mean", "median", "std", "sum", "count"])
        .reset_index()
    )
    summary["month_name"] = summary["month"].apply(lambda m: months[m - 1])
    return summary


def compute_seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute demand statistics grouped by season:
    1: Spring, 2: Summer, 3: Fall, 4: Winter.
    """
    season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
    summary = (
        df.groupby("season")["count"]
        .agg(["mean", "median", "std", "sum", "count"])
        .reset_index()
    )
    summary["season_name"] = summary["season"].map(season_map)
    return summary


def compute_workingday_vs_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare average hourly profiles on working days vs weekends/holidays.
    """
    data = ensure_temporal(df)
    summary = (
        data.groupby(["workingday", "hour"])["count"]
        .agg(mean_demand="mean", median_demand="median", std_demand="std")
        .reset_index()
    )
    summary["day_type"] = summary["workingday"].map({1: "Working Day", 0: "Weekend / Holiday"})
    return summary


def compute_casual_vs_registered_breakdown(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze the behavioral divergence between casual (recreational) riders
    and registered (commuter) subscribers.
    Note: Casual and registered are used strictly for descriptive behavioral insights.
    """
    if "casual" not in df.columns or "registered" not in df.columns:
        return {"available": False, "message": "Casual and registered columns not present in dataset"}

    total_casual = int(df["casual"].sum())
    total_registered = int(df["registered"].sum())
    total_rentals = int(df["count"].sum())

    casual_share = round(total_casual / total_rentals * 100.0, 2)
    registered_share = round(total_registered / total_rentals * 100.0, 2)

    data = ensure_temporal(df)
    workingday_split = (
        data.groupby("workingday")[["casual", "registered", "count"]]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )

    return {
        "available": True,
        "total_rentals": total_rentals,
        "total_casual": total_casual,
        "total_registered": total_registered,
        "casual_share_pct": casual_share,
        "registered_share_pct": registered_share,
        "workingday_profiles": workingday_split,
        "key_takeaway": (
            "Registered riders dominate weekday commute hours (7-9 AM, 5-7 PM), "
            "whereas casual riders peak on weekend afternoons with high temperature sensitivity."
        ),
    }


def generate_eda_summary_tables(df: pd.DataFrame) -> dict[str, Any]:
    """
    Consolidate all EDA summary tables into a single serializable bundle.
    """
    return {
        "hourly": compute_hourly_summary(df).to_dict(orient="records"),
        "day_of_week": compute_day_of_week_summary(df).to_dict(orient="records"),
        "monthly": compute_monthly_summary(df).to_dict(orient="records"),
        "seasonal": compute_seasonal_summary(df).to_dict(orient="records"),
        "user_breakdown": compute_casual_vs_registered_breakdown(df),
    }
