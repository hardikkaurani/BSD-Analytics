"""Analysis, EDA, and statistical testing subpackage."""

from bsd_analytics.analysis.eda import (
    compute_casual_vs_registered_breakdown,
    compute_day_of_week_summary,
    compute_hourly_summary,
    compute_monthly_summary,
    compute_seasonal_summary,
    compute_workingday_vs_weekend,
    generate_eda_summary_tables,
)

__all__ = [
    "compute_hourly_summary",
    "compute_day_of_week_summary",
    "compute_monthly_summary",
    "compute_seasonal_summary",
    "compute_workingday_vs_weekend",
    "compute_casual_vs_registered_breakdown",
    "generate_eda_summary_tables",
]
