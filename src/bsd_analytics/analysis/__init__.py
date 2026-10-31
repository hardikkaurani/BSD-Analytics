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
from bsd_analytics.analysis.statistics import (
    compute_correlation_matrix,
    generate_statistical_report,
    test_temperature_demand_hypothesis,
    test_weather_condition_anova,
    test_workingday_demand_difference,
)

__all__ = [
    "compute_hourly_summary",
    "compute_day_of_week_summary",
    "compute_monthly_summary",
    "compute_seasonal_summary",
    "compute_workingday_vs_weekend",
    "compute_casual_vs_registered_breakdown",
    "generate_eda_summary_tables",
    "compute_correlation_matrix",
    "test_temperature_demand_hypothesis",
    "test_workingday_demand_difference",
    "test_weather_condition_anova",
    "generate_statistical_report",
]
