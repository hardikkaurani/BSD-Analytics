"""
Plotly visualization components for the BSD-Analytics Streamlit dashboard.
Provides clean, consistent styling across all interactive chart figures.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TEMPLATE = "plotly_white"
PRIMARY_COLOR = "#2563EB"
ACCENT_COLOR = "#F59E0B"
SECONDARY_COLOR = "#10B981"


def create_hourly_trend_chart(hourly_df: pd.DataFrame) -> go.Figure:
    """Line chart showing average demand throughout the 24 hours of the day."""
    fig = go.Figure()

    if "count_mean" in hourly_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hourly_df["hour"],
                y=hourly_df["count_mean"],
                mode="lines+markers",
                name="Total Demand",
                line=dict(color=PRIMARY_COLOR, width=3),
                marker=dict(size=6),
            )
        )

    if "registered_mean" in hourly_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hourly_df["hour"],
                y=hourly_df["registered_mean"],
                mode="lines",
                name="Registered (Commuter)",
                line=dict(color="#64748B", width=2, dash="dot"),
            )
        )

    if "casual_mean" in hourly_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hourly_df["hour"],
                y=hourly_df["casual_mean"],
                mode="lines",
                name="Casual (Leisure)",
                line=dict(color=ACCENT_COLOR, width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title="Average Hourly Rental Volume",
        xaxis_title="Hour of Day (0-23)",
        yaxis_title="Average Bike Rentals / Hour",
        template=TEMPLATE,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_seasonal_bar_chart(seasonal_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing mean demand across Spring, Summer, Fall, Winter."""
    fig = px.bar(
        seasonal_df,
        x="season_name",
        y="mean",
        color="season_name",
        color_discrete_sequence=["#38BDF8", "#F59E0B", "#F97316", "#818CF8"],
        labels={"season_name": "Season", "mean": "Average Hourly Rentals"},
        title="Seasonal Demand Comparison",
        template=TEMPLATE,
    )
    fig.update_layout(showlegend=False)
    return fig


def create_weather_scatter_chart(
    df: pd.DataFrame, x_col: str = "temp", sample_size: int = 2500
) -> go.Figure:
    """Scatter plot with trendline showing meteorological impact on demand."""
    sample_df = df.sample(min(len(df), sample_size), random_state=42)

    labels = {
        "temp": "Temperature (Celsius)",
        "atemp": "Apparent 'Feels-Like' Temp (Celsius)",
        "humidity": "Relative Humidity (%)",
        "windspeed": "Windspeed (km/h)",
        "count": "Bike Rental Count",
    }

    fig = px.scatter(
        sample_df,
        x=x_col,
        y="count",
        color="season",
        color_continuous_scale="Viridis",
        opacity=0.5,
        title=f"Impact of {labels.get(x_col, x_col)} on Rental Demand",
        labels=labels,
        template=TEMPLATE,
    )
    return fig


def create_actual_vs_predicted_scatter(
    y_true: pd.Series, y_pred: list[float] | pd.Series, sample_size: int = 1500
) -> go.Figure:
    """Parity plot comparing actual vs predicted demand."""
    df_plot = pd.DataFrame({"Actual": y_true, "Predicted": y_pred}).sample(
        min(len(y_true), sample_size), random_state=42
    )

    max_val = max(df_plot["Actual"].max(), df_plot["Predicted"].max())

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_plot["Actual"],
            y=df_plot["Predicted"],
            mode="markers",
            opacity=0.4,
            marker=dict(color=PRIMARY_COLOR, size=5),
            name="Predictions",
        )
    )

    # 45-degree parity reference line
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#EF4444", width=2, dash="dash"),
            name="Perfect Calibration",
        )
    )

    fig.update_layout(
        title="Actual vs Predicted Demand (Holdout Set)",
        xaxis_title="Actual Rental Count",
        yaxis_title="Predicted Rental Count",
        template=TEMPLATE,
    )
    return fig


def create_residual_distribution_chart(residuals: list[float] | pd.Series) -> go.Figure:
    """Histogram of regression residuals to assess error normality and zero-bias."""
    fig = px.histogram(
        x=residuals,
        nbins=60,
        title="Residual Error Distribution (Actual - Predicted)",
        labels={"x": "Prediction Error / Residual"},
        color_discrete_sequence=[PRIMARY_COLOR],
        template=TEMPLATE,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#EF4444", annotation_text="Zero Bias")
    return fig


def create_correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Interactive heatmap showing pairwise feature correlations."""
    fig = px.imshow(
        corr_df,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
        template=TEMPLATE,
    )
    return fig
