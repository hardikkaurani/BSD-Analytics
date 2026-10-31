"""
Page 5: Model Performance
Displays regression benchmark leaderboard, actual vs predicted calibration,
residual error distributions, and feature importance rankings.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from bsd_analytics.data.ingestion import load_raw_data
from bsd_analytics.models.train import chronological_train_test_split, train_all_models
from bsd_analytics.models.evaluate import compare_models, subgroup_error_analysis
from bsd_analytics.models.interpretability import extract_native_importance
from dashboard.components.charts import (
    create_actual_vs_predicted_scatter,
    create_residual_distribution_chart,
)

st.set_page_config(page_title="Model Performance — BSD-Analytics", layout="wide")

st.title("Model Performance & Diagnostics")
st.caption("Empirical benchmark comparison evaluated strictly on chronological holdout data")


@st.cache_data
def get_benchmark_results():
    df = load_raw_data()
    split = chronological_train_test_split(df)
    results = train_all_models(split)
    leaderboard = compare_models(results)
    return results, leaderboard


results, leaderboard = get_benchmark_results()

# 1. Benchmark Comparison Leaderboard
st.subheader("1. Candidate Model Leaderboard")
st.dataframe(
    leaderboard[["model_name", "mae", "rmse", "r2", "rmsle", "mean_bias"]].style.highlight_min(
        subset=["mae", "rmse", "rmsle"], color="#DCFCE7"
    ).highlight_max(subset=["r2"], color="#DCFCE7"),
    use_container_width=True,
)

best_model_res = results[3]  # GBDT / best ensemble

# 2. Calibration and Residuals
st.subheader("2. Prediction Calibration & Error Distributions")
col1, col2 = st.columns(2)

with col1:
    fig_parity = create_actual_vs_predicted_scatter(best_model_res.y_test, best_model_res.y_pred)
    st.plotly_chart(fig_parity, use_container_width=True)

with col2:
    residuals = best_model_res.y_test.values - best_model_res.y_pred
    fig_resid = create_residual_distribution_chart(residuals)
    st.plotly_chart(fig_resid, use_container_width=True)

# 3. Feature Importance Analysis
st.subheader("3. Feature Importance Analysis")
imp_df = extract_native_importance(best_model_res.estimator, best_model_res.feature_names)

fig_imp = px.bar(
    imp_df.head(10),
    x="relative_pct",
    y="feature",
    orientation="h",
    title="Top 10 Influential Features (Relative Importance %)",
    labels={"relative_pct": "Relative Importance (%)", "feature": "Feature Name"},
    template="plotly_white",
    color="relative_pct",
    color_continuous_scale="Blues",
)
fig_imp.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_imp, use_container_width=True)

# 4. Subgroup Performance
st.subheader("4. Segment Error Breakdown")
sub_err = subgroup_error_analysis(best_model_res)

col_s1, col_s2 = st.columns(2)
with col_s1:
    st.markdown("**Seasonal Holdout MAE:**")
    st.json(sub_err["seasonal_mae"])

with col_s2:
    st.markdown("**Weather Severity Holdout MAE:**")
    st.json(sub_err["weather_mae"])
