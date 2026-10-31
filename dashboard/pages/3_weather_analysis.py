"""
Page 3: Weather Insights
Analyzes temperature, humidity, windspeed, and precipitation severity impacts on demand.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from bsd_analytics.data.ingestion import load_raw_data
from bsd_analytics.analysis.statistics import compute_correlation_matrix, test_temperature_demand_hypothesis, test_weather_condition_anova
from dashboard.components.charts import create_weather_scatter_chart, create_correlation_heatmap

st.set_page_config(page_title="Weather Insights — BSD-Analytics", layout="wide")

st.title("Weather Insights & Elasticity")
st.caption("How ambient temperature, humidity, and weather conditions govern ridership volume")

df = load_raw_data()

# Sidebar Weather Filters
st.sidebar.subheader("Filter Weather Parameters")
temp_range = st.sidebar.slider("Temperature Range (°C)", float(df["temp"].min()), float(df["temp"].max()), (10.0, 35.0))
hum_range = st.sidebar.slider("Humidity Range (%)", float(df["humidity"].min()), float(df["humidity"].max()), (20.0, 90.0))

filtered_df = df[
    (df["temp"] >= temp_range[0]) & (df["temp"] <= temp_range[1]) &
    (df["humidity"] >= hum_range[0]) & (df["humidity"] <= hum_range[1])
]

st.info(f"Filtered records: **{len(filtered_df):,}** out of {len(df):,} total hourly observations.")

# 1. Weather Category Comparison
st.subheader("1. Ridership by Weather Severity Category")
weather_map = {
    1: "Clear / Few Clouds",
    2: "Mist / Cloudy",
    3: "Light Snow / Rain",
    4: "Heavy Storm / Torrential",
}
df_weather = df.copy()
df_weather["weather_label"] = df_weather["weather"].map(weather_map)
weather_agg = df_weather.groupby("weather_label")["count"].agg(["mean", "median", "count"]).reset_index()

fig_wbar = px.bar(
    weather_agg,
    x="weather_label",
    y="mean",
    color="weather_label",
    title="Average Hourly Demand by Weather Condition",
    labels={"weather_label": "Weather Condition", "mean": "Average Hourly Rentals"},
    template="plotly_white",
    color_discrete_sequence=["#38BDF8", "#94A3B8", "#F59E0B", "#EF4444"],
)
st.plotly_chart(fig_wbar, use_container_width=True)

# 2. Interactive Meteorological Exploration
st.subheader("2. Meteorological Scatter Analysis")
param = st.selectbox("Select Weather Parameter to Inspect", ["temp", "atemp", "humidity", "windspeed"], index=0)
st.plotly_chart(create_weather_scatter_chart(filtered_df, x_col=param), use_container_width=True)

# 3. Correlation Heatmap & Hypothesis Testing
col1, col2 = st.columns(2)
with col1:
    st.subheader("3. Feature Correlations")
    corr = compute_correlation_matrix(df)
    st.plotly_chart(create_correlation_heatmap(corr), use_container_width=True)

with col2:
    st.subheader("4. Statistical Significance")
    temp_test = test_temperature_demand_hypothesis(df)
    anova_test = test_weather_condition_anova(df)

    st.markdown(f"""
    **Temperature vs Demand Hypothesis:**
    - Pearson $r$: `{temp_test['pearson_r']}` ($p < 0.001$)
    - Spearman $\\rho$: `{temp_test['spearman_rho']}`
    - Status: **Statistically Significant**
    
    *Conclusion:* {temp_test['conclusion']}
    
    ---
    
    **Weather Category ANOVA:**
    - ANOVA $F$-Statistic: `{anova_test['anova_f_statistic']}` ($p < 0.001$)
    - Kruskal-Wallis $H$: `{anova_test['kruskal_h_statistic']}`
    - Status: **Statistically Significant**
    
    *Conclusion:* {anova_test['conclusion']}
    """)
