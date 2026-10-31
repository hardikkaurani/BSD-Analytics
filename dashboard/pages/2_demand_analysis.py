"""
Page 2: Demand Analysis
Detailed temporal distributions, daily trends, working day vs weekend comparisons,
and casual vs registered user profiles.
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from bsd_analytics.analysis.eda import (
    compute_casual_vs_registered_breakdown,
    compute_day_of_week_summary,
    compute_hourly_summary,
    compute_seasonal_summary,
    compute_workingday_vs_weekend,
)
from bsd_analytics.data.ingestion import load_raw_data
from dashboard.components.charts import create_hourly_trend_chart, create_seasonal_bar_chart

st.set_page_config(page_title="Demand Analysis — BSD-Analytics", layout="wide")

st.title("Demand Analysis")
st.caption("Deep-dive exploration of temporal distributions and ridership behaviors")

df = load_raw_data()

# 1. Hourly Profiles
st.subheader("1. Hourly Demand Patterns")
hourly_df = compute_hourly_summary(df)
st.plotly_chart(create_hourly_trend_chart(hourly_df), use_container_width=True)

# 2. Working Day vs Weekend
st.subheader("2. Working Day vs Weekend Dynamics")
work_df = compute_workingday_vs_weekend(df)
fig_work = px.line(
    work_df,
    x="hour",
    y="mean_demand",
    color="day_type",
    title="Average Hourly Demand: Working Days vs Weekends/Holidays",
    labels={
        "hour": "Hour of Day (0-23)",
        "mean_demand": "Average Rentals",
        "day_type": "Day Classification",
    },
    template="plotly_white",
    color_discrete_map={"Working Day": "#2563EB", "Weekend / Holiday": "#F59E0B"},
)
st.plotly_chart(fig_work, use_container_width=True)

# 3. Seasonal & Day of Week
col1, col2 = st.columns(2)
with col1:
    st.subheader("3. Seasonal Variations")
    seasonal_df = compute_seasonal_summary(df)
    st.plotly_chart(create_seasonal_bar_chart(seasonal_df), use_container_width=True)

with col2:
    st.subheader("4. Day of the Week Comparison")
    dow_df = compute_day_of_week_summary(df)
    fig_dow = px.bar(
        dow_df,
        x="day_name",
        y="mean",
        title="Average Rentals by Day of Week",
        labels={"day_name": "Day", "mean": "Average Hourly Demand"},
        template="plotly_white",
        color="mean",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_dow, use_container_width=True)

# 5. User Behavioral Breakdown
st.subheader("5. Casual vs Registered Ridership Breakdown")
user_data = compute_casual_vs_registered_breakdown(df)

col_u1, col_u2 = st.columns([1, 2])
with col_u1:
    st.write(
        f"**Total Registered Rides:** {user_data['total_registered']:,} ({user_data['registered_share_pct']}%)"
    )
    st.write(
        f"**Total Casual Rides:** {user_data['total_casual']:,} ({user_data['casual_share_pct']}%)"
    )
    st.info(user_data["key_takeaway"])

with col_u2:
    pie_df = pd.DataFrame(
        {
            "User Type": ["Registered Subscribers", "Casual Customers"],
            "Count": [user_data["total_registered"], user_data["total_casual"]],
        }
    )
    fig_pie = px.pie(
        pie_df,
        names="User Type",
        values="Count",
        title="Share of Total Ridership Volume",
        template="plotly_white",
        color_discrete_sequence=["#2563EB", "#F59E0B"],
    )
    st.plotly_chart(fig_pie, use_container_width=True)
