"""
Page 1: Executive Overview
Displays high-level KPIs, total rental volume, macro demand trends, and core findings.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from bsd_analytics.data.ingestion import load_raw_data
from bsd_analytics.analytics.insights import get_executive_kpis

st.set_page_config(page_title="Executive Overview — BSD-Analytics", layout="wide")

st.title("Executive Overview")
st.caption("Macro demand metrics, operational summary, and strategic findings")

df = load_raw_data()
kpis = get_executive_kpis(df)

# Top KPIs Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Historical Rides", f"{kpis['total_rentals']:,}")
c2.metric("Mean Hourly Demand", f"{kpis['avg_hourly_demand']} bikes/hr")
c3.metric("Peak Rush Hour", f"{kpis['peak_demand_hour']}:00", f"{kpis['peak_hour_avg']} avg/hr")
c4.metric("Average Daily Rides", f"{kpis['avg_daily_demand']:,} bikes/day")

st.markdown("---")

# Monthly / Macro Trend Chart
st.subheader("Historical Demand Growth (2011 - 2012)")
df_daily = df.copy()
df_daily["date"] = pd.to_datetime(df_daily["datetime"]).dt.date
daily_summary = df_daily.groupby("date")["count"].sum().reset_index()

fig_trend = px.line(
    daily_summary,
    x="date",
    y="count",
    title="Daily Total Bike Rentals Over Time",
    labels={"date": "Date", "count": "Total Daily Rentals"},
    template="plotly_white",
    color_discrete_sequence=["#2563EB"],
)
st.plotly_chart(fig_trend, use_container_width=True)

# Strategic Takeaways
st.subheader("Key Analytical Discoveries")
col_a, col_b = st.columns(2)

with col_a:
    st.info("""
    **1. Sustained Network Growth**
    Total daily ridership exhibited dramatic expansion from 2011 to 2012 as system awareness
    and dock density expanded across Washington D.C.
    
    **2. Bimodal Commuter Concentration**
    Weekdays are heavily anchored by commute spikes: 08:00 AM inbound to commercial cores,
    and 17:00-18:00 PM outbound toward residential transit hubs.
    """)

with col_b:
    st.info("""
    **3. Weather Elasticity**
    Temperature acts as an expansion catalyst up to ~32°C. In contrast, precipitation
    and freezing temperatures cause instantaneous volume compression exceeding 50%.
    
    **4. Distinct User Personas**
    Registered subscribers account for 81.2% of demand with rigid schedules,
    while casual riders represent 18.8% and concentrate predominantly on warm weekend afternoons.
    """)
