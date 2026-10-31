"""
Page 6: Business Insights & Fleet Allocation
Translates predictive patterns and descriptive findings into operational actions,
fleet rebalancing schedules, and revenue optimization strategies.
"""

import pandas as pd
import streamlit as st

from bsd_analytics.analytics.insights import generate_fleet_recommendations, get_executive_kpis
from bsd_analytics.data.ingestion import load_raw_data

st.set_page_config(page_title="Business Insights — BSD-Analytics", layout="wide")

st.title("Business Insights & Fleet Optimization")
st.caption("Operational strategies, rebalancing dispatch schedules, and fleet maintenance planning")

df = load_raw_data()
kpis = get_executive_kpis(df)
recommendations = generate_fleet_recommendations(df)

# Executive Snapshot
st.subheader("Operational Baseline Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Peak Rush Hour", f"{kpis['peak_demand_hour']}:00 hrs", "Commuter Surge")
c2.metric(
    "Low-Traffic Maintenance Window",
    f"{kpis['trough_demand_hour']}:00 - 05:00 hrs",
    "Minimal Impact",
)
c3.metric("Peak Ridership Season", kpis["peak_season"], "Full Fleet Staging")

st.markdown("---")

st.subheader("Evidence-Backed Operational Recommendations")

for idx, rec in enumerate(recommendations, 1):
    with st.expander(f"Strategy {idx}: {rec['category']}", expanded=(idx <= 2)):
        st.markdown(f"**Observed Evidence:** {rec['observation']}")
        st.markdown(f"**Operational Recommendation:** {rec['recommendation']}")
        st.success(f"**Projected Business Impact:** {rec['business_impact']}")

st.markdown("---")

st.subheader("Rebalancing Dispatch Schedule Blueprint")

schedule_data = pd.DataFrame(
    [
        {
            "Time Window": "05:30 - 06:45 AM",
            "Primary Activity": "Morning Inbound Staging",
            "Target Zones": "Metro Terminals, Residential Districts",
            "Objective": "Ensure >=85% dock fill in residential nodes before 7 AM rush.",
        },
        {
            "Time Window": "08:30 - 10:00 AM",
            "Primary Activity": "Commercial De-clustering",
            "Target Zones": "Central Business District, Tech Hubs",
            "Objective": "Evacuate full docks in commercial zones to prevent turn-aways.",
        },
        {
            "Time Window": "15:30 - 16:45 PM",
            "Primary Activity": "Evening Outbound Staging",
            "Target Zones": "Downtown Offices, Financial Districts",
            "Objective": "Pre-stage bikes at office towers before 5 PM departure spike.",
        },
        {
            "Time Window": "01:30 - 04:30 AM",
            "Primary Activity": "Maintenance & Battery Swapping",
            "Target Zones": "All Stations / Central Depot",
            "Objective": "Perform deep cleans, repairs, and battery charging during demand trough.",
        },
    ]
)

st.table(schedule_data)
