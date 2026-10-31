"""
BSD-Analytics: Interactive Streamlit Dashboard Main Entrypoint.
Product: Bike Sharing Demand Analytics
Identity: Understand demand. Predict smarter. Optimize urban mobility.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from bsd_analytics.analytics.insights import get_executive_kpis
from bsd_analytics.data.ingestion import load_raw_data

# Page configuration
st.set_page_config(
    page_title="BSD-Analytics — Bike Sharing Demand",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean CSS styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    """Load and cache the standard dataset for fast multi-page rendering."""
    return load_raw_data()


def main():
    st.sidebar.title("🚲 BSD-Analytics")
    st.sidebar.caption("Demand Intelligence & Machine Learning")
    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    **Navigation**
    - Use the pages menu above to explore:
      1. Executive Overview
      2. Demand Analysis
      3. Weather Insights
      4. Demand Forecasting
      5. Model Performance
      6. Business Insights
    """)
    st.sidebar.markdown("---")
    st.sidebar.info("Dataset: Capital Bikeshare / Kaggle (17,379 hourly observations)")

    # Welcome landing view if accessed on home
    st.markdown('<div class="main-header">BSD-Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Understand demand. Predict smarter. Optimize urban mobility.</div>',
        unsafe_allow_html=True,
    )

    df = get_cached_dataset()
    kpis = get_executive_kpis(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Historical Rentals", f"{kpis['total_rentals']:,}")
    with col2:
        st.metric("Average Hourly Rentals", f"{kpis['avg_hourly_demand']} bikes/hr")
    with col3:
        st.metric(
            "Peak Rush Hour", f"{kpis['peak_demand_hour']}:00 hrs", f"{kpis['peak_hour_avg']} avg"
        )
    with col4:
        st.metric("Peak Demand Season", kpis["peak_season"], f"{kpis['peak_season_avg']} avg/hr")

    st.markdown("---")
    st.subheader("System Overview")
    st.write("""
    **BSD-Analytics** is a production-grade machine learning and business intelligence system
    designed to assist urban mobility operators, transit authorities, and fleet managers.
    
    ### Key Analytical Capabilities:
    - **Descriptive Analytics:** Granular temporal, meteorological, and behavioral segmentation.
    - **Inferential Statistics:** Rigorous parametric and non-parametric hypothesis testing.
    - **Predictive Regression:** Benchmark evaluation across multiple models with chronological validation.
    - **Actionable Optimization:** Practical recommendations for fleet distribution and maintenance.
    """)

    st.markdown("👈 **Select a page from the sidebar to begin exploring.**")


if __name__ == "__main__":
    main()
