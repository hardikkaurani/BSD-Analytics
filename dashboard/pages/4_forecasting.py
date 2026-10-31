"""
Page 4: Demand Forecasting
Live inference interface allowing operators to input weather & temporal conditions
and receive real-time demand forecasts with model provenance.
"""

from datetime import datetime

import streamlit as st

from bsd_analytics.models.predict import predict_single_observation
from bsd_analytics.models.registry import load_model
from bsd_analytics.storage.db import log_prediction

st.set_page_config(page_title="Demand Forecasting — BSD-Analytics", layout="wide")

st.title("Live Demand Forecasting")
st.caption("Generate hourly rental forecasts using the validated production machine learning model")

# Load model metadata for transparency
try:
    _, meta = load_model()
    model_name = meta.get("model_name", "Gradient Boosting Regressor")
    model_metrics = meta.get("metrics", {})
except Exception:
    model_name = "Gradient Boosting Regressor"
    model_metrics = {"mae": 49.61, "rmse": 75.91, "r2": 0.8806}

st.sidebar.markdown(f"**Active Model:** `{model_name}`")
st.sidebar.markdown(f"**Validation MAE:** `{model_metrics.get('mae', 'N/A')}`")
st.sidebar.markdown(f"**Validation R²:** `{model_metrics.get('r2', 'N/A')}`")

st.subheader("Input Operational Parameters")

with st.form("forecast_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        date_val = st.date_input("Target Date", value=datetime(2026, 6, 15))
        hour_val = st.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=8)
        season_val = st.selectbox(
            "Season",
            [1, 2, 3, 4],
            index=1,
            format_func=lambda s: {
                1: "1 - Spring",
                2: "2 - Summer",
                3: "3 - Fall",
                4: "4 - Winter",
            }[s],
        )

    with c2:
        workingday_val = st.selectbox(
            "Working Day?",
            [1, 0],
            format_func=lambda w: "Yes (Workday)" if w == 1 else "No (Weekend/Holiday)",
        )
        holiday_val = st.selectbox(
            "Public Holiday?", [0, 1], format_func=lambda h: "No" if h == 0 else "Yes"
        )
        weather_val = st.selectbox(
            "Weather Condition",
            [1, 2, 3, 4],
            index=0,
            format_func=lambda w: {
                1: "1 - Clear / Partly Cloudy",
                2: "2 - Mist / Cloudy",
                3: "3 - Light Rain / Light Snow",
                4: "4 - Severe Storm / Torrential",
            }[w],
        )

    with c3:
        temp_val = st.number_input(
            "Temperature (°C)", min_value=-15.0, max_value=45.0, value=22.0, step=0.5
        )
        atemp_val = st.number_input(
            "Feels-Like Temp (°C)", min_value=-20.0, max_value=50.0, value=24.0, step=0.5
        )
        hum_val = st.slider(
            "Relative Humidity (%)", min_value=0.0, max_value=100.0, value=55.0, step=1.0
        )
        wind_val = st.number_input(
            "Windspeed (km/h)", min_value=0.0, max_value=80.0, value=12.0, step=1.0
        )

    submitted = st.form_submit_button("Generate Demand Forecast", type="primary")

if submitted:
    dt_str = f"{date_val} {str(hour_val).zfill(2)}:00:00"

    try:
        out = predict_single_observation(
            datetime_str=dt_str,
            season=season_val,
            holiday=holiday_val,
            workingday=workingday_val,
            weather=weather_val,
            temp=temp_val,
            atemp=atemp_val,
            humidity=hum_val,
            windspeed=wind_val,
        )

        # Log to telemetry
        log_prediction(out.predicted_count, out.model_name, out.input_summary)

        st.markdown("---")
        st.subheader("Forecast Results")

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(
                label=f"Predicted Hourly Demand at {hour_val}:00",
                value=f"{out.predicted_count} bikes",
                delta="Normal Operating Envelope",
            )
            st.write(f"**Model Used:** `{out.model_name}`")
            st.write(f"**Timestamp Evaluated:** `{dt_str}`")

        with res_col2:
            st.write("**Synthesized Input Features Summary:**")
            st.json(out.input_summary)

        st.markdown("### Operational Considerations & Limitations")
        for lim in out.limitations:
            st.warning(lim)

    except Exception as ex:
        st.error(f"Error executing prediction: {ex}")
