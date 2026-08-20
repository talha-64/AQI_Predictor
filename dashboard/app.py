"""
Streamlit dashboard for the AQI Predictor.
Run with: streamlit run dashboard/app.py
"""
import os
import sys
import json
import logging
import requests
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dashboard")

st.set_page_config(page_title="Lahore AQI Predictor", page_icon="AQI", layout="wide")

CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


@st.cache_data(ttl=60, show_spinner="Loading predictions...")
def load_predictions():
    json_path = os.path.join(os.path.dirname(__file__), "..", "data", "latest_predictions.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return data, None
        except Exception as e:
            logger.error(f"Error reading local predictions JSON: {e}")

    return None, "File 'data/latest_predictions.json' not found. Ensure batch inference has run."


@st.cache_data(ttl=1800, show_spinner="Loading historical trend...")
def fetch_recent_history_openmeteo():
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=31.5497&longitude=74.3436&past_days=14&hourly=us_aqi"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return pd.DataFrame({
                "datetime_utc": pd.to_datetime(data["hourly"]["time"]),
                "epa_aqi": data["hourly"]["us_aqi"]
            })
    except Exception as e:
        logger.error(f"Error fetching history from Open-Meteo: {e}")
    return pd.DataFrame()


def render_forecast_card(col, label, day_data):
    color = CATEGORY_COLORS.get(day_data["category"], "#888888")
    with col:
        st.markdown(f"**{label}**")
        st.markdown(
            f"""<div style="background-color:{color}22;border:2px solid {color};
            border-radius:12px;padding:16px;text-align:center;">
            <div style="font-size:36px;font-weight:bold;">{day_data['aqi']:.0f}</div>
            <div style="font-size:14px;">{day_data['category']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        if day_data.get("hazardous_alert"):
            st.error("Hazardous level -- limit outdoor exposure")


def main():
    st.title("Lahore Air Quality Index -- 3-Day Forecast")
    st.caption("Chained LightGBM models blended with persistence -- trained on 5 years of hourly AQI + weather data")

    predictions, error = load_predictions()
    if predictions is None:
        st.error(f"Could not load predictions: {error}")
        return

    st.caption(f"Last updated: {predictions.get('generated_at', 'N/A')} UTC")

    cols = st.columns(4)
    render_forecast_card(cols[0], "Current", {
        "aqi": predictions["current_aqi"], 
        "category": predictions["current_category"], 
        "hazardous_alert": False
    })
    for h, col in zip(config.HORIZONS, cols[1:]):
        render_forecast_card(col, f"Day {h}", predictions["forecast"][f"day_{h}"])

    if any(predictions["forecast"][f"day_{h}"]["hazardous_alert"] for h in config.HORIZONS):
        st.error("Hazardous AQI levels forecasted in the next 3 days. Consider limiting outdoor activity.")

    st.divider()

    st.subheader("Recent AQI trend (last 14 days)")
    history = fetch_recent_history_openmeteo()
    if not history.empty:
        st.line_chart(history.set_index("datetime_utc")["epa_aqi"])
    else:
        st.info("Historical trend unavailable.")


if __name__ == "__main__":
    main()