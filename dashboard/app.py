"""
Streamlit dashboard for the AQI Predictor.
Run with: streamlit run dashboard/app.py
"""
import os
import sys
import logging
from pathlib import Path

# Force-disable unstable Arrow Flight gRPC before Hopsworks imports
os.environ["HOPSWORKS_NO_ARROW_FLIGHT"] = "true"

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st
import hopsworks

from src import config, model_trainer

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


@st.cache_resource(ttl=3600)
def get_hopsworks_project():
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )


def read_feature_group_fast(fg):
    """Fast offline feature reader using DuckDB over REST.
    
    Avoids Hive/Spark fallbacks which cause multi-minute hangs in cloud apps.
    """
    try:
        # 1. Primary: Try direct online store query (sub-second)
        return fg.read(online=True)
    except Exception as e:
        logger.warning(f"Online read failed ({e}). Falling back to DuckDB REST engine...")
        # 2. Fallback: Fast DuckDB engine over HTTP REST instead of slow Hive/Spark
        return fg.select_all().read(
            dataframe_type="pandas",
            read_options={
                "use_duckdb": True,
                "arrow_flight_config": {"disable_flight": True},
            },
        )


@st.cache_data(ttl=300, show_spinner="Loading latest predictions...")
def load_predictions():
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        pred_fg = fs.get_feature_group("aqi_predictions_fg", version=1)
        
        df_preds = read_feature_group_fast(pred_fg)

        if df_preds is None or df_preds.empty:
            return None, None

        latest_row = df_preds.sort_values("created_at_unix").iloc[-1]

        result = {
            "generated_at": str(latest_row["generated_at"]),
            "current_aqi": float(latest_row["current_aqi"]),
            "current_category": str(latest_row["current_category"]),
            "forecast": {
                "day_1": {
                    "aqi": float(latest_row["day1_aqi"]),
                    "category": str(latest_row["day1_category"]),
                    "hazardous_alert": bool(latest_row["day1_hazardous"]),
                },
                "day_2": {
                    "aqi": float(latest_row["day2_aqi"]),
                    "category": str(latest_row["day2_category"]),
                    "hazardous_alert": bool(latest_row["day2_hazardous"]),
                },
                "day_3": {
                    "aqi": float(latest_row["day3_aqi"]),
                    "category": str(latest_row["day3_category"]),
                    "hazardous_alert": bool(latest_row["day3_hazardous"]),
                },
            },
        }
        return result, None
    except Exception as e:
        logger.error(f"Error reading predictions from Hopsworks: {e}")
        return None, str(e)


@st.cache_data(ttl=1800, show_spinner="Loading historical trend...")
def load_recent_history(hours: int = 24 * 14):
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        fg = fs.get_feature_group(config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
        
        # Read with DuckDB over REST
        df = read_feature_group_fast(fg)
        
        if "datetime_utc" in df.columns:
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
            df = df.sort_values("datetime_utc")
            return df.tail(hours)[["datetime_utc", "epa_aqi"]]
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
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
        if error:
            st.error(f"Could not read predictions from Hopsworks: {error}")
        else:
            st.warning("No predictions available in Hopsworks yet. Run `pipelines/04_batch_inference.py` first.")
        return

    st.caption(f"Last updated: {predictions['generated_at']} UTC")

    cols = st.columns(4)
    render_forecast_card(cols[0], "Current", {
        "aqi": predictions["current_aqi"], "category": predictions["current_category"], "hazardous_alert": False
    })
    for h, col in zip(config.HORIZONS, cols[1:]):
        render_forecast_card(col, f"Day {h}", predictions["forecast"][f"day_{h}"])

    if any(predictions["forecast"][f"day_{h}"]["hazardous_alert"] for h in config.HORIZONS):
        st.error("Hazardous AQI levels forecasted in the next 3 days. Consider limiting outdoor activity.")

    st.divider()

    st.subheader("Recent AQI trend (last 14 days)")
    history = load_recent_history()
    if not history.empty:
        st.line_chart(history.set_index("datetime_utc")["epa_aqi"])
    else:
        st.info("History currently unavailable.")


if __name__ == "__main__":
    main()