"""
Streamlit dashboard for the AQI Predictor.
Run with: streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st
import hopsworks

from src import config, model_trainer
from src.utils import get_logger, read_features

logger = get_logger("dashboard")

st.set_page_config(page_title="Lahore AQI Predictor", page_icon="🌫️", layout="wide")

CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


@st.cache_data(ttl=300)
def load_predictions():
    try:
        project = hopsworks.login(
            project=config.HOPSWORKS_PROJECT_NAME,
            api_key_value=config.HOPSWORKS_API_KEY,
        )
        fs = project.get_feature_store()
        pred_fg = fs.get_feature_group("aqi_predictions_fg", version=1)
        df_preds = pred_fg.read()
        
        if df_preds.empty:
            return None
            
        # Get the latest prediction row by event timestamp
        latest_row = df_preds.sort_values("created_at_unix").iloc[-1]
        
        return {
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
            }
        }
    except Exception as e:
        logger.error(f"Error reading predictions from Hopsworks: {e}")
        return None


@st.cache_data(ttl=3600)
def load_recent_history(hours: int = 24 * 14):
    df = read_features()
    return df.sort_values("datetime_utc").tail(hours)[["datetime_utc", "epa_aqi"]]


@st.cache_resource
def load_models_and_features():
    return model_trainer.load_models()


def render_forecast_card(col, label, day_data, is_current=False):
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
            st.error("⚠️ Hazardous level — limit outdoor exposure")


def main():
    st.title("🌫️ Lahore Air Quality Index — 3-Day Forecast")
    st.caption("Chained LightGBM models blended with persistence · trained on 5 years of hourly AQI + weather data")

    predictions = load_predictions()
    if predictions is None:
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
        st.error("🚨 Hazardous AQI levels forecasted in the next 3 days. Consider limiting outdoor activity.")

    st.divider()

    st.subheader("Recent AQI trend (last 14 days)")
    try:
        history = load_recent_history()
        st.line_chart(history.set_index("datetime_utc")["epa_aqi"])
    except Exception as e:
        st.info(f"History unavailable: {e}")

    st.divider()

    st.subheader("What's driving the Day 1 forecast? (SHAP)")
    render_shap_section()


def render_shap_section():
    try:
        import shap

        models, feature_cols, feats_per_horizon = load_models_and_features()
        history = read_features().sort_values("datetime_utc")
        sample = history[feats_per_horizon[1]].dropna().tail(500)

        if len(sample) < 20:
            st.info("Not enough history yet for a SHAP explanation.")
            return

        explainer = shap.TreeExplainer(models["model_day1"])
        shap_values = explainer.shap_values(sample)

        importance = pd.Series(
            abs(shap_values).mean(axis=0), index=feats_per_horizon[1]
        ).sort_values(ascending=False).head(10)

        st.bar_chart(importance)
        st.caption("Mean absolute SHAP value — top 10 features driving the Day 1 forecast")
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")


if __name__ == "__main__":
    main()