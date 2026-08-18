"""
Streamlit dashboard for the AQI Predictor.
Run with: streamlit run dashboard/app.py
"""
import os
import sys
import logging
from pathlib import Path

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


@st.cache_resource
def get_hopsworks_project():
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )


def read_feature_group_robust(fg):
    """
    Same fallback pattern used in pipelines/04_batch_inference.py: try the
    online store first (goes through Hopsworks' REST API, not the offline
    Arrow Flight/DuckDB service), then fall back to the Hive/Spark backend.
    Deliberately avoids the plain fg.read() default and any
    "use_duckdb"/arrow-flight-disable options -- those route through the
    offline Arrow Flight/DuckDB query service, which has a known
    server-side crash ("release unlocked lock") as of Aug 2026.
    """
    try:
        return fg.read(online=True)
    except Exception as e:
        logger.warning(f"Online read failed ({e}). Falling back to Hive/Spark backend...")
        return fg.read(read_options={"use_hive": True})


@st.cache_data(ttl=300)
def load_predictions():
    try:
        project = get_hopsworks_project()
        fs = project.get_feature_store()
        pred_fg = fs.get_feature_group("aqi_predictions_fg", version=1)
        df_preds = read_feature_group_robust(pred_fg)

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
        # Was: silently returned None with no visible reason. Now the actual
        # exception comes back so the UI can show it instead of a generic
        # "no predictions" message that looks the same whether the feature
        # group is genuinely empty or Hopsworks itself is failing.
        return None, str(e)


@st.cache_data(ttl=3600)
def load_recent_history(hours: int = 24 * 14):
    project = get_hopsworks_project()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(config.FEATURE_GROUP_NAME, version=config.FEATURE_GROUP_VERSION)
    df = read_feature_group_robust(fg)
    df = df.sort_values("datetime_utc")
    return df.tail(hours)[["datetime_utc", "epa_aqi"]]


@st.cache_resource
def load_models_and_features():
    # Was: model_trainer.load_models() -- that function doesn't exist.
    # The real function (see src/model_trainer.py) is load_models_local(),
    # which reads from the local models/ directory written by
    # 03_training_pipeline.py's save_models_local(). Only used by the
    # (currently disabled) SHAP section below.
    return model_trainer.load_models_local(in_dir=os.path.join(os.path.dirname(__file__), "..", "models"))


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
            st.error("Hazardous level -- limit outdoor exposure")


def main():
    st.title("Lahore Air Quality Index -- 3-Day Forecast")
    st.caption("Chained LightGBM models blended with persistence -- trained on 5 years of hourly AQI + weather data")

    predictions, error = load_predictions()
    if predictions is None:
        if error:
            st.error(f"Could not read predictions from Hopsworks: {error}")
            st.caption("If this mentions 'release unlocked lock' or an Arrow Flight error, this is a known "
                       "server-side issue with Hopsworks' offline query service -- rerun in a few minutes, "
                       "or check that pipelines/04_batch_inference.py's fallback logic is actually being hit.")
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
    try:
        history = load_recent_history()
        st.line_chart(history.set_index("datetime_utc")["epa_aqi"])
    except Exception as e:
        st.info(f"History unavailable: {e}")

    st.divider()

    st.subheader("What's driving the Day 1 forecast? (SHAP)")
    # render_shap_section()


# def render_shap_section():
#     try:
#         import shap
#
#         models, feature_cols, feats_per_horizon = load_models_and_features()
#         history = load_recent_history(hours=24 * 30)
#         sample = history[feats_per_horizon[1]].dropna().tail(500)
#
#         if len(sample) < 20:
#             st.info("Not enough history yet for a SHAP explanation.")
#             return
#
#         explainer = shap.TreeExplainer(models["model_day1"])
#         shap_values = explainer.shap_values(sample)
#
#         importance = pd.Series(
#             abs(shap_values).mean(axis=0), index=feats_per_horizon[1]
#         ).sort_values(ascending=False).head(10)
#
#         st.bar_chart(importance)
#         st.caption("Mean absolute SHAP value -- top 10 features driving the Day 1 forecast")
#     except Exception as e:
#         st.info(f"SHAP explanation unavailable: {e}")


if __name__ == "__main__":
    main()