"""
Streamlit dashboard for the AQI Predictor.
Run with: streamlit run dashboard/app.py
"""
import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from src import config
# NOTE: the dashboard no longer imports `hopsworks` or talks to it at all --
# both predictions and history now come from local files written hourly by
# pipelines/04_batch_inference.py (see load_predictions/load_recent_history
# below). This also removes the HOPSWORKS_NO_ARROW_FLIGHT env var that was
# here before -- it wasn't a real recognized setting in the hsfs client, so
# it wasn't doing anything anyway. If you re-add any live Hopsworks read
# here later, reuse the online-first / use_hive-fallback pattern from
# pipelines/04_batch_inference.py's fetch_recent_features -- don't
# reintroduce plain fg.read() or the "use_duckdb"/disable_flight combo,
# both route through the Arrow Flight service that's been unreliable.

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


@st.cache_data(ttl=60, show_spinner="Loading latest predictions...")
def load_predictions():
    json_path = config.PREDICTIONS_PATH

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            return data, None
        except Exception as e:
            logger.error(f"Error reading local predictions JSON: {e}")

    return None, "File 'predictions/latest_predictions.json' not found. Ensure batch inference has run."


@st.cache_data(ttl=1800, show_spinner="Loading historical trend...")
def load_recent_history():
    """
    Reads the small local CSV (datetime_utc, epa_aqi) maintained by
    pipelines/04_batch_inference.py's update_recent_history_snapshot(),
    instead of querying Hopsworks live.

    This used to query the full aqi_hourly_fg feature group (60+ columns)
    directly from the dashboard on every page load. Locally that was slow
    (MySQL lock timeouts, then Arrow Flight retries/timeouts); on Streamlit
    Cloud it hung indefinitely, since Arrow Flight's gRPC connection
    commonly gets silently blocked by restricted outbound networks rather
    than cleanly rejected -- so it just spins forever instead of erroring.
    Reading a file that's already in the repo sidesteps both problems, and
    the file itself is maintained incrementally (one row appended per hour,
    not re-queried), so Hopsworks only gets touched by the pipeline, never
    by the dashboard -- worth keeping in mind on the free tier.
    """
    csv_path = config.RECENT_HISTORY_PATH

    if not os.path.exists(csv_path):
        logger.warning(f"'{csv_path}' not found -- has pipelines/04_batch_inference.py run yet?")
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
        return df.sort_values("datetime_utc")
    except Exception as e:
        logger.error(f"Error reading local history CSV: {e}")
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
            st.error(f"Could not load predictions: {error}")
        else:
            st.warning("No predictions available yet. Run `pipelines/04_batch_inference.py` first.")
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
    history = load_recent_history()
    if not history.empty:
        st.line_chart(history.set_index("datetime_utc")["epa_aqi"])
    else:
        st.info("History currently unavailable.")


if __name__ == "__main__":
    main()