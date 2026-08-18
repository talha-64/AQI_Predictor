import sys
import os
import tempfile
import logging
from pathlib import Path
import pandas as pd

# Environment setting to bypass Hopsworks Arrow Flight client issues
os.environ["HOPSWORKS_NO_ARROW_FLIGHT"] = "true"

# Compute project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import hopsworks

if os.name == "nt":
    os.environ["TMPDIR"] = tempfile.gettempdir()
    try:
        os.makedirs(r"C:\tmp", exist_ok=True)
    except Exception:
        pass

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)
from src.data_fetcher import fetch_current_air_pollution, fetch_weather_forecast_openmeteo
from src.utils import parse_openweather_record
from src.feature_engineering import engineer_features_and_targets, apply_live_lead_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Longest engineered window is the 168h EMA/lag (epa_aqi_ema_168h, epa_aqi_lag_168h)
HISTORY_HOURS_NEEDED = 250


def fetch_historical_context(fg) -> pd.DataFrame:
    """Reads historical feature context safely without triggering Arrow Flight gRPC errors."""
    logger.info("Fetching historical feature context from Hopsworks...")
    
    # Attempt 1: Online Read via HTTPS REST (port 443, bypassing Arrow Flight gRPC port 5005)
    try:
        logger.info("Attempting online storage read...")
        df_history = fg.read(online=True)
        if not df_history.empty:
            return df_history
    except Exception as e:
        logger.warning(f"Online read unavailable ({e}). Falling back to Flight-disabled offline query...")

    # Attempt 2: Offline Read forcing DuckDB engine & explicitly disabling gRPC Flight client
    try:
        df_history = fg.select_all().read(
            dataframe_type="pandas",
            read_options={
                "use_duckdb": True,
                "arrow_flight_config": {"disable_flight": True}
            }
        )
        return df_history
    except Exception as e_off:
        logger.error(f"Offline feature read failed ({e_off}). Proceeding with empty context history.")
        return pd.DataFrame()


def main():
    logger.info("=== Starting Phase 2: Live Feature Ingestion Pipeline ===")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )
    feature_store = project.get_feature_store()
    fg = feature_store.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    # 1. Read recent historical features safely to calculate EMA/lag/rolling stats correctly
    df_history = fetch_historical_context(fg)
    
    if not df_history.empty and "timestamp_unix" in df_history.columns:
        df_history["timestamp_unix"] = pd.to_numeric(df_history["timestamp_unix"])
        df_history = df_history.sort_values("timestamp_unix").tail(HISTORY_HOURS_NEEDED)

    # 2. Fetch live pollution reading
    logger.info("Fetching current pollution reading...")
    raw_pollution = fetch_current_air_pollution()

    # 3. Fetch weather forecast (covers current hour through 72h+ ahead)
    logger.info("Fetching weather forecast (current hour + next 72h+)...")
    df_forecast = fetch_weather_forecast_openmeteo(forecast_days=4)

    parsed_row = parse_openweather_record(raw_pollution)
    components = parsed_row.pop("raw_components_ugm3", {})
    
    pollution_dt = pd.to_datetime(parsed_row["datetime_utc"])
    if pollution_dt.tzinfo is not None:
        pollution_dt = pollution_dt.tz_localize(None)

    # Match current weather closest to current pollution timestamp
    df_forecast_dt = pd.to_datetime(df_forecast["datetime_utc"])
    if df_forecast_dt.dt.tz is not None:
        df_forecast_dt = df_forecast_dt.dt.tz_localize(None)

    nearest_idx = (df_forecast_dt - pollution_dt).abs().idxmin()
    current_weather = df_forecast.loc[nearest_idx].to_dict()
    current_weather.pop("datetime_utc", None)

    # Merge pollution + raw pollutant components + current weather into live record
    live_row = {**parsed_row, **components, **current_weather}
    df_live = pd.DataFrame([live_row])

    # 4. Combine with recent history and engineer features (is_training=False)
    if not df_history.empty:
        df_combined = pd.concat([df_history, df_live], ignore_index=True)
    else:
        df_combined = df_live

    df_combined["datetime_utc"] = pd.to_datetime(df_combined["datetime_utc"], utc=True)
    df_engineered = engineer_features_and_targets(df_combined, is_training=False)

    latest_feature_row = df_engineered.iloc[[-1]].copy()

    # 5. Overwrite proxy lead columns with actual forecast values
    latest_feature_row = apply_live_lead_features(latest_feature_row, df_forecast)

    # 6. Align output to Feature Group schema
    fg_schema_cols = [f.name for f in fg.features]
    latest_feature_row_aligned = latest_feature_row[
        [col for col in latest_feature_row.columns if col in fg_schema_cols]
    ].copy()

    missing_schema_cols = [c for c in fg_schema_cols if c not in latest_feature_row_aligned.columns]
    if missing_schema_cols:
        logger.warning(f"Columns in Feature Group schema but missing from live row: {missing_schema_cols}")

    # Cast integer columns to float64 if required by Hopsworks schema
    for col in ["month", "is_smog_season"]:
        if col in latest_feature_row_aligned.columns:
            latest_feature_row_aligned[col] = latest_feature_row_aligned[col].astype("float64")

    logger.info("Inserting engineered live feature vector into Hopsworks...")
    fg.insert(latest_feature_row_aligned, write_options={"wait_for_job": False})

    if "epa_aqi" in latest_feature_row.columns:
        logger.info(f"Live EPA AQI: {latest_feature_row['epa_aqi'].iloc[0]}")
    logger.info("=== Phase 2 Completed Successfully! ===")


if __name__ == "__main__":
    main()