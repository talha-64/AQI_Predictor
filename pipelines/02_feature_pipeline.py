import sys
import os
import tempfile
import logging
from pathlib import Path
import pandas as pd

os.environ["HOPSWORKS_NO_ARROW_FLIGHT"] = "true"

import hopsworks

if os.name == "nt":
    os.environ["TMPDIR"] = tempfile.gettempdir()
    try:
        os.makedirs(r"C:\tmp", exist_ok=True)
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


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

# Longest engineered window is the 168h EMA/lag (epa_aqi_ema_168h, epa_aqi_lag_168h).
# Was tail(100) — too short, would've silently produced NaN for every 168h feature
# on every live row regardless of how much history sat in the feature store.
HISTORY_HOURS_NEEDED = 250


def main():
    logger.info("=== Starting Phase 2: Live Feature Ingestion Pipeline ===")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )
    feature_store = project.get_feature_store()
    fg = feature_store.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    # 1. Read recent historical features to calculate EMA/lag/rolling stats correctly
    # Disables Arrow Flight service to fall back to direct HTTP/DuckDB streaming
    df_history = fg.read()
    df_history = df_history.sort_values("timestamp_unix").tail(HISTORY_HOURS_NEEDED)

    # 2. Fetch live pollution reading
    logger.info("Fetching current pollution reading...")
    raw_pollution = fetch_current_air_pollution()

    # 3. Fetch weather forecast (covers current hour through 72h+ ahead).
    #    Was: fetch_24h_weather_forecast(), which only returned forecast_*_24h
    #    values and NEVER populated the live row's own current temp/pressure/
    #    wind/rain — meaning every live-inserted row had those as NaN, which
    #    would have broken wind vectors, stagnation_index, pressure_delta_24h,
    #    and rain_cumsum_72h the moment enough live rows accumulated to matter.
    logger.info("Fetching weather forecast (current hour + next 72h+)...")
    df_forecast = fetch_weather_forecast_openmeteo(forecast_days=4)

    parsed_row = parse_openweather_record(raw_pollution)
    components = parsed_row.pop("raw_components_ugm3", {})
    pollution_dt = pd.to_datetime(parsed_row["datetime_utc"]).tz_localize(None) if pd.to_datetime(parsed_row["datetime_utc"]).tzinfo else pd.to_datetime(parsed_row["datetime_utc"])

    nearest_idx = (df_forecast["datetime_utc"] - pollution_dt).abs().idxmin()
    current_weather = df_forecast.loc[nearest_idx].to_dict()
    current_weather.pop("datetime_utc", None)

    # Merge pollution + raw pollutant components + current weather into one live row
    live_row = {**parsed_row, **components, **current_weather}
    df_live = pd.DataFrame([live_row])

    # 4. Combine with recent history and engineer features (is_training=False —
    #    no target computation attempted for a row with no future data yet)
    df_combined = pd.concat([df_history, df_live], ignore_index=True)
    df_combined["datetime_utc"] = pd.to_datetime(df_combined["datetime_utc"], utc=True)
    df_engineered = engineer_features_and_targets(df_combined, is_training=False)

    latest_feature_row = df_engineered.iloc[[-1]].copy()

    # 5. Overwrite the proxy lead columns (NaN at the live edge anyway) with
    #    real forecast values
    latest_feature_row = apply_live_lead_features(latest_feature_row, df_forecast)

    # 6. Align to Feature Group schema
    fg_schema_cols = [f.name for f in fg.features]
    latest_feature_row_aligned = latest_feature_row[
        [col for col in latest_feature_row.columns if col in fg_schema_cols]
    ]

    missing_schema_cols = [c for c in fg_schema_cols if c not in latest_feature_row_aligned.columns]
    if missing_schema_cols:
        logger.warning(f"Columns in Feature Group schema but missing from this live row: {missing_schema_cols}")

    logger.info("Inserting engineered live feature vector into Hopsworks...")
    # Cast integer columns to float64 if the target schema expects double
    for col in ["month", "is_smog_season"]:
        if col in latest_feature_row_aligned.columns:
            latest_feature_row_aligned[col] = latest_feature_row_aligned[col].astype("float64")

    fg.insert(latest_feature_row_aligned, write_options={"wait_for_job": False})

    logger.info(f"Live EPA AQI: {latest_feature_row['epa_aqi'].iloc[0]}")
    logger.info("=== Phase 2 Completed Successfully! ===")


if __name__ == "__main__":
    main()
