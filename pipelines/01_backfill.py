import os
import tempfile
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import hopsworks

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    LATITUDE,
    LONGITUDE,
    TARGET_COLS,
)
from src.data_fetcher import fetch_historical_air_pollution_batch, fetch_historical_weather_openmeteo
from src.feature_engineering import build_full_feature_pipeline

if os.name == "nt":
    os.environ["TMPDIR"] = tempfile.gettempdir()
    try:
        os.makedirs(r"C:\tmp", exist_ok=True)
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sanitize_dataframe_for_hopsworks(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitizes DataFrame to prevent Spark materialization and schema crashes in Hopsworks."""
    df = df.copy()

    df = df.sort_values("timestamp_unix").reset_index(drop=True)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.tz_localize(None)

    # Was: [c for c in df.columns if c.startswith("epa_aqi_lead_")]
    # Now uses the notebook's rolling-average target columns.
    target_cols = [c for c in TARGET_COLS if c in df.columns]

    # Drop rows where targets are legitimately missing (end of series) BEFORE any fill
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    # Only fill feature columns, never targets
    feature_cols = [c for c in df.columns if c not in target_cols]
    df[feature_cols] = df[feature_cols].bfill().ffill()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fill_numeric = [c for c in numeric_cols if c not in target_cols]
    df[fill_numeric] = df[fill_numeric].fillna(0.0)

    df = df.dropna(subset=["timestamp_unix", "datetime_utc"]).reset_index(drop=True)
    df["timestamp_unix"] = df["timestamp_unix"].astype("int64")

    for col in numeric_cols:
        if col != "timestamp_unix":
            df[col] = df[col].astype("float64")

    if "primary_pollutant" in df.columns:
        df["primary_pollutant"] = df["primary_pollutant"].astype(str)

    return df


def main(backfill_days: int):
    logger.info("=== Phase 3: Historical Data Backfill ===")

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=backfill_days)

    end_unix = int(now_utc.timestamp())
    start_unix = int(start_utc.timestamp())

    logger.info(f"Target Window: {start_utc.isoformat()} -> {now_utc.isoformat()} ({backfill_days} days)")

    # 1. Fetch historical pollution
    raw_payload = fetch_historical_air_pollution_batch(
        start_unix=start_unix,
        end_unix=end_unix,
        lat=LATITUDE,
        lon=LONGITUDE,
        chunk_days=30,
    )

    record_count = len(raw_payload.get("list", []))
    if record_count == 0:
        logger.error("No pollution records retrieved. Aborting backfill pipeline.")
        sys.exit(1)

    # 2. Fetch historical weather (Open-Meteo ERA5 reanalysis, no API key needed)
    logger.info("Fetching historical weather data for the same window (Open-Meteo)...")
    weather_records = fetch_historical_weather_openmeteo(
        start_unix=start_unix,
        end_unix=end_unix,
        lat=LATITUDE,
        lon=LONGITUDE,
    )

    if len(weather_records) == 0:
        logger.warning(
            "No weather records retrieved — proceeding with pollution-only features. "
            "Model quality will suffer without meteorological features."
        )

    # 3. Build and engineer features (pollution + weather merged, reindexed to hourly,
    #    notebook-aligned feature set + rolling-average targets)
    df_features = build_full_feature_pipeline(
        raw_payload,
        weather_records=weather_records,
        is_training=True,
    )

    # 4. Sanitize for Spark ingestion
    df_clean = sanitize_dataframe_for_hopsworks(df_features)
    logger.info(
        f"Sanitized DataFrame shape for insertion: {df_clean.shape} | Nulls remaining: {df_clean.isnull().sum().sum()}"
    )

    # 5. Authenticate with Hopsworks
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )
    feature_store = project.get_feature_store()

    # 6. Re-get or create the Feature Group
    aqi_fg = feature_store.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp_unix"],
        event_time="datetime_utc",
        description="Hourly raw pollutant readings, weather features, cyclical time features, EMA/lag/volatility statistics, and multi-horizon rolling-average target variables.",
        online_enabled=True,
        time_travel_format="HUDI",
    )

    logger.info("Inserting sanitized DataFrame into Hopsworks...")
    aqi_fg.insert(
        features=df_clean,
        write_options={"wait_for_job": False},
    )

    logger.info("=== Phase 3: Historical Data Backfill Triggered Successfully! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Was hardcoded to 730 days (2 years). The notebook trained on ~5 years
    # of history (since Nov 2020) to reach the reported R² numbers — default
    # here matches that. Override with --days if you want a shorter backfill.
    parser.add_argument("--days", type=int, default=1900, help="How many days of history to backfill (~5yr default)")
    args = parser.parse_args()
    main(args.days)
