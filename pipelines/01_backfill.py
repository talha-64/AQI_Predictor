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
    """Sanitizes DataFrame to strictly conform to Apache Spark / Hudi schema rules."""
    df = df.copy()

    # 1. Clean & standardize column names (lowercase, no spaces, no special characters)
    df.columns = [c.lower().replace("-", "_").replace(" ", "_").replace(".", "_") for c in df.columns]

    # 2. Convert event time strictly to naive UTC timestamp (Spark TimestampType)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.tz_localize(None)

    # 3. Target columns cleanup
    target_cols = [c.lower() for c in TARGET_COLS if c.lower() in df.columns]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    # 4. Fill missing values in feature columns
    feature_cols = [c for c in df.columns if c not in target_cols]
    df[feature_cols] = df[feature_cols].bfill().ffill()

    # 5. Drop any row missing primary key or event time
    df = df.dropna(subset=["timestamp_unix", "datetime_utc"]).reset_index(drop=True)

    # 6. Enforce strict type conversions (int64 for keys, float32 for metrics)
    df["timestamp_unix"] = df["timestamp_unix"].astype("int64")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "timestamp_unix":
            df[col] = df[col].astype("float32")

    # Replace any infinite values generated during rolling calculations
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)

    return df


def main(backfill_days: int):
    logger.info("=== Phase 3: Historical Data Backfill ===")

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=backfill_days)

    end_unix = int(now_utc.timestamp())
    start_unix = int(start_utc.timestamp())

    logger.info(f"Target Window: {start_utc.isoformat()} -> {now_utc.isoformat()} ({backfill_days} days)")

    # 1. Fetch pollution
    raw_payload = fetch_historical_air_pollution_batch(
        start_unix=start_unix,
        end_unix=end_unix,
        lat=LATITUDE,
        lon=LONGITUDE,
        chunk_days=30,
    )

    if len(raw_payload.get("list", [])) == 0:
        logger.error("No pollution records retrieved. Aborting backfill pipeline.")
        sys.exit(1)

    # 2. Fetch weather
    logger.info("Fetching historical weather data from Open-Meteo...")
    weather_records = fetch_historical_weather_openmeteo(
        start_unix=start_unix,
        end_unix=end_unix,
        lat=LATITUDE,
        lon=LONGITUDE,
    )

    # 3. Build features
    df_features = build_full_feature_pipeline(
        raw_payload,
        weather_records=weather_records,
        is_training=True,
    )

    # 4. Sanitize DataFrame
    df_clean = sanitize_dataframe_for_hopsworks(df_features)
    logger.info(f"Sanitized DataFrame shape for insertion: {df_clean.shape} | Nulls: {df_clean.isnull().sum().sum()}")

   # 5. Authenticate with Hopsworks
    project = hopsworks.login(
        host="eu-west.cloud.hopsworks.ai",
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )
    feature_store = project.get_feature_store()

    # 6. Create or retrieve Feature Group
    aqi_fg = feature_store.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp_unix"],
        event_time="datetime_utc",
        description="Hourly raw pollutant readings, weather features, cyclical time features, and multi-horizon targets.",
        online_enabled=True,
        time_travel_format="HUDI",  # Mandatory for Hopsworks 5.0
    )

    logger.info("Inserting sanitized DataFrame into Hopsworks...")
    aqi_fg.insert(
        features=df_clean,
        write_options={"wait_for_job": False},
    )

    logger.info("=== Phase 3: Historical Data Backfill Completed Successfully! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=1900, help="How many days of history to backfill")
    args = parser.parse_args()
    main(args.days)