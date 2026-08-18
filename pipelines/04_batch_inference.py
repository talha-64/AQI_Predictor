import sys
import os
import tempfile
import joblib
import logging
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if os.name == "nt":
    os.environ["TMPDIR"] = tempfile.gettempdir()
    try:
        os.makedirs(r"C:\tmp", exist_ok=True)
    except Exception:
        pass

import hopsworks
import pandas as pd

from src.config import (
    HOPSWORKS_API_KEY,
    HOPSWORKS_PROJECT_NAME,
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    MODEL_NAME_TEMPLATE,
    NON_FEATURE_COLS,
    HORIZONS,
)
from src.feature_engineering import get_final_feature_list
from src.model_trainer import feats_used_per_horizon, predict_next_3_days, load_models_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PREDICTIONS_FG_NAME = "aqi_predictions_fg"
PREDICTIONS_FG_VERSION = 1

AQI_CATEGORIES = [
    ("Good", 0, 50),
    ("Moderate", 51, 100),
    ("Unhealthy for Sensitive Groups", 101, 150),
    ("Unhealthy", 151, 200),
    ("Very Unhealthy", 201, 300),
    ("Hazardous", 301, 500),
]
HAZARD_ALERT_THRESHOLD = 200

# History depth needed for the longest engineered window (168h EMA/lag), with margin.
HISTORY_HOURS_NEEDED = 250


def aqi_category(value: float) -> str:
    for name, low, high in AQI_CATEGORIES:
        if low <= value <= high:
            return name
    return "Hazardous"


def fetch_latest_features(fg) -> pd.DataFrame:
    """
    Reads recent feature rows from the Feature Group. Tries the ONLINE store
    first (goes through Hopsworks' REST API to RonDB, not the offline Arrow
    Flight/DuckDB service) -- this is the path that actually succeeded in
    testing, just somewhat slow (~2-3 min). Falls back to the Hive/Spark
    backend, which is a genuinely different code path from Arrow Flight
    (per Hopsworks docs: read_options={"use_hive": True}), not another
    route through the same service.

    Deliberately does NOT attempt fg.read() (default) or any
    "use_duckdb"/arrow-flight-disable combination -- those all route through
    the offline Arrow Flight/DuckDB query service, which has a known
    server-side crash ("release unlocked lock") as of Aug 2026. If that
    service gets fixed on Hopsworks' end later, the Hive fallback below can
    be replaced with the plain offline fg.read() for speed.
    """
    try:
        logger.info("Attempting online feature group read...")
        df_features = fg.read(online=True)
    except Exception as e:
        logger.warning(f"Online read failed ({e}). Falling back to Hive/Spark backend...")
        df_features = fg.read(read_options={"use_hive": True})

    if df_features is None or df_features.empty:
        raise ValueError("Retrieved empty DataFrame from Feature Group -- cannot run inference.")

    if "timestamp_unix" in df_features.columns:
        df_features["timestamp_unix"] = pd.to_numeric(df_features["timestamp_unix"])
        df_features = df_features.sort_values("timestamp_unix").tail(HISTORY_HOURS_NEEDED).reset_index(drop=True)
    else:
        df_features = df_features.tail(HISTORY_HOURS_NEEDED).reset_index(drop=True)

    return df_features


def load_models_from_registry(mr):
    """Downloads and loads all 3 horizon models from the Hopsworks Model
    Registry. Falls back to the local models/ dir (written by
    03_training_pipeline.py's save_models_local) if the registry download
    fails or a model.pkl can't be loaded from the downloaded directory."""
    models = {}
    try:
        for idx in HORIZONS:
            model_name = MODEL_NAME_TEMPLATE.format(horizon=idx)
            logger.info(f"Fetching registered model '{model_name}'...")
            model_meta = mr.get_model(model_name, version=1)
            model_dir = model_meta.download()

            model_path = os.path.join(model_dir, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"'model.pkl' not found in downloaded artifact at {model_dir}")

            models[f"model_day{idx}"] = joblib.load(model_path)

        logger.info("All 3 horizon models loaded from Hopsworks Model Registry.")
        return models
    except Exception as e:
        logger.warning(f"Could not load models from Hopsworks Model Registry ({e}). Falling back to local models/ dir.")
        local_models, _, _ = load_models_local(in_dir=os.path.join(ROOT_DIR, "models"))
        return local_models


def main():
    logger.info("=== Starting Batch Inference Pipeline ===")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )
    feature_store = project.get_feature_store()
    mr = project.get_model_registry()

    # 1. Load all 3 horizon models
    models = load_models_from_registry(mr)

    # 2. Fetch recent features from Hopsworks (need history for EMA/lag context,
    #    not just the single latest row)
    logger.info(f"Fetching recent feature records from '{FEATURE_GROUP_NAME}_v{FEATURE_GROUP_VERSION}'...")
    fg = feature_store.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df_features = fetch_latest_features(fg)

    # Forward-fill short feature gaps only -- never fabricate epa_aqi or any target
    fill_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    df_features[fill_cols] = df_features[fill_cols].ffill()

    latest_row = df_features.tail(1).reset_index(drop=True)
    latest_ts = str(latest_row["datetime_utc"].iloc[0])
    current_aqi = float(latest_row["epa_aqi"].iloc[0])
    logger.info(f"Latest observation timestamp: {latest_ts} | Current EPA AQI: {current_aqi}")

    # 3. Reconstruct the exact same pruned feature list used at training time.
    #    get_final_feature_list is a pure function of the column list, so as
    #    long as feature_engineering.py hasn't changed since training, this
    #    reproduces the training-time feature set without needing to ship
    #    feature-list metadata alongside the model artifacts.
    all_feature_cols = [
        c for c in df_features.columns
        if c not in NON_FEATURE_COLS and str(df_features[c].dtype) in ["float64", "int64", "int32", "float32"]
    ]
    final_features = get_final_feature_list(all_feature_cols)
    feats_per_horizon = feats_used_per_horizon(final_features)
    logger.info(f"Using {len(final_features)} features for inference.")

    # 4. Chained + persistence-blended 3-day forecast
    logger.info("Running batch inference across horizons...")
    predictions = predict_next_3_days(models, latest_row, feats_per_horizon)

    now_utc = datetime.now(timezone.utc)
    pred_data = {
        "city": "lahore",
        "generated_at": latest_ts,
        "created_at_unix": int(now_utc.timestamp() * 1000),
        "current_aqi": round(current_aqi, 1),
        "current_category": aqi_category(current_aqi),
        "day1_aqi": float(predictions[1]),
        "day1_category": aqi_category(predictions[1]),
        "day1_hazardous": bool(predictions[1] >= HAZARD_ALERT_THRESHOLD),
        "day2_aqi": float(predictions[2]),
        "day2_category": aqi_category(predictions[2]),
        "day2_hazardous": bool(predictions[2] >= HAZARD_ALERT_THRESHOLD),
        "day3_aqi": float(predictions[3]),
        "day3_category": aqi_category(predictions[3]),
        "day3_hazardous": bool(predictions[3] >= HAZARD_ALERT_THRESHOLD),
    }
    pred_df = pd.DataFrame([pred_data])

    # 5. Insert predictions into Hopsworks Feature Group
    # time_travel_format="HUDI" is required here -- without it, this client
    # defaults to DELTA, which fails on environments without the delta
    # library installed (no Spark).
    logger.info(f"Writing prediction record to Hopsworks feature group '{PREDICTIONS_FG_NAME}'...")
    pred_fg = feature_store.get_or_create_feature_group(
        name=PREDICTIONS_FG_NAME,
        version=PREDICTIONS_FG_VERSION,
        primary_key=["city", "generated_at"],
        event_time="created_at_unix",
        description="Stores live multi-horizon AQI predictions generated by batch inference.",
        time_travel_format="HUDI",
    )
    pred_fg.insert(pred_df, write_options={"wait_for_job": False})

    logger.info("\n" + "=" * 50)
    logger.info("           3-DAY AIR QUALITY FORECAST              ")
    logger.info("=" * 50)
    logger.info(f"Current AQI: {pred_data['current_aqi']} ({pred_data['current_category']})")
    for h in HORIZONS:
        aqi_val = pred_data[f"day{h}_aqi"]
        cat_val = pred_data[f"day{h}_category"]
        flag = "  [HAZARDOUS ALERT]" if pred_data[f"day{h}_hazardous"] else ""
        logger.info(f"Day {h}: {aqi_val:.1f} ({cat_val}){flag}")
    logger.info("=" * 50)
    logger.info("=== Batch Inference Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    main()