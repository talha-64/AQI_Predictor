import sys
import os
import json
import joblib
import logging
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import hopsworks
import numpy as np
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

PREDICTIONS_PATH = os.path.join(ROOT_DIR, "predictions", "latest_predictions.json")

# Same category thresholds a dashboard needs — kept local to this script
# rather than added to src/utils.py again, to avoid another round of
# re-uploading that file. Move into utils.py if the dashboard also needs it.
AQI_CATEGORIES = [
    ("Good", 0, 50),
    ("Moderate", 51, 100),
    ("Unhealthy for Sensitive Groups", 101, 150),
    ("Unhealthy", 151, 200),
    ("Very Unhealthy", 201, 300),
    ("Hazardous", 301, 500),
]
HAZARD_ALERT_THRESHOLD = 200  # alert at/above "Unhealthy"


def aqi_category(value: float) -> str:
    for name, low, high in AQI_CATEGORIES:
        if low <= value <= high:
            return name
    return "Hazardous"


def load_models_from_registry(mr):
    """Downloads and loads all 3 horizon models from the Hopsworks Model
    Registry. Falls back to the local models/ dir (written by
    03_training_pipeline.py's save_models_local) if the registry is
    unreachable — useful when running 03 and 04 back-to-back in the same
    workflow job, where the local files are already sitting on disk."""
    models = {}
    try:
        for idx in HORIZONS:
            model_name = MODEL_NAME_TEMPLATE.format(horizon=idx)
            logger.info(f"Fetching registered model '{model_name}'...")
            model_meta = mr.get_model(model_name, version=1)
            model_dir = model_meta.download()
            models[f"model_day{idx}"] = joblib.load(os.path.join(model_dir, "model.pkl"))
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

    # 2. Read latest feature row from the Feature Store
    fg = feature_store.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df_features = fg.read()
    df_features = df_features.sort_values("timestamp_unix").reset_index(drop=True)

    # Forward-fill short feature gaps only (never fabricate epa_aqi itself)
    fill_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    df_features[fill_cols] = df_features[fill_cols].ffill()

    latest_row = df_features.tail(1).reset_index(drop=True)
    latest_ts = latest_row["datetime_utc"].iloc[0]
    current_aqi = float(latest_row["epa_aqi"].iloc[0])

    # 3. Reconstruct the exact same pruned feature list used at training time.
    #    get_final_feature_list is a pure function of the column list, so as
    #    long as feature_engineering.py hasn't changed since training, this
    #    reproduces the training-time feature set without needing to ship
    #    feature-list metadata alongside the model artifacts.
    all_feature_cols = [
        c for c in df_features.columns
        if c not in NON_FEATURE_COLS and str(df_features[c].dtype) in ["float64", "int64"]
    ]
    final_features = get_final_feature_list(all_feature_cols)
    feats_per_horizon = feats_used_per_horizon(final_features)

    # 4. Chained + persistence-blended 3-day forecast
    predictions = predict_next_3_days(models, latest_row, feats_per_horizon)

    result = {
        "generated_at": str(latest_ts),
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_aqi": round(current_aqi, 1),
        "current_category": aqi_category(current_aqi),
        "forecast": {
            f"day_{h}": {
                "aqi": predictions[h],
                "category": aqi_category(predictions[h]),
                "hazardous_alert": predictions[h] >= HAZARD_ALERT_THRESHOLD,
            }
            for h in HORIZONS
        },
    }

    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("\n" + "=" * 50)
    logger.info("           3-DAY AIR QUALITY FORECAST              ")
    logger.info("=" * 50)
    logger.info(f"Current AQI: {result['current_aqi']} ({result['current_category']})")
    for h in HORIZONS:
        day = result["forecast"][f"day_{h}"]
        flag = "  ⚠️ HAZARDOUS ALERT" if day["hazardous_alert"] else ""
        logger.info(f"Day {h}: {day['aqi']} ({day['category']}){flag}")
    logger.info("=" * 50)
    logger.info(f"Predictions written to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
