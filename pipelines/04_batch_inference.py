import sys
import os
import json
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
    PREDICTIONS_PATH,
    RECENT_HISTORY_PATH,
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

# Was HISTORY_HOURS_NEEDED = 250. Prediction only needs the latest row --
# its EMA/lag columns were already computed once by 02_feature_pipeline.py
# and stored on that row. This small window is just a forward-fill safety
# net in case the very latest row has a stray NaN in some column.
PREDICTION_READ_HOURS = 24

# Dashboard's "recent trend" chart wants 14 days of history. Rather than
# re-querying this whole window from Hopsworks every hour, the chart data
# is maintained as a rolling window on disk (see update_recent_history_snapshot):
# each hourly run appends just the one new row and drops whatever's fallen
# outside the window. Hopsworks only gets queried for the full window once,
# the very first time this pipeline runs (when the file doesn't exist yet).
CHART_HISTORY_HOURS = 24 * 14


def aqi_category(value: float) -> str:
    for name, low, high in AQI_CATEGORIES:
        if low <= value <= high:
            return name
    return "Hazardous"


def fetch_recent_features(fg, read_hours: int) -> pd.DataFrame:
    """
    Reads the last `read_hours` rows from the Feature Group. Tries the
    ONLINE store first (goes through Hopsworks' REST API to RonDB, not the
    offline Arrow Flight/DuckDB service) -- proven to work, just somewhat
    slow (~2-3 min regardless of window size, since the slowness is
    server-side, not proportional to rows). Falls back to the Hive/Spark
    backend, a genuinely different code path from Arrow Flight (per
    Hopsworks docs: read_options={"use_hive": True}).

    Deliberately does NOT attempt fg.read() (default) or any
    "use_duckdb"/arrow-flight-disable combination -- those all route through
    the offline Arrow Flight/DuckDB query service, which has a known
    server-side crash ("release unlocked lock") as of Aug 2026.
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
        df_features = df_features.sort_values("timestamp_unix").tail(read_hours).reset_index(drop=True)
    else:
        df_features = df_features.tail(read_hours).reset_index(drop=True)

    return df_features


def update_recent_history_snapshot(fg, latest_row: pd.DataFrame):
    """
    Maintains predictions/recent_history.csv as a rolling 14-day window,
    updated incrementally:

      - First run ever (no file yet): does ONE larger Hopsworks query
        (CHART_HISTORY_HOURS rows) to seed the full window.
      - Every run after that: appends just the single latest row already
        fetched for prediction -- no extra Hopsworks query at all -- then
        drops whatever has aged out past the 14-day cutoff.

    This keeps steady-state Hopsworks load to one small query per hour
    (from fetch_recent_features for prediction) instead of a 336-row query
    every run just for the dashboard chart.
    """
    # NOTE: datetime_utc can come back tz-naive or tz-aware depending on which
    # Hopsworks backend served the read (online/RonDB vs. Hive/Spark fallback --
    # see fetch_recent_features). An earlier version of this normalization
    # converted everything to tz-AWARE, but pandas can still silently produce
    # a mixed-dtype (object) column when concatenating two "aware" columns
    # whose tz representations aren't identical -- which is exactly what
    # caused a "Cannot compare tz-naive and tz-aware timestamps" crash here
    # despite that normalization being in place. Converting to tz-naive UTC
    # instead (via utc=True, then stripping the tz) is strictly safer: two
    # naive datetime64 columns can never mismatch this way. Do not switch
    # this back to a "detect and conditionally localize" approach.
    def _to_naive_utc(series):
        return pd.to_datetime(series, utc=True).dt.tz_localize(None)

    new_row = latest_row[["datetime_utc", "epa_aqi"]].copy()
    new_row["datetime_utc"] = _to_naive_utc(new_row["datetime_utc"])

    if os.path.exists(RECENT_HISTORY_PATH):
        hist = pd.read_csv(RECENT_HISTORY_PATH)
        hist["datetime_utc"] = _to_naive_utc(hist["datetime_utc"])
        combined = pd.concat([hist, new_row], ignore_index=True)
        # Dedupe in case this hour's row already exists (e.g. pipeline re-run) --
        # keep the newer version.
        combined = combined.drop_duplicates(subset="datetime_utc", keep="last")
    else:
        logger.info(
            f"No existing history snapshot found -- doing a one-time "
            f"{CHART_HISTORY_HOURS}-row backfill query from Hopsworks..."
        )
        backfill = fetch_recent_features(fg, read_hours=CHART_HISTORY_HOURS + 10)
        combined = backfill[["datetime_utc", "epa_aqi"]].copy()
        combined["datetime_utc"] = _to_naive_utc(combined["datetime_utc"])
        combined = pd.concat([combined, new_row], ignore_index=True)
        combined = combined.drop_duplicates(subset="datetime_utc", keep="last")

    combined = combined.sort_values("datetime_utc")

    # Rolling window by TIME, not row count -- stays correct even if a run
    # is occasionally missed (the window just has a gap, not a crash).
    cutoff = combined["datetime_utc"].max() - pd.Timedelta(hours=CHART_HISTORY_HOURS)
    combined = combined[combined["datetime_utc"] > cutoff]

    os.makedirs(os.path.dirname(RECENT_HISTORY_PATH), exist_ok=True)
    combined.to_csv(RECENT_HISTORY_PATH, index=False)
    logger.info(
        f"History snapshot now has {len(combined)} rows "
        f"({combined['datetime_utc'].min()} -> {combined['datetime_utc'].max()})"
    )


def get_latest_model_version(mr, model_name: str) -> int:
    """
    Hopsworks auto-increments the model version each time a new one is
    registered under the same name (register_model_to_hopsworks in
    model_trainer.py, called fresh by 03_training_pipeline.py every day).
    Hardcoding version=1 would mean inference keeps serving the very first
    model forever, no matter how many times it's retrained -- silently
    defeating the point of daily retraining, with no error to notice it by.
    This looks up the actual latest version instead.
    """
    models = mr.get_models(model_name)
    if not models:
        raise ValueError(f"No registered versions found for model '{model_name}'")
    latest = max(models, key=lambda m: m.version)
    return latest.version


def load_models_from_registry(mr):
    models = {}
    try:
        for idx in HORIZONS:
            model_name = MODEL_NAME_TEMPLATE.format(horizon=idx)
            latest_version = get_latest_model_version(mr, model_name)
            logger.info(f"Fetching registered model '{model_name}' v{latest_version}...")
            model_meta = mr.get_model(model_name, version=latest_version)
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

    models = load_models_from_registry(mr)

    # Small window only -- see PREDICTION_READ_HOURS comment above for why
    # this doesn't need the full history anymore.
    logger.info(f"Fetching recent feature records from '{FEATURE_GROUP_NAME}_v{FEATURE_GROUP_VERSION}'...")
    fg = feature_store.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df_features = fetch_recent_features(fg, read_hours=PREDICTION_READ_HOURS)

    fill_cols = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    df_features[fill_cols] = df_features[fill_cols].ffill()

    latest_row = df_features.tail(1).reset_index(drop=True)
    latest_ts = str(latest_row["datetime_utc"].iloc[0])
    current_aqi = float(latest_row["epa_aqi"].iloc[0])
    logger.info(f"Latest observation timestamp: {latest_ts} | Current EPA AQI: {current_aqi}")

    # Update the dashboard's rolling 14-day history snapshot. Only queries
    # Hopsworks again if this is the very first run ever (no file yet) --
    # otherwise this is a pure local file operation, zero extra requests.
    update_recent_history_snapshot(fg, latest_row)

    all_feature_cols = [
        c for c in df_features.columns
        if c not in NON_FEATURE_COLS and str(df_features[c].dtype) in ["float64", "int64", "int32", "float32"]
    ]
    final_features = get_final_feature_list(all_feature_cols)
    feats_per_horizon = feats_used_per_horizon(final_features)
    logger.info(f"Using {len(final_features)} features for inference.")

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

    # Insert predictions into Hopsworks Feature Group
    pred_df = pd.DataFrame([pred_data])
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

    json_payload = {
        "generated_at": pred_data["generated_at"],
        "current_aqi": pred_data["current_aqi"],
        "current_category": pred_data["current_category"],
        "forecast": {
            "day_1": {
                "aqi": pred_data["day1_aqi"],
                "category": pred_data["day1_category"],
                "hazardous_alert": pred_data["day1_hazardous"],
            },
            "day_2": {
                "aqi": pred_data["day2_aqi"],
                "category": pred_data["day2_category"],
                "hazardous_alert": pred_data["day2_hazardous"],
            },
            "day_3": {
                "aqi": pred_data["day3_aqi"],
                "category": pred_data["day3_category"],
                "hazardous_alert": pred_data["day3_hazardous"],
            },
        },
    }

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_PATH, "w") as f:
        json.dump(json_payload, f, indent=2)

    logger.info(f"Successfully saved snapshot to {PREDICTIONS_PATH}")

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