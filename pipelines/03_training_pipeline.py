import sys
import os
import tempfile
import logging
from pathlib import Path

if os.name == "nt":
    os.environ["TMPDIR"] = tempfile.gettempdir()
    try:
        os.makedirs(r"C:\tmp", exist_ok=True)
    except Exception:
        pass

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
    FEATURE_VIEW_NAME,
    FEATURE_VIEW_VERSION,
    MODEL_NAME_TEMPLATE,
    TARGET_MAPPING,
    TARGET_COLS,
    NON_FEATURE_COLS,
    HORIZONS,
)
from src.feature_engineering import get_final_feature_list
from src.model_trainer import (
    chronological_split,
    train_chained_models,
    evaluate_predictions,
    generate_shap_summary,
    register_model_to_hopsworks,
    save_models_local,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def read_full_feature_history(fg) -> pd.DataFrame:
    """
    Reads the FULL feature group (all historical rows) for training.

    Was: fg.read() (defaults to Arrow Flight, which has had a server-side
    crash bug) falling back to read_options={"use_hive": True} -- but the
    Hive/Spark path depends on the OFFLINE MATERIALIZATION JOB having
    succeeded, and that job has been failing.

    Now: tries the ONLINE store first instead. Since this feature group's
    primary key is timestamp_unix (every hour is a distinct key, nothing
    gets overwritten), the online/RonDB store holds the full row history
    too, not just "latest value per key" -- so this can serve a full
    training read without depending on the broken materialization job at
    all. Falls back to Hive/Spark only if the online read itself fails.

    Caveat: pulling ~40k+ rows through the online JDBC path is a lot more
    than the few hundred rows this pattern was originally proven on in
    04_batch_inference.py -- it may be slow (multiple minutes) or hit the
    same MySQL lock-wait timeout under load. It's the best available path
    right now, not a guaranteed-fast one.
    """
    try:
        logger.info("Attempting online feature group read (full history)...")
        df = fg.read(online=True)
        logger.info(f"Online read succeeded: {len(df)} rows.")
        return df
    except Exception as e:
        logger.warning(f"Online read failed ({e}). Falling back to Hive/Spark backend...")
        df = fg.read(read_options={"use_hive": True})
        return df


def main():
    logger.info("=== Starting Phase 4: Chained Multi-Horizon Training Pipeline ===")

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY,
    )
    feature_store = project.get_feature_store()

    fg = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    # Feature view now carries all three horizon targets as labels (was a
    # single 24h-lead target). Kept as a Hopsworks registry artifact for
    # versioning/lineage — actual training data still comes from
    # read_full_feature_history below, same as your original pipeline did.
    try:
        fv = feature_store.get_feature_view(name=FEATURE_VIEW_NAME, version=FEATURE_VIEW_VERSION)
    except Exception:
        query = fg.select_all()
        fv = feature_store.create_feature_view(
            name=FEATURE_VIEW_NAME,
            version=FEATURE_VIEW_VERSION,
            query=query,
            labels=TARGET_COLS,
            description="Feature View for chained 3-day (24h/48h/72h rolling-average) EPA AQI forecasting.",
        )

    logger.info("Reading feature matrix from Hopsworks Feature Group...")
    df = read_full_feature_history(fg)

    df = df.sort_values("timestamp_unix").reset_index(drop=True)
    logger.info(f"Rows read from Feature Group: {len(df)}")

    all_feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and str(df[c].dtype) in ["float64", "int64"]
    ]
    final_features = get_final_feature_list(all_feature_cols)
    logger.info(f"Engineered columns: {len(all_feature_cols)} -> pruned to {len(final_features)} final features")

    # 1. Drop rows missing ANY of the three targets, before any filling, so we
    #    never train against a label that was fabricated by ffill/bfill.
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    logger.info(f"Rows remaining after dropping missing targets: {len(df)}")

    # 2. Forward-fill short gaps in feature columns only (never targets).
    df[final_features] = df[final_features].ffill()

    # 3. Drop any rows where features are still unfillable (leading NaNs from
    #    the longest lag/EMA windows at the very start of the series).
    before = len(df)
    df = df.dropna(subset=final_features).reset_index(drop=True)
    logger.info(f"Rows dropped for unfillable feature NaNs: {before - len(df)}")

    if len(df) < 500:
        logger.warning(
            f"Only {len(df)} usable rows after cleaning — small training set for hourly "
            f"time-series data. Verify the backfill and weather merge ran correctly."
        )

    split_idx = int(len(df) * 0.80)
    train_dates = pd.to_datetime(df["datetime_utc"].iloc[:split_idx])
    test_dates = pd.to_datetime(df["datetime_utc"].iloc[split_idx:])
    logger.info(f"Train period: {train_dates.min()} -> {train_dates.max()}")
    logger.info(f"Test period:  {test_dates.min()} -> {test_dates.max()}")

    # Persistence baseline per horizon — the bar every model needs to clear.
    # AQI in Lahore is highly persistent (smog season runs for weeks), so this
    # is a genuinely strong baseline, not a strawman.
    for label, target_col in TARGET_MAPPING.items():
        y_test = df[target_col].iloc[split_idx:].values
        naive_pred = df["epa_aqi"].iloc[split_idx:].values
        naive_metrics = evaluate_predictions(y_test, naive_pred)
        logger.info(f"Naive baseline — {label}: RMSE {naive_metrics['rmse']}, MAE {naive_metrics['mae']}, R2 {naive_metrics['r2']}")

    train_df, test_df = chronological_split(df, test_frac=0.20)

    logger.info("Training chained multi-horizon LightGBM models + persistence blend...")
    models, metrics_df, feats_per_horizon, test_df = train_chained_models(train_df, test_df, final_features)

    logger.info("\n" + "=" * 60)
    logger.info("        CHAINED MODEL + PERSISTENCE BLEND RESULTS         ")
    logger.info("=" * 60)
    print(metrics_df.to_string(index=False))
    logger.info("=" * 60)

    # Local safety-net save (models/final_feature_list.pkl/feats_per_horizon.pkl)
    save_models_local(models, final_features, feats_per_horizon, out_dir="models")

    # Register each horizon's model in the Hopsworks Model Registry
    for idx in HORIZONS:
        label = list(TARGET_MAPPING.keys())[idx - 1]
        target_col = TARGET_MAPPING[label]
        feats = feats_per_horizon[idx]
        row_metrics = metrics_df.iloc[idx - 1]

        # Ensure missing chained prediction features exist in DataFrames
        for f in feats:
            if f not in train_df.columns:
                train_df[f] = 0.0
            if f not in test_df.columns:
                test_df[f] = 0.0

        shap_path = generate_shap_summary(
            models[f"model_day{idx}"],
            train_df[feats].tail(500),
            output_path=f"shap_summary_day{idx}.png",
        )

        register_model_to_hopsworks(
            model=models[f"model_day{idx}"],
            metrics={"r2": float(row_metrics["R2"]), "mae": float(row_metrics["MAE"]), "rmse": float(row_metrics["RMSE"])},
            input_example=test_df[feats].head(1),
            shap_plot_path=shap_path,
            model_name=MODEL_NAME_TEMPLATE.format(horizon=idx),
            description=f"Chained LightGBM + persistence blend, {label}, target={target_col}",
        )

    logger.info("\n=== Phase 4 Completed Successfully! ===")


if __name__ == "__main__":
    main()