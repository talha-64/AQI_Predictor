import os
import time
import tempfile
import joblib
import logging
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
import hopsworks

from src.config import HOPSWORKS_PROJECT_NAME, HOPSWORKS_API_KEY
from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kept from your original implementation — model-agnostic, no notebook
# dependency.
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def generate_shap_summary(model, X_sample: pd.DataFrame, output_path: str = "shap_summary.png") -> str:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to '{output_path}'")
    except Exception as e:
        logger.warning(f"Could not generate SHAP plot: {e}")
        output_path = ""
    return output_path


def register_model_to_hopsworks(
    model,
    metrics: dict,
    input_example: pd.DataFrame,
    shap_plot_path: str,
    model_name: str,
    description: str,
    max_retries: int = 3,
):
    """Saves a single model artifact and registers it in the Hopsworks Model
    Registry with retry logic and directory protection."""
    project = hopsworks.login(project=HOPSWORKS_PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()

    temp_dir = tempfile.mkdtemp()

    try:
        joblib.dump(model, os.path.join(temp_dir, "model.pkl"))

        if shap_plot_path and os.path.exists(shap_plot_path):
            shutil.copy(shap_plot_path, os.path.join(temp_dir, "shap_summary.png"))

        hw_model = mr.python.create_model(
            name=model_name,
            metrics=metrics,
            description=description,
            input_example=input_example,
        )

        for attempt in range(1, max_retries + 1):
            try:
                # keep_original_files=True prevents HSML from prematurely deleting the local temp folder
                hw_model.save(temp_dir, keep_original_files=True)
                logger.info(f"Model registered in Hopsworks as '{model_name}'.")
                break
            except Exception as e:
                if attempt == max_retries:
                    raise e
                logger.warning(
                    f"Upload failed for '{model_name}' (Attempt {attempt}/{max_retries}). "
                    f"Retrying in 5 seconds... Error: {e}"
                )
                time.sleep(5)

    finally:
        # Clean up local temporary folder safely after all upload attempts finish
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
# ---------------------------------------------------------------------------
# NEW — chained multi-horizon LightGBM + persistence blend
# (replaces the Ridge/RandomForest/XGBoost/LightGBM "champion" bake-off,
# to match final_1_0.ipynb cell 12)
# ---------------------------------------------------------------------------
def chronological_split(df_clean: pd.DataFrame, test_frac: float = 0.2):
    split_idx = int(len(df_clean) * (1 - test_frac))
    train_df = df_clean.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_clean.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


def feats_used_per_horizon(base_features: list) -> dict:
    return {
        idx: base_features + [f"pred_day{h}" for h in range(1, idx)]
        for idx in range(1, len(config.HORIZONS) + 1)
    }


def train_chained_models(train_df: pd.DataFrame, test_df: pd.DataFrame, base_features: list):
    """
    Trains one LightGBM model per horizon, chaining prior horizons'
    predictions in as extra features (Day 2's model sees Day 1's prediction,
    Day 3's sees Day 1 and Day 2's), then blends each model's prediction
    with simple persistence (today's AQI carried forward), weighted by
    config.BLEND_WEIGHT.

    Returns (models, metrics_df, feats_per_horizon, test_df) — test_df comes
    back with pred_day1/2/3 columns attached, useful for diagnostics/SHAP.
    """
    models = {}
    eval_rows = []

    train_df = train_df.copy()
    test_df = test_df.copy()

    for idx, (label, target_col) in enumerate(config.TARGET_MAPPING.items(), start=1):
        feats = base_features + [f"pred_day{h}" for h in range(1, idx)]

        model = LGBMRegressor(**config.LGBM_PARAMS)
        model.fit(train_df[feats], train_df[target_col])
        models[f"model_day{idx}"] = model

        train_pred = np.clip(model.predict(train_df[feats]), 0, 500)
        test_pred = np.clip(model.predict(test_df[feats]), 0, 500)
        train_df[f"pred_day{idx}"] = train_pred
        test_df[f"pred_day{idx}"] = test_pred

        pred_naive = test_df["epa_aqi"].values
        pred_blend = config.BLEND_WEIGHT * test_pred + (1 - config.BLEND_WEIGHT) * pred_naive
        y_actual = test_df[target_col].values

        metrics = evaluate_predictions(y_actual, pred_blend)
        eval_rows.append({
            "Horizon": label,
            "Model weight": config.BLEND_WEIGHT,
            "MAE": metrics["mae"],
            "RMSE": metrics["rmse"],
            "R2": metrics["r2"],
        })

    metrics_df = pd.DataFrame(eval_rows)
    logger.info(f"Training complete:\n{metrics_df.to_string(index=False)}")
    return models, metrics_df, feats_used_per_horizon(base_features), test_df


def save_models_local(models: dict, feature_cols: list, feats_per_horizon: dict, out_dir: str = "models"):
    """Local joblib save — a lightweight safety net alongside Hopsworks
    registration, so 04_batch_inference.py can run even if Hopsworks is
    briefly unavailable. New addition, not present in your original code."""
    os.makedirs(out_dir, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(out_dir, f"{name}.pkl"))
    joblib.dump(feature_cols, os.path.join(out_dir, "final_feature_list.pkl"))
    joblib.dump(feats_per_horizon, os.path.join(out_dir, "feats_per_horizon.pkl"))
    logger.info(f"Saved {len(models)} models + metadata locally to {out_dir}")


def load_models_local(in_dir: str = "models"):
    models = {
        f"model_day{h}": joblib.load(os.path.join(in_dir, f"model_day{h}.pkl"))
        for h in config.HORIZONS
    }
    feature_cols = joblib.load(os.path.join(in_dir, "final_feature_list.pkl"))
    feats_per_horizon = joblib.load(os.path.join(in_dir, "feats_per_horizon.pkl"))
    return models, feature_cols, feats_per_horizon


def predict_next_3_days(models: dict, feature_row: pd.DataFrame, feats_per_horizon: dict) -> dict:
    """feature_row: single-row DataFrame with all engineered features for the
    current timestamp. Returns {horizon: blended_aqi_prediction}."""
    row = feature_row.copy()
    current_aqi = float(row["epa_aqi"].iloc[0])

    raw_model_preds = {}
    blended_preds = {}

    for idx in config.HORIZONS:
        feats = feats_per_horizon[idx]
        for h in range(1, idx):
            row[f"pred_day{h}"] = raw_model_preds[h]

        model = models[f"model_day{idx}"]
        pred = float(np.clip(model.predict(row[feats])[0], 0, 500))
        raw_model_preds[idx] = pred

        blended = config.BLEND_WEIGHT * pred + (1 - config.BLEND_WEIGHT) * current_aqi
        blended_preds[idx] = round(blended, 1)

    return blended_preds
