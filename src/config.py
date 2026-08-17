import os
from dotenv import load_dotenv
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = ROOT_DIR / "predictions"
PREDICTIONS_PATH = PREDICTIONS_DIR / "latest_predictions.json"

# Ensure directory exists
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_API_URL = os.getenv("HOPSWORKS_API_URL")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

CITY_LAT = float(os.getenv("CITY_LATITUDE", "31.480961"))
CITY_LON = float(os.getenv("CITY_LONGITUDE", "74.363350"))
LOCATION_NAME = os.getenv("LOCATION_NAME", "Lahore")

LATITUDE = CITY_LAT
LONGITUDE = CITY_LON

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
OPENWEATHER_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
OPENWEATHER_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"  # kept for reference; no longer used for lead features, see data_fetcher.fetch_weather_forecast_openmeteo

FEATURE_GROUP_NAME = "aqi_hourly_fg"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_hourly_fv"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "aqi_forecast_model"

# ---------------------------------------------------------------------------
# NOTEBOOK-ALIGNED TARGETS (final_1_0.ipynb)
# ---------------------------------------------------------------------------
# Superseded: single-point leads (epa_aqi_lead_24h/48h/72h) and PRIMARY_TARGET
# below are no longer what the model is trained against. Kept commented out
# rather than deleted so the old target definition is visible if you ever
# need to compare against it.
# PRIMARY_TARGET = "epa_aqi"
# FORECAST_HORIZONS = [24, 48, 72]

TARGET_MAPPING = {
    "Day 1 Forecast (Next 24h Mean)": "target_day1_avg_aqi",
    "Day 2 Forecast (Next 48h Mean)": "target_day2_avg_aqi",
    "Day 3 Forecast (Next 72h Mean)": "target_day3_avg_aqi",
}
TARGET_COLS = list(TARGET_MAPPING.values())
HORIZONS = [1, 2, 3]

LAG_HOURS = [24, 48, 72, 168]          # AR lags on epa_aqi (was [1, 2, 24, 48])
EMA_SPANS = {"12h": 12, "72h": 72, "168h": 168}
EMA_SOURCE_COLS = ["pm2_5", "pm10", "epa_aqi", "co", "no2"]
ROLLING_WINDOWS = [24, 72]              # rolling std windows on epa_aqi (was [3, 24])
SMOG_MONTHS = [10, 11, 12, 1, 2]        # Lahore winter smog season

# ---------------------------------------------------------------------------
# Feature pruning — validated in the notebook (ablation + collinearity check)
# ---------------------------------------------------------------------------
# doy_sin/cos acted as a "which calendar date does this look like" shortcut
# rather than genuine AQI dynamics — dropped.
LEAK_AND_SUSPECT_COLS = ["dayofyear_sin", "dayofyear_cos"]

# Redundant/collinear AQI-family columns (corr 0.79-0.98 with each other in
# the notebook's correlation check) — keep only one "current", one short EMA,
# one long EMA, one volatility measure; everything else in the AQI family
# (lags, other EMA spans, other pollutant raw readings) gets dropped.
AQI_FAMILY_KEEP = ["epa_aqi", "epa_aqi_ema_12h", "epa_aqi_ema_168h", "aqi_std_24h"]

NON_FEATURE_COLS = [
    "datetime_utc", "timestamp_unix", "latitude", "longitude",
    "primary_pollutant", "sub_indices",
] + TARGET_COLS

# Future-weather "lead" columns. During training these are proxied by
# shifting real historical weather backward (valid — the future already
# happened for any historical row). At live inference time they come from
# an actual weather forecast instead — see
# feature_engineering.apply_forecast_lead_features and
# data_fetcher.fetch_weather_forecast_openmeteo.
LEAD_HOURS = [24, 48, 72]
LEAD_WEATHER_VARS = ["wind_speed", "temp", "rain", "humidity"]
LEAD_COLS = [f"{v}_lead_{h}h" for v in LEAD_WEATHER_VARS for h in LEAD_HOURS]

# ---------------------------------------------------------------------------
# Model — chained multi-horizon LightGBM + persistence blend
# (final_1_0.ipynb cell 12 "FINAL MODEL: CHAINED MODEL + PERSISTENCE BLEND")
# ---------------------------------------------------------------------------
LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
BLEND_WEIGHT = 0.4  # final_pred = BLEND_WEIGHT * model_pred + (1 - BLEND_WEIGHT) * current_aqi

MODEL_NAME_TEMPLATE = "aqi_lgbm_day{horizon}"

