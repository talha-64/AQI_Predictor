import math
from datetime import datetime, timezone
from typing import Dict, Any, Union

import numpy as np
import pandas as pd

import logging

def get_logger(name: str = "aqi_dashboard") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
# ---------------------------------------------------------------------------
# NOTEBOOK-ALIGNED EPA AQI (final_1_0.ipynb) — PM2.5-only piecewise formula.
# This is what the model's target (`epa_aqi`) is computed with everywhere in
# the pipeline now, so training/backfill/live-inference all agree.
# ---------------------------------------------------------------------------
def _calc_piecewise_aqi(c: float, i_low: int, i_high: int, c_low: float, c_high: float) -> int:
    return round(((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low)


def pm25_to_epa_aqi(pm25: float) -> Union[float, int]:
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    elif pm25 <= 12.0:
        return _calc_piecewise_aqi(pm25, 0, 50, 0.0, 12.0)
    elif pm25 <= 35.4:
        return _calc_piecewise_aqi(pm25, 51, 100, 12.1, 35.4)
    elif pm25 <= 55.4:
        return _calc_piecewise_aqi(pm25, 101, 150, 35.5, 55.4)
    elif pm25 <= 150.4:
        return _calc_piecewise_aqi(pm25, 151, 200, 55.5, 150.4)
    elif pm25 <= 250.4:
        return _calc_piecewise_aqi(pm25, 201, 300, 150.5, 250.4)
    elif pm25 <= 500.4:
        return _calc_piecewise_aqi(pm25, 301, 500, 250.5, 500.4)
    else:
        return 500


# ---------------------------------------------------------------------------
# LEGACY / ALTERNATE — full multi-pollutant EPA AQI (max sub-index across
# PM2.5, PM10, O3, CO, SO2, NO2). This is the more textbook-complete EPA
# definition, and is kept here since it's solid, well-tested code — but it
# is NOT what the current model was trained against, so it's not called
# anywhere in the pipeline. If you want to switch the model over to this
# definition later, you'd need to retrain from scratch (the target changes)
# and rerun the validation work (naive baseline, ablations, weight sweep)
# against it.
# ---------------------------------------------------------------------------
BREAKPOINTS = {
    "pm2_5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 354.4, 301, 400),
        (354.5, 500.4, 401, 500),
    ],
    "pm10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "o3": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "co": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
    "so2": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ],
    "no2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
}


def truncate(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def calculate_sub_aqi(pollutant: str, concentration: float) -> Union[int, None]:
    if pollutant not in BREAKPOINTS:
        return None
    if pollutant == "pm2_5":
        c = truncate(concentration, 1)
    elif pollutant == "pm10":
        c = truncate(concentration, 0)
    elif pollutant == "o3":
        c = truncate(concentration, 3)
    elif pollutant == "co":
        c = truncate(concentration, 1)
    elif pollutant in ["so2", "no2"]:
        c = truncate(concentration, 0)
    else:
        c = concentration

    for c_low, c_high, i_low, i_high in BREAKPOINTS[pollutant]:
        if c_low <= c <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low)
    return None


def convert_openweather_units(components: Dict[str, float]) -> Dict[str, float]:
    return {
        "pm2_5": components.get("pm2_5", 0.0),
        "pm10": components.get("pm10", 0.0),
        "o3": components.get("o3", 0.0) / 1995.8,
        "co": components.get("co", 0.0) / 1145.0,
        "so2": components.get("so2", 0.0) / 2.62,
        "no2": components.get("no2", 0.0) / 1.88,
    }


def calculate_overall_epa_aqi(components: Dict[str, float]) -> Dict[str, Any]:
    """Not currently used by the pipeline — see module docstring above."""
    converted_components = convert_openweather_units(components)
    sub_indices = {}
    for pollutant, value in converted_components.items():
        sub_index = calculate_sub_aqi(pollutant, value)
        if sub_index is not None:
            sub_indices[pollutant] = sub_index
    if not sub_indices:
        return {"epa_aqi": None, "main_pollutant": None, "sub_indices": {}}
    main_pollutant = max(sub_indices, key=sub_indices.get)
    return {"epa_aqi": sub_indices[main_pollutant], "main_pollutant": main_pollutant, "sub_indices": sub_indices}


# ---------------------------------------------------------------------------
# Timestamp utilities
# ---------------------------------------------------------------------------
def timestamp_to_utc_datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def timestamp_to_iso(ts: int) -> str:
    return timestamp_to_utc_datetime(ts).isoformat()


def parse_openweather_record(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a single record from the OpenWeather Air Pollution API payload.
    epa_aqi is now computed with the notebook's PM2.5-only formula (was the
    multi-pollutant max-sub-index calc) so it matches what the model was
    trained and validated against.
    """
    entry = payload["list"][0]
    dt_timestamp = entry["dt"]
    components = entry["components"]

    epa_aqi = pm25_to_epa_aqi(components.get("pm2_5"))

    return {
        "latitude": payload["coord"]["lat"],
        "longitude": payload["coord"]["lon"],
        "timestamp_unix": dt_timestamp,
        "datetime_utc": timestamp_to_iso(dt_timestamp),
        "openweather_aqi_scale": entry["main"]["aqi"],  # Scale 1-5, informational only
        "epa_aqi": epa_aqi,                              # Scale 0-500 — the model target
        "raw_components_ugm3": components,
    }

def read_features() -> pd.DataFrame:
    """Reads the primary feature matrix from Hopsworks Feature Store."""
    import hopsworks
    from src import config

    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(
        name=config.FEATURE_GROUP_NAME,
        version=config.FEATURE_GROUP_VERSION,
    )
    return fg.read()