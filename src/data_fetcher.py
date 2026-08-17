import time
import numpy as np
import pandas as pd
import requests
from typing import Dict, Any, List
import logging
from datetime import datetime, timezone
from src.config import (
    OPENWEATHER_API_KEY,
    OPENWEATHER_BASE_URL,
    OPENWEATHER_HISTORY_URL,
    OPENWEATHER_FORECAST_URL,
    LATITUDE,
    LONGITUDE,
)

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Added "precipitation" (-> rain) — the notebook's feature set relies on it
# (rain_cumsum_72h, rain_lead_24h/48h/72h). It was missing from the original
# variable list, which would have silently left those features as NaN/0.
OPEN_METEO_HOURLY_VARS = [
    "temperature_2m",
    "surface_pressure",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
]


def fetch_current_air_pollution(
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    api_key: str = OPENWEATHER_API_KEY
) -> Dict[str, Any]:
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable is not set.")

    params = {"lat": lat, "lon": lon, "appid": api_key}
    response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_historical_air_pollution_chunk(
    start_unix: int,
    end_unix: int,
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    api_key: str = OPENWEATHER_API_KEY
) -> Dict[str, Any]:
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable is not set.")

    params = {
        "lat": lat,
        "lon": lon,
        "start": start_unix,
        "end": end_unix,
        "appid": api_key,
    }
    response = requests.get(OPENWEATHER_HISTORY_URL, params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def fetch_historical_air_pollution_batch(
    start_unix: int,
    end_unix: int,
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    chunk_days: int = 30,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    Fetches historical air pollution data in chunks, with retry/backoff per chunk.
    Raises RuntimeError if any chunk fails permanently, rather than silently
    returning gappy data (gaps corrupt lag/rolling/lead feature calculations).
    """
    all_records = []
    seconds_per_chunk = chunk_days * 86400
    current_start = start_unix
    failed_chunks = []

    while current_start < end_unix:
        current_end = min(current_start + seconds_per_chunk, end_unix)
        logger.info(f"Fetching historical pollution batch: {current_start} -> {current_end}")

        for attempt in range(max_retries):
            try:
                payload = fetch_historical_air_pollution_chunk(
                    start_unix=current_start,
                    end_unix=current_end,
                    lat=lat,
                    lon=lon,
                )
                all_records.extend(payload.get("list", []))
                break
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                logger.warning(
                    f"Chunk {current_start}-{current_end} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
        else:
            failed_chunks.append((current_start, current_end))
            logger.error(f"Chunk {current_start}-{current_end} failed after {max_retries} retries.")

        current_start = current_end + 1
        time.sleep(0.5)

    if failed_chunks:
        raise RuntimeError(
            f"Historical pollution backfill incomplete: {len(failed_chunks)} chunk(s) failed permanently "
            f"at {failed_chunks}. Aborting rather than training on data with silent time gaps."
        )

    return {"coord": {"lat": lat, "lon": lon}, "list": all_records}


def fetch_historical_weather_openmeteo(
    start_unix: int,
    end_unix: int,
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Fetches historical hourly weather from Open-Meteo's free Historical Weather API
    (ERA5 reanalysis, no API key required, no subscription tier).
    """
    start_date = datetime.fromtimestamp(start_unix, tz=timezone.utc).strftime("%Y-%m-%d")
    end_date = datetime.fromtimestamp(end_unix, tz=timezone.utc).strftime("%Y-%m-%d")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }

    logger.info(f"Fetching historical weather from Open-Meteo: {start_date} -> {end_date}")

    data = None
    for attempt in range(max_retries):
        try:
            response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"Open-Meteo fetch failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    if data is None:
        raise RuntimeError(f"Open-Meteo historical weather fetch failed after {max_retries} retries.")

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    if not times:
        logger.warning("Open-Meteo returned no hourly data for the requested window.")
        return []

    n = len(times)
    temp_arr = hourly.get("temperature_2m", [None] * n)
    pressure_arr = hourly.get("surface_pressure", [None] * n)
    humidity_arr = hourly.get("relative_humidity_2m", [None] * n)
    wind_speed_arr = hourly.get("wind_speed_10m", [None] * n)
    wind_deg_arr = hourly.get("wind_direction_10m", [None] * n)
    clouds_arr = hourly.get("cloud_cover", [None] * n)
    rain_arr = hourly.get("precipitation", [None] * n)

    records = []
    for i, t in enumerate(times):
        dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
        ts = int(dt.timestamp())

        records.append({
            "timestamp_unix": ts,
            "temp": temp_arr[i],
            "pressure": pressure_arr[i],
            "humidity": humidity_arr[i],
            "wind_speed": wind_speed_arr[i],
            "wind_deg": wind_deg_arr[i],
            "clouds": clouds_arr[i],
            "rain": rain_arr[i],
        })

    logger.info(f"Open-Meteo returned {len(records)} hourly weather records.")
    return records


def fetch_weather_forecast_openmeteo(
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    forecast_days: int = 4,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetches hourly weather forecast (true hourly resolution, not OpenWeather's
    3-hour steps) covering the current hour through `forecast_days` days ahead.
    Used to build real _lead_24h/48h/72h weather features at live inference
    time (see feature_engineering.apply_forecast_lead_features), replacing
    the old fetch_24h_weather_forecast (24h-only, 3-hour resolution) below.

    Using the same provider (Open-Meteo) for both historical training data
    and live forecasts also avoids a train/serve mismatch you'd get from
    mixing two different weather data sources.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(OPEN_METEO_HOURLY_VARS),
        "forecast_days": forecast_days,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }

    data = None
    for attempt in range(max_retries):
        try:
            response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            break
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt
            logger.warning(f"Open-Meteo forecast fetch failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    if data is None or "hourly" not in data:
        raise RuntimeError(f"Open-Meteo forecast fetch failed after {max_retries} retries.")

    hourly = data["hourly"]
    df = pd.DataFrame({
        "datetime_utc": pd.to_datetime(hourly["time"], utc=True).tz_localize(None),
        "temp": hourly.get("temperature_2m"),
        "pressure": hourly.get("surface_pressure"),
        "humidity": hourly.get("relative_humidity_2m"),
        "wind_speed": hourly.get("wind_speed_10m"),
        "wind_deg": hourly.get("wind_direction_10m"),
        "clouds": hourly.get("cloud_cover"),
        "rain": hourly.get("precipitation"),
    })
    logger.info(f"Fetched {len(df)}h of forecast weather from Open-Meteo.")
    return df


def fetch_24h_weather_forecast(
    lat: float = LATITUDE,
    lon: float = LONGITUDE,
    api_key: str = OPENWEATHER_API_KEY
) -> Dict[str, float]:
    """DEPRECATED — superseded by fetch_weather_forecast_openmeteo, which
    covers the full 24/48/72h lead horizon at true hourly resolution instead
    of just 24h at 3-hour steps, and uses the same provider as historical
    training data. Left in place only in case anything else still imports it."""
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY environment variable is not set.")

    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(OPENWEATHER_FORECAST_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        forecast_24h = data["list"][8]  # index 8 = 24 hours ahead (8 * 3h steps)

        temp = forecast_24h["main"]["temp"]
        humidity = forecast_24h["main"]["humidity"]
        pressure = forecast_24h["main"]["pressure"]
        wind_speed = forecast_24h["wind"]["speed"]
        wind_deg = forecast_24h["wind"]["deg"]
        clouds = forecast_24h["clouds"]["all"]
        pop = forecast_24h.get("pop", 0.0)

        wind_rad = np.radians(wind_deg)
        wind_u = wind_speed * np.cos(wind_rad)
        wind_v = wind_speed * np.sin(wind_rad)

        return {
            "forecast_temp_24h": temp,
            "forecast_humidity_24h": humidity,
            "forecast_pressure_24h": pressure,
            "forecast_wind_speed_24h": wind_speed,
            "forecast_wind_u_24h": wind_u,
            "forecast_wind_v_24h": wind_v,
            "forecast_clouds_24h": clouds,
            "forecast_pop_24h": pop,
        }
    except Exception as e:
        logger.error(f"Error fetching 24h weather forecast: {e}")
        raise e
