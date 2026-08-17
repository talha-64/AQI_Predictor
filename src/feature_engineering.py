import pandas as pd
import numpy as np
import logging
from src.utils import parse_openweather_record
from src import config

logger = logging.getLogger(__name__)


def compute_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """hour_sin/cos + doy_sin/cos, matching final_1_0.ipynb. (Previously also
    computed dayofweek_sin/cos — dropped for exact fidelity to the notebook's
    feature set; add back if you want to experiment with weekly patterns.)"""
    df = df.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_localize(None)

    hour = df["datetime_utc"].dt.hour
    dayofyear = df["datetime_utc"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    # doy_sin/cos are computed (needed downstream) but dropped from the final
    # feature list by get_final_feature_list — see config.LEAK_AND_SUSPECT_COLS.
    # The notebook's ablation test found these acted as a "which calendar
    # date does this look like" shortcut rather than real AQI dynamics.
    df["dayofyear_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)

    df["month"] = df["datetime_utc"].dt.month
    df["is_smog_season"] = df["month"].isin(config.SMOG_MONTHS).astype(int)

    return df


def compute_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Converts wind speed and direction into orthogonal U/V vectors."""
    df = df.copy()
    if "wind_speed" in df.columns and "wind_deg" in df.columns:
        wind_rad = np.radians(df["wind_deg"])
        df["wind_u"] = df["wind_speed"] * np.cos(wind_rad)
        df["wind_v"] = df["wind_speed"] * np.sin(wind_rad)
    return df


def apply_forecast_lead_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds `{var}_lead_{h}h` columns for h in config.LEAD_HOURS by shifting
    real historical weather backward. Valid for training/backfill rows only
    (the "future" already happened for any historical row). For live
    inference, these get overwritten by apply_live_lead_features() with
    values from an actual weather forecast instead.
    """
    df = df.copy()
    for var in config.LEAD_WEATHER_VARS:
        if var not in df.columns:
            logger.warning(f"'{var}' not found in df — cannot build lead features for it.")
            continue
        for h in config.LEAD_HOURS:
            df[f"{var}_lead_{h}h"] = df[var].shift(-h)
    return df


def apply_live_lead_features(df_latest: pd.DataFrame, df_forecast: pd.DataFrame) -> pd.DataFrame:
    """
    Live-inference counterpart to apply_forecast_lead_proxy: overwrites the
    `_lead_*` columns on the latest row(s) with real forecast values instead
    of the shift-based proxy (which would be NaN anyway at the live edge of
    the data, since there's no real future to shift from).

    df_latest: engineered feature row(s), most recent timestamp last.
    df_forecast: output of data_fetcher.fetch_weather_forecast_openmeteo(),
                 must cover at least config.LEAD_HOURS ahead of the latest
                 timestamp in df_latest.
    """
    df = df_latest.copy()
    latest_ts = df["datetime_utc"].iloc[-1]
    fc = df_forecast.sort_values("datetime_utc").reset_index(drop=True)

    for h in config.LEAD_HOURS:
        target_ts = latest_ts + pd.Timedelta(hours=h)
        if len(fc) == 0:
            continue
        idx = (fc["datetime_utc"] - target_ts).abs().idxmin()
        for var in config.LEAD_WEATHER_VARS:
            col = f"{var}_lead_{h}h"
            if var in fc.columns:
                df.loc[df.index[-1], col] = fc.loc[idx, var]
    return df


def engineer_features_and_targets(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Computes the full notebook-aligned feature set: cyclical time, wind
    vectors, stagnation index, EMAs, AR lags, volatility, smog-season flag,
    rain accumulation, forecast-lead proxies, and (for training) the
    rolling-average multi-horizon targets.

    Assumes df is already at a true, gap-free hourly cadence (see
    build_full_feature_pipeline's reindexing step).
    """
    df = df.copy()
    df = df.sort_values("timestamp_unix").reset_index(drop=True)

    # 1. Cyclical time features & wind vectors
    df = compute_cyclical_time_features(df)
    df = compute_wind_vectors(df)

    # 2. Atmospheric stagnation & pressure delta
    if "pm2_5" in df.columns and "wind_speed" in df.columns:
        df["stagnation_index"] = df["pm2_5"] / (df["wind_speed"] + 0.5)
    if "pressure" in df.columns:
        df["pressure_delta_24h"] = df["pressure"] - df["pressure"].shift(24)

    # 3. Multi-scale EMAs (12h / 72h / 168h) on pollutant + AQI columns
    for col in config.EMA_SOURCE_COLS:
        if col in df.columns:
            for label, span in config.EMA_SPANS.items():
                df[f"{col}_ema_{label}"] = df[col].ewm(span=span).mean()

    # 4. Autoregressive lags on epa_aqi (24h / 48h / 72h / 168h)
    if "epa_aqi" in df.columns:
        for lag in config.LAG_HOURS:
            df[f"epa_aqi_lag_{lag}h"] = df["epa_aqi"].shift(lag)

        # Volatility (uncertainty signal for longer horizons)
        df["aqi_std_24h"] = df["epa_aqi"].rolling(window=24).std()
        df["aqi_std_72h"] = df["epa_aqi"].rolling(window=72).std()

        # Momentum
        df["aqi_change_rate_24h"] = df["epa_aqi"] - df["epa_aqi"].shift(24)

    # 5. Rain — pollution-clearing signal
    if "rain" in df.columns:
        df["rain_cumsum_72h"] = df["rain"].rolling(window=72).sum()

    # 6. Future-weather lead features (training proxy — see apply_live_lead_features
    #    for the live-inference counterpart, called separately by the pipeline)
    df = apply_forecast_lead_proxy(df)

    # 7. Daily-average targets (only meaningful during training/backfill —
    #    live rows won't have future AQI yet, so these come out NaN there,
    #    which is expected and handled by the training pipeline's dropna)
    if is_training and "epa_aqi" in df.columns:
        indexer_24h = pd.api.indexers.FixedForwardWindowIndexer(window_size=24)
        df["target_day1_avg_aqi"] = df["epa_aqi"].shift(-1).rolling(window=indexer_24h).mean()
        df["target_day2_avg_aqi"] = df["epa_aqi"].shift(-25).rolling(window=indexer_24h).mean()
        df["target_day3_avg_aqi"] = df["epa_aqi"].shift(-49).rolling(window=indexer_24h).mean()

    return df


def get_final_feature_list(all_feature_cols: list) -> list:
    """Applies the notebook's validated feature pruning: drop day-of-year
    calendar features (ablation), drop redundant/collinear AQI-family
    columns (correlation check), keep everything else."""
    ablated = [c for c in all_feature_cols if c not in config.LEAK_AND_SUSPECT_COLS]

    aqi_family = [c for c in ablated if "aqi" in c.lower() or c in ["pm2_5", "pm10"]]
    aqi_drop = [c for c in aqi_family if c not in config.AQI_FAMILY_KEEP]

    return [c for c in ablated if c not in aqi_drop]


def build_full_feature_pipeline(
    raw_payload: dict,
    weather_records: list = None,
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Parses raw pollution API batch records into a DataFrame, merges in
    historical weather (if provided), reindexes to a gap-free hourly
    timeline, and applies feature engineering.
    """
    records = raw_payload.get("list", [])
    coord = raw_payload.get("coord", {"lat": 0.0, "lon": 0.0})

    parsed_rows = []
    for item in records:
        single_payload = {"coord": coord, "list": [item]}
        row = parse_openweather_record(single_payload)
        raw_comp = row.pop("raw_components_ugm3", {})
        for pollutant, val in raw_comp.items():
            row[pollutant] = val
        parsed_rows.append(row)

    df_raw = pd.DataFrame(parsed_rows)

    if weather_records:
        df_weather = pd.DataFrame(weather_records)
        df_raw = df_raw.merge(df_weather, on="timestamp_unix", how="left")
        n_missing = df_raw["temp"].isna().sum() if "temp" in df_raw.columns else len(df_raw)
        logger.info(f"Weather merge complete. Rows still missing weather: {n_missing} / {len(df_raw)}")
    else:
        logger.warning("No weather_records provided — proceeding with pollution-only features.")

    df_raw["datetime_utc"] = pd.to_datetime(df_raw["datetime_utc"], utc=True).dt.tz_localize(None)
    df_raw = df_raw.drop_duplicates(subset=["datetime_utc"]).set_index("datetime_utc").sort_index()

    full_index = pd.date_range(df_raw.index.min(), df_raw.index.max(), freq="h")
    n_gap_hours = len(full_index) - len(df_raw)
    if n_gap_hours > 0:
        logger.warning(f"Reindexing introduced {n_gap_hours} missing hour(s) to fill via interpolation.")

    df_raw = df_raw.reindex(full_index)
    df_raw.index.name = "datetime_utc"
    df_raw = df_raw.reset_index()
    df_raw["timestamp_unix"] = (df_raw["datetime_utc"].astype("int64") // 10**9).astype("int64")

    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
    df_raw[numeric_cols] = df_raw[numeric_cols].interpolate(method="linear", limit=3)

    if "latitude" in df_raw.columns:
        df_raw["latitude"] = df_raw["latitude"].ffill().bfill()
    if "longitude" in df_raw.columns:
        df_raw["longitude"] = df_raw["longitude"].ffill().bfill()

    df_engineered = engineer_features_and_targets(df_raw, is_training=is_training)
    return df_engineered
