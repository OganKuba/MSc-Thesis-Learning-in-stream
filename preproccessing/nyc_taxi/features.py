

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

_WARMUP_ROWS = 24

_US_HOLIDAYS = frozenset([
    (1, 1),   # New Year's Day
    (1, 15),  # MLK Day (approx.)
    (2, 19),  # Presidents Day (approx.)
    (5, 27),  # Memorial Day (approx.)
    (6, 19),  # Juneteenth
    (7, 4),   # Independence Day
    (9, 2),   # Labor Day (approx.)
    (11, 11), # Veterans Day
    (11, 28), # Thanksgiving (approx.)
    (12, 25), # Christmas
])


def _is_holiday(dt_series: pd.Series) -> pd.Series:
    return pd.array(
        [int((m, d) in _US_HOLIDAYS)
         for m, d in zip(dt_series.dt.month, dt_series.dt.day)],
        dtype="int8",
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"zone_id", "timestamp", "trip_count"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

    dt = df["timestamp"]


    hour = dt.dt.hour
    dow = dt.dt.dayofweek

    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    df["cos_dow"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype("int8")
    df["is_holiday"] = _is_holiday(dt)
    df["hour_since_midnight"] = (hour + dt.dt.minute / 60).astype("float32")


    grp = df.groupby("zone_id")["trip_count"]

    df["trip_count_lag1"] = grp.shift(1)
    df["trip_count_lag2"] = grp.shift(2)

    df["rolling_mean_4h"] = grp.transform(
        lambda s: s.shift(1).rolling(4, min_periods=4).mean()
    )
    df["rolling_mean_12h"] = grp.transform(
        lambda s: s.shift(1).rolling(12, min_periods=12).mean()
    )
    df["rolling_mean_24h"] = grp.transform(
        lambda s: s.shift(1).rolling(24, min_periods=24).mean()
    )
    df["rolling_std_24h"] = grp.transform(
        lambda s: s.shift(1).rolling(24, min_periods=24).std()
    )
    df["delta_trip_count"] = df["trip_count"] - df["trip_count_lag1"]


    df["zone_type"] = df["zone_id"].map(config.ZONE_TYPE).fillna(3).astype("int8")

    pivot = df.pivot_table(
        index="timestamp", columns="zone_id", values="trip_count", aggfunc="sum"
    )

    def _neighbor_avg(row: pd.Series) -> pd.Series:
        result = {}
        for zone in pivot.columns:
            neighbors = config.ZONE_NEIGHBORS.get(int(zone), [])
            avail = [n for n in neighbors if n in pivot.columns]
            result[zone] = row[avail].mean() if avail else float("nan")
        return pd.Series(result)

    neighbor_pivot = pivot.apply(_neighbor_avg, axis=1)
    neighbor_long = (
        neighbor_pivot.reset_index()
        .melt(id_vars="timestamp", var_name="zone_id", value_name="neighbor_avg_demand")
    )
    neighbor_long["zone_id"] = neighbor_long["zone_id"].astype(int)
    df = df.merge(neighbor_long, on=["timestamp", "zone_id"], how="left")


    expanding_median = grp.transform(lambda s: s.shift(1).expanding().median())
    df["demand_level"] = np.where(
        df["trip_count"] > expanding_median, "HIGH", "LOW"
    )


    df["_row_num"] = df.groupby("zone_id").cumcount()
    df = df[df["_row_num"] >= _WARMUP_ROWS].drop(columns=["_row_num"])

    before = len(df)
    feature_cols = [
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "is_weekend", "is_holiday", "hour_since_midnight",
        "trip_count_lag1", "trip_count_lag2",
        "rolling_mean_4h", "rolling_mean_12h", "rolling_mean_24h",
        "rolling_std_24h", "delta_trip_count",
        "zone_type", "neighbor_avg_demand",
        "demand_level",
    ]
    df = df.dropna(subset=[c for c in feature_cols if c != "demand_level"])
    dropped = before - len(df)
    if dropped > 0:
        logger.debug("Dropped %d rows with NaN features", dropped)

    df = df.sort_values(["timestamp", "zone_id"]).reset_index(drop=True)
    return df


def compute_all_features(
    processed_dir: str = config.PROCESSED_DIR,
) -> pd.DataFrame:

    in_path = os.path.join(processed_dir, "nyc_taxi_aggregated.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"'{in_path}' not found. Run nyc_taxi.aggregate first."
        )

    logger.info("Loading %s …", in_path)
    raw = pd.read_csv(in_path, parse_dates=["timestamp"])

    logger.info("Computing features …")
    featured = add_features(raw)

    out_path = os.path.join(processed_dir, "nyc_taxi_features.csv")
    featured.to_csv(out_path, index=False)
    logger.info("nyc_taxi_features.csv saved → %s  (%d rows)", out_path, len(featured))
    return featured


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = compute_all_features()
    print(df.shape)
    print(df.dtypes)
    print(df["demand_level"].value_counts())
