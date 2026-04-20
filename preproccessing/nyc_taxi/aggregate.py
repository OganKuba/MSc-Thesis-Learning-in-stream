import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def aggregate_file(path):
    df = pd.read_parquet(path, columns=[
        "tpep_pickup_datetime", "tpep_dropoff_datetime",
        "PULocationID", "trip_distance", "fare_amount", "tip_amount",
    ])

    df = df.dropna(subset=["PULocationID"])
    df = df[df["trip_distance"].between(0.1, 100)]
    df = df[df["fare_amount"].between(1, 500)]
    df = df[df["PULocationID"].astype(int).isin(config.NYC_ZONES)]

    if df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "zone_id", "trip_count",
            "avg_trip_distance", "avg_trip_duration",
            "avg_fare_amount", "avg_tip_pct",
        ])

    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
    df = df.dropna(subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"])

    df["trip_duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0

    df = df[df["trip_duration"].between(1, 180)]

    year_str = Path(path).stem.split("_")[-1]
    file_year = int(year_str.split("-")[0])
    file_month = int(year_str.split("-")[1])
    df = df[
        (df["tpep_pickup_datetime"].dt.year == file_year)
        & (df["tpep_pickup_datetime"].dt.month == file_month)
    ]

    if df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "zone_id", "trip_count",
            "avg_trip_distance", "avg_trip_duration",
            "avg_fare_amount", "avg_tip_pct",
        ])

    df["timestamp"] = df["tpep_pickup_datetime"].dt.floor("h")
    df["tip_pct"] = df["tip_amount"] / df["fare_amount"] * 100
    df["zone_id"] = df["PULocationID"].astype(int)

    grp = df.groupby(["zone_id", "timestamp"])
    result = grp.agg(
        trip_count=("zone_id", "size"),
        avg_trip_distance=("trip_distance", "mean"),
        avg_trip_duration=("trip_duration", "mean"),
        avg_fare_amount=("fare_amount", "mean"),
        avg_tip_pct=("tip_pct", "mean"),
    ).reset_index()

    return result[["timestamp", "zone_id", "trip_count",
                   "avg_trip_distance", "avg_trip_duration",
                   "avg_fare_amount", "avg_tip_pct"]]


def aggregate_all():
    taxi_dir = os.path.join(config.RAW_DIR, "nyc_taxi")
    parquet_files = sorted(Path(taxi_dir).glob("yellow_tripdata_*.parquet"))

    frames = [aggregate_file(p) for p in parquet_files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["timestamp", "zone_id"]).reset_index(drop=True)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "nyc_taxi_aggregated.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} rows → {out_path}")
    return combined


if __name__ == "__main__":
    aggregate_all()
