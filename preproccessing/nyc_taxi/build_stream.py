

import logging
import os
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.arff_writer import dataframe_to_arff

logger = logging.getLogger(__name__)

RELATION_NAME = "nyc_taxi_stream"
TARGET_COL = "demand_level"
DROP_COLS = {"timestamp", "zone_id", "pickup_zone", "pickup_hour",
             "demand", "trip_count"}  # identifiers / raw demand


def build_stream(
    processed_dir: str = config.PROCESSED_DIR,
    arff_dir: str = config.ARFF_DIR,
) -> pd.DataFrame:

    features_path = os.path.join(processed_dir, "nyc_taxi_features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"'{features_path}' not found. "
            "Run nyc_taxi.features.compute_all_features() first."
        )

    logger.info("Loading %s …", features_path)
    df = pd.read_csv(features_path, parse_dates=["timestamp"])


    os.makedirs(arff_dir, exist_ok=True)

    arff_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    arff_path = os.path.join(arff_dir, "nyc_taxi.arff")
    dataframe_to_arff(
        df=arff_df,
        relation_name=RELATION_NAME,
        target_col=TARGET_COL,
        output_path=arff_path,
    )
    logger.info("ARFF saved → %s", arff_path)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = build_stream()
    print(f"\n=== NYC Taxi stream statistics ===")
    print(f"Total rows   : {len(df):,}")
    print(f"Zones        : {df['zone_id'].nunique()}")
    if "timestamp" in df.columns:
        print(f"Time range   : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"\nTarget distribution:")
    print(df["demand_level"].value_counts().to_string())
    print(f"\nFeature dtypes (first 10):")
    print(df.dtypes.head(10).to_string())
