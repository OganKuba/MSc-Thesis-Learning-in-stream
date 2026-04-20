import logging
import os
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from utils.arff_writer import dataframe_to_arff

logger = logging.getLogger(__name__)

RELATION_NAME = "nhts_stream"
TARGET_COL = "mode_target"
DROP_COLS = {
    "houseid", "personid", "tdaydate", "strttime",
    "nhts_year", "edition_year",
}


def build_stream(
    processed_dir: str = config.PROCESSED_DIR,
    arff_dir: str = config.ARFF_DIR,
) -> pd.DataFrame:

    features_path = os.path.join(processed_dir, "nhts_features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"'{features_path}' not found. "
            "Run nhts.features.compute_all_features() first."
        )

    logger.info("Loading %s …", features_path)
    df = pd.read_csv(features_path, low_memory=False)


    os.makedirs(arff_dir, exist_ok=True)

    arff_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    arff_path = os.path.join(arff_dir, "nhts.arff")
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
    print(f"\n=== NHTS stream statistics ===")
    print(f"Total rows   : {len(df):,}")
    if "edition_boundary" in df.columns:
        print(f"Edition dist.:")
        print(df["edition_boundary"].value_counts().sort_index().to_string())
    print(f"\nTarget distribution:")
    print(df["mode_target"].value_counts().to_string())
    print(f"\nFeature dtypes:")
    print(df.dtypes.to_string())
