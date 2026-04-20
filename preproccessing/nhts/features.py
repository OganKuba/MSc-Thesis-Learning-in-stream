import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

_TARGET_MAP: dict[int, str] = {}
for _code in config.NHTS_PRIVATE_VEHICLE_CODES:
    _TARGET_MAP[_code] = "PRIVATE_VEHICLE"
for _code in config.NHTS_TRANSIT_CODES:
    _TARGET_MAP[_code] = "PUBLIC_TRANSIT"
for _code in config.NHTS_ACTIVE_CODES:
    _TARGET_MAP[_code] = "ACTIVE"

_AGE_BINS = [-1, 17, 34, 54, 64, 120]
_AGE_LABELS = [0, 1, 2, 3, 4]

_NHTS_MISSING_CODES = [-1, -7, -8, -9]

_NHTS_NUMERIC_COLS = [
    "trpmiles", "tdtrpn", "trvlcmin", "numontrp", "whyto", "trippurp",
    "strttime", "travday", "r_age", "per_r_age", "worker", "per_worker",
    "driver", "educ", "per_educ", "hhsize", "hh_hhsize",
    "hhvehcnt", "hh_hhvehcnt", "urbrur", "hh_urbrur", "msacat", "hh_msacat",
    "wrkcount", "hh_wrkcount",
]

_WHYTO_BUCKET = {
    1: 1, 2: 2, 3: 2, 4: 2, 5: 6,
    6: 7, 7: 9, 8: 3, 9: 3, 10: 3,
    11: 4, 12: 4, 13: 5, 14: 4, 15: 6,
    16: 6, 17: 6, 18: 8, 19: 6, 97: 9,
}

_TRIPPURP_STR_BUCKET = {
    "HBW": 2, "HBO": 9, "NHB": 9, "HBSHOP": 4, "HBSOCREC": 6,
    "HBSCHOOL": 3, "HBC": 6, "HBMED": 8,
}


def _to_numeric_clean(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s.isin(_NHTS_MISSING_CODES))


def _bucket_trip_purpose(whyto_series, trippurp_series):
    out = whyto_series.map(_WHYTO_BUCKET) if whyto_series is not None else None

    if trippurp_series is not None:
        if trippurp_series.dtype == object:
            fallback = trippurp_series.astype(str).str.upper().map(_TRIPPURP_STR_BUCKET)
        else:
            tp_num = pd.to_numeric(trippurp_series, errors="coerce")
            fallback = tp_num.where((tp_num >= 1) & (tp_num <= 10))

        if out is None:
            out = fallback
        else:
            out = out.fillna(fallback)

    return out


def _first_available(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _impute(df, numeric_cols, cat_cols):
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
    return df


def add_features(df):
    if "trptrans" not in df.columns:
        raise KeyError("'trptrans' column required but not found in DataFrame.")

    df = df.copy()

    for col in _NHTS_NUMERIC_COLS:
        if col in df.columns:
            df[col] = _to_numeric_clean(df[col])

    df["trptrans"] = pd.to_numeric(df["trptrans"], errors="coerce")
    df["mode_target"] = df["trptrans"].map(_TARGET_MAP)
    df = df.dropna(subset=["mode_target"])

    dist_col = _first_available(df, ["trpmiles", "tdtrpn"])
    df["trip_distance"] = (
        df[dist_col].clip(0, 200) if dist_col else float("nan")
    )

    dur_col = _first_available(df, ["trvlcmin"])
    df["trip_duration"] = (
        df[dur_col].clip(0, 300) if dur_col else float("nan")
    )

    np_col = _first_available(df, ["numontrp"])
    df["num_persons"] = (
        df[np_col].clip(1, 20) if np_col else float("nan")
    )

    whyto_s = df["whyto"] if "whyto" in df.columns else None
    trippurp_s = df["trippurp"] if "trippurp" in df.columns else None
    df["trip_purpose"] = _bucket_trip_purpose(whyto_s, trippurp_s)

    st_col = _first_available(df, ["strttime"])
    df["departure_hour"] = (
        (df[st_col] // 100).clip(0, 23) if st_col else float("nan")
    )

    td_col = _first_available(df, ["travday"])
    df["day_of_week"] = df[td_col] if td_col else float("nan")

    age_col = _first_available(df, ["r_age", "per_r_age"])
    if age_col:
        age = df[age_col].clip(0, 120)
        df["age_group"] = pd.cut(
            age, bins=_AGE_BINS, labels=_AGE_LABELS, right=True
        ).astype("Int16")
    else:
        df["age_group"] = float("nan")

    wk_col = _first_available(df, ["worker", "per_worker"])
    if wk_col:
        w = df[wk_col]
        df["worker"] = w.eq(1).astype("Int8").mask(w.isna())
    else:
        df["worker"] = float("nan")

    dr_col = _first_available(df, ["driver"])
    if dr_col:
        d = df[dr_col]
        df["driver"] = d.eq(1).astype("Int8").mask(d.isna())
    else:
        df["driver"] = float("nan")

    ed_col = _first_available(df, ["educ", "per_educ"])
    df["education"] = (
        df[ed_col].clip(1, 5) if ed_col else float("nan")
    )

    hs_col = _first_available(df, ["hhsize", "hh_hhsize"])
    df["hh_size"] = (
        df[hs_col].clip(1, 20) if hs_col else float("nan")
    )

    hv_col = _first_available(df, ["hhvehcnt", "hh_hhvehcnt"])
    df["hh_vehicles"] = (
        df[hv_col].clip(0, 20) if hv_col else float("nan")
    )

    ur_col = _first_available(df, ["urbrur", "hh_urbrur", "msacat", "hh_msacat"])
    if ur_col:
        ur_raw = df[ur_col]
        df["urban_rural"] = ur_raw.eq(1).astype("Int8").mask(ur_raw.isna())
    else:
        df["urban_rural"] = float("nan")

    hw_col = _first_available(df, ["wrkcount", "hh_wrkcount"])
    df["hh_workers"] = (
        df[hw_col].clip(0, 20) if hw_col else float("nan")
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        df["avg_speed"] = np.where(
            df["trip_duration"] > 0,
            df["trip_distance"] / (df["trip_duration"] / 60.0),
            0.0,
        )

    df["urban_x_vehicles"] = (
        df["urban_rural"].astype(float) * df["hh_vehicles"].astype(float)
    )

    numeric_feats = [
        "trip_distance", "trip_duration", "num_persons", "trip_purpose",
        "departure_hour", "day_of_week", "hh_size", "hh_vehicles",
        "hh_workers", "avg_speed", "urban_x_vehicles",
    ]
    cat_feats = ["age_group", "worker", "driver", "education", "urban_rural"]
    df = _impute(df, numeric_feats, cat_feats)

    sort_keys = [c for c in ("edition_year", "tdaydate", "strttime") if c in df.columns]
    if sort_keys:
        df = df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    if "edition_year" in df.columns:
        ed = pd.to_numeric(df["edition_year"], errors="coerce")
        df["edition_boundary"] = (
            (ed != ed.shift(1)) & ed.shift(1).notna()
        ).astype("Int8")
    else:
        df["edition_boundary"] = pd.Series(0, index=df.index, dtype="Int8")

    feature_cols = [
        "trip_distance", "trip_duration", "num_persons", "trip_purpose",
        "departure_hour", "day_of_week", "age_group", "worker", "driver",
        "education", "hh_size", "hh_vehicles", "urban_rural", "hh_workers",
        "avg_speed", "urban_x_vehicles", "edition_boundary",
        "mode_target",
    ]
    available = [c for c in feature_cols if c in df.columns]
    df = df[available].copy()

    return df


def compute_all_features(processed_dir=config.PROCESSED_DIR):
    in_path = os.path.join(processed_dir, "nhts_joined.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"'{in_path}' not found. Run nhts.load.load_all() first."
        )

    logger.info("Loading %s …", in_path)
    raw = pd.read_csv(in_path, low_memory=False)

    logger.info("Computing features …")
    featured = add_features(raw)

    out_path = os.path.join(processed_dir, "nhts_features.csv")
    featured.to_csv(out_path, index=False)
    logger.info("nhts_features.csv saved → %s  (%d rows)", out_path, len(featured))
    return featured


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = compute_all_features()
    print(df.shape)
    print(df.dtypes)
    print("\nmode_target:")
    print(df["mode_target"].value_counts())
    print("\nedition_boundary:")
    print(df["edition_boundary"].value_counts())
    print("indices where edition_boundary==1:",
          df.index[df["edition_boundary"] == 1].tolist())
    print("\nday_of_week:")
    print(df["day_of_week"].value_counts().sort_index())
    print("\ntrip_purpose:")
    print(df["trip_purpose"].value_counts().sort_index())
    print("\nNaN counts:")
    print(df.isna().sum()[df.isna().sum() > 0])
