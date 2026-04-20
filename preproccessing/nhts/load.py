import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


TRIP_COLS = ["trpmiles", "trvlcmin", "numontrp", "trippurp", "whyto",
             "strttime", "tdaydate", "trptrans", "travday"]

PERSON_COLS = ["r_age", "worker", "driver", "educ"]
HH_COLS     = ["hhsize", "hhvehcnt", "urbrur", "wrkcount"]

TRIP_KEYS   = ["houseid", "personid"]
HH_KEY      = "houseid"
SORT_COLS   = ["tdaydate", "strttime"]

FILE_NAMES = {
    2009: {"trip": "DAYV2PUB.csv", "person": "PERV2PUB.csv", "hh": "HHV2PUB.csv"},
    2017: {"trip": "trippub.csv",  "person": "perpub.csv",   "hh": "hhpub.csv"},
    2022: {"trip": "tripv2pub.csv","person": "perv2pub.csv", "hh": "hhv2pub.csv"},
}

RENAME = {
    2009: {
        "whytrp1s": "trippurp",
        "trvl_min": "trvlcmin",
    },
    2017: {},
    2022: {},
}


def _find(edition_dir, filename):
    target = filename.lower()
    for p in edition_dir.iterdir():
        if p.is_file() and p.name.lower() == target:
            return p
    raise FileNotFoundError(f"{filename} not found in {edition_dir}")


def _read(path, edition):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower()
    df = df.rename(columns=RENAME.get(edition, {}))
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def _select(df, wanted, keys):
    keep = [c for c in keys + wanted if c in df.columns]
    return df[keep].copy()


def load_edition(edition):
    edition_dir = Path(config.RAW_DIR) / config.NHTS_LOCAL_DIRS[edition]
    files = FILE_NAMES[edition]

    trips   = _read(_find(edition_dir, files["trip"]),   edition)
    persons = _read(_find(edition_dir, files["person"]), edition)
    hh      = _read(_find(edition_dir, files["hh"]),     edition)

    trips["trptrans"] = pd.to_numeric(trips["trptrans"], errors="coerce")
    trips = trips[trips["trptrans"].isin(config.NHTS_VALID_TRPTRANS)]

    trips   = _select(trips,   TRIP_COLS,   TRIP_KEYS)
    persons = _select(persons, PERSON_COLS, TRIP_KEYS)
    hh      = _select(hh,      HH_COLS,     [HH_KEY])

    trips = trips.merge(persons, on=TRIP_KEYS, how="left")
    trips = trips.merge(hh,      on=HH_KEY,    how="left")

    trips["edition_year"] = edition
    trips = trips.sort_values(SORT_COLS).reset_index(drop=True)
    trips = trips.loc[:, ~trips.columns.duplicated()]
    print(f"{edition}: {len(trips)} rows after filter")
    return trips


def load_all():
    frames = [load_edition(ed) for ed in sorted(config.NHTS_EDITIONS)]
    combined = pd.concat(frames, ignore_index=True, sort=False)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_DIR, "nhts_joined.csv")
    combined.to_csv(out_path, index=False)
    print(f"Saved {len(combined)} rows → {out_path}")
    return combined


if __name__ == "__main__":
    load_all()
