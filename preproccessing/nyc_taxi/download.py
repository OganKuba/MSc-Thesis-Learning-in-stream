import os
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def download_month(year, month):
    out_dir = os.path.join(config.RAW_DIR, "nyc_taxi")
    os.makedirs(out_dir, exist_ok=True)

    fname = f"yellow_tripdata_{year}-{month:02d}.parquet"
    dest = Path(out_dir) / fname

    if dest.exists():
        return dest

    url = f"{config.NYC_TLC_BASE_URL}/yellow_tripdata_{year}-{month:02d}.parquet"
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {fname} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def download_all():
    paths = []
    for year in config.NYC_YEARS:
        for month in config.NYC_MONTHS:
            p = download_month(year, month)
            paths.append(p)
    print(f"Total files: {len(paths)}")
    return paths


if __name__ == "__main__":
    download_all()
