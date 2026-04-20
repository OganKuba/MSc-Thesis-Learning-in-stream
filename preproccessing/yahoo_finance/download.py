import os
import time
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def download_ticker(ticker, start, end, interval):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()
    df["Ticker"] = ticker
    return df


def download_all():
    out_dir = os.path.join(config.RAW_DIR, "yahoo")
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for ticker in config.YAHOO_TICKERS:
        dest = os.path.join(out_dir, f"{ticker}.csv")
        df = download_ticker(
            ticker,
            start=config.YAHOO_START_DATE,
            end=config.YAHOO_END_DATE,
            interval=config.YAHOO_INTERVAL,
        )
        df.reset_index().to_csv(dest, index=False)
        results[ticker] = df
        print(f"{ticker}: {len(df)} rows")
        time.sleep(0.5)

    frames = []
    for ticker, df in results.items():
        frame = df.reset_index()
        frame["Ticker"] = ticker
        frames.append(frame)
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(out_dir, "all_tickers.csv"), index=False)
    print(f"Saved {len(all_df)} rows for {len(results)} tickers")

    return results


if __name__ == "__main__":
    download_all()
