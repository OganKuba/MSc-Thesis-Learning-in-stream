

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import ta

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

logger = logging.getLogger(__name__)

_WARMUP_ROWS = 26
_UP_THRESHOLD = 0.005
_DOWN_THRESHOLD = -0.005


def add_features(df: pd.DataFrame) -> pd.DataFrame:

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required OHLCV columns: {missing}")

    df = df.copy().sort_index()

    if len(df) <= _WARMUP_ROWS + 1:
        logger.debug("Series too short (%d rows) — returning empty DataFrame", len(df))
        return df.iloc[0:0].copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["SMA_5"] = close.rolling(5, min_periods=5).mean()
    df["SMA_20"] = close.rolling(20, min_periods=20).mean()
    df["EMA_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]

    rsi = ta.momentum.RSIIndicator(close=close, window=14)
    df["RSI_14"] = rsi.rsi()

    rolling_std_20 = close.rolling(20, min_periods=20).std()
    df["BB_width"] = 4 * rolling_std_20 / df["SMA_20"]

    atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14)
    df["ATR_14"] = atr.average_true_range()

    low_14 = low.rolling(14, min_periods=14).min()
    high_14 = high.rolling(14, min_periods=14).max()
    hl_range = high_14 - low_14
    df["Stoch_K"] = 100 * (close - low_14) / hl_range.replace(0, np.nan)
    df["Stoch_D"] = df["Stoch_K"].rolling(3, min_periods=3).mean()
    df["Momentum_10"] = close - close.shift(10)

    obv = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume)
    df["OBV"] = obv.on_balance_volume()

    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std()
    df["Volume_zscore"] = (volume - vol_mean) / (vol_std + 1e-9)

    df["daily_return"] = close.pct_change(1)
    df["log_return"] = np.log(close / close.shift(1))
    df["volatility_20"] = df["daily_return"].rolling(20, min_periods=20).std()

    for lag in range(1, 6):
        df[f"daily_return_lag{lag}"] = df["daily_return"].shift(lag)
        df[f"RSI_14_lag{lag}"] = df["RSI_14"].shift(lag)
        df[f"Volume_zscore_lag{lag}"] = df["Volume_zscore"].shift(lag)


    next_return = close.shift(-1) / close - 1
    df["target"] = np.select(
        [next_return > _UP_THRESHOLD, next_return < _DOWN_THRESHOLD],
        ["UP", "DOWN"],
        default="NEUTRAL",
    )

    df = df.iloc[_WARMUP_ROWS:-1]

    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped > 0:
        logger.debug("Dropped %d rows with NaN features", dropped)

    return df


def compute_all_features(
    raw_dir: str = config.RAW_DIR,
    processed_dir: str = config.PROCESSED_DIR,
) -> pd.DataFrame:

    all_path = os.path.join(raw_dir, "yahoo", "all_tickers.csv")
    if not os.path.exists(all_path):
        raise FileNotFoundError(
            f"'{all_path}' not found. Run yahoo_finance.download first."
        )

    logger.info("Reading %s …", all_path)
    raw = pd.read_csv(all_path, parse_dates=["Date"])
    raw = raw.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for ticker, grp in raw.groupby("Ticker", sort=True):
        try:
            ohlcv = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
            featured = add_features(ohlcv)
            featured = featured.reset_index()
            featured["Ticker"] = ticker
            frames.append(featured)
            logger.debug("  %s: %d feature rows", ticker, len(featured))
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)

    if not frames:
        raise RuntimeError("No ticker produced valid features.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "yahoo_features.csv")
    combined.to_csv(out_path, index=False)
    logger.info(
        "yahoo_features.csv saved → %s  (%d rows, %d tickers)",
        out_path, len(combined), combined["Ticker"].nunique(),
    )
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = compute_all_features()
    print(df.shape)
    print(df.dtypes)
    print(df["target"].value_counts())
