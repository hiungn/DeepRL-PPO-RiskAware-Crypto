"""
Download and preprocess OHLCV data for crypto assets.

Walk-forward splits:
  Window 1: train [2020, 2023), test [2023, 2024)
  Window 2: train [2021, 2024), test [2024, 2025)

Assets: BTC, ETH, BNB + CASH (modeled inside env)
Features (6): Close, MA7, MA21, RSI14, MACD, Volatility14
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["BTC-USD", "ETH-USD", "BNB-USD"]

START_DATE = "2020-01-01"
END_DATE   = "2025-06-01"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../../data")

WALK_FORWARD_SPLITS = [
    {"train_start": "2020-01-01", "train_end": "2023-01-01",
     "test_start":  "2023-01-01", "test_end":  "2024-01-01", "id": 1},
    {"train_start": "2021-01-01", "train_end": "2024-01-01",
     "test_start":  "2024-01-01", "test_end":  "2025-01-01", "id": 2},
]

INDICATOR_WINDOW_NORM = 60   # rolling z-score window
EXTREME_RETURN_THRESH = 0.5  # drop days where |1d return| > 50% 

FEATURE_COLS = ["Close", "MA7", "MA21", "RSI", "MACD", "Volatility"]


def _download_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for 1 ticker, normalize column names, cache to CSV."""
    cache_path = os.path.join(DATA_DIR, f"{ticker.replace('-','_')}_{start[:4]}_{end[:4]}.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.columns = [c.strip().title() for c in df.columns]

    if "Close" not in df.columns:
        for c in df.columns:
            if "close" in c.lower():
                df = df.rename(columns={c: "Close"})
                break
    if "Volume" not in df.columns:
        for c in df.columns:
            if "vol" in c.lower():
                df = df.rename(columns={c: "Volume"})
                break

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna().copy()
    df.to_csv(cache_path)
    return df


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators for 1 asset DataFrame.
    Features: Close, MA7, MA21, RSI14, MACD, Volatility14
    """
    close = df["Close"].astype(float)

    df = df.copy()
    df["MA7"]        = close.rolling(7,  min_periods=1).mean()
    df["MA21"]       = close.rolling(21, min_periods=1).mean()
    df["Return1d"]   = close.pct_change()
    df["Volatility"] = df["Return1d"].rolling(14, min_periods=2).std()

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs    = gain / (loss + 1e-8)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (EMA12 - EMA26)
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    df = df.bfill().fillna(0)
    return df


def _rolling_zscore(df: pd.DataFrame, cols: list, window: int = INDICATOR_WINDOW_NORM) -> pd.DataFrame:
    """
    Rolling z-score normalization per column.
    z_t = (x_t - mean_{t-window:t}) / (std_{t-window:t} + eps)
    No look-ahead: only uses past data up to t.
    """
    df = df.copy()
    for c in cols:
        roll_mean = df[c].rolling(window, min_periods=10).mean()
        roll_std  = df[c].rolling(window, min_periods=10).std()
        df[c] = (df[c] - roll_mean) / (roll_std + 1e-8)
    df = df.bfill().fillna(0)
    return df


def load_all_data(
    normalize: bool = True,
    tickers:   list = None,
    start:     str  = START_DATE,
    end:       str  = END_DATE,
) -> dict:
    """
    Load and preprocess data for N assets.

    Returns
    dict: {ticker_short: pd.DataFrame}
          columns = FEATURE_COLS + ["CloseRaw", "Return1d"]
          index   = DatetimeIndex, aligned (inner join) across all assets
    """
    if tickers is None:
        tickers = DEFAULT_TICKERS

    raw = {}
    for ticker in tickers:
        df  = _download_single(ticker, start, end)
        df  = _add_indicators(df)
        key = ticker.split("-")[0]

        # Drop days with extreme price gaps (data anomalies, e.g. BNB 2021-02-19)
        mask   = df["Return1d"].abs() > EXTREME_RETURN_THRESH
        n_drop = mask.sum()
        if n_drop > 0:
            dropped = df.index[mask].tolist()
            print(f"[data_utils] {key}: dropping {n_drop} day(s) with |return|>{EXTREME_RETURN_THRESH*100:.0f}%: "
                  f"{[str(d.date()) for d in dropped]}")
            df = df[~mask].copy()

        raw[key] = df

    # Align on common trading days (inner join)
    keys = list(raw.keys())
    common_idx = raw[keys[0]].index
    for key in keys[1:]:
        common_idx = common_idx.intersection(raw[key].index)
    common_idx = common_idx.sort_values()

    result = {}
    for key, df in raw.items():
        df = df.loc[common_idx].copy()
        df["CloseRaw"] = df["Close"].copy()
        if normalize:
            df = _rolling_zscore(df, FEATURE_COLS)
        result[key] = df

    n_assets = len(result)
    print(f"[data_utils] Loaded {n_assets} assets, {len(common_idx)} aligned days "
          f"({common_idx[0].date()} to {common_idx[-1].date()})")
    print(f"[data_utils] Assets: {list(result.keys())}")
    return result


def get_split(data: dict, split: dict) -> tuple:
    """
    Slice data dict by 1 walk-forward split.

    Returns
    (train_data, test_data) : each is dict {ticker_short: pd.DataFrame}
    """
    train_data, test_data = {}, {}
    for key, df in data.items():
        train_data[key] = df.loc[split["train_start"]:split["train_end"]].iloc[:-1].copy()
        test_data[key]  = df.loc[split["test_start"] :split["test_end"] ].iloc[:-1].copy()

    n_train = len(next(iter(train_data.values())))
    n_test  = len(next(iter(test_data.values())))
    print(f"[data_utils] Split W{split['id']}: train={n_train}d, test={n_test}d")
    return train_data, test_data


if __name__ == "__main__":
    data = load_all_data(normalize=True)
    for split_cfg in WALK_FORWARD_SPLITS:
        train, test = get_split(data, split_cfg)
        print(f"  BTC train: {train['BTC'].shape}, test: {test['BTC'].shape}")
