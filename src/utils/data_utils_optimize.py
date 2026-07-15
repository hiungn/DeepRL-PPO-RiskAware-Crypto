"""
data_utils_optimize.py
Single-asset data pipeline for futures RL trading.

Differences from data_utils.py:
  - Single asset (default BTC-USD) instead of multi-asset dict
  - 8 features instead of 6: adds Volume_zscore + synthetic Funding_rate
  - load_single_asset() returns a single DataFrame
  - get_split() takes DataFrame, returns (train_df, test_df)

Walk-forward splits (same as original):
  Window 1: train [2020, 2023), test [2023, 2024)
  Window 2: train [2021, 2024), test [2024, 2025)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKER = "BTC-USD"

START_DATE = "2020-01-01"
END_DATE   = "2025-06-01"
DATA_DIR   = os.path.join(os.path.dirname(__file__), "../../data")

WALK_FORWARD_SPLITS = [
    {"train_start": "2020-01-01", "train_end": "2023-01-01",
     "test_start":  "2023-01-01", "test_end":  "2024-01-01", "id": 1},
    {"train_start": "2021-01-01", "train_end": "2024-01-01",
     "test_start":  "2024-01-01", "test_end":  "2025-01-01", "id": 2},
]

INDICATOR_WINDOW_NORM = 60
EXTREME_RETURN_THRESH = 0.5

FEATURE_COLS = [
    "Close", "MA7", "MA21", "RSI", "MACD", "Volatility",
    "Volume_zscore", "Funding_rate",
]
N_FEATURES = len(FEATURE_COLS)  # 8


# ──────────────────────────────────────────────────────────────
# Download & cache (reused from original)
# ──────────────────────────────────────────────────────────────

def _download_single(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV for 1 ticker, cache to CSV."""
    cache_path = os.path.join(
        DATA_DIR,
        f"{ticker.replace('-', '_')}_{start[:4]}_{end[:4]}.csv",
    )
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = yf.download(
        ticker, start=start, end=end,
        interval="1d", progress=False, auto_adjust=True,
    )

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


# ──────────────────────────────────────────────────────────────
# Feature engineering (extended to 8 features)
# ──────────────────────────────────────────────────────────────

def _add_indicators(
    df: pd.DataFrame,
    funding_base_rate: float = 0.0003,
) -> pd.DataFrame:
    """
    Compute 8 features for a single asset.

    Original 6: Close, MA7, MA21, RSI, MACD, Volatility
    New 2:      Volume_zscore, Funding_rate
    """
    close  = df["Close"].astype(float)
    volume = df["Volume"].astype(float) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    df = df.copy()

    # ── Original 6 features ──
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

    # MACD
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # ── New feature 1: Volume z-score ──
    # log(Volume + 1) to compress dynamic range, then rolling z-score later
    df["Volume_zscore"] = np.log1p(volume)

    # ── New feature 2: Synthetic funding rate ──
    # Perpetual futures: longs pay shorts when price > MA21 (bullish consensus)
    #                    shorts pay longs when price < MA21 (bearish consensus)
    # This is a simplified model; real funding rates come from exchange API.
    ma21 = df["MA21"]
    df["Funding_rate"] = funding_base_rate * np.sign(close - ma21)

    df = df.bfill().fillna(0)
    return df


# ──────────────────────────────────────────────────────────────
# Normalization (reused from original)
# ──────────────────────────────────────────────────────────────

def _rolling_zscore(
    df: pd.DataFrame,
    cols: list,
    window: int = INDICATOR_WINDOW_NORM,
) -> pd.DataFrame:
    """
    Rolling z-score: z_t = (x_t - mean_{t-W:t}) / (std_{t-W:t} + eps)
    No look-ahead: only uses past data up to t.
    """
    df = df.copy()
    for c in cols:
        roll_mean = df[c].rolling(window, min_periods=10).mean()
        roll_std  = df[c].rolling(window, min_periods=10).std()
        df[c] = (df[c] - roll_mean) / (roll_std + 1e-8)
    df = df.bfill().fillna(0)
    return df


# ──────────────────────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────────────────────

def load_single_asset(
    ticker:    str   = DEFAULT_TICKER,
    normalize: bool  = True,
    start:     str   = START_DATE,
    end:       str   = END_DATE,
    funding_base_rate: float = 0.0003,
) -> pd.DataFrame:
    """
    Load and preprocess data for a single asset.

    Returns
    -------
    pd.DataFrame
        columns = FEATURE_COLS + ["CloseRaw", "Return1d"]
        index   = DatetimeIndex
    """
    df = _download_single(ticker, start, end)
    df = _add_indicators(df, funding_base_rate=funding_base_rate)

    short_name = ticker.split("-")[0]

    # Drop extreme return days
    mask   = df["Return1d"].abs() > EXTREME_RETURN_THRESH
    n_drop = mask.sum()
    if n_drop > 0:
        dropped = df.index[mask].tolist()
        print(
            f"[data_utils_optimize] {short_name}: dropping {n_drop} day(s) "
            f"with |return|>{EXTREME_RETURN_THRESH*100:.0f}%: "
            f"{[str(d.date()) for d in dropped]}"
        )
        df = df[~mask].copy()

    # Save raw close BEFORE normalization (for log-return in env)
    df["CloseRaw"] = df["Close"].copy()

    # Normalize features (rolling z-score)
    if normalize:
        # Normalize all features except Funding_rate (already small scale)
        norm_cols = [c for c in FEATURE_COLS if c != "Funding_rate"]
        df = _rolling_zscore(df, norm_cols)

    n_days = len(df)
    print(
        f"[data_utils_optimize] Loaded {short_name}: {n_days} days "
        f"({df.index[0].date()} to {df.index[-1].date()})"
    )
    return df


# ──────────────────────────────────────────────────────────────
# Walk-forward split
# ──────────────────────────────────────────────────────────────

def get_split(df: pd.DataFrame, split: dict) -> tuple:
    """
    Slice single-asset DataFrame by 1 walk-forward split.

    Returns
    -------
    (train_df, test_df) : tuple of pd.DataFrame
    """
    train_df = df.loc[split["train_start"]:split["train_end"]].iloc[:-1].copy()
    test_df  = df.loc[split["test_start"] :split["test_end"] ].iloc[:-1].copy()

    print(
        f"[data_utils_optimize] Split W{split['id']}: "
        f"train={len(train_df)}d, test={len(test_df)}d"
    )
    return train_df, test_df


# ──────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_single_asset(normalize=True)
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFeature stats (first 5 rows):")
    print(df[FEATURE_COLS].head())
    print(f"\nFunding rate sample:")
    print(df["Funding_rate"].describe())

    for split_cfg in WALK_FORWARD_SPLITS:
        train, test = get_split(df, split_cfg)
        print(f"  W{split_cfg['id']} train: {train.shape}, test: {test.shape}")
