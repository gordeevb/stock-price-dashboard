"""
Simplified analysis utilities: returns, rolling volatility, SMAs, and KPIs.

Exports:
    enrich(df) -> pd.DataFrame
    summarize_kpis(df) -> dict

Notes:
- Uses 'Adj Close' for all return math.
- Rolling volatility: sample stdev over 30 days, annualized by sqrt(252).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def _require_adj_close(df: pd.DataFrame) -> None:
    if "Adj Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Adj Close'.")


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a NEW dataframe with:
      ret_d, ret_w, ret_m, vol_30d_raw, vol_30d, sma_20, sma_50
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    _require_adj_close(df)

    out = df.copy()
    p = out["Adj Close"].astype("float64")

    # Returns
    out["ret_d"] = p.pct_change()
    out["ret_w"] = p.pct_change(5) * 100.0
    out["ret_m"] = p.pct_change(21) * 100.0

    # Volatility (30d)
    out["vol_30d_raw"] = out["ret_d"].rolling(window=30, min_periods=30).std()
    out["vol_30d"] = out["vol_30d_raw"] * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Moving averages
    out["sma_20"] = p.rolling(window=20, min_periods=20).mean()
    out["sma_50"] = p.rolling(window=50, min_periods=50).mean()

    return out


def summarize_kpis(df: pd.DataFrame) -> dict:
    """
    Assumes df already enriched.
    Returns:
        {
          'last_price', 'ret_1w_pct', 'ret_1m_pct',
          'ytd_pct', 'vol_30d_ann'
        }
    """
    _require_adj_close(df)
    if df.empty:
        return {k: None for k in ["last_price", "ret_1w_pct", "ret_1m_pct", "ytd_pct", "vol_30d_ann"]}

    last = df.iloc[-1]
    last_price = float(last["Adj Close"])

    # 1W/1M (already %)
    ret_1w = float(last["ret_w"]) if "ret_w" in df.columns else np.nan
    ret_1m = float(last["ret_m"]) if "ret_m" in df.columns else np.nan

    # YTD
    curr_year = df.index[-1].year
    first_this_year = df[df.index.year == curr_year].head(1)
    if not first_this_year.empty:
        p0 = float(first_this_year["Adj Close"].iloc[0])
        ytd_pct = (last_price - p0) / p0 * 100.0 if p0 else np.nan
    else:
        ytd_pct = np.nan

    # Latest 30d annualized vol
    vol_30d_ann = float(df["vol_30d"].iloc[-1]) if "vol_30d" in df.columns else np.nan

    def clean(x):
        return None if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

    return {
        "last_price": clean(last_price),
        "ret_1w_pct": clean(ret_1w),
        "ret_1m_pct": clean(ret_1m),
        "ytd_pct": clean(ytd_pct),
        "vol_30d_ann": clean(vol_30d_ann),
    }
