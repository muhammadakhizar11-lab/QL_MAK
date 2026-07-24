"""Yahoo Finance equity data: SPY / SPX daily OHLCV.

Thin wrapper over yfinance with the QL_MAK conventions: tidy lowercase
columns, UTC-naive DatetimeIndex, empty frame (never an exception) on a
failed pull, and local CSV caching so research re-runs are offline.
"""
from __future__ import annotations

import pandas as pd

from ..config import CACHE, get_logger

log = get_logger(__name__)

_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def pull_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Daily OHLCV for one ticker, columns [open, high, low, close, adj_close, volume].

    ``adj_close`` is split- and dividend-adjusted (use it for returns);
    unadjusted OHLC is kept for range-based volatility estimators, which
    need actual traded prices, not back-adjusted ones.
    """
    safe = ticker.replace("^", "_")
    cache_file = CACHE / f"yahoo_{safe}.csv"
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return df.loc[str(start): str(end) if end else None]

    import yfinance as yf

    log.info("Yahoo %s: fetching %s -> %s", ticker, start, end or "latest")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=False, progress=False, threads=False)
    if df is None or df.empty:
        log.warning("Yahoo %s: pull returned EMPTY", ticker)
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=lambda c: str(c).lower().replace(" ", "_"))
    df = df[[c for c in _COLS if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"

    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)
    return df
