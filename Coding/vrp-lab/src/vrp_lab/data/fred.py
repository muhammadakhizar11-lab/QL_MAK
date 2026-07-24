"""FRED client via the free fredgraph.csv endpoint (no API key).

Uses curl (not requests) as the HTTP transport: FRED's Akamai bot protection
fingerprints the TLS/HTTP client and silently stalls connections from Python's
requests/urllib3 stack while letting curl's fingerprint through. Same approach
as QL_MAK's data_pull.py.

Series used here:
  VIXCLS  — VIX index close (cross-check against CBOE's own history)
  DGS10   — 10Y Treasury constant-maturity yield (risk-free proxy)
  T10YIE  — 10Y breakeven inflation (macro context)
"""
from __future__ import annotations

import io
import subprocess

import pandas as pd

from ..config import CACHE, get_logger

log = get_logger(__name__)

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def pull_fred(series_id: str, use_cache: bool = True) -> pd.Series:
    """Fetch a single FRED series as a float Series indexed by date.

    Parameters
    ----------
    series_id : FRED series code, e.g. ``"DGS10"``.
    use_cache : if True, read/write ``data/cache/fred_<id>.csv``.

    Returns
    -------
    pd.Series named ``series_id`` with a DatetimeIndex named ``date``.
    Missing observations (FRED encodes them as ``"."``) are dropped.

    Example
    -------
    >>> dgs10 = pull_fred("DGS10")   # doctest: +SKIP
    >>> dgs10.tail()                 # doctest: +SKIP
    """
    cache_file = CACHE / f"fred_{series_id}.csv"
    if use_cache and cache_file.exists():
        log.debug("FRED %s: cache hit %s", series_id, cache_file)
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return cached.iloc[:, 0].rename(series_id)

    log.info("FRED %s: fetching via curl", series_id)
    result = subprocess.run(
        ["curl", "-s", "--max-time", "30", FRED_CSV.format(sid=series_id)],
        capture_output=True, text=True, check=True,
    )
    raw = pd.read_csv(io.StringIO(result.stdout))
    date_col, val_col = raw.columns[0], raw.columns[1]
    raw[date_col] = pd.to_datetime(raw[date_col])
    raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")
    s = raw.set_index(date_col)[val_col].dropna()
    s.index.name = "date"
    s.name = series_id

    if s.empty:
        log.warning("FRED %s: pull returned EMPTY", series_id)
    elif use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_csv(cache_file)
    return s
