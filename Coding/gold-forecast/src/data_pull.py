"""
data_pull.py — fetch all raw series from FREE public sources.

  * Market data (OHLCV): Yahoo Finance via `yfinance`.
  * Macro series:        FRED via the free `fredgraph.csv` endpoint (no API key).

Everything lands in data/raw/ as one CSV per series, plus a manifest.json that
records source, pull timestamp, row count, date span, and any warnings. Nothing
here silently swallows an empty pull — a series that comes back empty is recorded
as such so the validation stage (and you) can see it.

Run:  python -m src.data_pull
"""
from __future__ import annotations

import io
import json
import warnings
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import RAW, load_config, ensure_dirs

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_UA = {"User-Agent": "Mozilla/5.0 (research; gold-forecast)"}


# --------------------------------------------------------------------------- #
# Yahoo (market data)
# --------------------------------------------------------------------------- #
def pull_yahoo(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Return a tidy OHLCV frame (DatetimeIndex, UTC-naive) or empty on failure."""
    import yfinance as yf

    df = yf.download(
        ticker, start=start, end=end,
        auto_adjust=False, progress=False, threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance can return a MultiIndex column frame for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "adj close", "volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df


# --------------------------------------------------------------------------- #
# FRED (macro) — free CSV endpoint, no key
# --------------------------------------------------------------------------- #
def pull_fred(series_id: str) -> pd.Series:
    """Return a single FRED series as a pandas Series (name=series_id)."""
    r = requests.get(FRED_CSV.format(sid=series_id), headers=_UA, timeout=30)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    # fredgraph.csv columns: first is the date (name varies), second is the value.
    date_col, val_col = raw.columns[0], raw.columns[1]
    raw[date_col] = pd.to_datetime(raw[date_col])
    # FRED encodes missing as ".".
    raw[val_col] = pd.to_numeric(raw[val_col], errors="coerce")
    s = raw.set_index(date_col)[val_col].dropna()
    s.index.name = "date"
    s.name = series_id
    return s


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _span(idx: pd.Index) -> list:
    if len(idx) == 0:
        return [None, None]
    return [str(idx.min().date()), str(idx.max().date())]


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs()
    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]  # None => yfinance/FRED default to latest

    manifest: dict = {
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": start, "end": end, "series": {},
    }

    # ---- Yahoo tickers (flatten all groups) ----
    yahoo_map: dict[str, str] = {}
    for group in cfg["yahoo"].values():
        yahoo_map.update(group)

    for ticker, alias in yahoo_map.items():
        warn = None
        try:
            df = pull_yahoo(ticker, start, end)
        except Exception as e:  # network / parsing — record, keep going
            df, warn = pd.DataFrame(), f"pull_error: {e!r}"

        fname = f"{alias}.csv"
        if df.empty:
            warn = warn or "EMPTY — no data returned"
        else:
            df.to_csv(RAW / fname)

        manifest["series"][alias] = {
            "source": "yahoo", "ticker": ticker, "file": fname if not df.empty else None,
            "rows": int(len(df)), "span": _span(df.index), "warning": warn,
        }
        print(f"[yahoo] {ticker:<10} {alias:<18} rows={len(df):<6} {warn or ''}")

    # ---- FRED series ----
    for sid, meta in cfg["fred"].items():
        alias = meta["name"]
        warn = None
        try:
            s = pull_fred(sid)
        except Exception as e:
            s, warn = pd.Series(dtype=float), f"pull_error: {e!r}"

        fname = f"{alias}.csv"
        if s.empty:
            warn = warn or "EMPTY — no data returned"
        else:
            s.to_frame().to_csv(RAW / fname)

        manifest["series"][alias] = {
            "source": "fred", "series_id": sid, "freq": meta["freq"],
            "release_lag_months": meta["release_lag_months"],
            "file": fname if not s.empty else None,
            "rows": int(len(s)), "span": _span(s.index), "warning": warn,
        }
        print(f"[fred]  {sid:<10} {alias:<18} rows={len(s):<6} {warn or ''}")

    with open(RAW / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Loud, honest summary of anything that failed or looks thin.
    empties = [a for a, m in manifest["series"].items() if m.get("warning")]
    if empties:
        warnings.warn(f"{len(empties)} series had warnings: {empties}", stacklevel=2)
    print(f"\nManifest -> {RAW / 'manifest.json'}")
    return manifest


if __name__ == "__main__":
    run()
