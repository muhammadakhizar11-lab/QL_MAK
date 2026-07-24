"""CBOE data: VIX index history and VX futures settlement prices.

Sources (free, no key):
  * VIX daily OHLC: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv
  * VX per-contract history: https://cdn.cboe.com/data/us/futures/market_statistics/
        historical_data/VX/VX_{expiry:%Y-%m-%d}.csv
    (one CSV per contract, keyed by its final settlement date; CBOE only hosts
    contracts from ~2013 onward at this endpoint — earlier history is archived
    and noted as a limitation in docs/DATA_SOURCES.md).

Downloads go through curl for the same Akamai-fingerprinting reason as FRED.
Everything is cached under data/cache/ so a full re-run is offline.
"""
from __future__ import annotations

import io
import subprocess

import pandas as pd

from ..backtest.calendar import vx_expiries
from ..config import CACHE, get_logger, load_config

log = get_logger(__name__)


def _curl(url: str, timeout: int = 30) -> str:
    result = subprocess.run(
        ["curl", "-s", "-f", "--max-time", str(timeout), url],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def pull_vix_history(use_cache: bool = True) -> pd.DataFrame:
    """VIX index daily OHLC from CBOE, columns [open, high, low, close].

    Returns a frame with DatetimeIndex named ``date``. This is the cash index —
    not tradeable, but it anchors the term structure and the IV leg of the VRP.
    """
    cache_file = CACHE / "vix_history.csv"
    if use_cache and cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    url = load_config()["data"]["cboe"]["vix_history_url"]
    log.info("CBOE VIX history: fetching %s", url)
    raw = pd.read_csv(io.StringIO(_curl(url)))
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw.set_index("date")[["open", "high", "low", "close"]].astype(float)

    if df.empty:
        log.warning("CBOE VIX history: pull returned EMPTY")
    elif use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)
    return df


def pull_vx_contract(expiry: pd.Timestamp, use_cache: bool = True) -> pd.DataFrame:
    """Daily settlement history for one VX contract, keyed by its expiry date.

    Returns columns [settle, volume, open_interest] indexed by trade date,
    with the contract ``expiry`` attached in an ``expiry`` column. Empty frame
    if CBOE has no file for that contract (pre-2013 history, or future expiry).
    """
    cache_file = CACHE / f"vx_{expiry:%Y-%m-%d}.csv"
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df["expiry"] = pd.Timestamp(expiry)
        return df

    url_tpl = load_config()["data"]["cboe"]["vx_settlements_url"]
    url = url_tpl.format(expiry=f"{expiry:%Y-%m-%d}")
    try:
        text = _curl(url)
        raw = pd.read_csv(io.StringIO(text))
    except (subprocess.CalledProcessError, pd.errors.ParserError) as e:
        log.warning("CBOE VX %s: no data (%s)", f"{expiry:%Y-%m-%d}", type(e).__name__)
        return pd.DataFrame()

    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    if "trade_date" not in raw.columns or raw.empty:
        log.warning("CBOE VX %s: unexpected schema %s", f"{expiry:%Y-%m-%d}", list(raw.columns))
        return pd.DataFrame()

    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    df = raw.set_index("trade_date")[["settle", "total_volume", "open_interest"]].rename(
        columns={"total_volume": "volume"}
    ).astype(float)
    df.index.name = "date"
    df = df[df["settle"] > 0]  # zero settle rows = no trading yet

    if use_cache and not df.empty:
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)
    df["expiry"] = pd.Timestamp(expiry)
    return df


def pull_vx_curve(start: str, end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Stacked settlement history for every VX contract expiring in [start, end].

    Long format: index=trade date, columns [settle, volume, open_interest, expiry].
    Feed this to ``volatility.term_structure.build_curve`` to get VX1/VX2 series.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    # include contracts expiring shortly after `end` so the front month is
    # always defined on the last trade dates
    expiries = vx_expiries(start, pd.Timestamp(end) + pd.DateOffset(months=3))
    frames = [pull_vx_contract(e, use_cache=use_cache) for e in expiries]
    frames = [f for f in frames if not f.empty]
    if not frames:
        log.warning("VX curve: no contract data in [%s, %s]", start, end)
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out.loc[str(start):str(end)]
