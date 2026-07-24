"""Empirical VRP study: dataset assembly + conditioning analysis.

``build_dataset`` is the one function that touches the network (through the
cached data layer); everything else takes frames in and returns stats, so the
notebooks stay thin and the logic stays testable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PROCESSED, get_logger, load_config
from ..data import cboe, fred, yfinance
from ..data.validators import validate_frame
from ..volatility import term_structure as ts
from ..volatility.vrp import iv_from_vix, vrp_panel

log = get_logger(__name__)


def build_dataset(start: str | None = None, end: str | None = None,
                  save: bool = True) -> pd.DataFrame:
    """Assemble the aligned daily research dataset.

    Columns: spy ohlcv, spy_ret, vix (CBOE), ust10y, vx1/vx2/dte1/dte2 +
    term-structure fields (where VX data exists, ~2013+), and the VRP panel
    (ex-post + trailing, 21d and 63d).
    """
    cfg = load_config()
    start = start or cfg["date_range"]["start"]
    end = end or cfg["date_range"]["end"]

    spy = yfinance.pull_ohlcv("SPY", start, end)
    validate_frame(spy, "spy", required_cols=["open", "high", "low", "close", "adj_close"])
    vix = cboe.pull_vix_history()["close"].loc[str(start):]
    ust10y = fred.pull_fred("DGS10").loc[str(start):]

    df = spy.copy()
    df["spy_ret"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
    df["vix"] = vix.reindex(df.index).ffill(limit=cfg["data"]["validation"]["max_ffill_gap_days"])
    df["ust10y"] = ust10y.reindex(df.index).ffill(limit=10)

    panel = vrp_panel(iv_from_vix(df["vix"]), df["spy_ret"],
                      horizons=cfg["volatility"]["vrp_horizons"],
                      zscore_window=cfg["volatility"]["zscore_window"])
    df = df.join(panel)

    vx = cboe.pull_vx_curve(start, end)
    if not vx.empty:
        curve = ts.build_curve(vx, vix)
        df = df.join(curve.drop(columns=["vix"], errors="ignore"))
        df = df.join(ts.roll_yield(curve))
        df["ts_slope"] = ts.slope(curve)
        df["inverted"] = ts.inversion_flag(curve)
    else:
        log.warning("no VX futures data — term-structure columns skipped")

    if save:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        out = PROCESSED / "vrp_dataset.csv"
        df.to_csv(out)
        log.info("dataset -> %s (%d rows)", out, len(df))
    return df


def unconditional_stats(vrp_expost: pd.Series) -> pd.Series:
    """Headline numbers for the note: mean, share positive, tails."""
    v = 100 * vrp_expost.dropna()
    return pd.Series({
        "mean_vol_pts": v.mean(),
        "median_vol_pts": v.median(),
        "share_positive": (v > 0).mean(),
        "p5": v.quantile(0.05),
        "p95": v.quantile(0.95),
        "min": v.min(),
        "skew": v.skew(),
        "n_days": len(v),
    }, name="vrp_expost_21d")


def conditional_table(vrp_expost: pd.Series, conditioner: pd.Series,
                      bins: list[float] | None = None,
                      labels: list[str] | None = None) -> pd.DataFrame:
    """Mean/median/tail of subsequent VRP conditioned on a variable observable
    at t (VIX level, slope, trailing VRP z...). This is the honest version of
    'when is the premium fat': condition on t, measure over (t, t+21]."""
    df = pd.DataFrame({"vrp": 100 * vrp_expost, "cond": conditioner}).dropna()
    if bins is not None:
        df["bucket"] = pd.cut(df["cond"], bins=bins, labels=labels)
    else:
        df["bucket"] = pd.qcut(df["cond"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    g = df.groupby("bucket", observed=True)["vrp"]
    return pd.DataFrame({
        "mean": g.mean(), "median": g.median(),
        "share_positive": g.apply(lambda x: (x > 0).mean()),
        "p5": g.quantile(0.05), "worst": g.min(), "n": g.size(),
    })
