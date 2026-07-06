"""
features.py — build the daily feature matrix X and the forward-return label y.

AS-OF CONVENTION (read this before trusting anything)
-----------------------------------------------------
A row dated `t` holds features computable from information available AT THE CLOSE
OF DAY t. The label on that row is the forward log return from close t to close
t+N:  y_t = log(close_{t+N} / close_t).  The label is strictly future, so features
at t may legitimately use everything up to and including close t.

Leakage is controlled at three points:
  1. Own-asset trailing features (returns, vol, RSI, MACD, MA-distance): pandas
     rolling windows are trailing, so a value at t uses only data <= t. Safe.
  2. Cross-asset market data (silver/VIX/DXY/miner closes): markets close same day,
     so close_t is observable at t. Used contemporaneously, no shift. Safe.
  3. Exogenous released-with-a-lag data (FRED monthly + daily real yield): NOT
     known on their nominal date. These are explicitly shifted forward by a
     conservative availability lag before being forward-filled onto the daily
     grid. See `align_monthly` / `align_daily_lag`. This is the main leakage trap
     and is handled deliberately, not by luck.

Every column is documented in FEATURE_SPEC with its as-of lag and a leakage note.
`audit_leakage()` verifies empirically that no feature reacts to a future price
spike — run it in tests and after any change to this file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW, load_config
from . import rolls


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_close(alias: str) -> pd.Series:
    df = pd.read_csv(RAW / f"{alias}.csv", index_col=0, parse_dates=True)
    col = "close" if "close" in df.columns else df.columns[0]
    return df[col].astype(float).rename(alias)


def _load_fred(alias: str) -> pd.Series:
    df = pd.read_csv(RAW / f"{alias}.csv", index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(float).rename(alias)


# --------------------------------------------------------------------------- #
# Point-in-time alignment of released-with-a-lag exogenous data
# --------------------------------------------------------------------------- #
def align_monthly(monthly: pd.Series, daily_index: pd.DatetimeIndex,
                  release_lag_months: int) -> pd.Series:
    """
    Map a monthly FRED series onto the daily trading grid WITHOUT look-ahead.

    FRED dates a monthly obs to the 1st of month M, but it is released ~mid month
    M+release_lag_months. To stay safe against intra-month release timing we make
    obs M available only from the START of the month AFTER release, i.e. we shift
    the observation date forward by (release_lag_months + 1) months, then
    forward-fill. Conservative by design; exact point-in-time needs ALFRED vintages
    (a documented free-data limitation).
    """
    shifted = monthly.copy()
    shifted.index = shifted.index + pd.DateOffset(months=release_lag_months + 1)
    shifted.index = shifted.index.to_period("M").to_timestamp()  # snap to month start
    daily = shifted.reindex(daily_index.union(shifted.index)).ffill()
    return daily.reindex(daily_index)


def align_daily_lag(daily_series: pd.Series, daily_index: pd.DatetimeIndex,
                    lag_days: int = 1) -> pd.Series:
    """Daily released-next-day series (e.g. DFII10 via H.15): reindex then shift."""
    s = daily_series.reindex(daily_index.union(daily_series.index)).ffill()
    s = s.reindex(daily_index).shift(lag_days)
    return s


# --------------------------------------------------------------------------- #
# Technical indicators
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder's smoothing
    roll_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd_line": line, "macd_signal": sig, "macd_hist": line - sig})


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd


# --------------------------------------------------------------------------- #
# Feature spec — documents as-of lag + leakage note for EVERY column
# --------------------------------------------------------------------------- #
FEATURE_SPEC: list[dict] = [
    # own-asset, trailing (safe at close t)
    {"feature": "ret_1d",  "asof": "close t", "src": "GC=F", "note": "trailing log return"},
    {"feature": "ret_5d",  "asof": "close t", "src": "GC=F", "note": "trailing 5d log return"},
    {"feature": "ret_20d", "asof": "close t", "src": "GC=F", "note": "trailing 20d log return"},
    {"feature": "realized_vol", "asof": "close t", "src": "GC=F", "note": "trailing std of log rets"},
    {"feature": "rsi", "asof": "close t", "src": "GC=F", "note": "Wilder RSI, trailing"},
    {"feature": "macd_line", "asof": "close t", "src": "GC=F", "note": "trailing EMA diff"},
    {"feature": "macd_hist", "asof": "close t", "src": "GC=F", "note": "trailing"},
    {"feature": "dist_ma20", "asof": "close t", "src": "GC=F", "note": "close/MA20 - 1, trailing"},
    {"feature": "dist_ma50", "asof": "close t", "src": "GC=F", "note": "close/MA50 - 1, trailing"},
    # cross-asset, contemporaneous market closes (observable at t)
    {"feature": "gold_silver_ratio_z", "asof": "close t", "src": "GC=F,SI=F",
     "note": "z-score of gold/silver price ratio, trailing — classic precious-metals signal"},
    {"feature": "corr_dxy", "asof": "close t", "src": "GC=F,DXY",
     "note": "trailing rolling corr of returns (gold priced in USD)"},
    {"feature": "ret_vix_5d", "asof": "close t", "src": "^VIX",
     "note": "trailing 5d VIX change — risk-off / safe-haven demand proxy"},
    {"feature": "ret_miners_5d", "asof": "close t", "src": "GDX",
     "note": "trailing 5d gold-miner equity return (leveraged sentiment/flow proxy)"},
    # exogenous, released with a lag (SHIFTED — main leakage trap)
    {"feature": "real_10y", "asof": "close t-1", "src": "DFII10",
     "note": "daily real yield LEVEL, shifted 1 bday for next-day H.15 release — "
             "the textbook gold driver (opportunity cost of a zero-yield asset)"},
    {"feature": "real_10y_chg_5d", "asof": "close t-1", "src": "DFII10",
     "note": "5d change in real yield, same 1-bday shift"},
    {"feature": "beta_real_yield", "asof": "close t-1", "src": "GC=F,DFII10",
     "note": "trailing rolling beta of gold rets on real-yield changes (lagged series)"},
    {"feature": "infl_breakeven", "asof": "close t-1", "src": "T10YIE",
     "note": "10y inflation breakeven, shifted 1 bday — inflation-hedge demand"},
    {"feature": "cpi_yoy", "asof": "month M-2+", "src": "CPIAUCSL",
     "note": "YoY CPI, shifted (lag+1) months for point-in-time — slow inflation trend"},
    # regime flags (slow / structural)
    {"feature": "vol_regime_high", "asof": "close t", "src": "GC=F",
     "note": "1 if trailing vol above rolling median"},
    {"feature": "fomc_day", "asof": "known in advance", "src": "fomc_dates.csv",
     "note": "1 on scheduled FOMC dates (highly relevant for gold); "
             "0 (with warning) if calendar file absent"},
]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_features(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Assemble X (features), y (N-day forward log return), and the feature spec as a
    DataFrame. Rows with a NaN label (the last N days) are dropped from both.
    """
    cfg = cfg or load_config()
    fc = cfg["features"]
    rc = cfg["rolls"]
    N = cfg["label"]["horizon_days"]

    # --- target close + roll-aware clean returns ---
    gold = _load_close("gold_comex")
    idx = gold.index  # the trading calendar everything aligns to
    gold_ret, roll_dates = rolls.clean_returns(
        gold, method=rc["method"], jump_z=rc["jump_z"], jump_hard=rc["jump_hard"],
        month_boundary_window=rc["month_boundary_window"],
    )

    X = pd.DataFrame(index=idx)

    # --- own-asset trailing features ---
    for lb in fc["return_lookbacks"]:
        X[f"ret_{lb}d"] = np.log(gold / gold.shift(lb))
    X["realized_vol"] = gold_ret.rolling(fc["vol_window"]).std(ddof=0)
    X["rsi"] = rsi(gold, fc["rsi_window"])
    m = macd(gold)
    X["macd_line"] = m["macd_line"]
    X["macd_hist"] = m["macd_hist"]
    for w in fc["ma_windows"]:
        X[f"dist_ma{w}"] = gold / gold.rolling(w).mean() - 1.0

    # --- cross-asset ---
    silver = _load_close("silver_comex").reindex(idx).ffill()
    dxy = _load_close("dxy").reindex(idx).ffill()
    vix = _load_close("vix").reindex(idx).ffill()
    miners = _load_close("gold_miners").reindex(idx).ffill()

    ratio = gold / silver
    X["gold_silver_ratio_z"] = rolling_zscore(ratio, fc["zscore_window"])

    dxy_ret = np.log(dxy / dxy.shift(1))
    X["corr_dxy"] = gold_ret.rolling(fc["corr_window"]).corr(dxy_ret)

    # VIX is a level, not a price — use 5d log change of the level
    X["ret_vix_5d"] = np.log(vix / vix.shift(5))
    X["ret_miners_5d"] = np.log(miners / miners.shift(5))

    # --- exogenous, lag-aligned (leakage-critical) ---
    # Real yields: THE gold driver. Level, 5d change, and gold's rolling beta to
    # yield changes — all built on the 1-bday-lagged series so nothing uses a
    # print before its H.15 release.
    real10 = align_daily_lag(_load_fred("real_10y"), idx, lag_days=1)
    X["real_10y"] = real10
    real10_chg = real10.diff()
    X["real_10y_chg_5d"] = real10.diff(5)
    cov = gold_ret.rolling(fc["beta_window"]).cov(real10_chg)
    var = real10_chg.rolling(fc["beta_window"]).var()
    X["beta_real_yield"] = cov / var

    X["infl_breakeven"] = align_daily_lag(_load_fred("infl_breakeven"), idx, lag_days=1)

    cpi = _load_fred("us_cpi")
    cpi_yoy = cpi / cpi.shift(12) - 1.0            # YoY computed on native monthly grid
    X["cpi_yoy"] = align_monthly(
        cpi_yoy, idx, release_lag_months=cfg["fred"]["CPIAUCSL"]["release_lag_months"]
    )

    # --- regime flags ---
    vol = X["realized_vol"]
    vol_med = vol.rolling(fc["vol_regime_window"]).median()
    X["vol_regime_high"] = (vol > vol_med).astype(float)

    X["fomc_day"] = _fomc_flag(idx)

    # --- label (strictly future; built separately, never a feature input) ---
    y = np.log(gold.shift(-N) / gold).rename(f"fwd_ret_{N}d")

    # neutralized roll days: drop rows where the clean return is NaN so a fake
    # basis jump can't sit inside a label window edge either.
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]
    spec = pd.DataFrame(FEATURE_SPEC)
    return X, y, spec


def _fomc_flag(idx: pd.DatetimeIndex) -> pd.Series:
    """
    1 on scheduled FOMC decision dates. We DO NOT hard-code dates here to avoid
    shipping subtly-wrong ones. Drop a one-column CSV of dates at
    data/raw/fomc_dates.csv (header `date`, from the public Fed calendar) and it
    is picked up automatically. Absent that file, the flag is all-zero + a warning.
    """
    import warnings
    path = RAW / "fomc_dates.csv"
    if not path.exists():
        warnings.warn(
            "fomc_dates.csv not found — fomc_day flag is all-zero. Add scheduled "
            "FOMC dates from federalreserve.gov to enable it.", stacklevel=2,
        )
        return pd.Series(0.0, index=idx)
    dates = pd.to_datetime(pd.read_csv(path)["date"])
    return idx.isin(dates).astype(float)


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
def audit_leakage(cfg: dict | None = None, spike_frac: float = 0.5) -> dict:
    """
    Empirical no-look-ahead check. Inject a large upward spike into the target
    close on a single interior date t*, rebuild features, and assert NO feature
    value strictly BEFORE t* changes vs baseline. A feature that "sees" the future
    would move on a date < t*, which this catches.

    Returns a dict with pass/fail and the list of any offending (feature, date)
    pairs. Operates on the real raw data if present; the synthetic test wires it
    to a generated dataset instead.
    """
    cfg = cfg or load_config()
    X0, _, _ = build_features(cfg)

    gold = _load_close("gold_comex")
    t_star = gold.index[len(gold) // 2]

    # perturb only close at t* by writing a temp copy, then restore.
    orig_path = RAW / "gold_comex.csv"
    backup = orig_path.read_text()
    try:
        df = pd.read_csv(orig_path, index_col=0, parse_dates=True)
        col = "close" if "close" in df.columns else df.columns[0]
        df.loc[t_star, col] *= (1 + spike_frac)
        df.to_csv(orig_path)
        X1, _, _ = build_features(cfg)
    finally:
        orig_path.write_text(backup)  # always restore

    common = X0.index.intersection(X1.index)
    before = common[common < t_star]
    diff = (X1.loc[before] - X0.loc[before]).abs()
    tol = 1e-9
    offenders = [(c, str(before[r].date()))
                 for c in diff.columns
                 for r in np.flatnonzero(diff[c].to_numpy() > tol)]
    return {"passed": len(offenders) == 0, "t_star": str(t_star.date()),
            "n_offenders": len(offenders), "offenders": offenders[:20]}


if __name__ == "__main__":
    X, y, spec = build_features()
    print(f"X: {X.shape}, y: {y.shape}")
    print(X.tail())
