"""Realized volatility estimators: close-to-close, Parkinson, Garman-Klass, Yang-Zhang.

All estimators:
  * take a rolling ``window`` in trading days (default 21 ~ 1 month),
  * return *annualized* volatility (sqrt-of-time with ``ann_days``, default 252),
  * are strictly trailing — the value at date t uses data up to and including t,
    never after (verified in tests/test_realized.py),
  * return NaN until a full window is available.

Efficiency comparisons below are relative variance of the estimator vs the
close-to-close baseline under GBM (lower is better).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN_DAYS = 252


def _check_ohlc(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"OHLC frame missing columns {missing}; have {list(df.columns)}")


def close_to_close(close: pd.Series, window: int = 21, ann_days: int = ANN_DAYS) -> pd.Series:
    r"""Close-to-close realized volatility (baseline).

    .. math::
        \hat\sigma^2 = \frac{1}{n-1}\sum_{i=1}^{n}(r_i-\bar r)^2,\qquad
        r_i = \ln(C_i/C_{i-1})

    Reference: textbook sample stdev of log returns, annualized by
    :math:`\sqrt{252}`.

    Prefer when: you only have closing prices, or you want the estimator
    whose assumptions are weakest (no dependence on intraday path). It is
    the least efficient — needs ~5x the observations of Garman-Klass for
    the same precision — and is noisy at short windows.

    Example
    -------
    >>> import numpy as np, pandas as pd
    >>> px = pd.Series(100 * np.exp(np.cumsum(np.full(300, 0.01))))
    >>> close_to_close(px).iloc[-1]  # constant returns -> ~0 vol
    0.0
    """
    r = np.log(close / close.shift(1))
    return r.rolling(window).std(ddof=1) * np.sqrt(ann_days)


def parkinson(ohlc: pd.DataFrame, window: int = 21, ann_days: int = ANN_DAYS) -> pd.Series:
    r"""Parkinson (1980) high-low range estimator.

    .. math::
        \hat\sigma^2 = \frac{1}{4n\ln 2}\sum_{i=1}^{n}\left(\ln\frac{H_i}{L_i}\right)^2

    Reference: Parkinson, M. (1980), "The Extreme Value Method for Estimating
    the Variance of the Rate of Return", *Journal of Business* 53(1).

    ~5x more efficient than close-to-close under continuous GBM, but biased
    *downward* by discrete sampling (observed high/low understate the true
    extremes) and ignores overnight jumps and drift.

    Prefer when: intraday range is reliable and the asset trades near-continuously;
    avoid for assets with large overnight gaps (use Yang-Zhang instead).
    """
    _check_ohlc(ohlc, ("high", "low"))
    hl2 = np.log(ohlc["high"] / ohlc["low"]) ** 2
    var = hl2.rolling(window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * ann_days)


def garman_klass(ohlc: pd.DataFrame, window: int = 21, ann_days: int = ANN_DAYS) -> pd.Series:
    r"""Garman-Klass (1980) OHLC estimator.

    .. math::
        \hat\sigma^2 = \frac{1}{n}\sum_{i=1}^{n}\left[
            \tfrac12\left(\ln\tfrac{H_i}{L_i}\right)^2
            - (2\ln 2 - 1)\left(\ln\tfrac{C_i}{O_i}\right)^2\right]

    Reference: Garman, M. & Klass, M. (1980), "On the Estimation of Security
    Price Volatilities from Historical Data", *Journal of Business* 53(1).

    ~7.4x more efficient than close-to-close under GBM with zero drift and
    continuous trading. Like Parkinson it assumes no overnight jump — the
    open-to-close term only spans the trading session.

    Prefer when: near-24h markets (futures, FX, crypto). For cash equities
    with overnight gaps prefer Yang-Zhang.
    """
    _check_ohlc(ohlc, ("open", "high", "low", "close"))
    hl2 = np.log(ohlc["high"] / ohlc["low"]) ** 2
    co2 = np.log(ohlc["close"] / ohlc["open"]) ** 2
    daily_var = 0.5 * hl2 - (2.0 * np.log(2.0) - 1.0) * co2
    var = daily_var.rolling(window).mean()
    return np.sqrt(var.clip(lower=0.0) * ann_days)


def rogers_satchell(ohlc: pd.DataFrame, window: int = 21, ann_days: int = ANN_DAYS) -> pd.Series:
    r"""Rogers-Satchell (1991) drift-independent estimator (Yang-Zhang component).

    .. math::
        \hat\sigma^2 = \frac{1}{n}\sum_i\left[
            \ln\tfrac{H_i}{C_i}\ln\tfrac{H_i}{O_i}
          + \ln\tfrac{L_i}{C_i}\ln\tfrac{L_i}{O_i}\right]

    Reference: Rogers, L.C.G. & Satchell, S.E. (1991), "Estimating Variance
    From High, Low and Closing Prices", *Annals of Applied Probability* 1(4).

    Unbiased under nonzero drift (Parkinson/Garman-Klass are not); still
    ignores overnight jumps. Mostly used as the intraday leg of Yang-Zhang.
    """
    _check_ohlc(ohlc, ("open", "high", "low", "close"))
    h, l, o, c = ohlc["high"], ohlc["low"], ohlc["open"], ohlc["close"]
    daily = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))
    var = daily.rolling(window).mean()
    return np.sqrt(var.clip(lower=0.0) * ann_days)


def yang_zhang(ohlc: pd.DataFrame, window: int = 21, ann_days: int = ANN_DAYS) -> pd.Series:
    r"""Yang-Zhang (2000): overnight + open-to-close + Rogers-Satchell combination.

    .. math::
        \hat\sigma^2_{YZ} = \sigma^2_{overnight} + k\,\sigma^2_{open-close}
                            + (1-k)\,\sigma^2_{RS},\qquad
        k = \frac{0.34}{1.34 + \frac{n+1}{n-1}}

    where :math:`\sigma^2_{overnight}` is the sample variance of
    :math:`\ln(O_i/C_{i-1})` and :math:`\sigma^2_{open-close}` that of
    :math:`\ln(C_i/O_i)` (both mean-adjusted).

    Reference: Yang, D. & Zhang, Q. (2000), "Drift-Independent Volatility
    Estimation Based on High, Low, Open, and Close Prices",
    *Journal of Business* 73(3).

    Minimum-variance unbiased among OHLC estimators, drift-independent, and
    the only one here that prices in overnight jumps. **Prefer this for real
    equity data** (SPY gaps overnight on macro news); its edge shrinks for
    24h markets where the overnight term ~ 0.
    """
    _check_ohlc(ohlc, ("open", "high", "low", "close"))
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))

    log_oc_prev = np.log(ohlc["open"] / ohlc["close"].shift(1))   # overnight
    log_co = np.log(ohlc["close"] / ohlc["open"])                 # open-to-close

    var_on = log_oc_prev.rolling(n).var(ddof=1)
    var_oc = log_co.rolling(n).var(ddof=1)

    h, l, o, c = ohlc["high"], ohlc["low"], ohlc["open"], ohlc["close"]
    rs_daily = (np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o))
    var_rs = rs_daily.rolling(n).mean()

    var = var_on + k * var_oc + (1.0 - k) * var_rs
    return np.sqrt(var.clip(lower=0.0) * ann_days)


ESTIMATORS = {
    "close_to_close": close_to_close,
    "parkinson": parkinson,
    "garman_klass": garman_klass,
    "rogers_satchell": rogers_satchell,
    "yang_zhang": yang_zhang,
}


def realized_vol(
    ohlc: pd.DataFrame,
    estimator: str = "yang_zhang",
    window: int = 21,
    ann_days: int = ANN_DAYS,
) -> pd.Series:
    """Dispatch to a named estimator. ``close_to_close`` accepts a frame too
    (uses its ``close`` column)."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator '{estimator}'; choose from {list(ESTIMATORS)}")
    if estimator == "close_to_close":
        return close_to_close(ohlc["close"], window=window, ann_days=ann_days)
    return ESTIMATORS[estimator](ohlc, window=window, ann_days=ann_days)
