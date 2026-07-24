r"""Position sizing: vol-targeting, Kelly fraction, hard notional caps.

The sizing inversion is the point: a naive short-vol strategy sizes UP when
VIX is high (more premium!). Vol-targeting does the opposite — position
scales with 1/instrument-vol, so exposure shrinks exactly when the left tail
fattens. Both a vol target and a notional cap apply; the binding one wins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_contracts(
    equity: float,
    price: float,
    daily_vol: float,
    vol_target_annual: float = 0.15,
    multiplier: float = 1000.0,
    max_notional_mult: float = 2.0,
    ann_days: int = 252,
) -> float:
    r"""Contracts n such that the position's annualized $-vol = target:

    .. math::
        n = \frac{\sigma_{target}\,E}{\sigma_{instr}^{ann}\cdot P\cdot m}

    with ``daily_vol`` the instrument's daily *return* vol (e.g. trailing 21d
    stdev of VX1 % changes), annualized inside. Capped so gross notional
    :math:`n\,P\,m \le` ``max_notional_mult`` :math:`\times E`. Returns a
    float — rounding to whole contracts is the caller's choice (the backtest
    rounds down, conservatively).
    """
    if price <= 0 or equity <= 0 or not np.isfinite(daily_vol) or daily_vol <= 0:
        return 0.0
    instr_ann_dollar_vol = daily_vol * np.sqrt(ann_days) * price * multiplier
    n = vol_target_annual * equity / instr_ann_dollar_vol
    n_cap = max_notional_mult * equity / (price * multiplier)
    return float(min(n, n_cap))


def kelly_fraction(mean_ret: float, var_ret: float, cap: float = 0.25) -> float:
    r"""Kelly-optimal fraction :math:`f^* = \mu/\sigma^2`, hard-capped.

    Full Kelly on estimated moments is a well-known way to blow up (estimation
    error compounds); the cap default (quarter-ish Kelly for typical inputs)
    is the deliberate, documented choice.
    """
    if var_ret <= 0 or not np.isfinite(mean_ret / var_ret):
        return 0.0
    return float(np.clip(mean_ret / var_ret, 0.0, cap))


def trailing_daily_vol(prices: pd.Series, window: int = 21) -> float:
    """Trailing daily return vol of an instrument price series (last value).

    NaN-safe: returns NaN if fewer than ``window`` observations, which
    vol_target_contracts maps to a zero position — no data, no trade.
    """
    r = prices.pct_change().dropna()
    if len(r) < window:
        return float("nan")
    return float(r.iloc[-window:].std(ddof=1))
