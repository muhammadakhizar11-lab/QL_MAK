r"""Strategy risk metrics. All of them — a short-vol book reporting only its
Sharpe is hiding the body. Conventions: daily returns in decimals, 252-day
annualization, excess returns where a risk-free series is supplied.

CVaR sign convention: reported as a (negative) return level — "CVaR95 = -2.1%"
means the mean of the worst 5% of days is -2.1%.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def _to_excess(returns: pd.Series, rf_annual: pd.Series | float | None) -> pd.Series:
    if rf_annual is None:
        return returns
    if isinstance(rf_annual, (int, float)):
        return returns - rf_annual / ANN
    daily_rf = rf_annual.reindex(returns.index).ffill() / 100.0 / ANN
    return returns - daily_rf.fillna(0.0)


def sharpe(returns: pd.Series, rf_annual: pd.Series | float | None = None) -> float:
    r""":math:`\sqrt{252}\,\bar r_e/\sigma(r_e)` on daily excess returns."""
    ex = _to_excess(returns, rf_annual).dropna()
    sd = ex.std(ddof=1)
    return float(np.sqrt(ANN) * ex.mean() / sd) if sd > 0 else float("nan")


def sortino(returns: pd.Series, rf_annual: pd.Series | float | None = None) -> float:
    r"""Sharpe with downside deviation: :math:`\sigma_d=\sqrt{\frac1n\sum \min(r_e,0)^2}`."""
    ex = _to_excess(returns, rf_annual).dropna()
    downside = np.sqrt((np.minimum(ex, 0.0) ** 2).mean())
    return float(np.sqrt(ANN) * ex.mean() / downside) if downside > 0 else float("nan")


def max_drawdown(equity: pd.Series) -> dict:
    """Peak-to-trough max drawdown with duration and recovery.

    Returns {mdd, peak_date, trough_date, recovery_date (None if unrecovered),
    duration_days (peak->trough), recovery_days (trough->recovery)}.
    """
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough_date = dd.idxmin()
    mdd = float(dd.loc[trough_date])
    peak_date = equity.loc[:trough_date].idxmax()
    after = equity.loc[trough_date:]
    recovered = after[after >= equity.loc[peak_date]]
    recovery_date = recovered.index[0] if len(recovered) else None
    return {
        "mdd": mdd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "duration_days": int((trough_date - peak_date).days),
        "recovery_days": int((recovery_date - trough_date).days) if recovery_date else None,
    }


def calmar(returns: pd.Series, equity: pd.Series | None = None) -> float:
    """Annualized return / |max drawdown| (geometric annualization)."""
    eq = equity if equity is not None else (1.0 + returns.fillna(0.0)).cumprod()
    years = len(returns.dropna()) / ANN
    if years <= 0:
        return float("nan")
    ann_ret = _annualized_return(eq, years)
    mdd = abs(max_drawdown(eq)["mdd"])
    return float(ann_ret / mdd) if mdd > 0 else float("nan")


def _annualized_return(eq: pd.Series, years: float) -> float:
    """Geometric annualized return; a non-positive terminal equity is ruin
    and reports -100% rather than a NaN from a negative fractional power."""
    total = eq.iloc[-1] / eq.iloc[0]
    if total <= 0:
        return -1.0
    return float(total ** (1 / years) - 1.0)


def cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Expected shortfall: mean return over the worst (1-level) tail of days."""
    r = returns.dropna()
    if r.empty:
        return float("nan")
    cutoff = r.quantile(1.0 - level)
    tail = r[r <= cutoff]
    return float(tail.mean()) if len(tail) else float("nan")


def summary(
    returns: pd.Series,
    equity: pd.Series | None = None,
    rf_annual: pd.Series | float | None = None,
) -> pd.Series:
    """The full table the spec demands — never report Sharpe alone."""
    r = returns.dropna()
    eq = equity if equity is not None else (1.0 + r).cumprod()
    dd = max_drawdown(eq)
    years = len(r) / ANN
    out = {
        "ann_return": _annualized_return(eq, years) if years > 0 else np.nan,
        "ann_vol": float(r.std(ddof=1) * np.sqrt(ANN)),
        "sharpe": sharpe(r, rf_annual),
        "sortino": sortino(r, rf_annual),
        "calmar": calmar(r, eq),
        "max_drawdown": dd["mdd"],
        "dd_duration_days": dd["duration_days"],
        "dd_recovery_days": dd["recovery_days"],
        "cvar_95": cvar(r, 0.95),
        "cvar_99": cvar(r, 0.99),
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
        "hit_rate": float((r > 0).mean()),
        "n_days": len(r),
    }
    return pd.Series(out, name="metrics")
