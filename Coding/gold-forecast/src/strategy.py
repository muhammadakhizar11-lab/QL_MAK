"""
strategy.py — the position/P&L engine for stage 4. Pure functions, no file I/O.

TIMING CONVENTION (the one thing that must not slip)
----------------------------------------------------
`signal_t` is computed from information available at the CLOSE of day t (stage 3
guarantees this for the OOS forecasts). The position set from `signal_t` earns
`fwd_ret_t = log(close_{t+1}/close_t)` — the same quantity stage 3 used as the
1-day label. Nothing here shifts returns backwards; the only shifts in this file
move INFORMATION forward (trailing vol, previous-day position for costs), never
returns backward. `tests/test_strategy.py` checks the alignment both ways
(foresight makes money, hindsight doesn't).

PIPELINE per strategy:  raw signal
  -> unit position   (long_flat: 1 if signal>0 else 0; long_short: sign)
  -> vol targeting   (scale by target_vol / trailing realized vol, clipped)
  -> drawdown stop   (flatten after a trailing-equity stop fires, with cooldown)
  -> net returns     (gross pos*ret minus cost_bps per unit turnover)

Costs are charged on |Δposition|, so vol-scaling drift pays costs too — a real
account rebalances those fractions. This is still an idealized daily-close
backtest: no slippage model, no intraday fills, no financing. Stated in the
report, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Position construction
# --------------------------------------------------------------------------- #
def unit_position(signal: pd.Series, mode: str = "long_flat") -> pd.Series:
    """Map a real-valued signal (centered at 0) to a unit position.

    long_flat:  1 when signal > 0, else 0   (no shorting)
    long_short: sign(signal), 0 stays 0
    NaN signal => flat (no forecast, no position).
    """
    s = signal.fillna(0.0)
    if mode == "long_flat":
        return (s > 0).astype(float)
    if mode == "long_short":
        return np.sign(s)
    raise ValueError(f"unknown mode: {mode}")


def vol_scale(position: pd.Series, asset_ret: pd.Series, target_ann: float | None,
              window: int, max_leverage: float, trading_days: int = 252) -> pd.Series:
    """Scale positions to a constant ex-ante vol target.

    Realized vol at t uses returns THROUGH t: `asset_ret` is the forward return
    (r_t = close_t -> close_{t+1}), so shift(1) makes the window end on the
    t-1 -> t return — fully known at close t. Leverage is clipped symmetrically.
    """
    if target_ann is None:
        return position.clip(-max_leverage, max_leverage)
    trailing = asset_ret.shift(1).rolling(window).std(ddof=0) * np.sqrt(trading_days)
    lever = (target_ann / trailing).clip(upper=max_leverage)
    # until the vol window fills, trade unlevered rather than not at all
    return position * lever.fillna(1.0)


def apply_stop(position: pd.Series, asset_ret: pd.Series, max_drawdown: float,
               cooldown_days: int) -> pd.Series:
    """Trailing-equity stop: when the strategy's own cumulative log return draws
    down more than `max_drawdown` from its running peak, go flat for
    `cooldown_days`, then resume. Stateful by nature -> explicit loop.

    The stop reacts to returns through t (r_{t-1} is the last one realized at
    close t), so flattening applies from day t onward — no peeking.
    """
    pos = position.to_numpy(dtype=float).copy()
    ret = asset_ret.to_numpy(dtype=float)
    equity = peak = 0.0
    cooldown = 0
    for t in range(len(pos)):
        if cooldown > 0:
            pos[t] = 0.0
            cooldown -= 1
        if t > 0 and not np.isnan(ret[t - 1]):
            equity += pos[t - 1] * ret[t - 1]      # realized at close t
            peak = max(peak, equity)
            if peak - equity > max_drawdown and cooldown == 0 and pos[t] != 0.0:
                pos[t] = 0.0
                cooldown = cooldown_days
                peak = equity                       # re-arm from the stopped level
    return pd.Series(pos, index=position.index)


def net_returns(position: pd.Series, asset_ret: pd.Series,
                cost_bps: float) -> pd.DataFrame:
    """Gross and net daily strategy returns. Turnover |Δpos| is charged at
    cost_bps one-way; day 0 pays for entering the initial position."""
    gross = position * asset_ret
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    cost = turnover * cost_bps * 1e-4
    return pd.DataFrame({
        "position": position, "gross": gross, "cost": cost, "net": gross - cost,
        "turnover": turnover,
    })


# --------------------------------------------------------------------------- #
# One call per strategy
# --------------------------------------------------------------------------- #
@dataclass
class StrategyResult:
    name: str
    daily: pd.DataFrame                       # position/gross/cost/net/turnover
    metrics: dict = field(default_factory=dict)


def run_strategy(name: str, signal: pd.Series, asset_ret: pd.Series,
                 sc: dict) -> StrategyResult:
    """signal + forward returns + strategy config -> positioned, costed result."""
    pos = unit_position(signal, sc["mode"])
    pos = vol_scale(pos, asset_ret, sc["vol_target_ann"], sc["vol_window"],
                    sc["max_leverage"], sc["trading_days"])
    if sc.get("stop"):
        pos = apply_stop(pos, asset_ret, sc["stop"]["max_drawdown"],
                         sc["stop"]["cooldown_days"])
    daily = net_returns(pos, asset_ret, sc["cost_bps"])
    return StrategyResult(name, daily, perf_metrics(daily, sc["trading_days"]))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def perf_metrics(daily: pd.DataFrame, trading_days: int = 252) -> dict:
    """Annualized performance on daily log returns, plus the honesty numbers:
    cost drag and the breakeven cost (the one-way bps at which net return hits
    zero — how much edge there really is over trading friction)."""
    net = daily["net"].dropna()
    gross = daily["gross"].dropna()
    n = len(net)
    if n == 0:
        return {}
    ann = trading_days / n
    mu, sd = net.mean() * trading_days, net.std(ddof=0) * np.sqrt(trading_days)
    cum = net.cumsum()
    dd = cum - cum.cummax()
    active = daily["position"].abs() > 1e-12
    ann_turnover = daily["turnover"].sum() * ann
    gross_total, cost_total = gross.sum(), daily["cost"].sum()
    return {
        "ann_ret_net": float(mu),
        "ann_ret_gross": float(gross.mean() * trading_days),
        "ann_vol": float(sd),
        "sharpe_net": float(mu / sd) if sd > 0 else float("nan"),
        "max_drawdown": float(-dd.min()),
        "hit_rate_active": float((net[active & (net != 0)] > 0).mean()) if active.any() else float("nan"),
        "exposure_avg": float(daily["position"].abs().mean()),
        "ann_turnover": float(ann_turnover),
        "cost_drag_ann": float(cost_total * ann),
        # one-way bps at which cumulative net = 0
        "breakeven_cost_bps": float(gross_total / daily["turnover"].sum() * 1e4)
                              if daily["turnover"].sum() > 0 else float("nan"),
        "n_days": int(n),
    }
