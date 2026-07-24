"""Cheap OTM put-spread tail hedge overlay (long 5-delta / short 2-delta).

Purpose: convert the short-vol book's return distribution from open-ended
negative skew into something bounded and insurable. It COSTS money — roughly
0.5-1% of hedged notional per month — and that drag is the honest price of
the left tail. The notebook reports Sharpe with and without it; the spec is
explicit that showing the trade-off is the point, even when the hedge hurts.

Mechanics: on the first bar (and each expiry), buy a put spread ``dte`` days
out sized to ``hedge_notional_frac`` of current equity, priced on the same
synthetic smile as the short book. Held to expiry, settled at intrinsic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.costs import OptionCostModel
from ..config import get_logger
from ..data.options import skewed_iv
from ..volatility.implied import bs_price, strike_from_delta

log = get_logger(__name__)

CONTRACT = 100.0


@dataclass
class TailOverlayConfig:
    long_put_delta: float = 0.05
    short_put_delta: float = 0.02
    dte: int = 30
    hedge_notional_frac: float = 1.0   # protect ~1x of book equity notionally
    skew_slope: float = 0.15
    q: float = 0.015


def run_tail_overlay(
    spy_close: pd.Series,
    iv: pd.Series,
    equity_ref: pd.Series,
    rf_annual: float = 0.02,
    cfg: TailOverlayConfig | None = None,
    opt_costs: OptionCostModel | None = None,
) -> pd.DataFrame:
    """Simulate the rolling put-spread overlay. Returns a daily frame with
    columns [overlay_pnl, overlay_value, premium_paid, rolled] — add
    ``overlay_pnl`` to the base strategy's dollar P&L to get the hedged book.

    equity_ref : the base strategy's equity curve; sizing tracks it so the
    hedge scales with the book it protects.
    """
    cfg = cfg or TailOverlayConfig()
    opt_costs = opt_costs or OptionCostModel()
    idx = spy_close.dropna().index.intersection(iv.dropna().index)
    spy_close, iv = spy_close.loc[idx], iv.loc[idx]
    equity_ref = equity_ref.reindex(idx).ffill()

    pos = None  # dict(k_long, k_short, expiry, contracts)
    prev_value = 0.0
    rows = []

    def _mark(s: float, iv_t: float, t: pd.Timestamp) -> float:
        tt = max((pos["expiry"] - t).days, 0) / 365.0
        v = 0.0
        for k, sign in ((pos["k_long"], 1.0), (pos["k_short"], -1.0)):
            iv_k = skewed_iv(iv_t, float(np.log(k / s)), cfg.skew_slope)
            v += sign * bs_price(s, k, tt, iv_k, rf_annual, cfg.q, "put")
        return v * pos["contracts"] * CONTRACT

    for i, t in enumerate(idx):
        s, iv_t = float(spy_close.loc[t]), float(iv.loc[t])
        pnl = 0.0
        premium_paid = 0.0
        rolled = False

        if pos is not None:
            value = _mark(s, iv_t, t)
            pnl += value - prev_value
            prev_value = value
            if (pos["expiry"] - t).days <= 0:
                pos = None  # settled at intrinsic via the final mark

        if pos is None:
            tt = cfg.dte / 365.0
            k_long = strike_from_delta(cfg.long_put_delta, s, tt, iv_t,
                                       rf_annual, cfg.q, "put")
            k_short = strike_from_delta(cfg.short_put_delta, s, tt, iv_t,
                                        rf_annual, cfg.q, "put")
            contracts = int(np.floor(
                cfg.hedge_notional_frac * float(equity_ref.loc[t]) / (s * CONTRACT)))
            if contracts >= 1:
                pos = {"k_long": k_long, "k_short": k_short,
                       "expiry": t + pd.Timedelta(days=cfg.dte),
                       "contracts": contracts}
                value = _mark(s, iv_t, t)
                per_leg_mid = value / (2 * contracts * CONTRACT)
                cost = 2 * opt_costs.cost(contracts, max(per_leg_mid, 0.01))
                premium_paid = value + cost   # cash out the door (reporting)
                # P&L at entry is only the cost: we pay `value` but hold an
                # asset marked at `value`; its decay/payoff flows via d(mark)
                pnl -= cost
                prev_value = value
                rolled = True

        rows.append({"date": t, "overlay_pnl": pnl,
                     "overlay_value": prev_value if pos else 0.0,
                     "premium_paid": premium_paid, "rolled": rolled})

    out = pd.DataFrame(rows).set_index("date")
    ann_drag = out["overlay_pnl"].sum() / max(len(idx) / 252, 1e-9)
    log.info("tail overlay: net P&L %.0f (%.0f/yr) over %d days",
             out["overlay_pnl"].sum(), ann_drag, len(idx))
    return out
