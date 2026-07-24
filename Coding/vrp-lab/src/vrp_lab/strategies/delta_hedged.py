"""Strategy 2: sell SPY straddles/strangles on synthetic chains, delta-hedge daily.

Runs on the synthetic Black-Scholes option path (data/options.py) — the
documented fallback since historical chains aren't available. What survives
the synthetic caveat: the *mechanics* (hedging, exits, decomposition, costs)
are real; the *P&L levels* inherit BS-world assumptions (flat smile through
time, VIX as the only IV input). Treat results as a lower-variance sketch of
the trade, not a tradeable track record.

Mechanics per the spec:
  entry  : sell ATM straddle (or symmetric strangle at ``strangle_delta``),
           ``dte_entry`` calendar days out, sized to ``notional_frac`` of
           equity, only while the regime gate is open.
  hedge  : daily (or every ``hedge_every`` days) trade SPY to flatten book delta.
  exit   : close at ``dte_exit`` DTE (gamma explosion zone) or when the
           unrealized loss exceeds ``stop_loss_premium_mult`` x premium.
  re-arm : a new position may open on the next bar after an exit.

Output history carries prior-close position greeks + dS/d_iv/dt so
risk/pnl_decomp.decompose() attributes every day's option P&L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.costs import EquityCostModel, OptionCostModel
from ..config import get_logger
from ..volatility.implied import bs_greeks, bs_price, strike_from_delta

log = get_logger(__name__)

CONTRACT = 100.0  # shares per option contract


@dataclass
class DeltaHedgedConfig:
    dte_entry: int = 30
    dte_exit: int = 5
    stop_loss_premium_mult: float = 2.0
    strangle_delta: float | None = None   # None => ATM straddle; e.g. 0.10
    hedge_every: int = 1                  # hedge every N trading days
    notional_frac: float = 0.25           # short-option notional / equity
    skew_slope: float = 0.15              # passed to the synthetic smile
    q: float = 0.015                      # SPY dividend yield approx


@dataclass
class _Leg:
    kind: str
    strike: float
    iv_entry: float


class _Position:
    def __init__(self, legs: list[_Leg], expiry: pd.Timestamp, contracts: int,
                 premium: float):
        self.legs = legs
        self.expiry = expiry
        self.contracts = contracts       # sold per leg
        self.premium = premium           # total collected, dollars

    def value_and_greeks(self, s: float, iv_atm: float, as_of: pd.Timestamp,
                         r: float, cfg: DeltaHedgedConfig) -> dict:
        """Mark the SHORT position: value = liability (positive dollars),
        greeks are position-scaled (contracts x 100 x short sign)."""
        from ..data.options import skewed_iv

        t = max((self.expiry - as_of).days, 0) / 365.0
        scale = -self.contracts * CONTRACT   # short
        value = 0.0
        agg = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for leg in self.legs:
            iv_k = skewed_iv(iv_atm, float(np.log(leg.strike / s)), cfg.skew_slope)
            value += bs_price(s, leg.strike, t, iv_k, r, cfg.q, leg.kind)
            g = bs_greeks(s, leg.strike, t, iv_k, r, cfg.q, leg.kind)
            for k in agg:
                agg[k] += g[k]
        return {
            "liability": value * self.contracts * CONTRACT,
            **{f"pos_{k}": v * scale for k, v in agg.items()},
        }


def run_delta_hedged(
    spy_close: pd.Series,
    iv: pd.Series,
    rf_annual: pd.Series | float = 0.02,
    gate: pd.Series | None = None,
    cfg: DeltaHedgedConfig | None = None,
    initial_capital: float = 1_000_000,
    opt_costs: OptionCostModel | None = None,
    eq_costs: EquityCostModel | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Day-loop simulator. Returns (history, trades).

    history columns: equity, cash, hedge_shares, option_pnl, hedge_pnl, costs,
    pos_delta/gamma/vega/theta (prior close), dS, d_iv, dt_years, in_position.
    All decisions at t use only prices/IV at or before t.
    """
    cfg = cfg or DeltaHedgedConfig()
    opt_costs = opt_costs or OptionCostModel()
    eq_costs = eq_costs or EquityCostModel()

    idx = spy_close.dropna().index.intersection(iv.dropna().index)
    spy_close, iv = spy_close.loc[idx], iv.loc[idx]
    gate = gate.reindex(idx).fillna(0.0) if gate is not None else pd.Series(1.0, index=idx)
    rf = (rf_annual.reindex(idx).ffill().fillna(0.0) / 100.0
          if isinstance(rf_annual, pd.Series) else pd.Series(rf_annual, index=idx))

    cash = float(initial_capital)
    hedge_shares = 0.0
    pos: _Position | None = None
    prev_mark: dict | None = None
    rows, trades = [], []

    for i, t in enumerate(idx):
        s, iv_t, r = float(spy_close.loc[t]), float(iv.loc[t]), float(rf.loc[t])
        day_costs = 0.0
        option_pnl = hedge_pnl = 0.0
        row: dict = {"date": t, "dS": 0.0, "d_iv": 0.0, "dt_years": 0.0,
                     "pos_delta": 0.0, "pos_gamma": 0.0, "pos_vega": 0.0,
                     "pos_theta": 0.0}

        if i > 0:
            s_prev = float(spy_close.iloc[i - 1])
            hedge_pnl = hedge_shares * (s - s_prev)
            cash += cash * r / 252.0  # interest on cash
            if pos is not None and prev_mark is not None:
                mark = pos.value_and_greeks(s, iv_t, t, r, cfg)
                option_pnl = -(mark["liability"] - prev_mark["liability"])
                row.update({
                    "dS": s - s_prev,
                    "d_iv": iv_t - float(iv.iloc[i - 1]),
                    "dt_years": (t - idx[i - 1]).days / 365.0,
                    "pos_delta": prev_mark["pos_delta"],
                    "pos_gamma": prev_mark["pos_gamma"],
                    "pos_vega": prev_mark["pos_vega"],
                    "pos_theta": prev_mark["pos_theta"],
                })
                prev_mark = mark

        # ---- exits ------------------------------------------------------ #
        if pos is not None:
            dte = (pos.expiry - t).days
            unrealized = pos.premium - prev_mark["liability"]
            stop = unrealized < -cfg.stop_loss_premium_mult * pos.premium
            if dte <= cfg.dte_exit or stop:
                per_leg = prev_mark["liability"] / (len(pos.legs) * pos.contracts * CONTRACT)
                for leg in pos.legs:
                    day_costs += opt_costs.cost(pos.contracts, per_leg)
                cash -= prev_mark["liability"] + day_costs
                trades.append({"date": t, "action": "close",
                               "reason": "stop_loss" if stop else "dte_exit",
                               "pnl_vs_premium": unrealized})
                pos, prev_mark = None, None

        # ---- entry ------------------------------------------------------ #
        elif gate.loc[t] > 0:
            expiry = t + pd.Timedelta(days=cfg.dte_entry)
            tt = cfg.dte_entry / 365.0
            if cfg.strangle_delta is None:
                legs = [_Leg("call", s, iv_t), _Leg("put", s, iv_t)]
            else:
                k_put = strike_from_delta(cfg.strangle_delta, s, tt, iv_t, r, cfg.q, "put")
                k_call = strike_from_delta(cfg.strangle_delta, s, tt, iv_t, r, cfg.q, "call")
                legs = [_Leg("call", k_call, iv_t), _Leg("put", k_put, iv_t)]
            equity_now = cash + hedge_shares * s
            contracts = int(np.floor(cfg.notional_frac * equity_now / (s * CONTRACT)))
            if contracts >= 1:
                pos = _Position(legs, expiry, contracts, premium=0.0)
                mark = pos.value_and_greeks(s, iv_t, t, r, cfg)
                premium = mark["liability"]  # what the buyer pays us
                for leg in legs:
                    day_costs += opt_costs.cost(
                        contracts, premium / (len(legs) * contracts * CONTRACT))
                pos.premium = premium
                cash += premium - day_costs
                prev_mark = mark
                trades.append({"date": t, "action": "open", "reason": "entry",
                               "contracts": contracts, "premium": premium})

        # ---- daily delta hedge ------------------------------------------ #
        if pos is not None and i % cfg.hedge_every == 0:
            book_delta = prev_mark["pos_delta"] + hedge_shares
            trade_qty = -book_delta
            if abs(trade_qty) >= 1.0:
                c = eq_costs.cost(trade_qty, s)
                cash -= trade_qty * s + c
                day_costs += c
                hedge_shares += trade_qty
        elif pos is None and hedge_shares != 0.0:
            c = eq_costs.cost(hedge_shares, s)
            cash += hedge_shares * s - c
            day_costs += c
            hedge_shares = 0.0

        liability = prev_mark["liability"] if pos is not None else 0.0
        row.update({
            "equity": cash + hedge_shares * s - liability,
            "cash": cash, "hedge_shares": hedge_shares,
            "option_pnl": option_pnl, "hedge_pnl": hedge_pnl,
            "costs": day_costs, "in_position": pos is not None,
        })
        rows.append(row)

    history = pd.DataFrame(rows).set_index("date")
    trade_log = pd.DataFrame(trades).set_index("date") if trades else pd.DataFrame()
    n_stops = sum(1 for tr in trades if tr.get("reason") == "stop_loss")
    log.info("delta-hedged run: %d trades (%d stop-outs), final equity %.0f",
             len(trades), n_stops, history["equity"].iloc[-1])
    return history, trade_log
