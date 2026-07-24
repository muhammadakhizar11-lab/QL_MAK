"""Options chains: live snapshots (yfinance) + synthetic historical chains.

DATA REALITY (spec order of preference, and why we land where we do):
  1. WRDS/OptionMetrics IvyDB — not available in this environment.
  2. yfinance chains — CURRENT snapshot only; usable for live-forward testing,
     useless for backtesting. Provided here as ``fetch_live_chain``.
  3. Deribit BTC/ETH — real historical option data; noted as the crypto pivot
     in docs, not implemented in v1.
  4. **Synthetic chains** (``synthetic_chain``) — Black-Scholes prices built
     from historical SPY closes with the VIX as the IV input plus a fixed
     parametric put skew. THIS IS A MODEL, NOT DATA. Every Phase-2 result
     carries this asterisk: no smile dynamics, no vol-of-vol, no bid/ask
     microstructure beyond the cost model, and 30-day flat IV applied to all
     tenors. See docs/DATA_SOURCES.md and METHODOLOGY.md for the full caveat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_logger
from ..volatility.implied import bs_greeks, bs_price

log = get_logger(__name__)

CHAIN_COLS = ["kind", "strike", "expiry", "dte", "mid", "iv",
              "delta", "gamma", "vega", "theta"]


def fetch_live_chain(ticker: str = "SPY", min_oi: int = 100,
                     max_rel_spread: float = 0.20) -> pd.DataFrame:
    """Current options chain snapshot via yfinance (LIVE-FORWARD USE ONLY —
    yfinance keeps no history; you cannot backtest with this).

    Applies the spec's liquidity filter: open interest >= min_oi and
    bid-ask spread <= max_rel_spread of mid.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot_hist = tk.history(period="1d")
    if spot_hist.empty:
        log.warning("live chain %s: no spot price", ticker)
        return pd.DataFrame()
    frames = []
    today = pd.Timestamp.today().normalize()
    for expiry_str in tk.options:
        expiry = pd.Timestamp(expiry_str)
        chain = tk.option_chain(expiry_str)
        for kind, df in (("call", chain.calls), ("put", chain.puts)):
            df = df.copy()
            df["kind"] = kind
            df["expiry"] = expiry
            df["dte"] = (expiry - today).days
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw["mid"] = (raw["bid"] + raw["ask"]) / 2.0
    liquid = (
        (raw["openInterest"].fillna(0) >= min_oi)
        & (raw["mid"] > 0)
        & ((raw["ask"] - raw["bid"]) / raw["mid"] <= max_rel_spread)
    )
    out = raw[liquid].rename(columns={"impliedVolatility": "iv"})
    log.info("live chain %s: %d liquid of %d quotes", ticker, len(out), len(raw))
    return out


def skewed_iv(atm_iv: float, moneyness: float, skew_slope: float = 0.15,
              smile_curv: float = 0.05) -> float:
    r"""Parametric smile for synthetic chains:

    .. math::
        \sigma(m) = \sigma_{ATM}\,(1 - s\,m + c\,m^2),\qquad m=\ln(K/S)

    ``skew_slope`` s > 0 makes low strikes (puts) richer — the equity index
    smirk. Values (s=0.15, c=0.05 per unit log-moneyness, i.e. ~1.5 vol pts
    per 10% down-strike at 20 vol) eyeballed from typical SPX 1M surfaces;
    fixed through time, which is exactly the synthetic path's key limitation
    (no skew dynamics -> no skew P&L).
    """
    m = moneyness
    return float(atm_iv * (1.0 - skew_slope * m + smile_curv * m * m))


def synthetic_chain(
    spot: float,
    atm_iv: float,
    expiry: pd.Timestamp,
    as_of: pd.Timestamp,
    r: float = 0.0,
    q: float = 0.015,
    strike_pct_range: tuple[float, float] = (0.70, 1.20),
    strike_step_pct: float = 0.01,
    skew_slope: float = 0.15,
) -> pd.DataFrame:
    """Black-Scholes chain for one expiry at one date. Columns = CHAIN_COLS.

    ``atm_iv`` should be the VIX/100 at ``as_of`` (a 30-day vol applied to
    whatever this chain's tenor is — flat term structure, another documented
    simplification). Time is ACT/365.
    """
    t = max((expiry - as_of).days, 0) / 365.0
    strikes = spot * np.arange(strike_pct_range[0], strike_pct_range[1] + 1e-9,
                               strike_step_pct)
    rows = []
    for k in strikes:
        iv_k = skewed_iv(atm_iv, float(np.log(k / spot)), skew_slope)
        for kind in ("call", "put"):
            greeks = bs_greeks(spot, k, t, iv_k, r, q, kind)
            rows.append({
                "kind": kind, "strike": float(k), "expiry": expiry,
                "dte": int((expiry - as_of).days),
                "mid": bs_price(spot, k, t, iv_k, r, q, kind),
                "iv": iv_k, **greeks,
            })
    return pd.DataFrame(rows, columns=CHAIN_COLS)
