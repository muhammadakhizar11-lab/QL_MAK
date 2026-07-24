r"""Greeks-based P&L attribution for an option book.

Daily Taylor expansion of the position value around yesterday's close:

.. math::
    \Delta V \approx \underbrace{\Delta\,\mathrm{d}S}_{delta}
      + \underbrace{\tfrac12\,\Gamma\,\mathrm{d}S^2}_{gamma}
      + \underbrace{\mathcal{V}\,\mathrm{d}\sigma}_{vega}
      + \underbrace{\Theta\,\mathrm{d}t}_{theta}
      + \text{residual}

For a *short, delta-hedged* position the delta term nets against the hedge
leg, so the economics reduce to **theta collected vs gamma paid** — and
E[theta + gamma] > 0 over time *is* the harvested VRP. The vega term is
mark-to-market noise from IV moves; the residual absorbs higher-order greeks
and discrete-hedging error and must stay small (asserted in tests).

Inputs are position-level greeks recorded at the *prior* close (point-in-time:
the greeks you actually knew when the day started).
"""
from __future__ import annotations

import pandas as pd

from ..config import get_logger

log = get_logger(__name__)

REQUIRED = ["pos_delta", "pos_gamma", "pos_vega", "pos_theta",
            "dS", "d_iv", "dt_years", "option_pnl"]


def decompose(history: pd.DataFrame) -> pd.DataFrame:
    """Attribute each day's option P&L to greeks.

    Parameters
    ----------
    history : per-day frame with columns
        pos_delta/gamma/vega/theta : position greeks at prior close
            (contract-scaled: already multiplied by contracts x 100 x sign)
        dS : underlying price change over the day
        d_iv : IV change (decimal, e.g. -0.01 = one vol point down)
        dt_years : year fraction elapsed (1/365 per calendar day here)
        option_pnl : actual mark-to-market P&L of the option leg

    Returns the input plus delta_pnl, gamma_pnl, vega_pnl, theta_pnl,
    explained, residual.
    """
    missing = [c for c in REQUIRED if c not in history.columns]
    if missing:
        raise ValueError(f"pnl_decomp: missing columns {missing}")
    out = history.copy()
    out["delta_pnl"] = out["pos_delta"] * out["dS"]
    out["gamma_pnl"] = 0.5 * out["pos_gamma"] * out["dS"] ** 2
    out["vega_pnl"] = out["pos_vega"] * out["d_iv"]
    out["theta_pnl"] = out["pos_theta"] * out["dt_years"]
    out["explained"] = out[["delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl"]].sum(axis=1)
    out["residual"] = out["option_pnl"] - out["explained"]

    tot = out["option_pnl"].abs().sum()
    if tot > 0:
        share = out["residual"].abs().sum() / tot
        log.info("pnl decomposition: residual = %.1f%% of gross option P&L", 100 * share)
        if share > 0.25:
            log.warning("large unexplained residual (%.1f%%) — check greeks scaling "
                        "or hedge frequency", 100 * share)
    return out


def attribution_summary(decomposed: pd.DataFrame) -> pd.Series:
    """Cumulative dollars by component. theta + gamma ~ the realized VRP."""
    cols = ["theta_pnl", "gamma_pnl", "vega_pnl", "delta_pnl", "residual", "option_pnl"]
    s = decomposed[cols].sum()
    s["theta_plus_gamma"] = s["theta_pnl"] + s["gamma_pnl"]
    return s.rename("cumulative_pnl")
