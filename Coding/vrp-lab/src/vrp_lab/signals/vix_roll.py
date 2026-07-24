"""VX futures roll/carry signal: short the front month while contango pays.

Signal logic (all inputs observable at t — the curve is built from same-day
settlements, the regime gate from same-day closes):

  ENTER short VX1 when
    * front slope (VX1 - VIX)/VIX > contango_entry_threshold (default 5%), and
    * VIX <= circuit breaker (default 30 — above that, carry is noise next to
      gap risk), and
    * regime gate is on (not Crisis, vol not accelerating).

  EXIT (target = 0) when
    * curve inverts (VX1 < VIX), or
    * VIX breaches the circuit breaker, or
    * the position has been held max_holding_days.

The output is a *desired exposure* series in [-1, 0]; sizing to contracts is
the job of strategies/sizing.py, and execution/roll mechanics belong to the
backtest engine. Keeping those layers separate is what makes each testable.
"""
from __future__ import annotations

import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


def roll_signal(
    curve: pd.DataFrame,
    exposure_mult: pd.Series,
    contango_entry_threshold: float = 0.05,
    backwardation_exit_threshold: float = -0.02,
    vix_circuit_breaker: float = 30.0,
    max_holding_days: int = 60,
) -> pd.DataFrame:
    """Generate the short-vol carry signal from a VX curve frame.

    Parameters
    ----------
    curve : output of ``term_structure.build_curve`` (needs vix, vx1 columns).
    exposure_mult : regime gate from ``signals.regime.exposure_multiplier``.

    Returns
    -------
    DataFrame with columns:
      ``target``  desired exposure in [-1, 0] (-1 = fully short front VX)
      ``slope``   front slope used for the entry decision
      ``reason``  human-readable state for the trade log
    """
    slope = (curve["vx1"] - curve["vix"]) / curve["vix"]
    mult = exposure_mult.reindex(curve.index).fillna(0.0)

    target = pd.Series(0.0, index=curve.index, name="target")
    reason = pd.Series("flat", index=curve.index, name="reason", dtype=object)
    holding = 0

    for t in curve.index:
        s = slope.loc[t]
        vix = curve.at[t, "vix"]
        if pd.isna(s) or pd.isna(vix):
            reason.loc[t] = "no_data"
            holding = 0
            continue

        in_position = holding > 0
        if vix > vix_circuit_breaker:
            reason.loc[t] = "circuit_breaker"
            holding = 0
        elif s < backwardation_exit_threshold and in_position:
            reason.loc[t] = "backwardation_exit"
            holding = 0
        elif in_position and holding >= max_holding_days:
            reason.loc[t] = "max_holding_exit"
            holding = 0
        elif in_position:
            # asymmetric gate: a closed gate halves held exposure but only the
            # hard exits above flatten it (flattening at the bucket edge sells
            # spike highs / re-buys lower — the measured whipsaw leak)
            target.loc[t] = -1.0 if mult.loc[t] > 0 else -0.5
            reason.loc[t] = "short_carry" if mult.loc[t] > 0 else "hold_reduced"
            holding += 1
        elif s > contango_entry_threshold and mult.loc[t] > 0:
            target.loc[t] = -mult.loc[t]
            reason.loc[t] = "short_carry"
            holding = 1
        else:
            reason.loc[t] = "regime_gate" if mult.loc[t] == 0 else "no_contango"
            holding = 0

    n_active = int((target < 0).sum())
    log.info("vix_roll: short on %d/%d days", n_active, len(target))
    return pd.DataFrame({"target": target, "slope": slope, "reason": reason})
