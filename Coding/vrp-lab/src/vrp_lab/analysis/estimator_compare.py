"""Compare RV estimators on real data: level agreement, dispersion, and
behavior in high-vol episodes (where they diverge most).

The GBM bias/efficiency story is in tests/test_realized.py against known
sigma; this module is the empirical companion on actual SPY bars, where
overnight gaps and drift break each estimator's assumptions differently.
"""
from __future__ import annotations

import pandas as pd

from ..volatility.realized import ESTIMATORS, realized_vol


def run_all(ohlc: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """All estimators on one OHLC frame -> wide frame of annualized vols."""
    return pd.DataFrame({
        name: realized_vol(ohlc, estimator=name, window=window)
        for name in ESTIMATORS
    })


def divergence_stats(estimates: pd.DataFrame,
                     baseline: str = "close_to_close") -> pd.DataFrame:
    """Mean level, mean and max abs gap to the baseline, correlation.

    Expect: range estimators sit below close-to-close on gappy equities
    (overnight variance invisible to them), Yang-Zhang closest to baseline
    level with far less sampling noise.
    """
    rows = {}
    base = estimates[baseline]
    for col in estimates.columns:
        gap = (estimates[col] - base).dropna()
        rows[col] = {
            "mean_level": estimates[col].mean(),
            "stdev_of_estimate": estimates[col].std(),
            "mean_gap_vs_cc": gap.mean(),
            "max_abs_gap_vs_cc": gap.abs().max(),
            "corr_vs_cc": estimates[col].corr(base),
        }
    return pd.DataFrame(rows).T


def high_vol_episodes(estimates: pd.DataFrame, vix: pd.Series,
                      crisis_min: float = 25.0) -> pd.DataFrame:
    """Same stats restricted to VIX > crisis_min days — the regime where the
    estimator choice actually changes decisions."""
    mask = vix.reindex(estimates.index) > crisis_min
    return divergence_stats(estimates[mask])
