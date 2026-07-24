"""Conditional VRP entry: trade only when the *trailing-proxy* VRP is rich
relative to its own recent history.

Uses vrp_trailing (observable) — never vrp_ex_post (future RV). The z-score
window and entry threshold come from config. This signal is a conditioner:
multiply it into another strategy's target rather than trading it alone.
"""
from __future__ import annotations

import pandas as pd

from ..volatility.vrp import rolling_zscore, vrp_trailing


def vrp_condition(
    iv: pd.Series,
    returns: pd.Series,
    rv_window: int = 21,
    zscore_window: int = 252,
    entry_z: float = 0.5,
) -> pd.DataFrame:
    """1.0 when trailing VRP z-score > entry_z, else 0.0 (plus diagnostics).

    Interpretation: a high z-score says implied is unusually expensive vs
    recent realized — the premium on offer is fat. A *negative* z-score after
    a spike often just means trailing RV is stale-high; the regime filter, not
    this signal, is responsible for that case.
    """
    v = vrp_trailing(iv, returns, rv_window)
    z = rolling_zscore(v, zscore_window)
    on = (z > entry_z).astype(float).rename("vrp_on")
    return pd.DataFrame({"vrp_trailing": v, "vrp_z": z, "vrp_on": on})
