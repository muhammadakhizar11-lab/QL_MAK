"""How the VRP behaves across volatility regimes — the empirical justification
for the regime gate.

Question the table answers: conditioned on the regime you can OBSERVE at t,
what did the premium over (t, t+21] look like? The classic result: the mean
premium is largest in high-VIX states, but so is the left tail — the extra
premium is payment for holding the bag through the spike. The gate's job is
to skip the days where the bag is already on fire (curve inverted, vol
accelerating), not to avoid high VIX per se.
"""
from __future__ import annotations

import pandas as pd

from ..signals.regime import Regime, accel_flag, vix_regime


def vrp_by_regime(vrp_expost: pd.Series, vix: pd.Series,
                  low_vol_max: float = 15.0, crisis_min: float = 25.0) -> pd.DataFrame:
    """Subsequent-VRP stats bucketed by the VIX regime observable at t."""
    reg = vix_regime(vix, low_vol_max, crisis_min).reindex(vrp_expost.index)
    df = pd.DataFrame({"vrp": 100 * vrp_expost,
                       "regime": reg.map(lambda r: r.value if isinstance(r, Regime) else r)})
    df = df.dropna()
    g = df.groupby("regime")["vrp"]
    out = pd.DataFrame({
        "mean": g.mean(), "median": g.median(),
        "share_positive": g.apply(lambda x: (x > 0).mean()),
        "p5": g.quantile(0.05), "worst": g.min(), "n_days": g.size(),
    })
    return out.reindex([r.value for r in Regime])


def vrp_by_curve_state(vrp_expost: pd.Series, inverted: pd.Series) -> pd.DataFrame:
    """Same conditioning on curve inversion — the crisis tell the strategy
    actually trades on."""
    df = pd.DataFrame({"vrp": 100 * vrp_expost, "inverted": inverted}).dropna()
    g = df.groupby("inverted")["vrp"]
    return pd.DataFrame({
        "mean": g.mean(), "median": g.median(),
        "share_positive": g.apply(lambda x: (x > 0).mean()),
        "p5": g.quantile(0.05), "worst": g.min(), "n_days": g.size(),
    })


def gate_hit_rate(returns: pd.Series, vix: pd.Series,
                  accel_ratio_max: float = 1.5, crisis_min: float = 25.0) -> pd.Series:
    """Of the worst-decile VRP days, what share did each gate component flag
    in advance? (Both flags are observable at t; the outcome is (t, t+21].)"""
    accel = accel_flag(returns, threshold=accel_ratio_max)
    crisis = vix > crisis_min
    return pd.Series({
        "crisis_days": int(crisis.sum()),
        "accel_days": int(accel.reindex(vix.index).fillna(False).sum()),
        "either": int((crisis | accel.reindex(vix.index).fillna(False)).sum()),
        "total_days": len(vix),
    }, name="gate_coverage")
