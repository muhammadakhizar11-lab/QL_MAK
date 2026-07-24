r"""Volatility Risk Premium: measurement (ex-post) vs signal (trailing proxy).

Definition (forward-looking, honest):

.. math::
    VRP(t,\tau) = IV(t,\tau) - RV(t, t+\tau)

where :math:`IV(t,\tau)` is implied vol observed at t for horizon τ and
:math:`RV(t,t+\tau)` is the volatility *subsequently realized* over [t, t+τ].

UNITS: everything in this module is in decimal annualized vol (0.20 = 20%).
Divide VIX by 100 before passing it in (``iv_from_vix`` does this).

THE LOOK-AHEAD DISTINCTION — read this before using anything here:

* ``vrp_ex_post``    uses **future** realized vol. At time t you cannot know
  it. It exists for *measurement and research only* (how big was the premium,
  when did it invert). It must NEVER feed a trading signal. The backtest
  engine's no-look-ahead guard will reject it.
* ``vrp_trailing``   replaces future RV with **trailing** RV — a proxy that
  is observable at t. This is the only tradeable variant. The proxy error
  (trailing RV lags future RV at turning points) is exactly why regime
  filters exist on top of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


def iv_from_vix(vix_close: pd.Series) -> pd.Series:
    """VIX quoted in vol points (20.0) -> decimal annualized IV (0.20).

    The VIX is the square root of the 30-day variance swap rate on SPX —
    a model-free 30-day IV. Note it is a *variance* swap vol, so it sits
    slightly above ATM Black-Scholes IV by the convexity of the smile.
    """
    return (vix_close / 100.0).rename("iv")


def forward_rv(returns: pd.Series, horizon: int) -> pd.Series:
    r"""Volatility realized over the *future* window (t, t+horizon].

    .. warning::
        LOOK-AHEAD BY CONSTRUCTION. The value at date t summarizes returns
        from t+1 through t+horizon. Measurement only — never a signal input.

    Parameters
    ----------
    returns : daily log returns; aggregated as
        :math:`RV = \sqrt{\frac{252}{\tau}\sum_{i=t+1}^{t+\tau} r_i^2}`
        (zero-mean convention).
    horizon : window length τ in trading days.
    """
    r2 = returns**2
    # rolling(h).mean() at t+h covers (t, t+h]; shifting back by h stamps it at t
    fwd_mean = r2.rolling(horizon).mean().shift(-horizon)
    out = np.sqrt(fwd_mean * 252.0)
    out.name = f"fwd_rv_{horizon}d"
    return out


def trailing_rv(returns: pd.Series, window: int) -> pd.Series:
    r"""Annualized vol realized over the *past* window (t-window, t]. Observable at t.

    :math:`RV = \sqrt{\frac{252}{n}\sum_{i=t-n+1}^{t} r_i^2}` (zero-mean
    convention, standard for vol trading horizons where drift is negligible).
    """
    out = np.sqrt((returns**2).rolling(window).mean() * 252.0)
    out.name = f"trail_rv_{window}d"
    return out


def vrp_ex_post(iv: pd.Series, returns: pd.Series, horizon: int) -> pd.Series:
    """Ex-post VRP(t, τ) = IV(t) - RV(t, t+τ).  MEASUREMENT ONLY (uses future RV).

    Both inputs in decimal vol. The last ``horizon`` observations are NaN
    because their future window is incomplete — they are unknowable, not zero.
    """
    iv_a, r_a = iv.align(returns, join="inner")
    out = iv_a - forward_rv(r_a, horizon)
    out.name = f"vrp_expost_{horizon}d"
    return out


def vrp_trailing(iv: pd.Series, returns: pd.Series, window: int) -> pd.Series:
    """Trailing-proxy VRP(t) = IV(t) - trailing RV(t-window, t).  Tradeable at t.

    This is the signal variant: every input is observable at t. Its bias is
    known — after a vol spike, trailing RV stays elevated while IV normalizes,
    so the proxy *understates* the true premium; before a spike it overstates
    it. Document and regime-filter accordingly.
    """
    iv_a, r_a = iv.align(returns, join="inner")
    out = iv_a - trailing_rv(r_a, window)
    out.name = f"vrp_trailing_{window}d"
    return out


def rolling_zscore(s: pd.Series, window: int = 252, min_periods: int | None = None) -> pd.Series:
    """Strictly trailing z-score: (x_t - trailing mean) / trailing std."""
    mp = min_periods or window // 2
    mean = s.rolling(window, min_periods=mp).mean()
    std = s.rolling(window, min_periods=mp).std(ddof=0)
    return ((s - mean) / std).rename(f"{s.name}_z")


def rolling_percentile(s: pd.Series, window: int = 252) -> pd.Series:
    """Trailing percentile rank of the latest value within its own window, in [0, 1]."""
    def _rank(x: np.ndarray) -> float:
        return float((x[:-1] <= x[-1]).mean())
    return s.rolling(window).apply(_rank, raw=True).rename(f"{s.name}_pct")


def vrp_panel(
    iv: pd.Series,
    returns: pd.Series,
    horizons: list[int] = (21, 63),
    zscore_window: int = 252,
) -> pd.DataFrame:
    """Convenience: ex-post + trailing VRP for each horizon, with z-scores.

    Columns ``vrp_expost_*`` are research-only (future RV); columns
    ``vrp_trailing_*`` and their z-scores are the tradeable set.
    """
    cols = {}
    for h in horizons:
        cols[f"vrp_expost_{h}d"] = vrp_ex_post(iv, returns, h)
        trail = vrp_trailing(iv, returns, h)
        cols[f"vrp_trailing_{h}d"] = trail
        cols[f"vrp_trailing_{h}d_z"] = rolling_zscore(trail, zscore_window)
    df = pd.DataFrame(cols)
    log.info("VRP panel: %d rows, horizons=%s", len(df), list(horizons))
    return df
