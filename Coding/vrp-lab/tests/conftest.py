"""Shared fixtures: synthetic GBM paths with known vol, synthetic VX curves.

All tests are hermetic — no network, no reads from data/. Anything a test
needs is generated here with fixed seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

ANN = 252


def make_gbm_ohlc(
    sigma: float,
    n_days: int = 5000,
    mu: float = 0.05,
    seed: int = 0,
    steps_per_day: int = 390,      # 1-min bars: keeps range-discretization bias <3% in vol
    overnight_frac: float = 0.0,   # share of daily variance realized overnight
    s0: float = 100.0,
) -> pd.DataFrame:
    """Simulate GBM intraday and aggregate to daily OHLC with known annual sigma.

    overnight_frac > 0 moves part of each day's variance into an open gap,
    which is what breaks Parkinson/Garman-Klass but not Yang-Zhang.
    """
    rng = np.random.default_rng(seed)
    dt_day = 1.0 / ANN
    var_day = sigma**2 * dt_day
    var_on = var_day * overnight_frac
    var_id = var_day * (1.0 - overnight_frac)
    dt_step = var_id / steps_per_day  # variance per intraday step (already sigma-scaled)

    idx = pd.bdate_range("2005-01-03", periods=n_days)
    o = np.empty(n_days); h = np.empty(n_days); l = np.empty(n_days); c = np.empty(n_days)
    price = s0
    drift_day = (mu - 0.5 * sigma**2) * dt_day
    for i in range(n_days):
        gap = rng.normal(0.0, np.sqrt(var_on)) if var_on > 0 else 0.0
        o[i] = price * np.exp(gap)
        steps = rng.normal(drift_day / steps_per_day, np.sqrt(dt_step), steps_per_day)
        path = o[i] * np.exp(np.cumsum(steps))
        h[i] = max(o[i], path.max())
        l[i] = min(o[i], path.min())
        c[i] = path[-1]
        price = c[i]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


@pytest.fixture(scope="session")
def gbm_20() -> pd.DataFrame:
    """Continuous GBM, sigma = 20%, no overnight gaps."""
    return make_gbm_ohlc(sigma=0.20, seed=42)


@pytest.fixture(scope="session")
def gbm_20_gaps() -> pd.DataFrame:
    """GBM, sigma = 20% total, 30% of variance overnight (equity-like)."""
    return make_gbm_ohlc(sigma=0.20, seed=7, overnight_frac=0.30)


@pytest.fixture()
def synthetic_curve() -> pd.DataFrame:
    """A tiny VX curve frame with known contango/backwardation days."""
    idx = pd.bdate_range("2020-01-01", periods=6)
    return pd.DataFrame(
        {
            "vix":  [15.0, 15.0, 15.0, 30.0, 30.0, 16.0],
            "vx1":  [16.5, 16.0, 15.2, 27.0, 28.0, 16.4],
            "vx2":  [17.5, 17.0, 15.1, 25.0, 26.0, 17.2],
            "dte1": [20, 19, 18, 17, 16, 15],
            "dte2": [48, 47, 46, 45, 44, 43],
        },
        index=idx,
    )


@pytest.fixture()
def flat_returns() -> pd.Series:
    idx = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(0.0, index=idx)
