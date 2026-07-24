"""Volatility regime detection: VIX-level buckets + realized-vol acceleration.

Regimes gate every strategy in this repo — the single most important line of
defense against the short-vol left tail. Two orthogonal detectors:

1. VIX level buckets:  LOW (<15), NORMAL (15-25), CRISIS (>25) by default.
2. Vol acceleration:   5-day RV / 21-day RV. When short-horizon realized vol
   runs well ahead of the monthly baseline, a spike is *in progress* even if
   the VIX level still looks benign (Feb 2018 gapped through the buckets in
   two sessions — acceleration fired earlier).

Every function is strictly causal: value at t uses data up to and including t.
"""
from __future__ import annotations

from enum import Enum

import pandas as pd

from ..config import get_logger
from ..volatility.vrp import trailing_rv

log = get_logger(__name__)


class Regime(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    CRISIS = "crisis"


def vix_regime(
    vix: pd.Series, low_vol_max: float = 15.0, crisis_min: float = 25.0
) -> pd.Series:
    """Bucket each day by VIX close. Returns a Series of Regime values."""
    out = pd.Series(Regime.NORMAL, index=vix.index, name="regime", dtype=object)
    out[vix < low_vol_max] = Regime.LOW
    out[vix > crisis_min] = Regime.CRISIS
    return out


def vol_acceleration(returns: pd.Series, fast: int = 5, slow: int = 21) -> pd.Series:
    """Ratio of fast to slow trailing RV. > 1 means vol is picking up *now*."""
    ratio = trailing_rv(returns, fast) / trailing_rv(returns, slow)
    return ratio.rename("vol_accel")


def accel_flag(returns: pd.Series, threshold: float = 1.5,
               fast: int = 5, slow: int = 21) -> pd.Series:
    """True when 5d RV exceeds ``threshold`` x 21d RV — cut risk regardless of
    the VIX level bucket."""
    return (vol_acceleration(returns, fast, slow) > threshold).rename("vol_accelerating")


def exposure_multiplier(
    vix: pd.Series,
    returns: pd.Series,
    low_vol_max: float = 15.0,
    crisis_min: float = 25.0,
    accel_ratio_max: float = 1.5,
    low_vol_exposure: float = 1.0,
    normal_exposure: float = 1.0,
    crisis_exposure: float = 0.0,
) -> pd.Series:
    """Combined regime gate in [0, 1]: bucket exposure, zeroed while vol is
    accelerating. Multiply any strategy's target position by this."""
    reg = vix_regime(vix, low_vol_max, crisis_min)
    mult = pd.Series(normal_exposure, index=reg.index, name="exposure_mult")
    mult[reg == Regime.LOW] = low_vol_exposure
    mult[reg == Regime.CRISIS] = crisis_exposure
    accel = accel_flag(returns, accel_ratio_max).reindex(mult.index).fillna(False)
    mult[accel] = 0.0
    n_off = int((mult == 0).sum())
    log.info("regime gate: flat on %d/%d days", n_off, len(mult))
    return mult
