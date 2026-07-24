"""25-delta put skew (risk reversal) as a risk/opportunity signal.

skew = IV(25Δ put) - IV(25Δ call). Steep skew = puts rich = more premium on
the put wing of a strangle (opportunity, if the regime gate agrees). Flattening
or inverting skew relative to its own history = crash protection being given
away or calls being chased — either way, elevated risk. Signal, not gospel.
"""
from __future__ import annotations

import pandas as pd

from ..volatility.implied import interpolate_smile


def risk_reversal_25d(chain: pd.DataFrame) -> float:
    """25-delta risk reversal from one expiry's chain snapshot.

    ``chain`` needs columns [kind, delta, iv]; deltas as signed BS deltas
    (puts negative). IVs are linearly interpolated in |delta| on each wing.
    Returns NaN if either wing lacks a bracket around 0.25.
    """
    puts = chain[chain["kind"] == "put"].copy()
    calls = chain[chain["kind"] == "call"].copy()
    if puts.empty or calls.empty:
        return float("nan")
    put_iv = interpolate_smile(puts["delta"].abs(), puts["iv"], 0.25)
    call_iv = interpolate_smile(calls["delta"].abs(), calls["iv"], 0.25)
    return float(put_iv - call_iv)


def skew_signal(skew_series: pd.Series, window: int = 252,
                steep_pct: float = 0.8, flat_pct: float = 0.2) -> pd.Series:
    """Classify each day's skew vs its trailing percentile: 'steep' (rich puts),
    'flat' (cheap crash protection — risk elevated), else 'normal'."""
    def _rank(x):
        return float((x[:-1] <= x[-1]).mean())
    pct = skew_series.rolling(window).apply(_rank, raw=True)
    out = pd.Series("normal", index=skew_series.index, name="skew_state", dtype=object)
    out[pct >= steep_pct] = "steep"
    out[pct <= flat_pct] = "flat"
    out[pct.isna()] = "warmup"
    return out
