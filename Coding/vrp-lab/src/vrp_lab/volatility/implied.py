r"""Implied volatility: Black-Scholes pricing/greeks, IV inversion, smile tools.

The VIX itself is our primary 30-day IV series (model-free variance-swap vol);
this module supplies the option-level machinery used by Phase 2: pricing
synthetic options, backing IV out of observed prices, and interpolating a
smile at fixed deltas for the skew signal.

Conventions: decimal vol (0.20), decimal rates, time in years (ACT/365-ish via
trading days / 252 is fine at these horizons), forward measure via Black-Scholes
with continuous rate ``r`` and zero dividend unless ``q`` given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def _d1(s: float, k: float, t: float, sigma: float, r: float, q: float) -> float:
    return (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def bs_price(
    s: float, k: float, t: float, sigma: float,
    r: float = 0.0, q: float = 0.0, kind: str = "call",
) -> float:
    r"""Black-Scholes-Merton European option price.

    .. math::
        C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2),\qquad
        d_{1,2} = \frac{\ln(S/K) + (r - q \pm \sigma^2/2)T}{\sigma\sqrt T}
    """
    if t <= 0:
        intrinsic = max(s - k, 0.0) if kind == "call" else max(k - s, 0.0)
        return float(intrinsic)
    d1 = _d1(s, k, t, sigma, r, q)
    d2 = d1 - sigma * np.sqrt(t)
    if kind == "call":
        return float(s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1))


def bs_greeks(
    s: float, k: float, t: float, sigma: float,
    r: float = 0.0, q: float = 0.0, kind: str = "call",
) -> dict[str, float]:
    r"""Black-Scholes greeks (per 1 underlying unit; theta per *year*).

    delta = :math:`e^{-qT}N(d_1)` (call) / :math:`-e^{-qT}N(-d_1)` (put);
    gamma = :math:`e^{-qT}\phi(d_1)/(S\sigma\sqrt T)`;
    vega  = :math:`S e^{-qT}\phi(d_1)\sqrt T` (per 1.00 of vol, i.e. per 100 pts);
    theta = standard BSM calendar decay.
    """
    if t <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    sqrt_t = np.sqrt(t)
    d1 = _d1(s, k, t, sigma, r, q)
    d2 = d1 - sigma * sqrt_t
    pdf = norm.pdf(d1)
    gamma = np.exp(-q * t) * pdf / (s * sigma * sqrt_t)
    vega = s * np.exp(-q * t) * pdf * sqrt_t
    if kind == "call":
        delta = np.exp(-q * t) * norm.cdf(d1)
        theta = (-s * np.exp(-q * t) * pdf * sigma / (2 * sqrt_t)
                 - r * k * np.exp(-r * t) * norm.cdf(d2)
                 + q * s * np.exp(-q * t) * norm.cdf(d1))
    else:
        delta = -np.exp(-q * t) * norm.cdf(-d1)
        theta = (-s * np.exp(-q * t) * pdf * sigma / (2 * sqrt_t)
                 + r * k * np.exp(-r * t) * norm.cdf(-d2)
                 - q * s * np.exp(-q * t) * norm.cdf(-d1))
    return {"delta": float(delta), "gamma": float(gamma),
            "vega": float(vega), "theta": float(theta)}


def implied_vol(
    price: float, s: float, k: float, t: float,
    r: float = 0.0, q: float = 0.0, kind: str = "call",
    lo: float = 1e-4, hi: float = 5.0,
) -> float:
    """Invert Black-Scholes for IV via Brent's method. NaN if the price sits
    outside no-arbitrage bounds (below intrinsic / above forward cap)."""
    try:
        return float(brentq(lambda sig: bs_price(s, k, t, sig, r, q, kind) - price, lo, hi,
                            xtol=1e-8, maxiter=200))
    except ValueError:
        return float("nan")


def strike_from_delta(
    delta: float, s: float, t: float, sigma: float,
    r: float = 0.0, q: float = 0.0, kind: str = "put",
) -> float:
    r"""Strike whose Black-Scholes delta equals ``delta`` (put delta passed positive).

    Solved in closed form: for a put, :math:`d_1 = -N^{-1}(|\Delta|e^{qT})`
    (for a call, :math:`d_1 = N^{-1}(\Delta e^{qT})`), then
    :math:`K = S\exp\{(r-q+\tfrac12\sigma^2)T - d_1\sigma\sqrt T\}`.
    """
    adj = abs(delta) * np.exp(q * t)
    if kind == "put":
        d1 = -norm.ppf(adj)
    else:
        d1 = norm.ppf(adj)
    return float(s * np.exp((r - q + 0.5 * sigma**2) * t - d1 * sigma * np.sqrt(t)))


def interpolate_smile(strikes: pd.Series, ivs: pd.Series, k: float) -> float:
    """Linear IV interpolation in strike; clamps to the closest end outside range.

    Deliberately simple (no SVI/SABR fitting) — documented limitation. Inputs
    must be same-length; NaN IVs are dropped.
    """
    df = pd.DataFrame({"k": strikes.values, "iv": ivs.values}).dropna().sort_values("k")
    if df.empty:
        return float("nan")
    return float(np.interp(k, df["k"].values, df["iv"].values))
