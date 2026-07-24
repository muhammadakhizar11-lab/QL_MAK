"""IV extraction correctness: round-trips, put-call parity, delta inversion."""
import numpy as np
import pytest

from vrp_lab.volatility import implied as iv


S, K, T, R, Q = 100.0, 100.0, 30 / 252, 0.03, 0.01


def test_bs_price_put_call_parity():
    sigma = 0.25
    c = iv.bs_price(S, K, T, sigma, R, Q, "call")
    p = iv.bs_price(S, K, T, sigma, R, Q, "put")
    lhs = c - p
    rhs = S * np.exp(-Q * T) - K * np.exp(-R * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("sigma", [0.05, 0.20, 0.80, 2.0])
@pytest.mark.parametrize("kind,k", [
    # ATM both sides, plus OTM wings — where IV is actually extracted.
    # Deep-ITM low-vol options price at intrinsic (vega ~ 0) and are not
    # numerically invertible; real pipelines invert the OTM side (as OptionMetrics does).
    ("call", 100.0), ("put", 100.0), ("call", 120.0), ("put", 80.0),
])
def test_iv_round_trip(sigma, kind, k):
    price = iv.bs_price(S, k, T, sigma, R, Q, kind)
    recovered = iv.implied_vol(price, S, k, T, R, Q, kind)
    assert recovered == pytest.approx(sigma, rel=1e-5)


def test_iv_nan_outside_arbitrage_bounds():
    # a call below intrinsic value has no BS vol
    assert np.isnan(iv.implied_vol(0.01, S, 80.0, T, 0.0, 0.0, "call"))


def test_expired_option_is_intrinsic():
    assert iv.bs_price(105.0, 100.0, 0.0, 0.3, kind="call") == 5.0
    assert iv.bs_price(105.0, 100.0, 0.0, 0.3, kind="put") == 0.0


def test_greeks_signs_and_magnitudes():
    g_call = iv.bs_greeks(S, K, T, 0.20, R, Q, "call")
    g_put = iv.bs_greeks(S, K, T, 0.20, R, Q, "put")
    assert 0.4 < g_call["delta"] < 0.7
    assert -0.6 < g_put["delta"] < -0.3
    assert g_call["gamma"] > 0 and g_put["gamma"] > 0
    assert g_call["gamma"] == pytest.approx(g_put["gamma"], rel=1e-12)
    assert g_call["vega"] == pytest.approx(g_put["vega"], rel=1e-12)
    assert g_call["theta"] < 0  # ATM long option decays


def test_delta_finite_difference():
    """Analytic delta must match a central finite difference of price."""
    sigma, eps = 0.3, 1e-4
    g = iv.bs_greeks(S, K, T, sigma, R, Q, "call")
    fd = (iv.bs_price(S + eps, K, T, sigma, R, Q, "call")
          - iv.bs_price(S - eps, K, T, sigma, R, Q, "call")) / (2 * eps)
    assert g["delta"] == pytest.approx(fd, abs=1e-6)


def test_strike_from_delta_round_trip():
    sigma, target = 0.22, 0.25
    k_put = iv.strike_from_delta(target, S, T, sigma, R, Q, "put")
    assert k_put < S  # 25-delta put is OTM
    d = iv.bs_greeks(S, k_put, T, sigma, R, Q, "put")["delta"]
    assert abs(d) == pytest.approx(target, rel=1e-6)


def test_interpolate_smile():
    import pandas as pd
    strikes = pd.Series([90.0, 100.0, 110.0])
    ivs = pd.Series([0.30, 0.20, 0.22])
    assert iv.interpolate_smile(strikes, ivs, 95.0) == pytest.approx(0.25)
    # clamps outside the observed range
    assert iv.interpolate_smile(strikes, ivs, 50.0) == pytest.approx(0.30)
