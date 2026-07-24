"""VRP point-in-time integrity: the ex-post variant must use exactly the
future window (and nothing else); the trailing variant must be causal."""
import numpy as np
import pandas as pd
import pytest

from vrp_lab.volatility import vrp


@pytest.fixture()
def spike_returns() -> pd.Series:
    """Zero returns everywhere except one huge move on a known date."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    r = pd.Series(0.0, index=idx)
    r.iloc[100] = 0.10  # 10% one-day move
    return r


def test_forward_rv_sees_only_future_window(spike_returns):
    fwd = vrp.forward_rv(spike_returns, horizon=21)
    spike_pos = 100
    # dates whose (t, t+21] window contains the spike: positions 79..99
    assert (fwd.iloc[spike_pos - 21: spike_pos] > 0).all()
    # the spike's own date: window is (100, 121] -> spike NOT included
    assert fwd.iloc[spike_pos] == 0.0
    # well before: zero
    assert (fwd.iloc[: spike_pos - 21] == 0.0).all()


def test_trailing_rv_is_causal(spike_returns):
    trail = vrp.trailing_rv(spike_returns, window=21)
    spike_pos = 100
    # value on the spike date includes it; the day before does not
    assert trail.iloc[spike_pos] > 0
    assert trail.iloc[spike_pos - 1] == 0.0
    # falls out of the window 21 days later
    assert trail.iloc[spike_pos + 21] == 0.0


def test_ex_post_vrp_last_horizon_days_are_nan(spike_returns):
    iv = pd.Series(0.20, index=spike_returns.index)
    out = vrp.vrp_ex_post(iv, spike_returns, horizon=21)
    assert out.iloc[-21:].isna().all(), "incomplete future windows must be NaN, not zero"
    assert out.iloc[:-21].notna().all()


def test_trailing_vrp_needs_no_future_data(spike_returns):
    """Trailing VRP at t must be identical whether or not future rows exist."""
    iv = pd.Series(0.20, index=spike_returns.index)
    full = vrp.vrp_trailing(iv, spike_returns, window=21)
    cut = vrp.vrp_trailing(iv.iloc[:150], spike_returns.iloc[:150], window=21)
    t = cut.index[-1]
    assert np.isclose(full.loc[t], cut.loc[t])


def test_vrp_sign_convention(flat_returns):
    """IV 20%, realized 0% -> VRP = +0.20 in both variants."""
    iv = pd.Series(0.20, index=flat_returns.index)
    expost = vrp.vrp_ex_post(iv, flat_returns, horizon=21).dropna()
    trail = vrp.vrp_trailing(iv, flat_returns, window=21).dropna()
    assert np.allclose(expost, 0.20)
    assert np.allclose(trail, 0.20)


def test_iv_from_vix_units():
    vix = pd.Series([20.0, 82.69])
    assert np.allclose(vrp.iv_from_vix(vix).values, [0.20, 0.8269])


def test_rolling_zscore_is_trailing():
    s = pd.Series(np.arange(400.0))
    z = vrp.rolling_zscore(s, window=100)
    tail = s.iloc[-100:]
    expected = (s.iloc[-1] - tail.mean()) / tail.std(ddof=0)
    assert np.isclose(z.iloc[-1], expected)


def test_rolling_percentile_bounds():
    rng = np.random.default_rng(3)
    s = pd.Series(rng.normal(size=600))
    p = vrp.rolling_percentile(s, window=252).dropna()
    assert (p >= 0).all() and (p <= 1).all()


def test_vrp_panel_columns(flat_returns):
    iv = pd.Series(0.2, index=flat_returns.index)
    panel = vrp.vrp_panel(iv, flat_returns, horizons=[21, 63])
    for h in (21, 63):
        assert f"vrp_expost_{h}d" in panel
        assert f"vrp_trailing_{h}d" in panel
        assert f"vrp_trailing_{h}d_z" in panel
