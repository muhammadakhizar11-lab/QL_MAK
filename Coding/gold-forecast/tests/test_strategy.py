"""
Hermetic unit tests for the stage-4 strategy engine (no files, no network).
Run:  pytest -q

The load-bearing tests are the timing ones: a signal that KNOWS the forward
return must make money, and a signal that only knows the PAST return must not
get credit for it. An off-by-one in either direction fails one of the pair.
"""
import numpy as np
import pandas as pd

from src import strategy

IDX = pd.bdate_range("2020-01-01", periods=500)
RNG = np.random.default_rng(11)


def _ret(sigma=0.01, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else RNG
    return pd.Series(rng.normal(0, sigma, len(IDX)), index=IDX)


def _sc(**over):
    sc = {"mode": "long_short", "vol_target_ann": None, "vol_window": 20,
          "max_leverage": 2.0, "cost_bps": 0.0, "stop": None, "trading_days": 252}
    sc.update(over)
    return sc


# --------------------------------------------------------------------------- #
# Timing alignment — the classic backtest off-by-one
# --------------------------------------------------------------------------- #
def test_foresight_signal_profits_and_reversed_loses():
    ret = _ret(seed=1)
    good = strategy.run_strategy("clairvoyant", ret, ret, _sc())
    bad = strategy.run_strategy("inverted", -ret, ret, _sc())
    # sign(r_t) * r_t = |r_t| > 0 every day
    assert (good.daily["net"].dropna() >= 0).all()
    assert good.metrics["ann_ret_net"] > 0
    assert bad.metrics["ann_ret_net"] < 0


def test_stale_signal_earns_nothing_systematic():
    """A signal equal to YESTERDAY'S return has no information about today's iid
    return — mean net return must be statistically indistinguishable from zero.
    If a timing bug shifted returns back one day this would look clairvoyant."""
    ret = _ret(seed=2)
    stale = strategy.run_strategy("stale", ret.shift(1), ret, _sc())
    daily = stale.daily["net"].dropna()
    tstat = daily.mean() / (daily.std(ddof=0) / np.sqrt(len(daily)))
    assert abs(tstat) < 3, f"stale signal shows systematic P&L (t={tstat:.1f}) — timing leak"


# --------------------------------------------------------------------------- #
# Position construction
# --------------------------------------------------------------------------- #
def test_long_flat_never_short_and_nan_is_flat():
    sig = pd.Series([1.0, -1.0, np.nan, 0.5], index=IDX[:4])
    pos = strategy.unit_position(sig, "long_flat")
    assert pos.tolist() == [1.0, 0.0, 0.0, 1.0]
    pos_ls = strategy.unit_position(sig, "long_short")
    assert pos_ls.tolist() == [1.0, -1.0, 0.0, 1.0]


def test_vol_targeting_hits_target_and_respects_cap():
    ret = _ret(sigma=0.02, seed=3)                     # ~32% annualized
    pos = pd.Series(1.0, index=IDX)
    scaled = strategy.vol_scale(pos, ret, target_ann=0.10, window=20,
                                max_leverage=2.0)
    realized = (scaled * ret).std(ddof=0) * np.sqrt(252)
    unscaled = ret.std(ddof=0) * np.sqrt(252)
    # a 20d trailing estimator is noisy and 1/vol is convex, so realized vol
    # overshoots the target somewhat; the property that matters is that ~32%
    # asset vol is pulled down to the target's neighborhood.
    assert 0.06 < realized < 0.16, f"realized {realized:.3f} not near 0.10 target"
    assert realized < unscaled / 2, "vol targeting barely reduced vol"
    # high target on low-vol asset must clip at the leverage cap
    calm = _ret(sigma=0.001, seed=4)
    capped = strategy.vol_scale(pos, calm, target_ann=0.50, window=20,
                                max_leverage=2.0)
    assert capped.max() <= 2.0 + 1e-12


def test_costs_charged_on_turnover():
    ret = pd.Series(0.0, index=IDX[:6])
    sig = pd.Series([1, 1, -1, -1, 1, 1], index=IDX[:6], dtype=float)
    res = strategy.run_strategy("flip", sig, ret, _sc(cost_bps=10.0))
    # entry (1) + two flips (2 each) = 5 units of turnover at 10bp
    assert np.isclose(res.daily["turnover"].sum(), 5.0)
    assert np.isclose(res.daily["cost"].sum(), 5.0 * 10e-4)
    assert np.isclose(res.metrics["ann_ret_gross"], 0.0)


def test_stop_flattens_after_drawdown_then_resumes():
    # long position into a relentless -2%/day slide: a 10% stop must fire and
    # keep the strategy flat during the cooldown
    n = 60
    idx = IDX[:n]
    ret = pd.Series([-0.02] * 20 + [0.0] * (n - 20), index=idx)
    pos = pd.Series(1.0, index=idx)
    stopped = strategy.apply_stop(pos, ret, max_drawdown=0.10, cooldown_days=10)
    assert stopped.iloc[0] == 1.0, "starts long"
    fired = np.flatnonzero(stopped.to_numpy() == 0.0)
    assert len(fired) > 0, "stop never fired on a 40% slide"
    first = fired[0]
    assert first <= 8, "stop should fire once cumulative loss exceeds 10%"
    assert (stopped.iloc[first:first + 10] == 0.0).all(), "cooldown not respected"
    # loss is bounded ~ stop level, far above the unstopped 40% loss
    assert (stopped * ret).sum() > -0.20


def test_metrics_sane_on_flat_strategy():
    ret = _ret(seed=5)
    res = strategy.run_strategy("flat", pd.Series(0.0, index=IDX), ret,
                                _sc(mode="long_flat"))
    m = res.metrics
    assert m["ann_ret_net"] == 0.0 and m["exposure_avg"] == 0.0
    assert m["max_drawdown"] == 0.0
