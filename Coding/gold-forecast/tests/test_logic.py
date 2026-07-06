"""
Hermetic unit tests for stage 1-2 logic (no network, no file I/O).
Run:  pytest -q
"""
import numpy as np
import pandas as pd

from src import rolls, features


def test_align_monthly_no_lookahead():
    """A monthly obs dated month M must not be visible before its release-safe date."""
    months = pd.date_range("2020-01-01", "2020-12-01", freq="MS")
    monthly = pd.Series(1.0, index=months)
    monthly.loc["2020-01-01"] = 999.0            # distinctive Jan value

    daily = pd.bdate_range("2020-01-01", "2020-12-31")
    aligned = features.align_monthly(monthly, daily, release_lag_months=1)

    # release_lag=1 -> shift (1+1)=2 months -> Jan value available from 1 Mar.
    jan_feb = aligned.loc["2020-01-01":"2020-02-29"]
    assert not (jan_feb == 999.0).any(), "Jan value leaked into Jan/Feb (look-ahead!)"
    mar = aligned.loc["2020-03-01":"2020-03-31"]
    assert (mar == 999.0).all(), "Jan value should be available from March onward"


def test_roll_detected_at_boundary_not_midmonth():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    close = pd.Series(100.0, index=idx)
    close += np.linspace(0, 5, len(close))        # gentle drift, no spikes

    boundary = idx[(idx.year == 2020) & (idx.month == 6)][-1]   # last bday of June
    midmonth = pd.Timestamp("2020-09-15")
    close.loc[close.index >= boundary] *= 1.10    # roll-like jump at boundary
    close.loc[midmonth] *= 1.10                   # same-size jump mid-month

    cand = rolls.detect_roll_candidates(close, jump_z=4.0, jump_hard=0.08,
                                        month_boundary_window=5)
    assert boundary in cand, "boundary jump should be a roll candidate"
    assert midmonth not in cand, "mid-month jump should NOT be flagged as a roll"


def test_neutralize_sets_roll_return_nan():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    close = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    roll_date = idx[10]
    clean = rolls.neutralize_roll_returns(close, pd.DatetimeIndex([roll_date]))
    assert np.isnan(clean.loc[roll_date])
    assert clean.drop(index=roll_date).notna().sum() > 0


def test_rsi_bounded():
    idx = pd.bdate_range("2020-01-01", "2021-01-01")
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(idx)))), index=idx)
    r = features.rsi(close, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_rolling_zscore_is_trailing():
    """Last z-score must be computable from the trailing window only (no future)."""
    s = pd.Series(np.arange(100.0))
    z = features.rolling_zscore(s, window=20)
    # recompute the final value by hand from the trailing 20 obs
    tail = s.iloc[-20:]
    expected = (s.iloc[-1] - tail.mean()) / tail.std(ddof=0)
    assert np.isclose(z.iloc[-1], expected)
