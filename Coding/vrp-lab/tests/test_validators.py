"""Validator behavior: hard failures raise, soft issues warn, ffill is bounded."""
import numpy as np
import pandas as pd
import pytest

from vrp_lab.data import validators as v


def _frame(n=50):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(0)
    return pd.DataFrame({"close": 100 + rng.normal(size=n).cumsum()}, index=idx)


def test_valid_frame_passes():
    rep = v.validate_frame(_frame(), "test", required_cols=["close"])
    assert rep.rows == 50 and rep.warnings == []


def test_empty_frame_raises():
    with pytest.raises(v.ValidationError, match="empty"):
        v.validate_frame(pd.DataFrame(), "test")


def test_unsorted_index_raises():
    df = _frame().iloc[::-1]
    with pytest.raises(v.ValidationError, match="not sorted"):
        v.validate_frame(df, "test")


def test_duplicate_dates_raise():
    df = pd.concat([_frame(), _frame().iloc[[10]]]).sort_index()
    with pytest.raises(v.ValidationError, match="duplicate"):
        v.validate_frame(df, "test")


def test_missing_column_raises():
    with pytest.raises(v.ValidationError, match="missing required column"):
        v.validate_frame(_frame(), "test", required_cols=["settle"])


def test_future_rows_vs_as_of_raise():
    """The point-in-time guard: data stamped after as_of is look-ahead."""
    df = _frame()
    with pytest.raises(v.ValidationError, match="look-ahead"):
        v.validate_frame(df, "test", as_of=df.index[25])


def test_stale_run_warns():
    df = _frame()
    df.iloc[10:30] = 55.5  # 20-day identical run
    rep = v.validate_frame(df, "test", required_cols=["close"], staleness_max_repeats=10)
    assert any("identical" in w for w in rep.warnings)


def test_bounded_ffill_respects_limit():
    idx = pd.bdate_range("2020-01-01", periods=20)
    s = pd.Series(1.0, index=idx)
    s.iloc[3:6] = np.nan    # 3-day gap: fillable at limit 5
    s.iloc[10:17] = np.nan  # 7-day gap: only first 5 filled
    out = v.bounded_ffill(s, max_gap_days=5)
    assert out.iloc[3:6].notna().all()
    assert out.iloc[10:15].notna().all()
    assert out.iloc[15:17].isna().all()


def test_align_business_days_reindexes():
    df = _frame(10).drop(index=pd.Timestamp("2020-01-08"))
    out = v.align_business_days(df)
    assert pd.Timestamp("2020-01-08") in out.index
    assert out.loc["2020-01-08", "close"] == df.loc["2020-01-07", "close"]
