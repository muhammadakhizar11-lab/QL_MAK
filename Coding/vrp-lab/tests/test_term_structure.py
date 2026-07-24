"""Term structure classifier, roll yield signs, and the VX expiry calendar
against known exchange dates."""
import pandas as pd

from vrp_lab.backtest import calendar as cal
from vrp_lab.volatility import term_structure as ts


def test_classifier_on_synthetic_curve(synthetic_curve):
    regime = ts.classify_regime(synthetic_curve)
    assert list(regime) == [
        "contango",       # 17.5 > 16.5 > 15
        "contango",
        "mixed",          # vx1 > vix but vx2 < vx1
        "backwardation",  # vx1 27 < vix 30
        "backwardation",
        "contango",
    ]


def test_inversion_flag(synthetic_curve):
    inv = ts.inversion_flag(synthetic_curve)
    assert inv.sum() == 2 and inv.iloc[3] and inv.iloc[4]


def test_roll_yield_sign(synthetic_curve):
    ry = ts.roll_yield(synthetic_curve)
    assert ry["roll_front_ann"].iloc[0] > 0    # contango -> positive carry
    assert ry["roll_front_ann"].iloc[3] < 0    # backwardation -> negative


def test_build_curve_point_in_time():
    """A contract must vanish from the curve on its own expiry date."""
    expiry = pd.Timestamp("2020-06-17")
    idx = pd.DatetimeIndex(["2020-06-15", "2020-06-16", "2020-06-17"])
    long = pd.DataFrame(
        {"settle": [30.0, 29.0, 28.0], "volume": 1.0, "open_interest": 1.0,
         "expiry": expiry},
        index=idx,
    )
    nxt = long.copy()
    nxt["settle"] = [31.0, 30.5, 30.0]
    nxt["expiry"] = pd.Timestamp("2020-07-22")
    vix = pd.Series(28.0, index=idx)
    curve = ts.build_curve(pd.concat([long, nxt]).sort_index(), vix)
    assert curve.loc["2020-06-16", "vx1"] == 29.0
    # on expiry day the June contract is gone; July becomes the front
    assert curve.loc["2020-06-17", "vx1"] == 30.0
    assert curve.loc["2020-06-17", "expiry1"] == pd.Timestamp("2020-07-22")


class TestVXExpiryCalendar:
    def test_known_expiries(self):
        # Standard months: Wednesday 30d before next month's 3rd Friday
        assert cal.vx_expiry(2020, 3) == pd.Timestamp("2020-03-18")
        assert cal.vx_expiry(2018, 2) == pd.Timestamp("2018-02-14")
        assert cal.vx_expiry(2024, 8) == pd.Timestamp("2024-08-21")
        # holiday-shifted SPX Fridays (verified against CBOE contract files):
        assert cal.vx_expiry(2025, 3) == pd.Timestamp("2025-03-18")  # Good Friday Apr 18
        assert cal.vx_expiry(2026, 5) == pd.Timestamp("2026-05-19")  # Juneteenth Jun 19

    def test_expiry_is_business_day(self):
        for y in range(2007, 2026):
            for m in range(1, 13):
                e = cal.vx_expiry(y, m)
                assert e.weekday() < 5

    def test_expiries_in_range(self):
        es = cal.vx_expiries("2020-01-01", "2020-12-31")
        assert len(es) == 12
        assert es.is_monotonic_increasing

    def test_roll_date_before_expiry(self):
        e = cal.vx_expiry(2020, 3)
        r = cal.roll_date(e, days_before=4)
        assert r < e
        assert 3 <= (e - r).days <= 7  # 4 business days = 4-7 calendar days
