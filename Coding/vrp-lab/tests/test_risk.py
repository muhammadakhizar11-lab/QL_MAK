"""Risk metrics against hand-calculated values; stress report slicing."""
import numpy as np
import pandas as pd
import pytest

from vrp_lab.risk import metrics as m
from vrp_lab.risk import stress
from vrp_lab.strategies import sizing


class TestMetrics:
    def test_sharpe_hand_calc(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        r = pd.Series([0.01, -0.005, 0.02, 0.002], index=idx)
        expected = np.sqrt(252) * r.mean() / r.std(ddof=1)
        assert m.sharpe(r) == pytest.approx(expected)

    def test_sortino_uses_downside_only(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        r = pd.Series([0.01, -0.01, 0.01, -0.02], index=idx)
        downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
        expected = np.sqrt(252) * r.mean() / downside
        assert m.sortino(r) == pytest.approx(expected)
        # all-positive returns -> no downside -> NaN, not inf
        assert np.isnan(m.sortino(pd.Series([0.01, 0.02], index=idx[:2])))

    def test_max_drawdown_hand_calc(self):
        idx = pd.bdate_range("2020-01-01", periods=6)
        eq = pd.Series([100, 120, 90, 95, 130, 125.0], index=idx)
        dd = m.max_drawdown(eq)
        assert dd["mdd"] == pytest.approx(90 / 120 - 1)      # -25%
        assert dd["peak_date"] == idx[1]
        assert dd["trough_date"] == idx[2]
        assert dd["recovery_date"] == idx[4]

    def test_unrecovered_drawdown(self):
        idx = pd.bdate_range("2020-01-01", periods=3)
        eq = pd.Series([100, 80, 70.0], index=idx)
        dd = m.max_drawdown(eq)
        assert dd["recovery_date"] is None and dd["recovery_days"] is None

    def test_cvar_hand_calc(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        r = pd.Series(np.zeros(100), index=idx)
        r.iloc[:5] = [-0.10, -0.08, -0.06, -0.04, -0.02]
        # worst 5% of 100 days = worst 5 days; quantile(0.05) interpolates,
        # so assert against the metric's own definition:
        cutoff = r.quantile(0.05)
        expected = r[r <= cutoff].mean()
        assert m.cvar(r, 0.95) == pytest.approx(expected)
        assert m.cvar(r, 0.95) < 0

    def test_excess_returns_with_rf_series(self):
        idx = pd.bdate_range("2020-01-01", periods=252)
        r = pd.Series(0.001, index=idx)
        rf = pd.Series(2.52, index=idx)  # 2.52% annual -> 0.0001/day
        ex_sharpe = m.sharpe(r, rf_annual=rf)
        # constant returns: std=0 -> nan; use noisy series instead
        rng = np.random.default_rng(0)
        r = pd.Series(0.001 + rng.normal(0, 0.01, 252), index=idx)
        assert m.sharpe(r, rf_annual=rf) < m.sharpe(r)

    def test_summary_has_all_fields(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(5e-4, 0.01, 300), index=idx)
        s = m.summary(r)
        for key in ("sharpe", "sortino", "calmar", "max_drawdown",
                    "cvar_95", "cvar_99", "worst_day", "skew", "kurtosis"):
            assert key in s.index


class TestSizing:
    def test_vol_target_inverts_with_vol(self):
        lo = sizing.vol_target_contracts(1e6, 15.0, daily_vol=0.02)
        hi = sizing.vol_target_contracts(1e6, 15.0, daily_vol=0.08)
        assert lo > hi, "position must SHRINK when instrument vol rises"

    def test_notional_cap_binds(self):
        n = sizing.vol_target_contracts(
            1e6, 15.0, daily_vol=1e-6, max_notional_mult=2.0)
        assert n * 15.0 * 1000 <= 2.0 * 1e6 + 1e-6

    def test_zero_on_bad_inputs(self):
        assert sizing.vol_target_contracts(1e6, 15.0, daily_vol=float("nan")) == 0.0
        assert sizing.vol_target_contracts(1e6, 0.0, daily_vol=0.02) == 0.0
        assert sizing.vol_target_contracts(-5.0, 15.0, daily_vol=0.02) == 0.0

    def test_kelly_capped(self):
        assert sizing.kelly_fraction(0.10, 0.01) == 0.25  # raw 10, capped
        assert sizing.kelly_fraction(-0.10, 0.01) == 0.0  # never short the edge
        assert sizing.kelly_fraction(0.0, 0.0) == 0.0


class TestStress:
    def _fake_curve(self):
        idx = pd.bdate_range("2018-01-02", "2018-03-30")
        eq = pd.Series(1e6, index=idx, dtype=float)
        # -12% hit across early Feb, partial recovery
        crash = idx[(idx >= "2018-02-02") & (idx <= "2018-02-09")]
        eq.loc[crash[0]:] = 1e6 * 0.94
        eq.loc[crash[-1]:] = 1e6 * 0.88
        eq.loc["2018-03-01":] = 1e6 * 0.93
        return pd.DataFrame({
            "equity": eq,
            "gross_notional": np.where(eq < 1e6, 0.0, 5e5),
            "margin_breach": False,
        }, index=idx)

    def test_stress_report_slices_event(self):
        events = {"volmageddon": {"start": "2018-01-26", "end": "2018-02-12",
                                  "label": "Feb 2018"}}
        rep = stress.stress_report(self._fake_curve(), events)
        row = rep.loc["volmageddon"]
        assert row["days_in_window"] > 0
        assert row["pnl_pct"] == pytest.approx(0.88 - 1.0, abs=1e-9)
        assert row["exposure_cut"]

    def test_uncovered_event_reports_zero_days(self):
        events = {"covid": {"start": "2020-02-19", "end": "2020-04-07"}}
        rep = stress.stress_report(self._fake_curve(), events)
        assert rep.loc["covid", "days_in_window"] == 0

    def test_survived(self):
        events = {"volmageddon": {"start": "2018-01-26", "end": "2018-02-12"}}
        rep = stress.stress_report(self._fake_curve(), events)
        assert stress.survived(rep)                    # -12% != ruin
        assert not stress.survived(rep, ruin_dd=-0.05) # unless the bar is 5%
