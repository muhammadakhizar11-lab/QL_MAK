"""RV estimators must recover known GBM sigma, be trailing-only, and rank
correctly on efficiency. Tolerances are looser for close-to-close (least
efficient) and tighter for Yang-Zhang, per the spec."""
import numpy as np
import pandas as pd
import pytest

from vrp_lab.volatility import realized as rv

SIGMA = 0.20
WINDOW = 252  # use a long window so estimator noise, not window noise, dominates


def _mean_estimate(est: pd.Series) -> float:
    return float(est.dropna().mean())


class TestRecovery:
    def test_close_to_close_recovers_sigma(self, gbm_20):
        est = rv.close_to_close(gbm_20["close"], window=WINDOW)
        assert _mean_estimate(est) == pytest.approx(SIGMA, rel=0.05)

    def test_parkinson_recovers_sigma(self, gbm_20):
        est = rv.parkinson(gbm_20, window=WINDOW)
        # biased slightly low by discrete intraday sampling — allow 8% low side
        assert _mean_estimate(est) == pytest.approx(SIGMA, rel=0.08)

    def test_garman_klass_recovers_sigma(self, gbm_20):
        est = rv.garman_klass(gbm_20, window=WINDOW)
        assert _mean_estimate(est) == pytest.approx(SIGMA, rel=0.08)

    def test_yang_zhang_recovers_sigma(self, gbm_20):
        est = rv.yang_zhang(gbm_20, window=WINDOW)
        assert _mean_estimate(est) == pytest.approx(SIGMA, rel=0.05)

    def test_yang_zhang_handles_overnight_gaps(self, gbm_20_gaps):
        """With 30% of variance overnight, range-only estimators miss vol;
        Yang-Zhang must still see all of it."""
        yz = _mean_estimate(rv.yang_zhang(gbm_20_gaps, window=WINDOW))
        pk = _mean_estimate(rv.parkinson(gbm_20_gaps, window=WINDOW))
        assert yz == pytest.approx(SIGMA, rel=0.05)
        assert pk < yz, "Parkinson should undershoot when variance moves overnight"


class TestEfficiency:
    def test_range_estimators_less_noisy_than_close_to_close(self, gbm_20):
        """Efficiency = lower sampling variance of the rolling estimate at a
        short window, continuous GBM."""
        w = 21
        cc = rv.close_to_close(gbm_20["close"], window=w).dropna().std()
        gk = rv.garman_klass(gbm_20, window=w).dropna().std()
        pk = rv.parkinson(gbm_20, window=w).dropna().std()
        assert gk < cc
        assert pk < cc


class TestHygiene:
    def test_trailing_only(self, gbm_20):
        """Value at t must not change when future rows are removed."""
        full = rv.yang_zhang(gbm_20, window=21)
        cut = rv.yang_zhang(gbm_20.iloc[:1000], window=21)
        t = cut.index[-1]
        assert np.isclose(full.loc[t], cut.loc[t], equal_nan=True)

    def test_nan_until_full_window(self, gbm_20):
        est = rv.close_to_close(gbm_20["close"], window=21)
        assert est.iloc[:21].isna().all()
        assert est.iloc[21:].notna().all()

    def test_missing_columns_raise(self, gbm_20):
        with pytest.raises(ValueError, match="missing columns"):
            rv.parkinson(gbm_20[["close"]])

    def test_dispatch(self, gbm_20):
        for name in rv.ESTIMATORS:
            out = rv.realized_vol(gbm_20, estimator=name, window=63)
            assert out.dropna().gt(0).all()
        with pytest.raises(ValueError, match="unknown estimator"):
            rv.realized_vol(gbm_20, estimator="garch")

    def test_zero_vol_series(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        flat = pd.DataFrame(
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx
        )
        for name in rv.ESTIMATORS:
            est = rv.realized_vol(flat, estimator=name, window=21).dropna()
            assert (est == 0).all(), f"{name} nonzero on a constant price"
