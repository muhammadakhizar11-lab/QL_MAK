"""Signal generation: determinism, causality, and regime/roll logic."""
import numpy as np
import pandas as pd
import pytest

from vrp_lab.signals import regime, skew, vix_roll, vrp_zscore


@pytest.fixture()
def vix_series():
    idx = pd.bdate_range("2020-01-01", periods=10)
    return pd.Series([12, 14, 16, 20, 24, 26, 40, 24, 14, 10.0], index=idx)


class TestRegime:
    def test_vix_buckets(self, vix_series):
        r = regime.vix_regime(vix_series)
        assert r.iloc[0] == regime.Regime.LOW          # 12
        assert r.iloc[3] == regime.Regime.NORMAL       # 20
        assert r.iloc[6] == regime.Regime.CRISIS       # 40

    def test_acceleration_flag_fires_on_spike(self):
        idx = pd.bdate_range("2020-01-01", periods=120)
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0, 0.005, 120), index=idx)
        r.iloc[-3:] = [0.04, -0.05, 0.045]  # sudden violence
        flag = regime.accel_flag(r, threshold=1.5)
        assert flag.iloc[-1]
        assert not flag.iloc[80]

    def test_exposure_multiplier_zero_in_crisis(self, vix_series):
        rets = pd.Series(0.0, index=vix_series.index)
        mult = regime.exposure_multiplier(vix_series, rets)
        assert mult.iloc[6] == 0.0   # VIX 40
        assert mult.iloc[0] == 1.0   # VIX 12, calm


class TestRollSignal:
    def _curve(self):
        idx = pd.bdate_range("2020-01-01", periods=8)
        return pd.DataFrame({
            "vix": [15, 15, 15, 15, 32, 15, 15, 15.0],
            "vx1": [16.0, 15.5, 16.2, 16.3, 30.0, 14.0, 16.5, 16.5],
            "vx2": [17.0] * 8,
            "dte1": [20] * 8, "dte2": [48] * 8,
        }, index=idx)

    def test_entry_exit_logic(self):
        curve = self._curve()
        gate = pd.Series(1.0, index=curve.index)
        sig = vix_roll.roll_signal(curve, gate, contango_entry_threshold=0.05)
        assert sig["target"].iloc[0] == -1.0          # slope 6.7% > 5%
        assert sig["reason"].iloc[4] == "circuit_breaker"   # VIX 32
        assert sig["target"].iloc[4] == 0.0
        # day 5: slope < 0 but we are flat, so it's "no_contango" not an exit
        assert sig["target"].iloc[5] == 0.0

    def test_holds_through_shallow_contango(self):
        """Once in, stay in while contango >= 0 even below entry threshold."""
        curve = self._curve()
        gate = pd.Series(1.0, index=curve.index)
        sig = vix_roll.roll_signal(curve, gate)
        # day 1 slope = 3.3% < 5% entry, but day 0 entered -> still short
        assert sig["target"].iloc[1] == -1.0

    def test_gate_blocks_entry(self):
        curve = self._curve()
        gate = pd.Series(0.0, index=curve.index)
        sig = vix_roll.roll_signal(curve, gate)
        assert (sig["target"] == 0).all()

    def test_deterministic(self):
        curve = self._curve()
        gate = pd.Series(1.0, index=curve.index)
        a = vix_roll.roll_signal(curve, gate)
        b = vix_roll.roll_signal(curve, gate)
        pd.testing.assert_frame_equal(a, b)


class TestVRPCondition:
    def test_on_when_rich(self):
        idx = pd.bdate_range("2018-01-01", periods=600)
        rets = pd.Series(0.01, index=idx)  # constant RV
        iv = pd.Series(0.20, index=idx)
        iv.iloc[-1] = 0.60  # implied triples on the last day
        out = vrp_zscore.vrp_condition(iv, rets)
        assert out["vrp_on"].iloc[-1] == 1.0
        assert out["vrp_on"].iloc[-2] == 0.0  # nothing special the day before


class TestSkew:
    def test_risk_reversal(self):
        chain = pd.DataFrame({
            "kind": ["put", "put", "call", "call"],
            "delta": [-0.10, -0.40, 0.10, 0.40],
            "iv": [0.32, 0.24, 0.18, 0.20],
        })
        rr = skew.risk_reversal_25d(chain)
        # put wing at |0.25| -> 0.28; call wing -> 0.19; rr = 0.09
        assert rr == pytest.approx(0.28 - 0.19)

    def test_skew_states(self):
        idx = pd.bdate_range("2018-01-01", periods=300)
        s = pd.Series(np.linspace(0.01, 0.10, 300), index=idx)  # steepening
        state = skew.skew_signal(s, window=252)
        assert state.iloc[-1] == "steep"
        assert state.iloc[100] == "warmup"
