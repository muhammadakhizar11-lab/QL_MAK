"""Greeks decomposition must explain actual option P&L (small residual),
and the delta-hedged simulator's accounting must be internally consistent."""
import numpy as np
import pandas as pd
import pytest

from vrp_lab.risk import pnl_decomp
from vrp_lab.strategies.delta_hedged import DeltaHedgedConfig, run_delta_hedged
from vrp_lab.strategies.tail_overlay import TailOverlayConfig, run_tail_overlay


@pytest.fixture(scope="module")
def calm_market():
    """Two years of mild GBM (12% realized) with IV fixed at 20% — a fat,
    steady VRP: the short-vol book should collect it."""
    idx = pd.bdate_range("2021-01-04", periods=504)
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0, 0.12 / np.sqrt(252), len(idx))
    spy = pd.Series(400 * np.exp(np.cumsum(rets)), index=idx)
    iv = pd.Series(0.20, index=idx)
    return spy, iv


@pytest.fixture(scope="module")
def hedged_run(calm_market):
    spy, iv = calm_market
    return run_delta_hedged(spy, iv, rf_annual=0.0, cfg=DeltaHedgedConfig())


class TestDecomposition:
    def test_components_sum_to_actual(self, hedged_run):
        history, _ = hedged_run
        d = pnl_decomp.decompose(history[history["in_position"]])
        gross = d["option_pnl"].abs().sum()
        assert gross > 0
        # Taylor residual (incl. discrete hedging error) small vs gross P&L
        assert d["residual"].abs().sum() / gross < 0.15

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="missing columns"):
            pnl_decomp.decompose(pd.DataFrame({"option_pnl": [1.0]}))

    def test_short_vol_signs(self, hedged_run):
        """Short options: theta collected (+), gamma paid (-), and with
        IV > subsequent RV the theta must dominate."""
        history, _ = hedged_run
        d = pnl_decomp.decompose(history[history["in_position"]])
        s = pnl_decomp.attribution_summary(d)
        assert s["theta_pnl"] > 0, "short position should collect decay"
        assert s["gamma_pnl"] < 0, "short gamma pays for realized moves"
        assert s["theta_plus_gamma"] > 0, "IV 20 vs RV 12 must net positive"

    def test_vega_zero_when_iv_constant(self, hedged_run):
        history, _ = hedged_run
        d = pnl_decomp.decompose(history)
        assert d["vega_pnl"].abs().sum() == pytest.approx(0.0)


class TestDeltaHedgedSimulator:
    def test_profitable_when_iv_rich(self, hedged_run):
        history, trades = hedged_run
        assert not trades.empty
        assert history["equity"].iloc[-1] > 1_000_000

    def test_hedge_keeps_delta_flat(self, hedged_run):
        """The hedge at t flattens against the position delta marked at t —
        which is recorded as pos_delta on row t+1. So for consecutive
        in-position days, hedge_shares(t) + pos_delta(t+1) ~ 0 up to the
        1-share trade granularity."""
        history, _ = hedged_run
        h = history[history["in_position"] & history["in_position"].shift(-1).fillna(False)]
        book = history["pos_delta"].shift(-1).loc[h.index] + h["hedge_shares"]
        assert book.abs().max() <= 1.0 + 1e-9

    def test_stop_loss_fires_on_crash(self):
        """A -25% two-day crash right after entry must trigger the premium
        stop, not ride to expiry."""
        idx = pd.bdate_range("2021-01-04", periods=60)
        spy = pd.Series(400.0, index=idx)
        spy.iloc[10] = 340.0
        spy.iloc[11:] = 300.0
        iv = pd.Series(0.20, index=idx)
        iv.iloc[10:] = 0.60  # vol doubles as it crashes
        _, trades = run_delta_hedged(spy, iv, rf_annual=0.0)
        closes = trades[trades["action"] == "close"]
        assert (closes["reason"] == "stop_loss").any()

    def test_gate_blocks_entries(self, calm_market):
        spy, iv = calm_market
        gate = pd.Series(0.0, index=spy.index)
        history, trades = run_delta_hedged(spy, iv, gate=gate)
        assert trades.empty
        assert not history["in_position"].any()

    def test_exit_by_dte(self, calm_market):
        """With no stops in a calm tape, positions exit at dte_exit not expiry."""
        spy, iv = calm_market
        history, trades = run_delta_hedged(spy, iv, rf_annual=0.0)
        closes = trades[trades["action"] == "close"]
        assert (closes["reason"] == "dte_exit").all()


class TestTailOverlay:
    def test_overlay_costs_in_calm_tape(self, calm_market):
        """No crash -> the hedge is pure drag (bounded, not catastrophic)."""
        spy, iv = calm_market
        eq = pd.Series(1_000_000.0, index=spy.index)
        out = run_tail_overlay(spy, iv, eq, cfg=TailOverlayConfig())
        total = out["overlay_pnl"].sum()
        years = len(spy) / 252
        assert total < 0
        assert abs(total) / 1_000_000 / years < 0.15  # <15%/yr drag sanity bound

    def test_overlay_pays_in_crash(self):
        """-30% crash inside the hedge window -> the put spread pays off."""
        idx = pd.bdate_range("2021-01-04", periods=40)
        spy = pd.Series(400.0, index=idx)
        spy.iloc[15:] = 280.0
        iv = pd.Series(0.18, index=idx)
        iv.iloc[15:] = 0.70
        eq = pd.Series(1_000_000.0, index=idx)
        out = run_tail_overlay(spy, iv, eq)
        assert out["overlay_pnl"].sum() > 0
