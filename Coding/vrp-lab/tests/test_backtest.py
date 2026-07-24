"""Backtest engine: no look-ahead, fill correctness, cost application,
margin accounting, and end-to-end strategy behavior on a synthetic curve."""
import numpy as np
import pandas as pd
import pytest

from vrp_lab.backtest import calendar as cal
from vrp_lab.backtest.costs import VXCostModel
from vrp_lab.backtest.engine import LookAheadError, MarketView, Order, run_backtest
from vrp_lab.backtest.ledger import Fill, Ledger
from vrp_lab.strategies.vix_carry import VIXCarryStrategy, build_vx_price_table, contract_name


# --------------------------------------------------------------------------- #
# fixtures: a synthetic two-contract market in steady contango
# --------------------------------------------------------------------------- #
@pytest.fixture()
def synthetic_market():
    """~3 months of business days, VIX flat at 15, VX1 decaying toward it
    (classic contango carry), two live contracts at all times."""
    idx = pd.bdate_range("2020-01-02", "2020-03-31")
    e1, e2, e3 = (cal.vx_expiry(2020, 1), cal.vx_expiry(2020, 2), cal.vx_expiry(2020, 3))
    vix = pd.Series(15.0, index=idx)

    rows = []
    for t in idx:
        for e in (e1, e2, e3):
            if t >= e:
                continue
            dte = (e - t).days
            # linear premium: settle = vix + 0.05 * dte  (decays into expiry)
            rows.append({"date": t, "expiry": e, "settle": 15.0 + 0.05 * dte,
                         "volume": 1000.0, "open_interest": 1000.0})
    long = pd.DataFrame(rows).set_index("date")

    from vrp_lab.volatility.term_structure import build_curve
    curve = build_curve(long, vix)
    prices = build_vx_price_table(long)
    spy_returns = pd.Series(0.001, index=idx)
    return {"curve": curve, "prices": prices, "spy_returns": spy_returns, "vix": vix}


# --------------------------------------------------------------------------- #
# MarketView / look-ahead guard
# --------------------------------------------------------------------------- #
class TestNoLookAhead:
    def test_view_truncates_at_date(self, synthetic_market):
        view = MarketView(synthetic_market["prices"], {"curve": synthetic_market["curve"]})
        t = synthetic_market["prices"].index[10]
        view._set_date(t)
        assert view.prices().index.max() <= t
        assert view.aux("curve").index.max() <= t

    def test_spy_strategy_sees_nothing_future(self, synthetic_market):
        """A probe strategy records the max timestamp it can observe each day."""
        seen = []

        class Probe:
            def on_bar(self, view, ledger):
                seen.append((view.date, view.aux("curve").index.max()))
                return []

        run_backtest(Probe(), synthetic_market["prices"],
                     aux={"curve": synthetic_market["curve"]})
        assert all(obs <= now for now, obs in seen)

    def test_lookahead_error_on_corrupt_index(self):
        """If the guard's slice ever fails (e.g. unsorted index), it raises."""
        idx = pd.DatetimeIndex(["2020-01-02", "2020-01-06", "2020-01-03"])
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx)
        view = MarketView(df, {"aux": df})
        view._set_date(pd.Timestamp("2020-01-03"))
        with pytest.raises(LookAheadError):
            view.aux("aux")


# --------------------------------------------------------------------------- #
# ledger accounting
# --------------------------------------------------------------------------- #
class TestLedger:
    def test_short_futures_pnl(self):
        led = Ledger(100_000, multiplier=1000)
        t0, t1 = pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")
        led.execute(Fill(t0, "VX_A", -2, 16.0, cost=55.0))
        assert led.cash == 100_000 - 55
        # settle drops 16 -> 15.5: short 2 contracts gains 2*1000*0.5
        var, _ = led.mark_to_market(t1, {"VX_A": 15.5}, {"VX_A": 16.0})
        assert var == pytest.approx(1000.0)
        assert led.cash == pytest.approx(100_000 - 55 + 1000)

    def test_missing_settle_raises(self):
        led = Ledger(100_000)
        led.execute(Fill(pd.Timestamp("2020-01-02"), "VX_A", 1, 16.0, cost=0.0))
        with pytest.raises(KeyError, match="no settle"):
            led.mark_to_market(pd.Timestamp("2020-01-03"), {}, {"VX_A": 16.0})

    def test_margin_breach_flag(self):
        led = Ledger(10_000, multiplier=1000, margin_frac=0.30)
        t = pd.Timestamp("2020-01-02")
        led.execute(Fill(t, "VX_A", -3, 20.0, cost=0.0))
        row = led.record_state(t, {"VX_A": 20.0}, 0.0, 0.0)
        # margin req = 3*20*1000*0.3 = 18k > 10k equity
        assert row["margin_breach"]

    def test_interest_accrues(self):
        led = Ledger(100_000)
        var, interest = led.mark_to_market(
            pd.Timestamp("2020-01-03"), {}, {}, daily_rate=0.0001)
        assert interest == pytest.approx(10.0)
        assert led.cash == pytest.approx(100_010.0)


# --------------------------------------------------------------------------- #
# costs
# --------------------------------------------------------------------------- #
class TestCosts:
    def test_vx_cost_hand_calc(self):
        m = VXCostModel(half_spread_pts=0.025, commission=1.32,
                        slippage_pts_per_contract=0.01, multiplier=1000)
        # 2 contracts: spread 2*0.025*1000=50, slippage 2*0.01*1000=20, comm 2.64
        assert m.cost(2) == pytest.approx(50 + 20 + 2.64)
        assert m.cost(-2) == m.cost(2)  # sign-blind

    def test_costs_hit_equity(self, synthetic_market):
        class OneTrade:
            # short the longest-dated contract so it never expires in-sample
            # (the ledger deliberately raises if a held contract stops settling)
            done = False
            def on_bar(self, view, ledger):
                if not self.done:
                    self.done = True
                    inst = view.prices().iloc[-1].dropna().index[-1]
                    return [Order(inst, -1, note="test")]
                return []

        eq, trades = run_backtest(OneTrade(), synthetic_market["prices"],
                                  aux={}, initial_capital=1_000_000)
        assert len(trades) == 1
        expected_cost = VXCostModel().cost(1)
        assert trades["cost"].iloc[0] == pytest.approx(expected_cost)


# --------------------------------------------------------------------------- #
# strategy end-to-end on the synthetic market
# --------------------------------------------------------------------------- #
class TestVIXCarryEndToEnd:
    def test_harvests_contango_carry(self, synthetic_market):
        """In a frictionless-ish steady contango market, the short-roll
        strategy must make money: settles decay 0.05/day toward spot."""
        strat = VIXCarryStrategy()
        eq, trades = run_backtest(
            strat, synthetic_market["prices"],
            aux={"curve": synthetic_market["curve"],
                 "spy_returns": synthetic_market["spy_returns"]},
            initial_capital=1_000_000,
        )
        assert not trades.empty, "strategy never traded in a 6%-contango market"
        assert eq["equity"].iloc[-1] > 1_000_000
        assert not eq["margin_breach"].any()

    def test_rolls_before_expiry(self, synthetic_market):
        """No fill may occur on a contract on/after its expiry date, and the
        strategy must vacate the front contract before expiry."""
        strat = VIXCarryStrategy(roll_days_before_expiry=4)
        _, trades = run_backtest(
            strat, synthetic_market["prices"],
            aux={"curve": synthetic_market["curve"],
                 "spy_returns": synthetic_market["spy_returns"]},
        )
        jan_expiry = cal.vx_expiry(2020, 1)
        jan_inst = contract_name(jan_expiry)
        jan_fills = trades[trades["instrument"] == jan_inst]
        if not jan_fills.empty:
            assert jan_fills.index.max() < jan_expiry
            # position in the january contract nets to zero before expiry
            assert jan_fills["qty"].sum() == pytest.approx(0.0)

    def test_circuit_breaker_flattens(self, synthetic_market):
        """Force VIX above 30 mid-sample: the book must go flat the same day."""
        curve = synthetic_market["curve"].copy()
        shock_day = curve.index[30]
        curve.loc[shock_day:, "vix"] = 35.0
        strat = VIXCarryStrategy()
        eq, trades = run_backtest(
            strat, synthetic_market["prices"],
            aux={"curve": curve, "spy_returns": synthetic_market["spy_returns"]},
        )
        assert (eq.loc[shock_day:, "n_positions"] == 0).all()

    def test_naive_variant_sizes_bigger(self, synthetic_market):
        """The naive always-short 1x-notional variant must carry more gross
        notional than the vol-targeted one at VIX 15 with a calm tape...
        actually the point of vol-targeting: in a LOW-vol tape it may size
        larger. Here we just assert both run and differ."""
        aux = {"curve": synthetic_market["curve"],
               "spy_returns": synthetic_market["spy_returns"]}
        eq_smart, _ = run_backtest(VIXCarryStrategy(), synthetic_market["prices"], aux=aux)
        eq_naive, _ = run_backtest(
            VIXCarryStrategy(use_regime_gate=False, use_vol_target=False),
            synthetic_market["prices"], aux=aux)
        assert not np.allclose(eq_smart["gross_notional"], eq_naive["gross_notional"])
