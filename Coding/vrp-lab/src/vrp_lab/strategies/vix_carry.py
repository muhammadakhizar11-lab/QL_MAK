"""Strategy 1: VX futures roll harvesting (event-driven, engine-compatible).

Short the front VX contract while the curve pays positive carry, gated by
regime and vol-targeted. All decisions use the MarketView (point-in-time by
construction); the strategy keeps only scalar state (holding-day counter).

Contract naming: instruments are ``VX_YYYY-MM-DD`` (expiry-stamped), matching
the columns of the price table built by ``build_vx_price_table``.

Naive-variant flags (for the notebook comparison the spec asks for):
  * ``use_regime_gate=False``  -> ignores Crisis/acceleration gating
  * ``use_vol_target=False``   -> fixed 1x notional instead of vol targeting
Running with both off is the "always-short" anti-pattern — kept only as the
baseline to show *why* the filters exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..backtest.engine import MarketView, Order
from ..backtest.ledger import Ledger
from ..config import get_logger
from ..signals.regime import Regime, vix_regime, vol_acceleration
from .sizing import trailing_daily_vol, vol_target_contracts

log = get_logger(__name__)


def contract_name(expiry: pd.Timestamp) -> str:
    return f"VX_{pd.Timestamp(expiry):%Y-%m-%d}"


def build_vx_price_table(vx_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long CBOE frame into engine format: one column per contract."""
    wide = vx_long.pivot_table(index=vx_long.index, columns="expiry", values="settle")
    wide.columns = [contract_name(c) for c in wide.columns]
    wide.index.name = "date"
    return wide


@dataclass
class VIXCarryStrategy:
    """Short-front-VX carry with regime gate and vol targeting."""

    contango_entry_threshold: float = 0.05
    backwardation_exit_threshold: float = -0.02   # hysteresis: a -0.1% blip is noise,
                                                  # exiting on it sells the spike low
    vix_circuit_breaker: float = 30.0
    max_holding_days: int = 60
    roll_days_before_expiry: int = 4

    low_vol_max: float = 15.0
    crisis_min: float = 25.0
    accel_ratio_max: float = 1.5

    vol_target_annual: float = 0.15
    max_notional_mult: float = 2.0
    multiplier: float = 1000.0

    use_regime_gate: bool = True
    use_vol_target: bool = True
    rebalance_threshold: float = 0.10   # skip resizes smaller than 10% of position

    _holding_days: int = field(default=0, init=False)

    # ------------------------------------------------------------------ #
    def _regime_state(self, view: MarketView, vix_today: float) -> tuple[bool, float]:
        """(entry_ok, hold_mult) from regime + acceleration.

        Asymmetric by design: entries require calm (not Crisis, vol not
        accelerating), but an existing position is only *flattened* by the
        hard exits (circuit breaker, deep backwardation). Flattening at the
        VIX-25 bucket edge systematically covers at spike highs and re-enters
        lower — a measured -30%/yr leak in choppy years (2021). While vol is
        accelerating, held exposure is halved instead.
        """
        if not self.use_regime_gate:
            return True, 1.0
        reg = vix_regime(pd.Series([vix_today]), self.low_vol_max, self.crisis_min).iloc[0]
        accel_fired = False
        spy_ret = view.aux("spy_returns")
        if len(spy_ret) >= 22:
            accel = vol_acceleration(spy_ret).iloc[-1]
            accel_fired = bool(np.isfinite(accel) and accel > self.accel_ratio_max)
        entry_ok = (reg != Regime.CRISIS) and not accel_fired
        hold_mult = 0.5 if accel_fired else 1.0
        return entry_ok, hold_mult

    def _size(self, view: MarketView, ledger: Ledger, price: float) -> float:
        """Whole contracts for a full-exposure position (before sign)."""
        if not self.use_vol_target:
            n = ledger.equity / (price * self.multiplier)  # 1x notional
            return float(np.floor(max(n, 0.0)))
        curve = view.aux("curve")
        dvol = trailing_daily_vol(curve["vx1"].dropna(), window=21)
        n = vol_target_contracts(
            equity=ledger.equity, price=price, daily_vol=dvol,
            vol_target_annual=self.vol_target_annual,
            multiplier=self.multiplier, max_notional_mult=self.max_notional_mult,
        )
        return float(np.floor(n))

    # ------------------------------------------------------------------ #
    def on_bar(self, view: MarketView, ledger: Ledger) -> list[Order]:
        curve = view.aux("curve")
        if view.date not in curve.index:
            return self._flatten(ledger, "no_curve_data")
        row = curve.loc[view.date]
        vix, vx1 = row.get("vix"), row.get("vx1")
        if pd.isna(vix) or pd.isna(vx1):
            return self._flatten(ledger, "missing_prices")

        slope = (vx1 - vix) / vix
        in_position = any(q != 0 for q in ledger.positions.values())

        # -- exits ------------------------------------------------------ #
        if vix > self.vix_circuit_breaker:
            self._holding_days = 0
            return self._flatten(ledger, "circuit_breaker")
        if slope < self.backwardation_exit_threshold and in_position:
            self._holding_days = 0
            return self._flatten(ledger, "backwardation_exit")
        if in_position and self._holding_days >= self.max_holding_days:
            self._holding_days = 0
            return self._flatten(ledger, "max_holding_exit")

        entry_ok, hold_mult = self._regime_state(view, float(vix))

        # -- entry / hold ----------------------------------------------- #
        if not in_position and (not entry_ok or slope <= self.contango_entry_threshold):
            return []
        # pick the contract: front unless inside the roll window
        use_second = row.get("dte1", 99) <= self.roll_days_before_expiry
        tenor = "2" if use_second else "1"
        expiry, price = row.get(f"expiry{tenor}"), row.get(f"vx{tenor}")
        if pd.isna(price) or pd.isna(expiry):
            return self._flatten(ledger, "no_tradeable_contract")
        inst = contract_name(expiry)

        target_qty = -hold_mult * self._size(view, ledger, float(price))
        self._holding_days = self._holding_days + 1 if in_position else 1

        orders = []
        # flatten anything that isn't the target contract (this IS the roll)
        for held, qty in ledger.positions.items():
            if held != inst and qty != 0:
                orders.append(Order(held, -qty, note="roll_out"))
        current = ledger.positions.get(inst, 0.0)
        delta = target_qty - current
        # daily vol-target drift churns spread costs; only resize on real moves
        is_resize = current != 0 and not orders
        min_delta = max(1.0, self.rebalance_threshold * abs(current)) if is_resize else 1.0
        if abs(delta) >= min_delta:
            note = "roll_in" if orders else ("entry" if current == 0 else "resize")
            orders.append(Order(inst, delta, note=note))
        return orders

    def _flatten(self, ledger: Ledger, note: str) -> list[Order]:
        return [Order(inst, -qty, note=note)
                for inst, qty in ledger.positions.items() if qty != 0]
