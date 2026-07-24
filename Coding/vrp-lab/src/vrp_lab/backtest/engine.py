"""Event-driven backtest engine. One day at a time, no vectorized shortcuts.

Daily loop, in order:
  1. mark-to-market yesterday's positions at today's settle (+ cash interest)
  2. hand the strategy a MarketView truncated at today — the no-look-ahead
     guard lives here: the view *cannot* produce a row stamped after `date`,
     and the engine asserts it on every access in debug runs
  3. execute the strategy's orders at today's settle plus transaction costs
  4. record end-of-day state (equity, notional, margin)

Fills at the daily settle are an explicit simplification (VX settles are
firm prices; we pay the spread via the cost model instead of modelling an
intraday fill). Documented in METHODOLOGY.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ..config import get_logger
from .costs import VXCostModel
from .ledger import Fill, Ledger

log = get_logger(__name__)


class LookAheadError(RuntimeError):
    """A data access requested rows after the current simulation date."""


@dataclass(frozen=True)
class Order:
    instrument: str
    qty: float           # signed contracts
    note: str = ""


class MarketView:
    """Point-in-time window onto the market data.

    Wraps the full price table and auxiliary frames but only ever returns
    rows with index <= the current date. Strategies receive *this*, never the
    raw frames — that is the structural no-look-ahead guarantee; the assert
    is a belt-and-braces runtime check on top.
    """

    def __init__(self, prices: pd.DataFrame, aux: dict[str, pd.DataFrame | pd.Series]):
        self._prices = prices
        self._aux = aux
        self._date: pd.Timestamp | None = None

    def _set_date(self, date: pd.Timestamp) -> None:
        self._date = date

    @property
    def date(self) -> pd.Timestamp:
        assert self._date is not None, "view used outside the event loop"
        return self._date

    def _guard(self, obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        out = obj.loc[: self.date]
        if len(out) and out.index.max() > self.date:
            raise LookAheadError(
                f"data window extends to {out.index.max()} past {self.date}"
            )
        return out

    def prices(self) -> pd.DataFrame:
        """All instrument prices up to and including the current date."""
        return self._guard(self._prices)

    def price(self, instrument: str) -> float:
        """Today's settle for one instrument (NaN if it did not trade)."""
        row = self._prices.loc[self.date] if self.date in self._prices.index else None
        return float(row[instrument]) if row is not None else float("nan")

    def aux(self, name: str) -> pd.DataFrame | pd.Series:
        """Auxiliary series/frame (curve, returns, rates) up to current date."""
        return self._guard(self._aux[name])


class Strategy(Protocol):
    """A strategy sees the point-in-time view + its own positions, returns orders."""

    def on_bar(self, view: MarketView, ledger: Ledger) -> list[Order]: ...


def run_backtest(
    strategy: Strategy,
    prices: pd.DataFrame,
    aux: dict[str, pd.DataFrame | pd.Series] | None = None,
    initial_capital: float = 1_000_000,
    cost_model: VXCostModel | None = None,
    daily_rates: pd.Series | None = None,
    margin_frac: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the event loop over every date in ``prices``.

    Parameters
    ----------
    prices : wide frame, one column per instrument, daily settles. NaN = did
        not trade that day (e.g. contract not yet listed / already expired).
    aux : named auxiliary data exposed to the strategy through the view.
    daily_rates : *daily* interest rate applied to cash (annual/252 upstream).

    Returns
    -------
    (equity_curve, trade_log) — both indexed by date.
    """
    cost_model = cost_model or VXCostModel()
    ledger = Ledger(initial_capital, multiplier=cost_model.multiplier,
                    margin_frac=margin_frac)
    view = MarketView(prices, aux or {})

    prev_settle: dict[str, float] = {}
    for date in prices.index:
        view._set_date(date)
        settle = prices.loc[date].dropna().to_dict()

        # 1. settle yesterday's book
        rate = float(daily_rates.get(date, 0.0)) if daily_rates is not None else 0.0
        variation, interest = ledger.mark_to_market(date, settle, prev_settle, rate)

        # 2. strategy decision on point-in-time data
        orders = strategy.on_bar(view, ledger)

        # 3. execution at settle + costs
        for o in orders:
            if abs(o.qty) < 1e-12:
                continue
            px = settle.get(o.instrument)
            if px is None:
                log.warning("%s: order for %s skipped — no price", date.date(), o.instrument)
                continue
            ledger.execute(Fill(date=date, instrument=o.instrument, qty=o.qty,
                                price=px, cost=cost_model.cost(o.qty), note=o.note))

        # 4. end-of-day snapshot
        ledger.record_state(date, settle, variation, interest)
        prev_settle = settle

    open_pos = dict(ledger.positions)
    if open_pos:
        log.info("backtest end: still open %s (marked, not force-closed)", open_pos)
    return ledger.equity_curve(), ledger.trade_log()
