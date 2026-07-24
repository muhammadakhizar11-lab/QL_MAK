"""Trade log, position tracking and cash accounting for futures-style
instruments (daily variation margin: positions carry no value between settles;
all P&L flows through cash).

Daily sequence (enforced by the engine):
  1. ``mark_to_market``  — variation margin on *yesterday's* positions at
     today's settle, plus interest on cash. Positions filled today are not
     yet in the book, so no same-day variation can be credited to them.
  2. ``execute`` fills   — at today's settle plus costs.
  3. ``record_state``    — equity snapshot after costs, margin check.

Margin is a SPAN-style approximation: a fixed fraction of gross notional.
A margin breach doesn't auto-liquidate — it is recorded so tests and stress
reports can assert on it, which is more informative than silently blowing up.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Fill:
    date: pd.Timestamp
    instrument: str
    qty: float          # signed; futures contracts or shares
    price: float        # fill price (settle for futures)
    cost: float         # dollars, >= 0
    note: str = ""


@dataclass
class Ledger:
    """Cash + positions + full trade history for one backtest run."""
    initial_capital: float
    multiplier: float = 1000.0        # $ per point for futures instruments
    margin_frac: float = 0.30

    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    _equity_rows: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_capital)

    def execute(self, fill: Fill) -> None:
        """Apply a fill: adjust position, charge costs. Futures exchange no
        premium at entry — only costs hit cash; P&L accrues via variation."""
        self.cash -= fill.cost
        new_qty = self.positions.get(fill.instrument, 0.0) + fill.qty
        if abs(new_qty) < 1e-12:
            self.positions.pop(fill.instrument, None)
        else:
            self.positions[fill.instrument] = new_qty
        self.fills.append(fill)
        log.debug("fill %s %s qty=%+.2f @ %.3f cost=%.2f",
                  fill.date.date(), fill.instrument, fill.qty, fill.price, fill.cost)

    def mark_to_market(
        self,
        date: pd.Timestamp,
        settle: dict[str, float],
        prev_settle: dict[str, float],
        daily_rate: float = 0.0,
    ) -> tuple[float, float]:
        """Variation margin on current (pre-fill) positions + interest on cash.

        Returns (variation, interest). Raises if an open position has no
        settle today — a silently missing price would corrupt the P&L.
        """
        variation = 0.0
        for inst, qty in self.positions.items():
            if inst not in settle:
                raise KeyError(f"{date.date()}: no settle price for open position {inst}")
            if inst not in prev_settle:
                raise KeyError(f"{date.date()}: no previous settle for {inst}")
            variation += qty * self.multiplier * (settle[inst] - prev_settle[inst])
        interest = self.cash * daily_rate
        self.cash += variation + interest
        return variation, interest

    def record_state(
        self, date: pd.Timestamp, settle: dict[str, float],
        variation: float, interest: float,
    ) -> dict:
        """End-of-day snapshot after fills: equity, notional, margin check."""
        gross_notional = sum(
            abs(q) * self.multiplier * settle[i] for i, q in self.positions.items()
        )
        margin_req = gross_notional * self.margin_frac
        row = {
            "date": date,
            "equity": self.cash,
            "variation": variation,
            "interest": interest,
            "gross_notional": gross_notional,
            "margin_req": margin_req,
            "margin_breach": margin_req > self.cash,
            "n_positions": len(self.positions),
        }
        if row["margin_breach"]:
            log.warning("%s: MARGIN BREACH req=%.0f equity=%.0f",
                        date.date(), margin_req, self.cash)
        self._equity_rows.append(row)
        return row

    @property
    def equity(self) -> float:
        return self.cash

    def equity_curve(self) -> pd.DataFrame:
        if not self._equity_rows:
            return pd.DataFrame()
        return pd.DataFrame(self._equity_rows).set_index("date")

    def trade_log(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame()
        return pd.DataFrame([vars(f) for f in self.fills]).set_index("date")
