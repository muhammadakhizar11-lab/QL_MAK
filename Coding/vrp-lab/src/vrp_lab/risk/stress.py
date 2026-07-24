"""Historical stress replay: how did the strategy live through the events
that kill short-vol books?

Events (from config/settings.yaml):
  * Feb 2018 "Volmageddon" — VIX doubled in a day; XIV (short-vol ETN) was
    terminated at a -96% loss.
  * Mar 2020 COVID — VIX printed 82; curve deeply inverted for weeks.
  * Aug 2024 JPY carry unwind — VIX spiked to ~65 intraday, fastest curve
    inversion since 2018.

This module doesn't re-simulate; it *slices an existing backtest* over each
window and reports P&L, drawdown, margin status, and whether the exposure was
actually cut. The honest question is stated in the output: did the regime
filter and vol targeting prevent ruin, or did we just get lucky?
"""
from __future__ import annotations

import pandas as pd

from ..config import get_logger, load_config
from .metrics import max_drawdown

log = get_logger(__name__)


def stress_report(
    equity_curve: pd.DataFrame,
    events: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Slice the backtest over each stress window.

    Parameters
    ----------
    equity_curve : engine output (needs columns equity, gross_notional,
        margin_breach; index = dates).
    events : {name: {start, end, label}}; defaults to config stress_events.

    Returns one row per event: pnl, pnl_pct, max_dd_pct, margin_breaches,
    max_gross_notional, exposure_cut (did gross notional hit 0 inside the
    window), and days_in_window (0 rows => backtest doesn't cover the event).
    """
    events = events or load_config()["stress_events"]
    rows = []
    for name, ev in events.items():
        window = equity_curve.loc[str(ev["start"]): str(ev["end"])]
        if window.empty:
            rows.append({"event": name, "label": ev.get("label", name),
                         "days_in_window": 0})
            continue
        eq = window["equity"]
        start_eq = float(eq.iloc[0])
        pnl = float(eq.iloc[-1] - start_eq)
        dd = max_drawdown(eq)["mdd"] if len(eq) > 1 else 0.0
        rows.append({
            "event": name,
            "label": ev.get("label", name),
            "days_in_window": len(window),
            "pnl": pnl,
            "pnl_pct": pnl / start_eq,
            "max_dd_pct": float(dd),
            "margin_breaches": int(window["margin_breach"].sum()),
            "max_gross_notional": float(window["gross_notional"].max()),
            "exposure_cut": bool((window["gross_notional"] == 0).any()),
        })
        log.info("stress %-12s pnl=%+.1f%%  maxDD=%.1f%%  breaches=%d",
                 name, 100 * pnl / start_eq, 100 * dd,
                 int(window["margin_breach"].sum()))
    return pd.DataFrame(rows).set_index("event")


def survived(report: pd.DataFrame, ruin_dd: float = -0.5) -> bool:
    """Ruin check: no margin breaches and no event drawdown beyond ruin_dd."""
    covered = report[report["days_in_window"] > 0]
    if covered.empty:
        return True
    ok_margin = (covered["margin_breaches"] == 0).all()
    ok_dd = (covered["max_dd_pct"] > ruin_dd).all()
    return bool(ok_margin and ok_dd)
