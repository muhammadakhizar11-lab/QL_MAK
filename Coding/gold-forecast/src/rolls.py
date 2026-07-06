"""
rolls.py — handle artificial jumps from continuous front-month futures rolls.

THE HONEST PROBLEM
------------------
`GC=F` (and `SI=F`, `HG=F`, ...) on Yahoo are *continuous front-month* tickers:
Yahoo stitches successive contract months together but does NOT back-adjust for
the price gap between the expiring and the new front contract. When the series
rolls, you get a jump in the price *level* that is not a real market move — it's
the basis between two contracts. Left untreated, that fake jump pollutes momentum,
volatility, and (worst of all) the forward-return LABEL.

True back-adjustment (Panama / ratio splice) needs the individual contract series
so you can compute the exact roll-day basis. Yahoo does not give us per-contract
histories for free. **This is a genuine free-data limitation** and is called out
in the write-up. What we CAN do defensibly with a single continuous series:

  1. Work in LOG RETURNS everywhere (the brief's stated minimum).
  2. DETECT roll candidates: return spikes that are (a) statistically extreme and
     (b) clustered near month boundaries, where COMEX metals contracts roll.
  3. NEUTRALIZE the detected roll-day return (default): set that day's return to
     NaN so it never enters a feature window or a label. This is conservative and
     avoids fabricating price history we don't actually have.

  4. (Optional) RATIO BACK-ADJUST: only meaningful if you supply a real roll
     schedule + the pre/post contract prices. Provided for completeness; not the
     default, precisely because inventing the basis from one series is not honest.

Detection is a HEURISTIC and will have false positives/negatives. It flags days
for inspection; it does not claim to perfectly reconstruct the roll calendar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def _near_month_boundary(idx: pd.DatetimeIndex, window: int) -> pd.Series:
    """True for trading days within `window` rows of a calendar-month change."""
    months = idx.to_period("M")
    is_change = np.empty(len(idx), dtype=bool)
    is_change[0] = False
    is_change[1:] = months[1:] != months[:-1]
    change_pos = np.flatnonzero(is_change)
    near = np.zeros(len(idx), dtype=bool)
    for p in change_pos:
        lo, hi = max(0, p - window), min(len(idx), p + window + 1)
        near[lo:hi] = True
    return pd.Series(near, index=idx)


def detect_roll_candidates(
    close: pd.Series,
    jump_z: float = 5.0,
    jump_hard: float = 0.08,
    month_boundary_window: int = 5,
) -> pd.DatetimeIndex:
    """
    Flag likely roll days. A day is a candidate if its log return is extreme by
    z-score AND it sits near a month boundary, OR it exceeds a hard magnitude near
    a month boundary. The month-boundary condition is what separates a *roll* from
    a genuine large market move (which can happen any day).
    """
    r = log_returns(close)
    z = (r - r.mean()) / r.std(ddof=0)
    near = _near_month_boundary(close.index, month_boundary_window)
    cand = ((z.abs() > jump_z) | (r.abs() > jump_hard)) & near
    return close.index[cand.fillna(False).to_numpy()]


def neutralize_roll_returns(close: pd.Series, roll_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Return a LOG-RETURN series with roll-day returns set to NaN. This is the
    default, honest treatment: the fake basis jump is removed from the return
    stream so it cannot leak into features or the label, without fabricating a
    replacement price path. Downstream code treats NaN returns as "no observation".
    """
    r = log_returns(close)
    r.loc[r.index.isin(roll_dates)] = np.nan
    return r


def ratio_backadjust(close: pd.Series, roll_schedule: pd.DataFrame) -> pd.Series:
    """
    OPTIONAL. Panama/ratio back-adjustment given a REAL roll schedule.

    roll_schedule: DataFrame indexed by roll date with columns
        ['price_old', 'price_new']  (front contract close pre-roll, new contract
        close on the same day). ratio = price_new / price_old; all prices strictly
        before the roll date are multiplied by the cumulative product of ratios.

    Only use this if you have sourced the per-contract prices (e.g. CME direct or
    paid data). Do NOT feed it guesses — that would manufacture history.
    """
    adj = close.copy().astype(float)
    for roll_date in sorted(roll_schedule.index, reverse=True):
        row = roll_schedule.loc[roll_date]
        ratio = float(row["price_new"]) / float(row["price_old"])
        adj.loc[adj.index < roll_date] *= ratio
    return adj


def clean_returns(
    close: pd.Series,
    method: str = "neutralize",
    jump_z: float = 5.0,
    jump_hard: float = 0.08,
    month_boundary_window: int = 5,
    roll_schedule: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """
    Entry point. Returns (clean_log_returns, detected_roll_dates).

    method="neutralize"  -> detect + NaN out roll-day returns (default, honest).
    method="ratio_backadjust" -> requires roll_schedule; returns adjusted returns.
    """
    roll_dates = detect_roll_candidates(close, jump_z, jump_hard, month_boundary_window)
    if method == "ratio_backadjust":
        if roll_schedule is None:
            raise ValueError(
                "ratio_backadjust requires a real roll_schedule with per-contract "
                "prices. With free single-series data you cannot honestly build one; "
                "use method='neutralize' instead."
            )
        adj = ratio_backadjust(close, roll_schedule)
        return log_returns(adj), roll_dates
    return neutralize_roll_returns(close, roll_dates), roll_dates
