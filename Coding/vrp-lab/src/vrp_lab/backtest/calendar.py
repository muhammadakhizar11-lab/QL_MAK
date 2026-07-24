"""Trading calendar, VX expiry rules, and roll-date computation.

VX futures final settlement (CBOE rule): the Wednesday that is 30 days prior
to the third Friday of the calendar month *following* the contract month.
If that third Friday is an exchange holiday, SPX expiration moves to the
preceding business day and the VX settlement moves with it (30 days prior to
that day instead). This bites in practice: Apr 2025's third Friday was Good
Friday and Jun 2026's is Juneteenth — hence the custom trading calendar below
(federal rules minus Columbus/Veterans Day, which markets trade through, plus
Good Friday, which they don't).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday,
                                    USFederalHolidayCalendar)
from pandas.tseries.offsets import CustomBusinessDay


class USTradingCalendar(AbstractHolidayCalendar):
    """US equity-market holidays, approximated from the federal calendar."""
    rules = [r for r in USFederalHolidayCalendar.rules
             if r.name not in ("Columbus Day", "Veterans Day")] + [GoodFriday]


_CAL = USTradingCalendar()
US_BDAY = CustomBusinessDay(calendar=_CAL)


def trading_days(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """US business days between start and end inclusive (federal-holiday calendar)."""
    return pd.bdate_range(start, end, freq=US_BDAY)


def third_friday(year: int, month: int) -> pd.Timestamp:
    """Third Friday of the given month (SPX standard expiration Friday)."""
    first = pd.Timestamp(year=year, month=month, day=1)
    # weekday(): Mon=0 ... Fri=4
    offset = (4 - first.weekday()) % 7
    return first + pd.Timedelta(days=offset + 14)


def vx_expiry(year: int, month: int) -> pd.Timestamp:
    """Final settlement date of the VX contract for the given contract month.

    30 calendar days before the next month's SPX expiration Friday — which is
    the third Friday, or the preceding business day when that Friday is a
    holiday (Good Friday / Juneteenth cases). The result is then itself
    adjusted back if it lands on a weekend/holiday.
    """
    nxt = dt.date(year, month, 1) + pd.DateOffset(months=1)
    friday = third_friday(nxt.year, nxt.month)
    fri_holidays = set(_CAL.holidays(friday - pd.Timedelta(days=7), friday))
    while friday.weekday() >= 5 or friday in fri_holidays:
        friday -= pd.Timedelta(days=1)
    expiry = friday - pd.Timedelta(days=30)
    holidays = set(_CAL.holidays(expiry - pd.Timedelta(days=7), expiry))
    while expiry.weekday() >= 5 or expiry in holidays:
        expiry -= pd.Timedelta(days=1)
    return expiry


def vx_expiries(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """All VX final-settlement dates whose expiry falls within [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out: list[pd.Timestamp] = []
    cur = start - pd.DateOffset(months=1)
    while cur <= end + pd.DateOffset(months=1):
        e = vx_expiry(cur.year, cur.month)
        if start <= e <= end:
            out.append(e)
        cur += pd.DateOffset(months=1)
    return pd.DatetimeIndex(sorted(set(out)))


def roll_date(expiry: pd.Timestamp, days_before: int = 4) -> pd.Timestamp:
    """Business day on which we roll out of a contract, `days_before` business
    days ahead of its expiry (spec: roll 3-5 days pre-expiry, never on expiry)."""
    return (expiry - days_before * US_BDAY).normalize()
