"""
fetch_fomc_dates.py — build data/raw/fomc_dates.csv from the Fed's own calendar.

features.py's `fomc_day` flag reads data/raw/fomc_dates.csv (one `date` column).
We refuse to hard-code dates (shipping subtly-wrong ones is worse than an
all-zero flag), so this script scrapes them from federalreserve.gov:

  * 2011-2020:  /monetarypolicy/fomchistorical{year}.htm   (h5 per meeting)
  * 2021+:      /monetarypolicy/fomccalendars.htm          (month/date divs)

Two deliberate choices, both about the feature's "known in advance" claim:
  1. We emit the DECISION day — the final day of each meeting, when the 2pm ET
     statement lands. That is the day that matters for gold.
  2. UNSCHEDULED and CANCELLED meetings are excluded. An emergency meeting
     (e.g. 2020-03-15) is not known in advance; flagging it would smuggle
     hindsight into a feature specced as ex-ante.

Uses curl as transport for consistency with data_pull.py.

Run:  python -m scripts.fetch_fomc_dates
"""
from __future__ import annotations

import re
import subprocess
import sys

import pandas as pd

from src.config import RAW, ensure_dirs

BASE = "https://www.federalreserve.gov/monetarypolicy"
HIST_YEARS = range(2010, 2021)          # per-year historical pages
_MONTHS = {m[:3].lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _get(url: str) -> str:
    r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                       capture_output=True, text=True, check=True)
    return r.stdout


def _month_num(token: str) -> int | None:
    return _MONTHS.get(token.strip()[:3].lower())


def _end_date(months: str, days: str, year: int) -> pd.Timestamp | None:
    """Resolve a meeting's month(s)/day(s) text to its FINAL (decision) day.

    months: "January" or "Jan/Feb" (cross-month meetings like Jan 31-Feb 1).
    days:   "28", "27-28", "31-1", possibly with a trailing '*' (SEP footnote).
    """
    days = re.sub(r"[^\d-]", "", days)          # strip '*' and stray notation
    day_parts = [p for p in days.split("-") if p]
    if not day_parts:
        return None
    end_day = int(day_parts[-1])
    month_parts = [p for p in re.split(r"[/-]", months) if p.strip()]
    end_month = _month_num(month_parts[-1])
    if end_month is None:
        return None
    # cross-month December->January meetings would roll the year; none exist in
    # the sample but be correct anyway
    if len(month_parts) > 1 and _month_num(month_parts[0]) == 12 and end_month == 1:
        year += 1
    return pd.Timestamp(year=year, month=end_month, day=end_day)


def parse_historical(html: str, year: int, skipped: list[str]) -> list[pd.Timestamp]:
    """h5 headings like 'January 27-28 Meeting - 2015'; cross-month meetings are
    written 'Jan/Feb 31-1 Meeting - 2017' (slash in the month token)."""
    out = []
    for h5 in re.findall(r"<h5[^>]*>([^<]+)</h5>", html):
        text = h5.strip()
        if "meeting" not in text.lower():
            continue                             # e.g. 'Conference Calls' headers
        if re.search(r"unscheduled|cancell?ed|notation", text, re.I):
            skipped.append(f"{year}: {text}")
            continue
        m = re.match(r"([A-Za-z/]+)\s+([\d-]+)", text)
        if not m:
            skipped.append(f"{year}: UNPARSED {text}")
            continue
        ts = _end_date(m.group(1), m.group(2), year)
        if ts is None:
            skipped.append(f"{year}: UNPARSED {text}")
        else:
            out.append(ts)
    return out


def parse_current(html: str, skipped: list[str]) -> list[pd.Timestamp]:
    """The combined calendar page: '{year} FOMC Meetings' panels, then per-meeting
    month/date div pairs in document order."""
    tokens = re.findall(
        r"(\d{4}) FOMC Meetings"
        r"|fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>"
        r"|fomc-meeting__date[^>]*>([^<]*)",
        html,
    )
    out, year, month = [], None, None
    for y, mon, day in tokens:
        if y:
            year = int(y)
        elif mon:
            month = mon.strip()
        elif year and month:
            if re.search(r"unscheduled|cancell?ed|notation", day, re.I):
                skipped.append(f"{year}: {month} {day.strip()}")
                month = None
                continue
            ts = _end_date(month, day, year)
            if ts is None:
                skipped.append(f"{year}: UNPARSED {month} {day.strip()}")
            else:
                out.append(ts)
            month = None
    return out


def main() -> int:
    ensure_dirs()
    skipped: list[str] = []
    dates: list[pd.Timestamp] = []

    for year in HIST_YEARS:
        html = _get(f"{BASE}/fomchistorical{year}.htm")
        got = parse_historical(html, year, skipped)
        dates += got
        print(f"[fomc] {year}: {len(got)} scheduled meetings")

    html = _get(f"{BASE}/fomccalendars.htm")
    got = parse_current(html, skipped)
    dates += got
    years = sorted({d.year for d in got})
    print(f"[fomc] {years[0]}-{years[-1]} (current page): {len(got)} scheduled meetings")

    if skipped:
        print(f"\nExcluded {len(skipped)} unscheduled/cancelled/unparsed entries:")
        for s in skipped:
            print(f"  - {s}")

    s = pd.Series(sorted(set(dates)), name="date")
    # sanity: the FOMC holds 8 scheduled meetings/year; flag any year outside 7-9
    per_year = s.groupby(s.dt.year).size()
    odd = per_year[(per_year < 7) | (per_year > 9)]
    if not odd.empty:
        print(f"\nWARNING: unusual meeting counts (expected ~8/yr): {odd.to_dict()}")

    out = RAW / "fomc_dates.csv"
    s.dt.date.to_frame().to_csv(out, index=False)
    print(f"\n{len(s)} decision dates ({s.min().date()} -> {s.max().date()}) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
