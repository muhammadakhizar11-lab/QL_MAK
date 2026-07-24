"""Defensive data validation: schema, monotonicity, staleness, gaps.

Philosophy (QL_MAK): fail loudly on structural problems (wrong columns,
unsorted or duplicated dates, future timestamps), log WARNINGs on quality
problems (gaps, stale runs), and never silently repair beyond a bounded
forward-fill.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import get_logger

log = get_logger(__name__)


class ValidationError(ValueError):
    """Structural data problem that must stop the pipeline."""


@dataclass
class ValidationReport:
    """Outcome of validate_frame: hard failures raise, soft issues collect here."""
    name: str
    rows: int
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("%s: %s", self.name, msg)


def validate_frame(
    df: pd.DataFrame,
    name: str,
    required_cols: list[str] | None = None,
    as_of: pd.Timestamp | None = None,
    staleness_max_repeats: int = 10,
) -> ValidationReport:
    """Validate a time-indexed frame. Raises ValidationError on structural
    problems; returns a ValidationReport listing soft quality warnings.

    Checks
    ------
    hard:  non-empty, DatetimeIndex, sorted, no duplicate dates,
           required columns present and numeric, no rows after ``as_of``
           (point-in-time guard — a row dated in the future relative to the
           evaluation date is look-ahead by construction).
    soft:  runs of identical values longer than ``staleness_max_repeats``
           (stale feed), NaN share per column.
    """
    if df.empty:
        raise ValidationError(f"{name}: empty frame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValidationError(f"{name}: index is {type(df.index).__name__}, want DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValidationError(f"{name}: index not sorted ascending")
    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()[:3]
        raise ValidationError(f"{name}: duplicate dates, e.g. {list(dupes)}")
    for col in required_cols or []:
        if col not in df.columns:
            raise ValidationError(f"{name}: missing required column '{col}'")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValidationError(f"{name}: column '{col}' is not numeric")
    if as_of is not None and df.index.max() > pd.Timestamp(as_of):
        raise ValidationError(
            f"{name}: contains rows after as_of={pd.Timestamp(as_of).date()} "
            f"(latest={df.index.max().date()}) — look-ahead"
        )

    report = ValidationReport(name=name, rows=len(df))
    for col in required_cols or df.select_dtypes("number").columns:
        s = df[col]
        nan_share = s.isna().mean()
        if nan_share > 0:
            report.warn(f"column '{col}' has {nan_share:.1%} NaNs")
        run = (s != s.shift()).cumsum()
        longest = int(s.groupby(run).transform("size").max())
        if longest >= staleness_max_repeats:
            report.warn(f"column '{col}' has a {longest}-day run of identical values (stale?)")
    return report


def bounded_ffill(s: pd.Series, max_gap_days: int = 5, name: str | None = None) -> pd.Series:
    """Forward-fill at most ``max_gap_days`` consecutive missing values.

    Gaps longer than the bound stay NaN (and are logged) — silently bridging
    a two-week data outage would fabricate a volatility regime.
    """
    filled = s.ffill(limit=max_gap_days)
    if filled.isna().any():
        log.warning("%s: %d NaNs remain after bounded ffill(limit=%d)",
                    name or s.name or "series", int(filled.isna().sum()), max_gap_days)
    return filled


def align_business_days(
    df: pd.DataFrame, start: str | None = None, end: str | None = None,
    max_gap_days: int = 5,
) -> pd.DataFrame:
    """Reindex to business days over [start, end] with bounded forward-fill."""
    start = start or df.index.min()
    end = end or df.index.max()
    idx = pd.bdate_range(start, end)
    out = df.reindex(idx)
    for col in out.columns:
        out[col] = bounded_ffill(out[col], max_gap_days=max_gap_days, name=str(col))
    out.index.name = "date"
    return out
