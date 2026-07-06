"""
validate.py — data-quality checks run BEFORE any modelling.

Checks per price series:
  * business-day gaps         (missing trading days vs a Mon-Fri calendar)
  * stale runs                (>= N identical consecutive closes -> suspicious)
  * return outliers           (|z| or |hard threshold| on daily log returns)
  * non-positive prices       (zero/negative closes -> bad prints)

Nothing is auto-corrected here. The job of this stage is to *surface and record*
problems so decisions about them are explicit downstream. Output is a dict of
findings plus a human-readable markdown report in reports/data_quality/.

Note on the business-day calendar: we use a plain Mon-Fri calendar, so exchange
holidays show up as "gaps". That is intentional and harmless — a holiday and a
genuine missing print look the same to a naive calendar, and treating the union
as "days to inspect" is the conservative choice. Swap in `pandas_market_calendars`
later if you want holiday-aware gap detection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPORTS, RAW, load_config, ensure_dirs


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def business_day_gaps(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Mon-Fri days present in the span but absent from the series index."""
    if len(idx) == 0:
        return pd.DatetimeIndex([])
    full = pd.bdate_range(idx.min(), idx.max())
    return full.difference(idx)


def stale_runs(close: pd.Series, min_run: int) -> list[dict]:
    """Find runs of >= min_run identical consecutive values."""
    vals = close.to_numpy()
    out: list[dict] = []
    if len(vals) == 0:
        return out
    # run-length encode equality with previous
    change = np.empty(len(vals), dtype=bool)
    change[0] = True
    change[1:] = vals[1:] != vals[:-1]
    grp = np.cumsum(change)
    s = pd.Series(grp, index=close.index)
    for _, block in close.groupby(s):
        if len(block) >= min_run:
            out.append({
                "start": str(block.index[0].date()),
                "end": str(block.index[-1].date()),
                "length": int(len(block)),
                "value": float(block.iloc[0]),
            })
    return out


def return_outliers(close: pd.Series, z_thresh: float, hard_thresh: float) -> pd.DataFrame:
    r = log_returns(close).dropna()
    if r.empty:
        return pd.DataFrame(columns=["date", "log_return", "z", "reason"])
    z = (r - r.mean()) / r.std(ddof=0)
    mask = (z.abs() > z_thresh) | (r.abs() > hard_thresh)
    hits = r[mask]
    reason = np.where(
        r[mask].abs() > hard_thresh,
        f"|ret|>{hard_thresh}",
        f"|z|>{z_thresh}",
    )
    return pd.DataFrame({
        "date": [d.date() for d in hits.index],
        "log_return": hits.to_numpy(),
        "z": z[mask].to_numpy(),
        "reason": reason,
    })


def nonpositive(close: pd.Series) -> list[str]:
    bad = close[close <= 0]
    return [str(d.date()) for d in bad.index]


def validate_series(name: str, df: pd.DataFrame, cfg_val: dict) -> dict:
    """Run all checks on one OHLCV frame. Uses 'close' if present, else first col."""
    col = "close" if "close" in df.columns else df.columns[0]
    close = df[col].astype(float)
    gaps = business_day_gaps(df.index)
    return {
        "name": name,
        "column_checked": col,
        "rows": int(len(close)),
        "span": [str(df.index.min().date()), str(df.index.max().date())] if len(df) else [None, None],
        "n_gaps": int(len(gaps)),
        "gap_sample": [str(d.date()) for d in gaps[:10]],
        "stale_runs": stale_runs(close, cfg_val["stale_min_run"]),
        "outliers": return_outliers(
            close, cfg_val["outlier_return_z"], cfg_val["outlier_return_hard"]
        ),
        "nonpositive_dates": nonpositive(close),
    }


def _md_section(f: dict) -> str:
    lines = [f"### {f['name']}",
             f"- rows: **{f['rows']}**, span: {f['span'][0]} → {f['span'][1]}, "
             f"column checked: `{f['column_checked']}`",
             f"- business-day gaps: **{f['n_gaps']}** "
             f"(sample: {', '.join(f['gap_sample']) or 'none'})",
             f"- non-positive prices: **{len(f['nonpositive_dates'])}**"]
    if f["stale_runs"]:
        lines.append(f"- stale runs (≥ threshold): **{len(f['stale_runs'])}**")
        for r in f["stale_runs"][:5]:
            lines.append(f"    - {r['start']}→{r['end']} ({r['length']}d @ {r['value']:.4g})")
    else:
        lines.append("- stale runs: none")
    n_out = len(f["outliers"])
    lines.append(f"- return outliers: **{n_out}**")
    for _, row in f["outliers"].head(5).iterrows():
        lines.append(f"    - {row['date']}: log-ret {row['log_return']:+.3f} "
                     f"(z={row['z']:+.1f}, {row['reason']})")
    return "\n".join(lines)


def run(cfg: dict | None = None) -> dict:
    """Validate every raw CSV listed in the manifest; write a markdown report."""
    import json
    cfg = cfg or load_config()
    ensure_dirs()
    manifest = json.loads((RAW / "manifest.json").read_text())

    findings: dict[str, dict] = {}
    for alias, meta in manifest["series"].items():
        if not meta.get("file"):
            continue
        df = pd.read_csv(RAW / meta["file"], index_col=0, parse_dates=True)
        findings[alias] = validate_series(alias, df, cfg["validation"])

    report = ["# Data Quality Report",
              f"_generated from {len(findings)} series_\n"]
    for f in findings.values():
        report.append(_md_section(f))
        report.append("")
    out_path = REPORTS / "data_quality" / "report.md"
    out_path.write_text("\n".join(report))
    print(f"Data quality report -> {out_path}")
    return findings


if __name__ == "__main__":
    run()
