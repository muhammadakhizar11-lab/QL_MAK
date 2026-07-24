"""Single-page HTML tearsheet: equity + drawdown, rolling metrics, monthly
heatmap, return distribution, full risk table. Self-contained (charts embedded
as base64 PNGs) so the file mails/commits cleanly.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import REPORTS, get_logger
from ..risk.metrics import summary
from . import plots

log = get_logger(__name__)


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _return_histogram(returns: pd.Series) -> plt.Figure:
    """Daily return distribution with a matched-moment normal overlay —
    the skew/kurtosis gap is the tail-risk exhibit."""
    from scipy.stats import norm

    r = 100 * returns.dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(r.min(), r.max(), 90)
    ax.hist(r, bins=bins, density=True, color=plots.C_BLUE, alpha=0.85, zorder=2)
    x = np.linspace(r.min(), r.max(), 400)
    ax.plot(x, norm.pdf(x, r.mean(), r.std()), color=plots.C_ORANGE,
            linewidth=1.6, label="normal, same mean/vol")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("daily return (%)", fontsize=10)
    ax.set_title(f"Return distribution — skew {r.skew():.2f}, "
                 f"excess kurtosis {r.kurtosis():.1f}", fontsize=11, loc="left")
    plots._ax_style(ax)
    fig.tight_layout()
    return fig


def _underwater(equity: pd.Series) -> plt.Figure:
    dd = equity / equity.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(dd.index, 100 * dd, 0, color=plots.C_RED, alpha=0.7, linewidth=0)
    ax.set_ylabel("drawdown (%)", fontsize=10)
    ax.set_title("Underwater chart", fontsize=11, loc="left")
    plots._ax_style(ax)
    fig.tight_layout()
    return fig


def _rolling_vol(returns: pd.Series, window: int = 63) -> plt.Figure:
    rv = 100 * returns.rolling(window).std(ddof=1) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(rv.index, rv, color=plots.C_YELLOW, linewidth=1.2)
    ax.set_ylabel(f"rolling {window}d vol (%)", fontsize=10)
    ax.set_title("Realized strategy volatility", fontsize=11, loc="left")
    plots._ax_style(ax)
    fig.tight_layout()
    return fig


def _metrics_table_html(returns: pd.Series, equity: pd.Series,
                        rf_annual: pd.Series | float | None) -> str:
    s = summary(returns, equity, rf_annual)
    fmt = {
        "ann_return": "{:+.2%}", "ann_vol": "{:.2%}", "sharpe": "{:.2f}",
        "sortino": "{:.2f}", "calmar": "{:.2f}", "max_drawdown": "{:+.2%}",
        "dd_duration_days": "{:.0f}", "dd_recovery_days": "{}",
        "cvar_95": "{:+.2%}", "cvar_99": "{:+.2%}", "worst_day": "{:+.2%}",
        "best_day": "{:+.2%}", "skew": "{:.2f}", "kurtosis": "{:.1f}",
        "hit_rate": "{:.1%}", "n_days": "{:.0f}",
    }
    rows = []
    for k, v in s.items():
        try:
            txt = fmt.get(k, "{}").format(v)
        except (ValueError, TypeError):
            txt = str(v)
        rows.append(f"<tr><td>{k}</td><td class='num'>{txt}</td></tr>")
    return "<table class='metrics'>" + "".join(rows) + "</table>"


def generate_tearsheet(
    equity: pd.Series,
    name: str = "strategy",
    rf_annual: pd.Series | float | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Build the one-page HTML tearsheet for an equity curve.

    Returns the written path (default: reports/tearsheet_<name>.html).
    """
    returns = equity.pct_change().dropna()
    imgs = {
        "equity": _fig_to_b64(plots.plot_equity_drawdown(equity, title=name)),
        "monthly": _fig_to_b64(plots.plot_monthly_heatmap(returns)),
        "sharpe": _fig_to_b64(plots.plot_rolling_sharpe(returns)),
        "vol": _fig_to_b64(_rolling_vol(returns)),
        "hist": _fig_to_b64(_return_histogram(returns)),
        "underwater": _fig_to_b64(_underwater(equity)),
    }
    table = _metrics_table_html(returns, equity, rf_annual)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{name} — tearsheet</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
         max-width: 1060px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; color: #52514e; margin-top: 28px; }}
  img {{ max-width: 100%; }}
  .row {{ display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-start; }}
  table.metrics {{ border-collapse: collapse; font-size: 13px; min-width: 300px; }}
  table.metrics td {{ border-bottom: 1px solid #e5e4de; padding: 4px 10px; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .note {{ color: #52514e; font-size: 12px; }}
</style></head><body>
<h1>{name} — tearsheet</h1>
<p class="note">Period {equity.index.min():%Y-%m-%d} → {equity.index.max():%Y-%m-%d}.
All metrics on daily data; Sharpe/Sortino vs risk-free where supplied.</p>
<h2>Equity &amp; drawdown</h2><img src="data:image/png;base64,{imgs['equity']}">
<h2>Risk metrics</h2>
<div class="row">{table}<img src="data:image/png;base64,{imgs['hist']}" style="max-width:56%"></div>
<h2>Monthly returns</h2><img src="data:image/png;base64,{imgs['monthly']}">
<h2>Rolling metrics</h2>
<img src="data:image/png;base64,{imgs['sharpe']}">
<img src="data:image/png;base64,{imgs['vol']}">
<h2>Time under water</h2><img src="data:image/png;base64,{imgs['underwater']}">
</body></html>"""

    out = Path(out_path) if out_path else REPORTS / f"tearsheet_{name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    log.info("tearsheet -> %s", out)
    return out
