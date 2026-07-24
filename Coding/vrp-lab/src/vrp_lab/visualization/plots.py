"""Publication-quality matplotlib charts for the VRP study and strategy results.

Conventions (kept consistent across every figure):
  * fixed categorical palette, assigned by slot order — a series keeps its
    color across figures (VIX/IV is always slot 1 blue, realized vol slot 3
    yellow, P&L slot 2 aqua, danger/crisis slot 6 red);
  * one y-axis per plot — never dual-axis; different scales get stacked panels;
  * recessive grid (light gray, behind the data), direct labels where they fit;
  * diverging colormap (RdBu: red = negative) only for signed quantities.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# validated categorical palette (fixed slot order — do not cycle/reorder)
C_BLUE = "#2a78d6"     # slot 1: implied vol / VIX / primary series
C_AQUA = "#1baf7a"     # slot 2: strategy P&L / secondary
C_YELLOW = "#eda100"   # slot 3: realized vol / comparison
C_GREEN = "#008300"    # slot 4
C_VIOLET = "#4a3aa7"   # slot 5
C_RED = "#e34948"      # slot 6: crisis / drawdown / negative
C_MAGENTA = "#e87ba4"  # slot 7
C_ORANGE = "#eb6834"   # slot 8
PALETTE = [C_BLUE, C_AQUA, C_YELLOW, C_GREEN, C_VIOLET, C_RED, C_MAGENTA, C_ORANGE]

GRID = dict(color="#d9d9d4", linewidth=0.6, zorder=0)
TEXT_2ND = "#52514e"


def _ax_style(ax: plt.Axes) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ac")
    ax.tick_params(colors=TEXT_2ND, labelsize=9)


def plot_vrp_timeseries(vrp_expost: pd.Series, vix: pd.Series,
                        crisis_min: float = 25.0) -> plt.Figure:
    """Ex-post VRP through time with crisis episodes shaded. The negative
    spikes ARE the story: the premium is insurance income, sold ahead of
    occasional catastrophic claims."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    crisis = (vix.reindex(vrp_expost.index) > crisis_min)
    for grp, block in crisis.groupby((crisis != crisis.shift()).cumsum()):
        if block.iloc[0]:
            ax.axvspan(block.index[0], block.index[-1], color=C_RED, alpha=0.10, zorder=0)
    ax.axhline(0, color="#b5b4ac", linewidth=0.8)
    ax.plot(vrp_expost.index, 100 * vrp_expost, color=C_BLUE, linewidth=1.0)
    mean = 100 * vrp_expost.mean()
    ax.axhline(mean, color=C_AQUA, linewidth=1.2, linestyle="--")
    ax.annotate(f"mean {mean:+.1f} vol pts", xy=(0.99, mean), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=9, color=C_AQUA)
    ax.set_ylabel("VRP = IV − subsequent RV (vol pts)", fontsize=10)
    ax.set_title("Volatility risk premium (ex-post, 21d) — red bands: VIX > 25",
                 fontsize=11, loc="left")
    _ax_style(ax)
    fig.tight_layout()
    return fig


def plot_vrp_histogram(vrp_expost: pd.Series) -> plt.Figure:
    """Distribution of the premium: positive mode, violent left tail."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = (100 * vrp_expost.dropna())
    neg, pos = x[x < 0], x[x >= 0]
    bins = np.linspace(x.min(), x.max(), 80)
    ax.hist(pos, bins=bins, color=C_BLUE, alpha=0.9, zorder=2)
    ax.hist(neg, bins=bins, color=C_RED, alpha=0.9, zorder=2,
            label=f"negative: {100 * (x < 0).mean():.0f}% of days")
    ax.axvline(x.mean(), color=C_AQUA, linestyle="--", linewidth=1.2)
    ax.annotate(f"mean {x.mean():+.1f}", xy=(x.mean(), 1), xycoords=("data", "axes fraction"),
                ha="left", va="top", fontsize=9, color=C_AQUA, xytext=(4, -4),
                textcoords="offset points")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("VRP (vol pts)", fontsize=10)
    ax.set_ylabel("days", fontsize=10)
    ax.set_title("The premium is earned on average — and occasionally taken back",
                 fontsize=11, loc="left")
    _ax_style(ax)
    fig.tight_layout()
    return fig


def plot_vrp_vs_vix(vrp_expost: pd.Series, vix: pd.Series) -> plt.Figure:
    """Premium vs VIX level: richer when vol is bid, and wilder."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df = pd.DataFrame({"vix": vix, "vrp": 100 * vrp_expost}).dropna()
    ax.scatter(df["vix"], df["vrp"], s=8, alpha=0.35, color=C_BLUE,
               edgecolors="none", zorder=2)
    med = df.groupby(pd.cut(df["vix"], bins=np.arange(10, 60, 5)), observed=True)["vrp"].median()
    centers = [iv.mid for iv in med.index]
    ax.plot(centers, med.values, color=C_ORANGE, linewidth=2.0, marker="o",
            markersize=5, zorder=3, label="median by VIX bucket")
    ax.axhline(0, color="#b5b4ac", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlabel("VIX level", fontsize=10)
    ax.set_ylabel("subsequent 21d VRP (vol pts)", fontsize=10)
    ax.set_title("Conditioning on the VIX level", fontsize=11, loc="left")
    _ax_style(ax)
    fig.tight_layout()
    return fig


def plot_estimator_comparison(estimates: dict[str, pd.Series]) -> plt.Figure:
    """RV estimators overlaid — divergence widens in high-vol episodes."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for (name, s), color in zip(estimates.items(), PALETTE):
        ax.plot(s.index, 100 * s, linewidth=1.0, label=name.replace("_", "-"),
                color=color)
    ax.set_ylabel("annualized RV (%)", fontsize=10)
    ax.set_title("Realized-vol estimators on the same underlying", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncols=len(estimates))
    _ax_style(ax)
    fig.tight_layout()
    return fig


def plot_equity_drawdown(equity: pd.Series, title: str = "Equity curve") -> plt.Figure:
    """Log-scale equity with drawdown panel below (never a second y-axis)."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, height_ratios=[3, 1])
    ax1.plot(equity.index, equity, color=C_AQUA, linewidth=1.2)
    ax1.set_yscale("log")
    ax1.set_ylabel("equity ($, log)", fontsize=10)
    ax1.set_title(title, fontsize=11, loc="left")
    dd = equity / equity.cummax() - 1.0
    ax2.fill_between(dd.index, 100 * dd, 0, color=C_RED, alpha=0.7, linewidth=0)
    ax2.set_ylabel("drawdown (%)", fontsize=10)
    trough = dd.idxmin()
    ax2.annotate(f"{100 * dd.min():.1f}%", xy=(trough, 100 * dd.min()),
                 fontsize=9, color=C_RED, ha="left", va="bottom",
                 xytext=(4, 2), textcoords="offset points")
    for ax in (ax1, ax2):
        _ax_style(ax)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(returns: pd.Series, window: int = 252) -> plt.Figure:
    """Rolling 1y Sharpe — the stability view a single headline number hides."""
    roll = (np.sqrt(252) * returns.rolling(window).mean()
            / returns.rolling(window).std(ddof=1))
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.axhline(0, color="#b5b4ac", linewidth=0.8)
    ax.plot(roll.index, roll, color=C_VIOLET, linewidth=1.2)
    ax.set_ylabel(f"rolling {window}d Sharpe", fontsize=10)
    ax.set_title("Rolling Sharpe", fontsize=11, loc="left")
    _ax_style(ax)
    fig.tight_layout()
    return fig


def plot_monthly_heatmap(returns: pd.Series) -> plt.Figure:
    """Calendar heatmap of monthly returns. Diverging RdBu: red = down."""
    monthly = (1 + returns.fillna(0)).groupby(
        [returns.index.year, returns.index.month]).prod() - 1
    tbl = monthly.unstack()
    tbl.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: len(tbl.columns)]
    lim = max(abs(np.nanmin(tbl.values)), abs(np.nanmax(tbl.values)), 1e-6)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(tbl) + 1.4))
    im = ax.imshow(tbl.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(tbl.columns)), tbl.columns, fontsize=9)
    ax.set_yticks(range(len(tbl.index)), tbl.index, fontsize=9)
    for i in range(tbl.shape[0]):
        for j in range(tbl.shape[1]):
            v = tbl.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{100 * v:.1f}", ha="center", va="center",
                        fontsize=8, color="#0b0b0b")
    ax.set_title("Monthly returns (%)", fontsize=11, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.7).ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def plot_pnl_attribution(decomposed: pd.DataFrame) -> plt.Figure:
    """Cumulative P&L by greek — the picture that shows WHERE the money
    came from (theta) and what it paid for (gamma)."""
    comp = decomposed[["theta_pnl", "gamma_pnl", "vega_pnl", "residual"]].cumsum()
    actual = decomposed["option_pnl"].cumsum()
    colors = {"theta_pnl": C_AQUA, "gamma_pnl": C_RED,
              "vega_pnl": C_VIOLET, "residual": C_YELLOW}
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col, color in colors.items():
        ax.plot(comp.index, comp[col], linewidth=1.2, color=color,
                label=col.replace("_pnl", "").replace("residual", "residual"))
    ax.plot(actual.index, actual, linewidth=1.6, color="#0b0b0b",
            linestyle=":", label="actual option P&L")
    ax.axhline(0, color="#b5b4ac", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, ncols=5)
    ax.set_ylabel("cumulative $", fontsize=10)
    ax.set_title("P&L attribution: theta collected vs gamma paid", fontsize=11, loc="left")
    _ax_style(ax)
    fig.tight_layout()
    return fig


def plot_gamma_vs_move(decomposed: pd.DataFrame, spy_close: pd.Series) -> plt.Figure:
    """Daily gamma P&L against the size of the underlying move: negative and
    convex — the visual definition of being short gamma."""
    move = spy_close.pct_change().abs().reindex(decomposed.index)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(100 * move, decomposed["gamma_pnl"], s=10, alpha=0.4,
               color=C_RED, edgecolors="none")
    ax.axhline(0, color="#b5b4ac", linewidth=0.8)
    ax.set_xlabel("|daily SPY move| (%)", fontsize=10)
    ax.set_ylabel("gamma P&L ($)", fontsize=10)
    ax.set_title("Short gamma: quiet days are free, big days are quadratic",
                 fontsize=11, loc="left")
    _ax_style(ax)
    fig.tight_layout()
    return fig
