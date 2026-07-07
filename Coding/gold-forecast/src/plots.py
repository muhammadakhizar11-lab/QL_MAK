"""
plots.py — stage 5: publication figures for the research note.

Reads only generated artifacts (equity_curves.csv, oos_predictions.csv,
horizon_sweep.json, raw series) and writes PNGs to reports/figures/. Nothing here
computes new results — if a number isn't already in a stage 3/4 artifact, it
doesn't belong in a figure.

Design system (static, light-surface figures):
  * Emphasis form — the story of most figures is "the ridge strategy vs context",
    so ridge wears the accent blue, the second signal aqua, and every benchmark a
    recessive gray. Color follows the entity across ALL figures (ridge is always
    blue), never its rank in one chart.
  * Palette: slots 1-2 of a CVD-validated categorical set (adjacent ΔE 24.2).
    Aqua is sub-3:1 on the light surface, so every series is direct-labeled and
    the note carries full value tables (the relief rule).
  * Marks: 2px lines, ~10%-opacity area washes, hairline solid gridlines, no top/
    right spines, text in ink tokens (never the series color for meaning).
  * One axis per plot — gold-vs-real-yields is two stacked panels on a shared
    x-axis, never a dual-axis chart.

Run:  python -m src.plots   (after src.evaluate and src.backtest)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PROCESSED, RAW, REPORTS, ensure_dirs

FIG = REPORTS / "figures"

# ---- palette (reference instance, light mode) ------------------------------ #
BLUE = "#2a78d6"      # categorical slot 1 — the ridge strategy, everywhere
AQUA = "#1baf7a"      # slot 2 — the logistic strategy, everywhere
GRAY_DARK = "#52514e"     # secondary ink — primary benchmark line
GRAY_MID = "#898781"      # muted ink — secondary benchmark line
GRAY_LIGHT = "#c3c2b7"    # baseline gray — tertiary benchmark line
INK = "#0b0b0b"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

SERIES_STYLE = {          # color follows the entity across every figure
    "reg_ridge": (BLUE, "ridge (signal)"),
    "clf_logistic": (AQUA, "logistic (signal)"),
    "long_always": (GRAY_DARK, "long always (vol-targeted)"),
    "momentum_20d": (GRAY_MID, "momentum 20d"),
    "gold_buy_hold_raw": (GRAY_LIGHT, "gold buy & hold (raw)"),
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 150,
        "font.family": "sans-serif", "font.size": 10,
        "text.color": INK, "axes.labelcolor": GRAY_DARK,
        "axes.edgecolor": GRAY_LIGHT, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": GRAY_MID, "ytick.color": GRAY_MID,
        "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlecolor": INK, "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })


def _endlabel(ax, x, y, text, color) -> None:
    """Direct label at a line's end — identity via the colored mark + ink text."""
    ax.annotate(f" {text}", (x, y), color=color, fontsize=9, fontweight="bold",
                va="center", annotation_clip=False)


def _save(fig, name: str) -> None:
    path = FIG / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {path}")


# --------------------------------------------------------------------------- #
# 1. Equity curves — emphasis: signals colored, benchmarks gray
# --------------------------------------------------------------------------- #
def fig_equity(equity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for name, (color, label) in SERIES_STYLE.items():
        if name in equity.columns:
            s = equity[name]
            ax.plot(s.index, s.values, color=color, label=label,
                    zorder=3 if name.startswith(("reg_", "clf_")) else 2)
            _endlabel(ax, s.index[-1], s.iloc[-1], label.split(" (")[0], color)
    ax.set_title("Out-of-sample equity — walk-forward signals vs benchmarks, net of 1bp costs")
    ax.set_ylabel("cumulative log return")
    ax.legend(loc="upper left", ncols=2)
    ax.margins(x=0.11)                      # room for the direct end-labels
    _save(fig, "equity_curves.png")


# --------------------------------------------------------------------------- #
# 2. Drawdown profile — the risk story, ridge vs raw gold
# --------------------------------------------------------------------------- #
def fig_drawdown(equity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for name in ("gold_buy_hold_raw", "reg_ridge"):
        color, label = SERIES_STYLE[name]
        dd = equity[name] - equity[name].cummax()
        ax.plot(dd.index, dd.values, color=color, label=label)
        ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.10)
        _endlabel(ax, dd.index[-1], dd.iloc[-1], label.split(" (")[0], color)
    ax.set_title("Drawdown from running peak — strategy vs owning gold outright")
    ax.set_ylabel("drawdown (log return)")
    ax.legend(loc="lower left")
    ax.margins(x=0.11)
    _save(fig, "drawdowns.png")


# --------------------------------------------------------------------------- #
# 3. Rolling IC — is the signal stable or one lucky stretch?
# --------------------------------------------------------------------------- #
def fig_rolling_ic(preds: pd.DataFrame, window: int = 252) -> None:
    # rolling Spearman: pandas rolling.corr is Pearson-only, but Pearson on
    # globally-ranked series is a standard rolling rank-correlation proxy
    r1 = preds["reg_ridge"].rank()
    r2 = preds["fwd_ret_actual"].rank()
    ic = r1.rolling(window).corr(r2)
    full = float(pd.Series(preds["reg_ridge"]).corr(preds["fwd_ret_actual"],
                                                    method="spearman"))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axhline(0, color=GRAY_LIGHT, linewidth=1)
    ax.plot(ic.index, ic.values, color=BLUE)      # single series — no legend box
    ax.axhline(full, color=GRAY_MID, linewidth=1)
    ax.annotate(f" full-sample IC {full:+.3f}", (ic.index[-1], full),
                color=GRAY_DARK, fontsize=9, va="bottom", annotation_clip=False)
    ax.set_title(f"Ridge information coefficient, rolling {window}d (Spearman, out-of-sample)")
    ax.set_ylabel("rank correlation")
    ax.margins(x=0.11)
    _save(fig, "rolling_ic.png")


# --------------------------------------------------------------------------- #
# 4. Horizon sweep — where the signal lives
# --------------------------------------------------------------------------- #
def fig_horizon_sweep(sweep: dict) -> None:
    horizons = sorted(sweep, key=int)
    ics = [sweep[h]["regression"]["ridge"]["ic_spearman"] for h in horizons]
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(horizons))
    # thin marks: ~0.16 of a ~150px slot keeps the bar ≤24px at 150dpi
    ax.bar(x, ics, width=0.16, color=BLUE, zorder=3)
    for xi, v in zip(x, ics):                     # value on every cap: it IS the data here
        ax.annotate(f"{v:+.3f}", (xi, v), ha="center", va="bottom",
                    fontsize=9, color=GRAY_DARK)
    ax.set_xticks(x, [f"{h}d" for h in horizons])
    ax.set_title("Ridge IC by label horizon — the signal is a 1-day effect")
    ax.set_ylabel("Spearman IC (OOS)")
    ax.set_xlabel("forward-return horizon")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    _save(fig, "horizon_sweep.png")


# --------------------------------------------------------------------------- #
# 5. Structural context — gold vs real yields (two panels, ONE axis each)
# --------------------------------------------------------------------------- #
def fig_gold_vs_real_yields() -> None:
    gold = pd.read_csv(RAW / "gold_comex.csv", index_col=0, parse_dates=True)["close"]
    real = pd.read_csv(RAW / "real_10y.csv", index_col=0, parse_dates=True).iloc[:, 0]
    real = real.loc[real.index >= gold.index.min()]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"hspace": 0.12})
    ax1.plot(gold.index, gold.values, color=BLUE)
    ax1.set_ylabel("gold, USD/oz")
    ax1.set_title("The 2022 regime break — gold rallied INTO sharply positive real yields")
    ax2.plot(real.index, real.values, color=GRAY_DARK)
    ax2.axhline(0, color=GRAY_LIGHT, linewidth=1)
    ax2.set_ylabel("10y real yield, % (DFII10)")

    # one shared annotation of the break, drawn on both panels
    break_date = pd.Timestamp("2022-03-01")
    for ax in (ax1, ax2):
        ax.axvline(break_date, color=GRAY_LIGHT, linewidth=1)
    ax1.annotate("  hiking cycle begins", (break_date, gold.max() * 0.97),
                 color=GRAY_MID, fontsize=9)
    _save(fig, "gold_vs_real_yields.png")


# --------------------------------------------------------------------------- #
def run() -> None:
    ensure_dirs()
    FIG.mkdir(parents=True, exist_ok=True)
    _style()

    equity = pd.read_csv(PROCESSED / "equity_curves.csv", index_col=0, parse_dates=True)
    preds = pd.read_csv(PROCESSED / "oos_predictions.csv", index_col=0, parse_dates=True)
    fig_equity(equity)
    fig_drawdown(equity)
    fig_rolling_ic(preds)

    sweep_path = REPORTS / "model" / "horizon_sweep.json"
    if sweep_path.exists():
        fig_horizon_sweep(json.loads(sweep_path.read_text()))
    else:
        print("[plots] horizon_sweep.json missing — run `python -m src.evaluate "
              "--horizons 1 5 10 20` first (figure skipped)")

    fig_gold_vs_real_yields()


if __name__ == "__main__":
    run()
