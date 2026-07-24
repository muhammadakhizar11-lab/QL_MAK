"""Generate and execute the five research notebooks.

Notebooks are built from source here (single source of truth, reviewable in
git) and executed against the local data cache — re-running this script
re-derives every figure in notebooks/ from scratch.

Run:  python -m scripts.build_notebooks [--no-execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = """\
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vrp_lab.config import load_config, PROCESSED, REPORTS
cfg = load_config()
%config InlineBackend.figure_format = "retina"
"""


def _nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3",
                              "language": "python"}
    for kind, src in cells:
        cell = (nbf.v4.new_markdown_cell(src) if kind == "md"
                else nbf.v4.new_code_cell(src))
        nb.cells.append(cell)
    return nb


# --------------------------------------------------------------------------- #
NB01 = [
    ("md", """\
# 01 — Measuring the Volatility Risk Premium

**Definition.** $VRP(t,\\tau) = IV(t,\\tau) - RV(t, t+\\tau)$ — implied vol observed
at $t$ minus the vol *subsequently realized* over $[t, t+\\tau]$.

Two variants live in `vrp_lab.volatility.vrp` and the distinction is the whole game:

| variant | RV leg | usable as a signal? |
|---|---|---|
| `vrp_ex_post` | future window $(t, t+\\tau]$ | **No** — look-ahead by construction. Measurement only. |
| `vrp_trailing` | trailing window $(t-\\tau, t]$ | Yes — everything observable at $t$. |

This notebook *measures* (ex-post variant). The trading notebooks only ever
consume the trailing variant."""),
    ("code", SETUP),
    ("code", """\
from vrp_lab.analysis.vrp_study import build_dataset, unconditional_stats, conditional_table
df = build_dataset(save=True)   # cached pulls; offline after first run
df[["close", "vix", "vrp_expost_21d", "vrp_trailing_21d"]].tail()"""),
    ("md", "## The premium through time\nRed bands are VIX > 25 episodes — "
           "the moments the insurance seller pays out."),
    ("code", """\
from vrp_lab.visualization.plots import (plot_vrp_timeseries, plot_vrp_histogram,
                                          plot_vrp_vs_vix)
plot_vrp_timeseries(df["vrp_expost_21d"], df["vix"]);"""),
    ("code", 'unconditional_stats(df["vrp_expost_21d"]).round(2)'),
    ("md", "The headline: **mean ≈ +3.6 vol points, positive on ~84% of days** — "
           "and a minimum of −69 points (Feb-2020 entry dates whose forward window "
           "caught the COVID crash). Selling vol collects a steady coupon and "
           "occasionally hands back years of it in a month."),
    ("code", 'plot_vrp_histogram(df["vrp_expost_21d"]);'),
    ("code", 'plot_vrp_vs_vix(df["vrp_expost_21d"], df["vix"]);'),
    ("md", "## Conditioning\nThe premium is *larger on average* when VIX is elevated "
           "— but the left tail grows faster than the mean. The extra carry is "
           "payment for spike risk, not free money."),
    ("code", """\
conditional_table(df["vrp_expost_21d"], df["vix"],
                  bins=[0, 15, 20, 25, 35, 100],
                  labels=["<15", "15-20", "20-25", "25-35", ">35"]).round(2)"""),
    ("md", "## Both horizons + the tradeable proxy\nThe trailing proxy tracks the "
           "ex-post premium on average but lags at turning points — after a spike "
           "it *understates* the premium (trailing RV stays elevated while IV "
           "normalizes). The regime filter, not the proxy, must carry that risk."),
    ("code", """\
ax = (100*df[["vrp_expost_21d", "vrp_trailing_21d"]]).rolling(63).mean().plot(
    figsize=(11, 4), color=["#2a78d6", "#eda100"], lw=1.2)
ax.axhline(0, color="#b5b4ac", lw=0.8); ax.set_ylabel("vol pts (63d smooth)")
ax.set_title("Ex-post premium vs trailing tradeable proxy", loc="left");"""),
]

NB02 = [
    ("md", """\
# 02 — Realized-vol estimator comparison

Four estimators (`vrp_lab.volatility.realized`), each with its paper reference
and formula in the docstrings. GBM recovery tests with known σ live in
`tests/test_realized.py`; this notebook is the *empirical* comparison on SPY,
where overnight gaps and drift break each estimator's assumptions differently."""),
    ("code", SETUP),
    ("code", """\
from vrp_lab.data.yfinance import pull_ohlcv
from vrp_lab.analysis.estimator_compare import run_all, divergence_stats, high_vol_episodes
from vrp_lab.data.cboe import pull_vix_history

spy = pull_ohlcv("SPY", cfg["date_range"]["start"])
vix = pull_vix_history()["close"]
est = run_all(spy, window=21)
divergence_stats(est).round(4)"""),
    ("md", "Close-to-close is the noisiest (`stdev_of_estimate`); the range-based "
           "estimators sit *below* it on SPY because overnight gaps carry variance "
           "they cannot see. Yang-Zhang recovers the level (overnight term) while "
           "keeping the range estimators' efficiency — the default everywhere in "
           "this repo."),
    ("code", """\
from vrp_lab.visualization.plots import plot_estimator_comparison
sl = slice("2019-06-01", "2020-12-31")
plot_estimator_comparison({k: est[k].loc[sl] for k in
                           ["close_to_close", "parkinson", "garman_klass", "yang_zhang"]});"""),
    ("md", "## Where the choice matters: crisis windows\nRestricted to VIX > 25 days, "
           "the dispersion across estimators widens to several vol points — enough to "
           "flip a marginal VRP signal."),
    ("code", 'high_vol_episodes(est, vix).round(4)'),
]

NB03 = [
    ("md", """\
# 03 — VIX futures carry backtest (2013–present)

Three variants of `strategies.vix_carry.VIXCarryStrategy` through the
event-driven engine (real CBOE settlements, transaction costs, margin):

1. **naive always-short** — no exits, no gate, 1x notional. The XIV replica.
2. **regime-filtered** — calm to enter, genuine crisis to exit (VIX-30 breaker,
   deep backwardation), acceleration halves exposure. Still 1x notional.
3. **filtered + vol-targeted** — same gates, position sized to a 15% annual
   vol budget with a 2x notional cap.

Re-derive with `python -m scripts.run_vix_carry`. **Read the conclusion before
the charts:** this is a negative-result notebook done honestly — front-month
VX carry post-2013 with survival constraints and real costs earns roughly cash."""),
    ("code", SETUP),
    ("code", """\
eq = pd.read_csv(PROCESSED / "equity_curves.csv", index_col=0, parse_dates=True)
results = pd.read_csv(REPORTS / "vix_carry_results.csv", index_col=0)
results.round(3)"""),
    ("md", "**The naive column is the exhibit**: −100% annualized (ruin on "
           "2018-02-05, equity through zero — precisely XIV's fate that afternoon), "
           "skew −11.6, kurtosis 346, 2,120 margin-breach days. A Sharpe printed on "
           "top of that distribution would be meaningless."),
    ("code", """\
ax = eq.clip(lower=1).plot(figsize=(11, 4.5), logy=True, lw=1.1,
        color=["#e34948", "#eda100", "#1baf7a"])
ax.set_ylabel("equity ($, log)"); ax.set_title("Three ways to be short vol", loc="left")
for i, ev in enumerate(["2018-02-05", "2020-03-16", "2024-08-05"]):
    ax.axvline(pd.Timestamp(ev), color="#52514e", lw=0.7, ls=":")
ax.legend(frameon=False);"""),
    ("code", """\
from vrp_lab.visualization.plots import plot_equity_drawdown, plot_rolling_sharpe, plot_monthly_heatmap
main = eq["filtered_vol_targeted"]
plot_equity_drawdown(main, title="filtered + vol-targeted");"""),
    ("code", """\
rets = main.pct_change().dropna()
plot_rolling_sharpe(rets);
plot_monthly_heatmap(rets);"""),
    ("md", "## P&L by term-structure regime"),
    ("code", """\
from vrp_lab.data import cboe
from vrp_lab.volatility.term_structure import build_curve, classify_regime
vix_full = cboe.pull_vix_history()["close"]
curve = build_curve(cboe.pull_vx_curve("2013-01-01"), vix_full)
reg = classify_regime(curve).reindex(rets.index).fillna("mixed")
pnl_by_regime = rets.groupby(reg).agg(["mean", "std", "count"])
pnl_by_regime["ann_mean_%"] = (100 * 252 * pnl_by_regime["mean"]).round(2)
pnl_by_regime"""),
    ("md", """\
## Honest conclusions

* The premium measured in notebook 01 is real; **capturing it via front-month
  futures is another matter.** Costs (~0.035 pts + commission per side, 24+
  legs/year), spike losses, and the survival-mandated exits consume most of it
  post-2013.
* Every risk control has a price: the VIX-30 circuit breaker covers near spike
  highs by construction. Removing it is how you become the naive column.
* Vol targeting cuts CVaR-99 from −17.6% to −3.7% and max DD from −85% to
  −23% — the distribution improves far more than the mean degrades.
* What would help: longer-dated tenors (VX4–VX7 carry with less gamma),
  entry conditioning on the VRP z-score, and the 2006–2012 sample this free
  data source can't reach (contango was fatter pre-2013)."""),
]

NB04 = [
    ("md", """\
# 04 — Delta-hedged option selling (synthetic chains)

**Read the caveat first.** No historical SPY chains are freely available, so
options here are Black-Scholes-priced from real SPY closes with the real VIX as
ATM IV and a *fixed* parametric skew (`data/options.py`). Real mechanics
(daily hedging, exits, costs, decomposition) on model prices: treat levels as a
sketch, not a track record. With IV set to VIX and BS pricing, a positive
theta−gamma spread is partly baked in whenever VIX > subsequent RV — which is
the very premium measured in notebook 01, but with none of the smile dynamics,
vol-of-vol marks, or gap microstructure a real book would carry.

Re-derive with `python -m scripts.run_delta_hedged`."""),
    ("code", SETUP),
    ("code", """\
results = pd.read_csv(REPORTS / "delta_hedged_results.csv", index_col=0)
hist = pd.read_csv(PROCESSED / "delta_hedged_history.csv", index_col=0, parse_dates=True)
results.round(3)"""),
    ("md", "## P&L decomposition — the measurement that matters\n"
           "$\\Delta V \\approx \\Delta\\,dS + \\tfrac12\\Gamma\\,dS^2 + \\nu\\,d\\sigma + \\Theta\\,dt$ — "
           "for a delta-hedged short book, **theta collected vs gamma paid is the "
           "realized VRP**. Residual = higher-order greeks + discrete hedging."),
    ("code", """\
from vrp_lab.risk.pnl_decomp import decompose, attribution_summary
d = decompose(hist[hist["in_position"]])
attribution_summary(d).round(0)"""),
    ("code", """\
from vrp_lab.visualization.plots import plot_pnl_attribution, plot_gamma_vs_move
plot_pnl_attribution(d);"""),
    ("code", """\
from vrp_lab.data.yfinance import pull_ohlcv
spy = pull_ohlcv("SPY", cfg["date_range"]["start"])["close"]
plot_gamma_vs_move(d, spy);"""),
    ("md", "Gamma P&L against |move| is negative and quadratic — the visual "
           "definition of short gamma. The hedge neutralizes *direction* (delta "
           "P&L ≈ 0 above), never *convexity*."),
    ("md", "## The tail hedge trade-off\nThe 5Δ/2Δ put spread costs Sharpe "
           "(1.39 → 0.44 here) and buys skew (−3.3 → −0.1) and a bounded left "
           "tail. In BS-world the hedge looks maximally unattractive — flat smile "
           "means crash wings are priced *without* crash premium, yet it still "
           "only halves the Sharpe. On a real smile you pay more; the point of "
           "showing it is the shape of the trade-off, not the exact number."),
    ("code", """\
cols = ["sharpe", "max_drawdown", "cvar_99", "skew", "worst_day"]
results.loc[cols].T.round(3)"""),
]

NB05 = [
    ("md", """\
# 05 — Stress testing & tail risk

Replay the (filtered + vol-targeted) VIX carry book through the three episodes
that define short-vol failure modes:

* **2018-02-05 "Volmageddon"** — VIX doubled in a day; XIV terminated at −96%.
* **Mar 2020 COVID** — VIX 82; curve inverted for weeks.
* **2024-08-05 JPY carry unwind** — VIX ~65 intraday, fastest inversion since 2018."""),
    ("code", SETUP),
    ("code", """\
from vrp_lab.risk.stress import stress_report, survived
eq = pd.read_csv(PROCESSED / "equity_curves.csv", index_col=0, parse_dates=True)
curves = pd.read_csv(REPORTS / "stress_report.csv", index_col=0)
curves.round(4)"""),
    ("md", "**Did the regime filter and vol-targeting prevent ruin?** Yes, in all "
           "three covered events: worst event drawdown −4.2% (COVID), zero margin "
           "breaches, and `exposure_cut=True` shows the book was flattened inside "
           "each window. Compare the naive book: dead on the first event."),
    ("code", """\
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
events = {"Volmageddon": ("2018-01-02", "2018-03-30"),
          "COVID":       ("2020-01-02", "2020-06-30"),
          "JPY unwind":  ("2024-07-01", "2024-09-30")}
for ax, (name, (a, b)) in zip(axes, events.items()):
    w = eq.loc[a:b]
    for col, color in zip(eq.columns, ["#e34948", "#eda100", "#1baf7a"]):
        ax.plot(w.index, w[col] / w[col].iloc[0], lw=1.1, color=color, label=col)
    ax.set_title(name, fontsize=10, loc="left"); ax.tick_params(labelsize=7)
axes[0].set_ylabel("equity (rebased)"); axes[0].legend(frameon=False, fontsize=7)
plt.tight_layout();"""),
    ("md", """\
## What the events teach

* **Feb 2018**: the failure is *speed* — the VIX level gate alone is too slow
  (VIX was 17 on Feb 1). The acceleration flag (5d ≫ 21d RV) and vol-targeted
  sizing are what kept the hit to ~3%: exposure was already small because
  realized vol had been creeping up.
* **Mar 2020**: slow-motion by comparison; the circuit breaker exits on the way
  up. Cost: the book sat out the historic contango of Apr–Dec 2020. Survival
  and upside come from the same switch.
* **Aug 2024**: the gate had the book *flat before the spike* (max gross
  notional 0 in the window) — vol had been accelerating since late July.
* Remaining unhedgeable risk: an overnight Volmageddon *from a calm tape* —
  a gap through every daily-frequency control. That residual is why position
  caps exist and why the tail overlay of notebook 04 is worth its drag."""),
]


NOTEBOOKS = {
    "01_vrp_measurement.ipynb": NB01,
    "02_estimator_comparison.ipynb": NB02,
    "03_vix_carry_backtest.ipynb": NB03,
    "04_delta_hedged_backtest.ipynb": NB04,
    "05_stress_testing.ipynb": NB05,
}


def main() -> None:
    execute = "--no-execute" not in sys.argv
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = _nb(cells)
        if execute:
            print(f"executing {name} ...")
            NotebookClient(nb, timeout=1200, kernel_name="python3",
                           resources={"metadata": {"path": str(ROOT)}}).execute()
        path = OUT / name
        nbf.write(nb, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
