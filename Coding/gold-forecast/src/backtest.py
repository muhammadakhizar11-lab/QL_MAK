"""
backtest.py — stage 4: run the strategy layer over the stitched OOS forecasts
and compare against benchmarks, all through the SAME machinery.

WHAT IS BEING TESTED
--------------------
Stage 3 established a thin 1-day-ahead signal (ridge IC ~ +0.11, logistic
AUC ~ 0.545). This stage asks the only question that matters for a strategy:
does that signal survive vol targeting, transaction costs, and a stop, and does
it beat doing nothing clever? Every strategy row — including the benchmarks —
goes through the identical position pipeline (unit position -> vol target ->
stop -> costs), so differences come from the SIGNAL, not the plumbing. One raw
buy-and-hold row (no machinery) is kept as the outside reference point.

Signals (from data/processed/oos_predictions.csv — walk-forward OOS only):
  * reg_ridge      : predicted 1d forward return (sign gives the position)
  * clf_logistic   : P(up) - 0.5
Benchmarks:
  * long_always    : signal ≡ 1 through the machinery (vol-targeted gold)
  * momentum_20d   : sign of the trailing 20d return, known at close t
  * gold_buy_hold  : raw, unlevered, no vol target/stop — the outside reference

This remains an idealized daily-close backtest: costs are a flat bps on
turnover; no slippage, financing, or intraday effects. GC futures round-trip
friction for small size is roughly 1-2bp, so cost_bps: 1.0 (one-way) is
realistic-to-slightly-generous. The breakeven-cost column shows how sensitive
each result is to that assumption.

Run:  python -m src.backtest   (requires `python -m src.evaluate` output)
"""
from __future__ import annotations

import json

import pandas as pd

from .config import PROCESSED, REPORTS, ensure_dirs, load_config
from .strategy import StrategyResult, run_strategy

PREDS_CSV = PROCESSED / "oos_predictions.csv"


def build_signals(preds: pd.DataFrame, wanted: list[str]) -> dict[str, pd.Series]:
    """Model forecasts -> zero-centered signals, plus the benchmark signals."""
    sig: dict[str, pd.Series] = {}
    for name in wanted:
        if name not in preds.columns:
            raise KeyError(f"{name} not in {PREDS_CSV.name} — re-run src.evaluate "
                           f"with this model included")
        s = preds[name]
        sig[name] = s - 0.5 if name.startswith("clf_") else s
    # benchmarks (same zero-centered convention)
    sig["long_always"] = pd.Series(1.0, index=preds.index)
    # trailing 20d return known at close t: fwd_ret is r(t -> t+1), so the sum of
    # its previous 20 values is log(close_t / close_{t-20}).
    sig["momentum_20d"] = preds["fwd_ret_actual"].shift(1).rolling(20).sum()
    return sig


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs()
    sc = cfg["strategy"]

    if not PREDS_CSV.exists():
        raise FileNotFoundError(f"{PREDS_CSV} missing — run `python -m src.evaluate` first")
    preds = pd.read_csv(PREDS_CSV, index_col=0, parse_dates=True).sort_index()
    ret = preds["fwd_ret_actual"]                     # r(t -> t+1), the 1d label

    results: list[StrategyResult] = [
        run_strategy(name, s, ret, sc) for name, s in build_signals(preds, sc["signals"]).items()
    ]
    # raw buy-and-hold: outside the machinery on purpose (reference point)
    raw_sc = {**sc, "mode": "long_flat", "vol_target_ann": None,
              "max_leverage": 1.0, "stop": None}
    results.append(run_strategy("gold_buy_hold_raw", pd.Series(1.0, index=preds.index),
                                ret, raw_sc))

    # --- persist ---
    out = REPORTS / "backtest"
    out.mkdir(parents=True, exist_ok=True)
    equity = pd.DataFrame({r.name: r.daily["net"].cumsum() for r in results})
    equity.to_csv(PROCESSED / "equity_curves.csv")
    summary = {
        "config": {k: sc[k] for k in ("mode", "vol_target_ann", "vol_window",
                                      "max_leverage", "cost_bps", "stop")},
        "span": [str(preds.index.min().date()), str(preds.index.max().date())],
        "strategies": {r.name: r.metrics for r in results},
    }
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    (out / "report.md").write_text(_render(summary))

    _print(summary)
    return summary


_COLS = [("ann_ret_net", "net ret/yr", 4), ("ann_ret_gross", "gross ret/yr", 4),
         ("ann_vol", "vol/yr", 4), ("sharpe_net", "Sharpe", 2),
         ("max_drawdown", "max DD", 3), ("hit_rate_active", "hit (active)", 3),
         ("exposure_avg", "avg |pos|", 2), ("ann_turnover", "turnover/yr", 1),
         ("cost_drag_ann", "cost drag/yr", 4), ("breakeven_cost_bps", "b/e cost bp", 1)]


def _render(s: dict) -> str:
    c = s["config"]
    L = [
        "# Backtest Report — Stage 4 (OOS walk-forward signals through the strategy layer)",
        "",
        f"_Span {s['span'][0]} → {s['span'][1]} · mode **{c['mode']}** · "
        f"vol target **{c['vol_target_ann']}** (window {c['vol_window']}, "
        f"max lever {c['max_leverage']}) · cost **{c['cost_bps']}bp** one-way · "
        f"stop {c['stop']}_",
        "",
        "> Idealized daily-close backtest: flat bps costs, no slippage/financing/"
        "intraday. All rows except `gold_buy_hold_raw` run through the identical "
        "position machinery, so differences are signal, not plumbing. "
        "`b/e cost bp` = the one-way cost at which the strategy's net return "
        "reaches zero — the distance between the edge and the friction.",
        "",
        "| strategy | " + " | ".join(h for _, h, _ in _COLS) + " |",
        "|---|" + "---|" * len(_COLS),
    ]
    for name, m in s["strategies"].items():
        row = " | ".join("nan" if m.get(k) != m.get(k) else f"{m[k]:.{nd}f}"
                         for k, _, nd in _COLS)
        L.append(f"| {name} | {row} |")
    L += [
        "",
        "### How to read this",
        "- The bar: a signal strategy must beat **both** benchmarks (`long_always`, "
        "`momentum_20d`) net of costs, on Sharpe and drawdown, before it counts.",
        "- `gold_buy_hold_raw` is unlevered long gold — the do-nothing reference. "
        "During a strong gold bull market it is a hard bar; say so rather than "
        "hiding behind the vol-targeted comparison.",
        "- Daily signals trade a lot. If the b/e cost is only a few bp above the "
        "assumed cost, the edge is fragile to execution quality — report it, "
        "don't tune it away.",
    ]
    return "\n".join(L) + "\n"


def _print(s: dict) -> None:
    print(f"\nBacktest {s['span'][0]} -> {s['span'][1]}  "
          f"(mode={s['config']['mode']}, cost={s['config']['cost_bps']}bp)\n")
    hdr = f"{'strategy':<18}" + "".join(f"{h:>13}" for _, h, _ in _COLS[:6])
    print(hdr)
    for name, m in s["strategies"].items():
        vals = "".join(
            f"{('nan' if m.get(k) != m.get(k) else format(m[k], f'.{nd}f')):>13}"
            for k, _, nd in _COLS[:6])
        print(f"{name:<18}{vals}")
    print(f"\nReport -> {REPORTS / 'backtest' / 'report.md'}")
    print(f"Equity -> {PROCESSED / 'equity_curves.csv'}")


if __name__ == "__main__":
    run()
