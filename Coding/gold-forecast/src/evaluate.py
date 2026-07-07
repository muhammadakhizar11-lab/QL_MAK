"""
evaluate.py — walk-forward, embargoed evaluation of the stage-3 models.

WHAT THIS ANSWERS (and what it does NOT)
----------------------------------------
It answers the README's gate: out-of-sample, does any model forecast the N-day
forward gold return, and does it BEAT THE NAIVE BENCHMARK? It is not a backtest —
there are no costs, no position sizing, no stops. Those are stage 4. The one
return-space number here (mean forward return by predicted sign) is a pre-cost
signal check, labelled as such, not a P&L claim.

WALK-FORWARD WITH AN EMBARGO (the leakage-critical bit)
-------------------------------------------------------
Training is an expanding window; we refit at every fold and predict the next
`test_days` rows out-of-sample. Because the label is an N-day FORWARD return,
row t's label depends on prices through t+N, so the last N training rows before a
test block have labels that reach into (or across) the boundary. We therefore
PURGE `embargo` rows (default = horizon N) immediately before each test block, so
no training label overlaps the test window. This is López de Prado purging for a
forward-labelled series. `tests/test_model.py` asserts the gap is real.

Everything a model learns (imputer/scaler moments, coefficients, tree splits) is
fit inside the fold on the train slice only — see models.py.

Run:  python -m src.evaluate                    # primary horizon (config label:)
      python -m src.evaluate --horizons 1 5 10 20   # horizon sweep -> comparison
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from .config import PROCESSED, REPORTS, ensure_dirs, load_config
from .features import build_features
from .models import build_classifiers, build_regressors


# --------------------------------------------------------------------------- #
# Walk-forward splitter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Fold:
    train_pos: np.ndarray   # integer positions into the sorted sample index
    test_pos: np.ndarray


def walk_forward_folds(n: int, min_train: int, test_days: int, embargo: int,
                       scheme: str = "expanding") -> list[Fold]:
    """Generate embargoed walk-forward folds over `n` time-ordered rows.

    For a test block starting at position `ts`, training uses rows [lo, ts-embargo):
    the `embargo` rows just before the test block are dropped so no training
    label (an N-day forward return) can peek into the test window. Expanding =>
    lo=0; rolling => lo=ts-embargo-min_train.
    """
    if embargo < 0 or test_days < 1 or min_train < 1:
        raise ValueError("min_train>=1, test_days>=1, embargo>=0 required")

    folds: list[Fold] = []
    ts = min_train + embargo               # first out-of-sample row
    while ts < n:
        te = min(ts + test_days, n)
        train_hi = ts - embargo            # exclusive; purges the embargo rows
        lo = 0 if scheme == "expanding" else max(0, train_hi - min_train)
        if train_hi - lo >= min_train:
            folds.append(Fold(np.arange(lo, train_hi), np.arange(ts, te)))
        ts = te
    return folds


# --------------------------------------------------------------------------- #
# Out-of-sample prediction collection
# --------------------------------------------------------------------------- #
def _run_head(X: pd.DataFrame, y: pd.Series, folds: list[Fold],
              factories: dict) -> pd.DataFrame:
    """Fit each model fresh per fold, predict its test block, and stitch the
    disjoint test blocks into one out-of-sample prediction column per model."""
    cols: dict[str, pd.Series] = {}
    for name, make in factories.items():
        parts = []
        for fold in folds:
            Xtr, ytr = X.iloc[fold.train_pos], y.iloc[fold.train_pos]
            Xte = X.iloc[fold.test_pos]
            model = make().fit(Xtr, ytr)
            parts.append(pd.Series(model.predict(Xte), index=Xte.index))
        cols[name] = pd.concat(parts)
    return pd.DataFrame(cols)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _reg_metrics(pred: pd.Series, actual: pd.Series, naive_rmse: float) -> dict:
    p, a = pred.to_numpy(), actual.to_numpy()
    rmse = float(np.sqrt(np.mean((p - a) ** 2)))
    # IC is undefined for a constant forecast (e.g. the naive-zero benchmark);
    # skip the corr call so it doesn't warn, report NaN.
    ic = float(pred.corr(actual, method="spearman")) if np.ptp(p) > 0 else float("nan")
    mask = (p != 0)                                           # naive predicts all 0
    hit = float(np.mean(np.sign(p[mask]) == np.sign(a[mask]))) if mask.any() else float("nan")
    # Pre-cost signal check (NOT a backtest): mean forward return taking the sign
    # of the prediction each day. Long-only = only when pred>0.
    ls = float(np.mean(np.sign(p) * a))
    lo = float(np.mean(a[p > 0])) if (p > 0).any() else float("nan")
    return {
        "rmse": rmse,
        "rmse_ratio_vs_naive": float(rmse / naive_rmse) if naive_rmse else float("nan"),
        "ic_spearman": ic,
        "hit_rate": hit,
        "signal_mean_ret_longshort": ls,
        "signal_mean_ret_longonly": lo,
    }


def _clf_metrics(prob: pd.Series, y_up: pd.Series, base_rate: float) -> dict:
    p = np.clip(prob.to_numpy(), 1e-6, 1 - 1e-6)
    yt = y_up.to_numpy().astype(int)
    acc = float(np.mean((p >= 0.5).astype(int) == yt))
    try:
        auc = float(roc_auc_score(yt, p)) if len(np.unique(yt)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    ll = float(log_loss(yt, p, labels=[0, 1]))
    majority = max(base_rate, 1 - base_rate)
    return {
        "accuracy": acc,
        "accuracy_vs_baserate": float(acc - majority),
        "auc": auc,
        "log_loss": ll,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _evaluate(cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Run the full embargoed walk-forward for cfg's horizon. Pure computation —
    returns (summary, oos_predictions) and persists nothing, so the horizon
    sweep can reuse it without clobbering the primary report."""
    N = cfg["label"]["horizon_days"]
    wf = cfg["model"]["walk_forward"]
    embargo = wf["embargo_days"] if wf["embargo_days"] is not None else N

    X, y_ret, _ = build_features(cfg)
    X = X.sort_index()
    y_ret = y_ret.reindex(X.index)
    y_up = (y_ret > 0).astype(int)

    folds = walk_forward_folds(
        len(X), min_train=wf["min_train_days"], test_days=wf["test_days"],
        embargo=embargo, scheme=wf["scheme"],
    )
    if not folds:
        raise RuntimeError(
            f"No walk-forward folds: {len(X)} rows < min_train ({wf['min_train_days']}) "
            f"+ embargo ({embargo}) + test ({wf['test_days']}). Lower min_train/test_days."
        )
    oos_index = np.concatenate([f.test_pos for f in folds])
    oos_dates = X.index[oos_index]

    # --- run both heads on the identical folds ---
    reg_pred = _run_head(X, y_ret, folds, build_regressors(cfg))
    clf_pred = _run_head(X, y_up, folds, build_classifiers(cfg))

    y_ret_oos = y_ret.loc[oos_dates]
    y_up_oos = y_up.loc[oos_dates]
    base_rate = float(y_up_oos.mean())
    naive_rmse = float(np.sqrt(np.mean(reg_pred["naive_zero"].sub(y_ret_oos) ** 2)))

    reg_metrics = {m: _reg_metrics(reg_pred[m], y_ret_oos, naive_rmse) for m in reg_pred.columns}
    clf_metrics = {m: _clf_metrics(clf_pred[m], y_up_oos, base_rate) for m in clf_pred.columns}

    summary = {
        "horizon_days": N,
        "embargo_days": embargo,
        "scheme": wf["scheme"],
        "n_folds": len(folds),
        "n_oos_rows": int(len(oos_dates)),
        "oos_span": [str(oos_dates.min().date()), str(oos_dates.max().date())],
        "train_span_first_fold": [str(X.index[folds[0].train_pos[0]].date()),
                                  str(X.index[folds[0].train_pos[-1]].date())],
        "oos_up_rate": base_rate,
        "naive_rmse": naive_rmse,
        "regression": reg_metrics,
        "classification": clf_metrics,
    }
    preds = pd.DataFrame({"fwd_ret_actual": y_ret_oos, "up_actual": y_up_oos})
    preds = preds.join(reg_pred.add_prefix("reg_")).join(clf_pred.add_prefix("clf_"))
    return summary, preds


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    ensure_dirs()
    summary, preds = _evaluate(cfg)

    # --- persist ---
    out = REPORTS / "model"
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PROCESSED / "oos_predictions.csv")
    with open(out / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    (out / "report.md").write_text(_render_report(summary))

    _print_console(summary)
    return summary


def _fmt(x: float, nd: int = 4) -> str:
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _render_report(s: dict) -> str:
    L = [
        "# Model Report — Stage 3 (walk-forward, out-of-sample)",
        "",
        f"_Forward horizon **{s['horizon_days']}d** · {s['scheme']} walk-forward · "
        f"{s['n_folds']} folds · embargo **{s['embargo_days']}d** (purged train/test overlap)_",
        "",
        f"- Out-of-sample rows: **{s['n_oos_rows']}**, span {s['oos_span'][0]} → {s['oos_span'][1]}",
        f"- First-fold train span: {s['train_span_first_fold'][0]} → {s['train_span_first_fold'][1]}",
        f"- OOS up-rate (base rate): **{s['oos_up_rate']:.3f}**",
        f"- Naive-benchmark RMSE (predict 0): **{_fmt(s['naive_rmse'], 5)}**",
        "",
        "> Not a backtest. No costs, sizing, or stops (stage 4). The two "
        "`signal_mean_ret_*` columns are pre-cost sanity checks on the forecast "
        "sign, not P&L.",
        "",
        "## Regression head — forecasting the forward log return",
        "",
        "Beat naive ⇒ **RMSE ratio < 1** and/or **IC > 0** with a hit-rate > 0.5.",
        "",
        "| model | RMSE | RMSE/naive | IC (Spearman) | hit-rate | mean ret L/S (pre-cost) | mean ret L-only |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, d in s["regression"].items():
        L.append(f"| {m} | {_fmt(d['rmse'],5)} | {_fmt(d['rmse_ratio_vs_naive'],3)} | "
                 f"{_fmt(d['ic_spearman'])} | {_fmt(d['hit_rate'],3)} | "
                 f"{_fmt(d['signal_mean_ret_longshort'],5)} | {_fmt(d['signal_mean_ret_longonly'],5)} |")
    L += [
        "",
        "## Classification head — forecasting direction (P up)",
        "",
        f"Beat naive ⇒ **accuracy > base rate ({max(s['oos_up_rate'], 1-s['oos_up_rate']):.3f})** "
        "and **AUC > 0.5**.",
        "",
        "| model | accuracy | acc − base-rate | AUC | log-loss |",
        "|---|---|---|---|---|",
    ]
    for m, d in s["classification"].items():
        L.append(f"| {m} | {_fmt(d['accuracy'],3)} | {_fmt(d['accuracy_vs_baserate'],3)} | "
                 f"{_fmt(d['auc'],3)} | {_fmt(d['log_loss'],4)} |")
    L += [
        "",
        "### How to read this",
        "- **IC ~ 0 and RMSE ratio ~ 1** on daily gold is the *expected, honest* "
        "outcome — short-horizon precious-metals returns are close to a random "
        "walk. The value of this stage is a credible, leakage-controlled ceiling, "
        "not a big number.",
        "- If XGBoost fails to clear naive while the linear/momentum baselines "
        "don't either, that is a finding to report, not a bug to tune away. "
        "Only escalate to the strategy layer (stage 4) on models that clear the bar.",
    ]
    return "\n".join(L) + "\n"


def _print_console(s: dict) -> None:
    print(f"\nWalk-forward: {s['n_folds']} folds, {s['n_oos_rows']} OOS rows "
          f"({s['oos_span'][0]} → {s['oos_span'][1]}), embargo {s['embargo_days']}d\n")
    print("REGRESSION           RMSE/naive     IC     hit")
    for m, d in s["regression"].items():
        print(f"  {m:<16} {_fmt(d['rmse_ratio_vs_naive'],3):>8} "
              f"{_fmt(d['ic_spearman']):>7} {_fmt(d['hit_rate'],3):>6}")
    print(f"\nCLASSIFICATION (base rate {max(s['oos_up_rate'],1-s['oos_up_rate']):.3f})"
          "   acc     AUC")
    for m, d in s["classification"].items():
        print(f"  {m:<16} {_fmt(d['accuracy'],3):>8} {_fmt(d['auc'],3):>7}")
    print(f"\nReport  -> {REPORTS / 'model' / 'report.md'}")
    print(f"Preds   -> {PROCESSED / 'oos_predictions.csv'}")


# --------------------------------------------------------------------------- #
# Horizon sweep — same harness, one row of headline metrics per horizon
# --------------------------------------------------------------------------- #
def sweep_horizons(horizons: list[int], cfg: dict | None = None) -> dict:
    """Re-run the full walk-forward per horizon. The embargo scales with N
    automatically (embargo_days: null => horizon), so longer labels get the
    wider purge they need. Writes reports/model/horizon_sweep.md; does NOT
    touch the primary report.md / oos_predictions.csv."""
    cfg = cfg or load_config()
    ensure_dirs()
    results: dict[int, dict] = {}
    for N in horizons:
        c = copy.deepcopy(cfg)
        c["label"]["horizon_days"] = N
        c["model"]["walk_forward"]["embargo_days"] = None   # follow the horizon
        s, _ = _evaluate(c)
        results[N] = s
        best_reg = min((d["rmse_ratio_vs_naive"], m) for m, d in s["regression"].items())
        print(f"[sweep] N={N:<3} folds={s['n_folds']:<3} "
              f"best RMSE/naive={best_reg[0]:.3f} ({best_reg[1]}), "
              f"ridge IC={s['regression']['ridge']['ic_spearman']:+.4f}, "
              f"logistic AUC={s['classification']['logistic']['auc']:.3f}")

    out = REPORTS / "model"
    out.mkdir(parents=True, exist_ok=True)
    (out / "horizon_sweep.md").write_text(_render_sweep(results))
    with open(out / "horizon_sweep.json", "w") as f:      # machine-readable, for plots
        json.dump(results, f, indent=2)
    print(f"\nSweep report -> {out / 'horizon_sweep.md'}")
    return results


def _render_sweep(results: dict[int, dict]) -> str:
    L = [
        "# Horizon Sweep — same embargoed walk-forward per label horizon",
        "",
        "_One full re-run per horizon; embargo = horizon in each. Baselines must "
        "beat naive before any horizon graduates to the strategy layer._",
        "",
        "**Caveat:** rows at horizon N overlap by N−1 days, so long-horizon IC/AUC "
        "are computed on fewer effectively-independent observations than the row "
        "count suggests — read the 20d line with proportionally wider error bars.",
        "",
        "## Regression head",
        "",
        "| N (days) | OOS rows | folds | best RMSE/naive (model) | ridge IC | ridge hit | xgb IC | xgb RMSE/naive |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for N, s in results.items():
        r = s["regression"]
        best_m = min(r, key=lambda m: r[m]["rmse_ratio_vs_naive"])
        L.append(
            f"| {N} | {s['n_oos_rows']} | {s['n_folds']} "
            f"| {_fmt(r[best_m]['rmse_ratio_vs_naive'],3)} ({best_m}) "
            f"| {_fmt(r['ridge']['ic_spearman'])} | {_fmt(r['ridge']['hit_rate'],3)} "
            f"| {_fmt(r['xgboost']['ic_spearman'])} | {_fmt(r['xgboost']['rmse_ratio_vs_naive'],3)} |")
    L += [
        "",
        "## Classification head",
        "",
        "| N (days) | base rate | logistic acc | logistic AUC | xgb acc | xgb AUC |",
        "|---|---|---|---|---|---|",
    ]
    for N, s in results.items():
        c = s["classification"]
        base = max(s["oos_up_rate"], 1 - s["oos_up_rate"])
        L.append(f"| {N} | {base:.3f} | {_fmt(c['logistic']['accuracy'],3)} "
                 f"| {_fmt(c['logistic']['auc'],3)} | {_fmt(c['xgboost']['accuracy'],3)} "
                 f"| {_fmt(c['xgboost']['auc'],3)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizons", type=int, nargs="+", default=None,
                    help="sweep these label horizons instead of the primary run")
    args = ap.parse_args()
    if args.horizons:
        sweep_horizons(args.horizons)
    else:
        run()
