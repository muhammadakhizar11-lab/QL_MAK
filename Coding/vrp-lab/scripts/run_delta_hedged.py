"""Run the delta-hedged option-selling strategy on real SPY/VIX history
(synthetic Black-Scholes chains — see data/options.py for the caveat).

Variants: ATM straddle vs 10-delta strangle, each with and without the
5d/2d put-spread tail overlay. Regime-gated entries throughout.

Writes: reports/delta_hedged_results.csv, data/processed/delta_hedged_history.csv,
reports/tearsheet_delta_hedged.html.

Run:  python -m scripts.run_delta_hedged
"""
from __future__ import annotations

import pandas as pd

from vrp_lab.analysis.vrp_study import build_dataset
from vrp_lab.config import PROCESSED, REPORTS, get_logger, load_config
from vrp_lab.risk.metrics import summary
from vrp_lab.risk.pnl_decomp import attribution_summary, decompose
from vrp_lab.signals.regime import exposure_multiplier
from vrp_lab.strategies.delta_hedged import DeltaHedgedConfig, run_delta_hedged
from vrp_lab.strategies.tail_overlay import run_tail_overlay
from vrp_lab.visualization.tearsheet import generate_tearsheet
from vrp_lab.volatility.vrp import iv_from_vix

log = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    df = build_dataset(save=False)
    spy = df["close"].dropna()          # unadjusted close = what options settle on
    iv = iv_from_vix(df["vix"]).dropna()
    rets = df["spy_ret"].dropna()
    reg = cfg["signals"]["regime"]
    gate = exposure_multiplier(
        df["vix"], rets,
        low_vol_max=reg["low_vol_max"], crisis_min=reg["crisis_min"],
        accel_ratio_max=reg["accel_ratio_max"],
    )

    variants = {
        "straddle": DeltaHedgedConfig(strangle_delta=None),
        "strangle_10d": DeltaHedgedConfig(strangle_delta=0.10),
    }
    rows = {}
    histories = {}
    for name, dcfg in variants.items():
        hist, trades = run_delta_hedged(spy, iv, rf_annual=df["ust10y"],
                                        gate=gate, cfg=dcfg)
        rets_v = hist["equity"].pct_change().dropna()
        stats = summary(rets_v, hist["equity"], rf_annual=df["ust10y"])
        stats["n_trades"] = len(trades)
        rows[name] = stats
        histories[name] = hist

        overlay = run_tail_overlay(spy, iv, hist["equity"])
        hedged_eq = (hist["equity"] + overlay["overlay_pnl"].cumsum()).dropna()
        stats_h = summary(hedged_eq.pct_change().dropna(), hedged_eq,
                          rf_annual=df["ust10y"])
        stats_h["n_trades"] = len(trades)
        rows[f"{name}_tail_hedged"] = stats_h

    results = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "delta_hedged_results.csv")
    print("\n=== Delta-hedged variants (SYNTHETIC options — see caveat) ===")
    print(results.round(3).to_string())

    main_hist = histories["straddle"]
    d = decompose(main_hist[main_hist["in_position"]])
    print("\n=== P&L attribution (straddle) ===")
    print(attribution_summary(d).round(0).to_string())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    main_hist.to_csv(PROCESSED / "delta_hedged_history.csv")
    generate_tearsheet(main_hist["equity"], name="delta_hedged",
                       rf_annual=df["ust10y"])


if __name__ == "__main__":
    main()
