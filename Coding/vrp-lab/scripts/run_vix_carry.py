"""Run the VIX carry backtest on real CBOE data (2013+) in three variants:
naive always-short, regime-filtered, and regime-filtered + vol-targeted.

Writes: reports/vix_carry_results.csv, reports/stress_report.csv,
reports/tearsheet_vix_carry.html, data/processed/equity_curves.csv.

Run:  python -m scripts.run_vix_carry
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vrp_lab.analysis.vrp_study import build_dataset
from vrp_lab.backtest.engine import run_backtest
from vrp_lab.config import PROCESSED, REPORTS, get_logger, load_config
from vrp_lab.data import cboe
from vrp_lab.risk.metrics import summary
from vrp_lab.risk.stress import stress_report, survived
from vrp_lab.strategies.vix_carry import VIXCarryStrategy, build_vx_price_table
from vrp_lab.visualization.tearsheet import generate_tearsheet
from vrp_lab.volatility.term_structure import build_curve

log = get_logger(__name__)

VARIANTS = {
    # truly always short: no exits of any kind, always in, 1x notional
    "naive_always_short": dict(use_regime_gate=False, use_vol_target=False,
                               contango_entry_threshold=-1e9,
                               backwardation_exit_threshold=-1e9,
                               vix_circuit_breaker=1e9,
                               max_holding_days=10**6),
    "regime_filtered": dict(use_regime_gate=True, use_vol_target=False),
    "filtered_vol_targeted": dict(use_regime_gate=True, use_vol_target=True),
}


def main() -> None:
    cfg = load_config()
    df = build_dataset(save=False)

    vx = cboe.pull_vx_curve("2013-01-01")
    vix = cboe.pull_vix_history()["close"]
    curve = build_curve(vx, vix)
    prices = build_vx_price_table(vx)
    # engine trades on curve dates only; align price table to them
    prices = prices.reindex(curve.index)
    spy_returns = df["spy_ret"].dropna()
    daily_rates = (df["ust10y"].reindex(prices.index).ffill() / 100 / 252)

    sig_cfg = cfg["signals"]["vix_roll"]
    reg_cfg = cfg["signals"]["regime"]
    strat_cfg = cfg["strategy"]
    bt_cfg = cfg["backtest"]

    curves, rows = {}, {}
    for name, overrides in VARIANTS.items():
        params = dict(
            contango_entry_threshold=sig_cfg["contango_entry_threshold"],
            backwardation_exit_threshold=sig_cfg["backwardation_exit_threshold"],
            vix_circuit_breaker=sig_cfg["vix_circuit_breaker"],
            max_holding_days=sig_cfg["max_holding_days"],
            roll_days_before_expiry=bt_cfg["roll_days_before_expiry"],
            low_vol_max=reg_cfg["low_vol_max"],
            crisis_min=reg_cfg["crisis_min"],
            accel_ratio_max=reg_cfg["accel_ratio_max"],
            vol_target_annual=strat_cfg["vol_target_annual"],
            max_notional_mult=strat_cfg["max_notional_mult"],
        )
        params.update(overrides)
        strat = VIXCarryStrategy(**params)
        eq, trades = run_backtest(
            strat, prices,
            aux={"curve": curve, "spy_returns": spy_returns},
            initial_capital=bt_cfg["initial_capital"],
            daily_rates=daily_rates,
            margin_frac=bt_cfg["margin"]["vx_margin_frac"],
        )
        rets = eq["equity"].pct_change().dropna()
        stats = summary(rets, eq["equity"], rf_annual=df["ust10y"])
        stats["n_trades"] = len(trades)
        stats["margin_breaches"] = int(eq["margin_breach"].sum())
        rows[name] = stats
        curves[name] = eq
        log.info("%-24s sharpe=%.2f mdd=%.1f%% cvar99=%.2f%% trades=%d",
                 name, stats["sharpe"], 100 * stats["max_drawdown"],
                 100 * stats["cvar_99"], len(trades))

    results = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "vix_carry_results.csv")
    print("\n=== VIX carry variants ===")
    print(results.round(3).to_string())

    main_eq = curves["filtered_vol_targeted"]
    stress = stress_report(main_eq)
    stress.to_csv(REPORTS / "stress_report.csv")
    print("\n=== Stress events (filtered + vol-targeted) ===")
    print(stress.round(3).to_string())
    print("\nSurvived all covered stress events:", survived(stress))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({k: v["equity"] for k, v in curves.items()}).to_csv(
        PROCESSED / "equity_curves.csv")
    generate_tearsheet(main_eq["equity"], name="vix_carry",
                       rf_annual=df["ust10y"])


if __name__ == "__main__":
    main()
