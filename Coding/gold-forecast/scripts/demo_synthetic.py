"""
demo_synthetic.py — prove the stage 1-2 logic works WITHOUT real data.

Generates synthetic market + macro series (same on-disk format as the real pull),
deliberately injecting:
  * a roll-like jump near a month boundary in gold,
  * a stale run of identical closes,
into data/raw/, then runs:
  validate -> roll detection -> feature build -> leakage audit
and asserts each check catches what it should. This is a smoke test for the
pipeline mechanics; it is NOT a claim about real gold data.

Run:  python -m scripts.demo_synthetic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RAW, ensure_dirs, load_config
from src import validate, rolls, features

RNG = np.random.default_rng(42)


def _gbm(index, s0=100.0, mu=0.0, sigma=0.015):
    steps = RNG.normal(mu, sigma, len(index))
    return pd.Series(s0 * np.exp(np.cumsum(steps)), index=index)


def _write_yahoo(alias, close):
    df = pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": RNG.integers(1e3, 1e5, len(close)),
    })
    df.index.name = "date"
    df.to_csv(RAW / f"{alias}.csv")


def _write_fred(alias, series):
    series.index.name = "date"
    series.to_frame(alias).to_csv(RAW / f"{alias}.csv")


def build_synthetic():
    ensure_dirs()
    bdays = pd.bdate_range("2016-01-01", "2024-12-31")

    gold = _gbm(bdays, 1800, mu=0.0002, sigma=0.010)

    # --- inject a roll-like jump near a month boundary ---
    # pick last business day of 2020-06 and bump the level +9%
    june_end = bdays[(bdays.year == 2020) & (bdays.month == 6)][-1]
    gold.loc[gold.index >= june_end] *= 1.09    # persistent level shift = fake roll
    injected_roll = june_end

    # --- inject a stale run (5 identical closes) ---
    stale_start_pos = bdays.get_loc(pd.Timestamp("2019-03-04"))
    stale_val = gold.iloc[stale_start_pos]
    gold.iloc[stale_start_pos:stale_start_pos + 5] = stale_val
    injected_stale = str(bdays[stale_start_pos].date())

    # --- inject a mid-month BAD PRINT (one-day +25% spike that reverts) ---
    # This is what the outlier check is for: a data error, not a roll. Mid-month
    # so the roll detector's month-boundary gate does NOT fire on it.
    badprint_date = pd.Timestamp("2021-09-15")
    gold.loc[badprint_date] *= 1.25
    injected_badprint = str(badprint_date.date())

    _write_yahoo("gold_comex", gold)
    # cross-asset — correlated-ish but independent enough
    _write_yahoo("silver_comex", _gbm(bdays, 22, sigma=0.018))
    _write_yahoo("dxy", _gbm(bdays, 95, sigma=0.004))
    _write_yahoo("vix", _gbm(bdays, 18, sigma=0.05))
    _write_yahoo("gold_miners", _gbm(bdays, 30, sigma=0.02))

    # FRED: daily real yield + daily breakeven + monthly CPI
    months = pd.date_range("2016-01-01", "2024-12-31", freq="MS")
    _write_fred("real_10y", pd.Series(RNG.normal(0.5, 0.05, len(bdays)).cumsum() * 0.01, index=bdays))
    _write_fred("infl_breakeven", pd.Series(2.0 + RNG.normal(0, 0.02, len(bdays)).cumsum(), index=bdays))
    _write_fred("us_cpi", pd.Series(240 * np.exp(np.linspace(0, 0.25, len(months))) + RNG.normal(0, 0.3, len(months)), index=months))

    return {"roll": injected_roll, "stale": injected_stale, "badprint": injected_badprint}


def main():
    cfg = load_config()
    injected = build_synthetic()
    print(f"Injected roll jump : {injected['roll'].date()}")
    print(f"Injected stale run : {injected['stale']}")
    print(f"Injected bad print : {injected['badprint']}\n")

    ok = True

    # 1) validation catches the bad print + stale run on gold
    #    (the roll is handled by the roll detector below, not the outlier check)
    df = pd.read_csv(RAW / "gold_comex.csv", index_col=0, parse_dates=True)
    f = validate.validate_series("gold_comex", df, cfg["validation"])
    outlier_dates = set(str(d) for d in f["outliers"]["date"])
    stale_starts = set(r["start"] for r in f["stale_runs"])
    v_outlier = injected["badprint"] in outlier_dates
    v_stale = injected["stale"] in stale_starts
    print(f"[validate] bad print flagged as outlier : {v_outlier}")
    print(f"[validate] stale run flagged            : {v_stale}")
    ok &= v_outlier and v_stale

    # 2) roll detector flags the injected month-boundary jump
    close = df["close"].astype(float)
    cand = rolls.detect_roll_candidates(
        close, jump_z=cfg["rolls"]["jump_z"], jump_hard=cfg["rolls"]["jump_hard"],
        month_boundary_window=cfg["rolls"]["month_boundary_window"],
    )
    r_hit = injected["roll"] in cand
    print(f"[rolls]    roll candidate detected      : {r_hit} "
          f"({len(cand)} candidate(s) total)")
    ok &= r_hit

    # 3) neutralization removes that day's return
    clean, _ = rolls.clean_returns(close, method="neutralize",
                                   jump_z=cfg["rolls"]["jump_z"],
                                   jump_hard=cfg["rolls"]["jump_hard"],
                                   month_boundary_window=cfg["rolls"]["month_boundary_window"])
    r_neut = bool(np.isnan(clean.loc[injected["roll"]]))
    print(f"[rolls]    roll-day return neutralized  : {r_neut}")
    ok &= r_neut

    # 4) features build + shapes sane
    X, y, spec = features.build_features(cfg)
    build_ok = X.shape[0] == y.shape[0] and X.shape[0] > 0 and not X.columns.duplicated().any()
    print(f"[features] built X={X.shape}, y={y.shape}, specced cols={len(spec)} : {build_ok}")
    ok &= build_ok

    # 5) leakage audit passes (no feature reacts to a future spike)
    audit = features.audit_leakage(cfg)
    print(f"[leakage]  audit passed                 : {audit['passed']} "
          f"(t*={audit['t_star']}, offenders={audit['n_offenders']})")
    if not audit["passed"]:
        print("           offenders:", audit["offenders"])
    ok &= audit["passed"]

    print("\n" + ("ALL CHECKS PASSED ✅" if ok else "SOME CHECKS FAILED ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
