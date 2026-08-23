# Gold Forecasting — Stages 1–6 (Data → Features → Models → Backtest → Figures → Note)

Research/portfolio project: forecast short-horizon gold returns from free public
data, then turn that into a backtested systematic strategy.
**This is a research demonstration, not a live trading system.**

This repo covers all six stages: **1** data + validation + roll handling,
**2** feature engineering + leakage audit, **3** baselines + embargoed
walk-forward modelling, **4** strategy layer + backtest, **5** figures, and
**6** the research note — **[`reports/research_note.md`](reports/research_note.md)**
is the finished write-up.

## Layout

```
gold-forecast/
├── config.yaml            # tickers, dates, model/strategy params — single source of truth
├── requirements.txt
├── src/
│   ├── config.py          # config loader + project paths
│   ├── data_pull.py       # yfinance (market) + free FRED CSV (macro) -> data/raw/ + manifest
│   ├── validate.py        # gaps, stale runs, outliers, bad prints -> reports/data_quality/
│   ├── rolls.py           # futures-roll detection + return neutralization (honest limits)
│   ├── features.py        # feature matrix X, forward-return label y, FEATURE_SPEC, leakage audit
│   ├── models.py          # model zoo: naive/momentum/ridge/logistic/XGBoost, two heads
│   ├── evaluate.py        # embargoed walk-forward -> reports/model/ (+ --horizons sweep)
│   ├── strategy.py        # position engine: vol targeting, drawdown stop, turnover costs
│   ├── backtest.py        # OOS signals vs benchmarks -> reports/backtest/
│   └── plots.py           # publication figures -> reports/figures/
├── scripts/
│   ├── demo_synthetic.py  # end-to-end smoke test on generated data (no network needed)
│   └── fetch_fomc_dates.py # scrape scheduled FOMC decision dates from federalreserve.gov
├── tests/                 # hermetic unit tests (rolls, leakage, embargo, strategy timing)
├── data/raw/  data/processed/   # gitignored; populated locally
└── reports/               # data_quality/, model/, backtest/ — all generated
```

## Run it

```bash
pip install -r requirements.txt
# macOS: xgboost needs OpenMP -> `brew install libomp` (no brew? see requirements.txt)

# 1. Pull data (needs internet — Yahoo + FRED) + the FOMC calendar
python -m src.data_pull
python -m scripts.fetch_fomc_dates

# 2. Validate — writes reports/data_quality/report.md
python -m src.validate

# 3. Build features + label (prints X/y shapes)
python -m src.features

# 4. Walk-forward model evaluation — writes reports/model/report.md
python -m src.evaluate
python -m src.evaluate --horizons 1 5 10 20   # label-horizon sweep

# 5. Strategy layer + backtest — writes reports/backtest/report.md
python -m src.backtest

# 6. Figures for the research note — writes reports/figures/*.png
python -m src.plots

# Verify the pipeline logic WITHOUT real data (generates synthetic faults):
python -m scripts.demo_synthetic   # NOTE: overwrites data/raw/ — re-pull after

# Unit tests:
pytest -q
```

`config.yaml` drives everything — tickers, thresholds, the label horizon
(`label.horizon_days`, primary = 1 after the sweep below), model and strategy
parameters.

## Why these features (the economics)

Gold is not an industrial-cycle metal — it pays no yield and has small industrial
demand relative to investment/jewellery/central-bank demand. The feature set
reflects that:

- **Real yields (`DFII10`) — the textbook driver.** Gold's opportunity cost is
  the real return on the risk-free alternative. Level, 5-day change, and gold's
  rolling beta to yield changes are all features.
- **Inflation expectations (`T10YIE`) and realized CPI (`CPIAUCSL`)** — the
  inflation-hedge demand channel (fast and slow versions).
- **USD (`DXY`) rolling correlation** — gold is priced in USD.
- **Gold/silver ratio z-score (`SI=F`)** — the classic precious-metals
  cross-signal, mean-reversion-ish.
- **VIX 5d change** — safe-haven / risk-off flow proxy.
- **Gold miners (`GDX`) 5d return** — leveraged sentiment/flow proxy.
- **FOMC-day flag** — monetary policy days matter more for gold than for almost
  any other commodity. Not hard-coded: `scripts/fetch_fomc_dates.py` scrapes the
  Fed's own calendar pages into `data/raw/fomc_dates.csv` (decision days only;
  unscheduled/cancelled meetings excluded, since the feature is specced as
  known-in-advance). Absent that file, the flag is all-zero with a warning.

Every feature's as-of lag is documented in `features.FEATURE_SPEC`.

## Design decisions that matter (the honest bits)

**As-of convention.** A row dated `t` holds features known at the **close of day
t**; the label is the forward log return `log(close_{t+N}/close_t)`, strictly
future. `features.audit_leakage()` empirically verifies no feature reacts to a
future price spike. The synthetic demo and unit tests both exercise it.

**Point-in-time macro (the main leakage trap).** Monthly CPI is dated to the 1st
of the month but released ~mid-following-month. We shift it forward by
`(release_lag_months + 1)` months before forward-filling to daily — conservative
by construction, guarantees no look-ahead. Exact point-in-time would need ALFRED
vintages (documented limitation). Daily FRED series (DFII10, T10YIE) are shifted
one business day for their next-day H.15 release.

**Futures rolls.** `GC=F` is a continuous front-month ticker that Yahoo does NOT
back-adjust. Gold futures roll through the active months (Feb/Apr/Jun/Aug/Oct/Dec),
injecting basis jumps at rolls. With only a single continuous series (no free
per-contract prices) true Panama back-adjustment isn't honestly possible. Default
treatment: detect roll candidates (extreme returns clustered at month boundaries)
and **neutralize** those returns (set to NaN) so they never enter a feature window
or label. A `ratio_backadjust` path exists for when/if you source a real roll
schedule. Gold's roll basis is typically small (deep liquidity, well-behaved carry-driven
contango) — expect few candidates flagged.

## Known free-data limitations (carry into the write-up)

- **`GC=F` is deep and liquid** — unlike thin tickers, data adequacy should not
  be a problem, but the validation report verifies rather than assumes.
- Free monthly macro without vintages → the conservative release-lag shift above.
- No intraday, no order book, no term structure, no positioning data (CFTC COT is
  free and weekly — a good later addition), no ETF flow data — daily close only.
- Central-bank gold purchases (a major structural driver since 2022) are
  quarterly, lagged, and revised — write-up context, not a model feature.

## Structural context (for the research note, not the model)

- **Real-rate regime**: gold's 2022–2024 resilience despite sharply positive real
  yields broke the textbook relationship — widely attributed to official-sector
  (central bank) buying. Worth a dedicated paragraph.
- **Central-bank accumulation**: sustained record purchases since 2022,
  concentrated in EM central banks — a structural demand shift, quarterly data,
  not tradeable at daily frequency.
- **Shariah angle**: AAOIFI's Shariah Standard No. 57 on Gold (2016) directly
  governs gold investment — spot/physical-backed instruments (e.g. IGLN-style
  ETCs) vs deferred-settlement futures raise distinct riba/qabd (possession)
  questions. A dedicated section fits naturally in the write-up.

## Stage 3 — what the models actually found

All evaluation is **walk-forward with an embargo**: expanding train window,
refit per fold, and the `horizon` rows straddling each train/test boundary are
purged so no overlapping forward label leaks across (López de Prado purging).
Two heads run on identical folds: regression on the forward log return, and
classification on its sign. The README's original bar stands: *baselines must
beat a naive benchmark before the ML results are taken seriously.*

Findings on real data (54 folds, ~3,390 OOS rows, 2013→2026 —
`reports/model/`):

- **The signal lives at the 1-day horizon and decays monotonically with N**
  (`--horizons 1 5 10 20` sweep). At N=1, ridge beats naive RMSE (ratio 0.997)
  with Spearman IC +0.108, and logistic beats the base rate (acc 0.543 vs
  0.521, AUC 0.545). At N=20 nothing beats naive. Consistent with a feature set
  of fast daily signals. N=1 is therefore the primary label.
- **The regularized linear models carry the signal; XGBoost does not.** XGBoost
  underperforms naive on RMSE at every horizon — the classic
  overfit-on-low-signal-data pattern. Reported as a finding, not tuned away.
- Magnitudes are honest: IC ~0.11 on daily gold is a thin edge, far from
  precise return prediction. Grinold gives IC×√252 ≈ 1.7 as the gross IR
  ceiling for such a signal — which is exactly the neighbourhood the stage-4
  backtest lands in.

## Stage 4 — strategy layer + backtest

`src/strategy.py` turns the **stitched walk-forward OOS forecasts** (the only
forecasts a live trader could have had) into positions: long/flat on the signal
sign → vol targeting (trailing window, leverage cap) → trailing drawdown stop
with cooldown → costs charged per unit turnover. Benchmarks run through the
*identical* machinery so differences are signal, not plumbing.

Headline (2013→2026, 1bp one-way costs, 10% vol target — `reports/backtest/`):
ridge nets ~10%/yr at Sharpe ~1.2 vs ~0.4 for both vol-targeted long gold and
raw buy-and-hold, with a breakeven cost of ~22bp — ~20× the realistic GC
friction, so the result is not an artifact of the cost assumption. Caveats that
belong in the write-up: idealized daily-close fills, flat bps costs, no
slippage/financing, and a single 13-year sample of one asset.

## Stages 5–6 — figures + the research note

`src/plots.py` renders the figures (equity curves, drawdowns, rolling IC,
horizon sweep, gold-vs-real-yields regime chart) from the generated artifacts —
nothing is recomputed at plot time. **The finished write-up is
[`reports/research_note.md`](reports/research_note.md)**: method, findings,
strategy results, the structural-context sections above (real-rate regime
break, central-bank buying, the AAOIFI Shariah Standard 57 angle), and the
full limitations list.
