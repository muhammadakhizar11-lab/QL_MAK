# Gold Forecasting — Stages 1–2 (Data + Features)

Research/portfolio project: forecast short-horizon gold returns from free public
data, then (later stages) turn that into a backtested systematic strategy.
**This is a research demonstration, not a live trading system.**

This repo currently covers **stage 1 (data + validation + roll handling)** and
**stage 2 (feature engineering + leakage audit)** only. Modelling and backtesting
(stages 3–6) come next and are intentionally not here yet.

## Layout

```
gold-forecast/
├── config.yaml            # tickers, dates, thresholds, feature params — single source of truth
├── requirements.txt
├── src/
│   ├── config.py          # config loader + project paths
│   ├── data_pull.py       # yfinance (market) + free FRED CSV (macro) -> data/raw/ + manifest
│   ├── validate.py        # gaps, stale runs, outliers, bad prints -> reports/data_quality/
│   ├── rolls.py           # futures-roll detection + return neutralization (honest limits)
│   └── features.py        # feature matrix X, forward-return label y, FEATURE_SPEC, leakage audit
├── scripts/
│   └── demo_synthetic.py  # end-to-end smoke test on generated data (no network needed)
├── tests/
│   └── test_logic.py      # hermetic unit tests (roll detection, point-in-time macro, etc.)
├── data/raw/  data/processed/   # gitignored; populated locally
└── reports/data_quality/        # generated validation report
```

## Run it

```bash
pip install -r requirements.txt

# 1. Pull data (needs internet — Yahoo + FRED)
python -m src.data_pull

# 2. Validate — writes reports/data_quality/report.md
python -m src.validate

# 3. Build features + label (prints X/y shapes)
python -m src.features

# Verify the pipeline logic WITHOUT real data (generates synthetic faults):
python -m scripts.demo_synthetic

# Unit tests:
pytest -q
```

`config.yaml` drives everything — change tickers, thresholds, the label horizon
(`label.horizon_days`, default 5), or feature windows there.

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
  any other commodity. Not hard-coded: drop a one-column
  `data/raw/fomc_dates.csv` (`date` header, from federalreserve.gov) and it's
  picked up; absent that, all-zero with a warning.

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

## Next (stages 3–6, not built yet)

Baselines (random walk / momentum / logistic) → walk-forward gradient boosting →
strategy layer (vol targeting, costs, stops) → backtest vs buy-and-hold and
momentum → plots + research note. Baselines must beat a naive benchmark before the
ML results are taken seriously.
