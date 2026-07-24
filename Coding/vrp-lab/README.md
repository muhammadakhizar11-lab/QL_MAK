# vrp-lab

Measurement, decomposition, and conditional harvesting of the **Volatility
Risk Premium** — with the tail-risk accounting done honestly.

**Thesis.** Implied volatility systematically overestimates subsequently
realized volatility. Measured here on SPX/VIX (2006–2026): the 21-day
ex-post premium averages **+3.6 vol points** and is positive on **84% of
days** — with skew −3.6 and a worst observation of **−69 points**. It is an
insurance premium: a steady coupon punctuated by catastrophic claims. This
repo measures that premium without look-ahead, then tries to harvest it two
ways under strict survival constraints, and reports what actually survives
costs and crises.

## Results summary

### The premium is real (notebook 01)

| ex-post VRP, 21d, 2006–2026 | |
|---|---|
| mean / median | +3.6 / +4.5 vol pts |
| share positive | 84% |
| 5th percentile / minimum | −7.1 / −68.9 vol pts |

Fatter when VIX is elevated — and so is the left tail. The extra carry is
payment for spike risk, not an anomaly.

### Harvesting it is another matter (notebooks 03–05)

VIX futures carry, event-driven backtest on real CBOE settlements with
transaction costs and margin, 2013-05 → 2026-07 (free archive limit):

| | naive always-short | regime-filtered | filtered + vol-targeted |
|---|---|---|---|
| ann. return | **−100% (ruin, 2018-02-05)** | −5.1% | +2.0% |
| ann. vol | 55% | 53% | 10.9% |
| max drawdown | −110% (through zero) | −85% | **−23%** |
| CVaR 99% | −19.1% | −17.6% | **−3.7%** |
| worst day | −112% | −35% | −8.4% |
| margin-breach days | 2,120 | 0 | **0** |

- The naive book **replicates XIV's death** on Feb 5 2018 — that column is
  the reason the regime gate and vol targeting are not optional.
- The disciplined book survives Volmageddon (−3.1%), COVID (−4.2%) and the
  Aug-2024 JPY unwind (+0.2%, flat *before* the spike) with zero margin
  breaches — and earns roughly cash. Post-2013 front-month carry, net of
  honest costs and survival constraints, is thin. That is the finding, and
  it matches the post-2012 academic replications.

### The premium isolated in greeks (notebook 04)

Delta-hedged short straddles on **synthetic** BSM chains (real SPY paths,
real VIX as IV — model, not data; caveat in `docs/METHODOLOGY.md` §8):
over 2006–2026, **theta collected +$1.61M vs gamma paid −$1.07M** per $1M
equity — the theta−gamma spread *is* the harvested VRP. The 5Δ/2Δ put-spread
tail hedge costs Sharpe (1.39 → 0.44) and buys skew (−3.3 → −0.1): the
trade-off, quantified.

## Repository tour

```
src/vrp_lab/
├── data/           # CBOE, FRED (curl transport), Yahoo, options; validators (PIT guard)
├── volatility/     # 4 RV estimators · BS/IV toolkit · term structure · VRP (ex-post vs trailing)
├── signals/        # regime gate · roll/carry · trailing-VRP z-score · 25Δ skew
├── strategies/     # VIX carry · delta-hedged selling · tail overlay · sizing
├── backtest/       # event-driven engine (no-look-ahead MarketView) · costs · ledger · calendar
├── risk/           # full metrics table · greeks P&L attribution · stress replay
├── analysis/       # dataset assembly · estimator comparison · regime conditioning
└── visualization/  # charts · one-page HTML tearsheet
notebooks/          # 01 measurement · 02 estimators · 03 carry · 04 delta-hedged · 05 stress
scripts/            # run_vix_carry · run_delta_hedged · build_notebooks (re-derive everything)
docs/               # METHODOLOGY · DATA_SOURCES · INTERVIEW_TALKING_POINTS
```

## Reproduce

```bash
pip install -e ".[dev]"
pytest                          # 112 hermetic tests, no network
python -m scripts.run_vix_carry       # real-data backtest -> reports/ + tearsheet
python -m scripts.run_delta_hedged    # synthetic-options strategy -> reports/
python -m scripts.build_notebooks     # regenerate + execute notebooks/
```

First run downloads and caches all data (CBOE VIX + ~160 VX contract files,
SPY, FRED); everything after is offline. No API keys.

## Design principles

- **No look-ahead, structurally.** Ex-post VRP (future RV) and trailing VRP
  (tradeable) are different functions with different names; the engine's
  `MarketView` makes future rows unreachable and a runtime guard raises on
  leaks. Tested, not asserted.
- **Event-driven backtests only.** Vectorized backtests hide look-ahead bugs.
- **Never short vol unconditionally.** Asymmetric regime gate: calm to enter
  (VIX < 25, vol not accelerating), genuine crisis to exit (VIX > 30, deep
  backwardation). Vol-targeted sizing shrinks positions when vol rises.
- **Report the distribution, not the Sharpe.** CVaR 95/99, worst day, skew,
  kurtosis, drawdown duration, margin breaches — a short-vol Sharpe without
  its left tail is marketing.
- **Honest costs, honest data.** ~23bp/side on VX; validated data with a
  point-in-time guard; the naive-ruin column stays in the table.

## Limitations (short list; full list in docs/METHODOLOGY.md)

Free VX archive starts mid-2013 (misses the fat 2006–2012 contango era);
Phase-2 options are synthetic (static smile — no skew dynamics); daily
frequency (gap risk between hedges is the residual risk, discussed in
notebook 05); flat 30% margin approximation; single underlying.
