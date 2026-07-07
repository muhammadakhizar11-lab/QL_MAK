# Backtest Report — Stage 4 (OOS walk-forward signals through the strategy layer)

_Span 2013-01-07 → 2026-07-02 · mode **long_flat** · vol target **0.1** (window 20, max lever 2.0) · cost **1.0bp** one-way · stop {'max_drawdown': 0.1, 'cooldown_days': 5}_

> Idealized daily-close backtest: flat bps costs, no slippage/financing/intraday. All rows except `gold_buy_hold_raw` run through the identical position machinery, so differences are signal, not plumbing. `b/e cost bp` = the one-way cost at which the strategy's net return reaches zero — the distance between the edge and the friction.

| strategy | net ret/yr | gross ret/yr | vol/yr | Sharpe | max DD | hit (active) | avg |pos| | turnover/yr | cost drag/yr | b/e cost bp |
|---|---|---|---|---|---|---|---|---|---|---|
| reg_ridge | 0.0999 | 0.1047 | 0.0861 | 1.16 | 0.225 | 0.548 | 0.46 | 48.4 | 0.0048 | 21.6 |
| clf_logistic | 0.0767 | 0.0817 | 0.0913 | 0.84 | 0.226 | 0.549 | 0.48 | 50.3 | 0.0050 | 16.2 |
| long_always | 0.0453 | 0.0462 | 0.1141 | 0.40 | 0.360 | 0.521 | 0.75 | 9.3 | 0.0009 | 50.0 |
| momentum_20d | 0.0052 | 0.0076 | 0.0875 | 0.06 | 0.405 | 0.512 | 0.40 | 24.1 | 0.0024 | 3.2 |
| gold_buy_hold_raw | 0.0690 | 0.0690 | 0.1694 | 0.41 | 0.477 | 0.523 | 1.00 | 0.1 | 0.0000 | 9284.8 |

### How to read this
- The bar: a signal strategy must beat **both** benchmarks (`long_always`, `momentum_20d`) net of costs, on Sharpe and drawdown, before it counts.
- `gold_buy_hold_raw` is unlevered long gold — the do-nothing reference. During a strong gold bull market it is a hard bar; say so rather than hiding behind the vol-targeted comparison.
- Daily signals trade a lot. If the b/e cost is only a few bp above the assumed cost, the edge is fragile to execution quality — report it, don't tune it away.
