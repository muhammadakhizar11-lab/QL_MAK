# Model Report — Stage 3 (walk-forward, out-of-sample)

_Forward horizon **1d** · expanding walk-forward · 54 folds · embargo **1d** (purged train/test overlap)_

- Out-of-sample rows: **3391**, span 2013-01-07 → 2026-07-02
- First-fold train span: 2010-01-04 → 2013-01-03
- OOS up-rate (base rate): **0.521**
- Naive-benchmark RMSE (predict 0): **0.01067**

> Not a backtest. No costs, sizing, or stops (stage 4). The two `signal_mean_ret_*` columns are pre-cost sanity checks on the forecast sign, not P&L.

## Regression head — forecasting the forward log return

Beat naive ⇒ **RMSE ratio < 1** and/or **IC > 0** with a hit-rate > 0.5.

| model | RMSE | RMSE/naive | IC (Spearman) | hit-rate | mean ret L/S (pre-cost) | mean ret L-only |
|---|---|---|---|---|---|---|
| naive_zero | 0.01067 | 1.000 | nan | nan | 0.00000 | nan |
| momentum | 0.01067 | 0.999 | 0.0136 | 0.513 | 0.00017 | 0.00034 |
| ridge | 0.01064 | 0.997 | 0.1076 | 0.537 | 0.00090 | 0.00098 |
| xgboost | 0.01085 | 1.017 | 0.0663 | 0.531 | 0.00060 | 0.00074 |

## Classification head — forecasting direction (P up)

Beat naive ⇒ **accuracy > base rate (0.521)** and **AUC > 0.5**.

| model | accuracy | acc − base-rate | AUC | log-loss |
|---|---|---|---|---|
| base_rate | 0.521 | 0.000 | 0.489 | 0.6932 |
| momentum | 0.521 | 0.000 | 0.489 | 0.6933 |
| logistic | 0.543 | 0.023 | 0.545 | 0.6946 |
| xgboost | 0.531 | 0.010 | 0.533 | 0.7044 |

### How to read this
- **IC ~ 0 and RMSE ratio ~ 1** on daily gold is the *expected, honest* outcome — short-horizon precious-metals returns are close to a random walk. The value of this stage is a credible, leakage-controlled ceiling, not a big number.
- If XGBoost fails to clear naive while the linear/momentum baselines don't either, that is a finding to report, not a bug to tune away. Only escalate to the strategy layer (stage 4) on models that clear the bar.
