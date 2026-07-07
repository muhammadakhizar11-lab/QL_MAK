# Horizon Sweep — same embargoed walk-forward per label horizon

_One full re-run per horizon; embargo = horizon in each. Baselines must beat naive before any horizon graduates to the strategy layer._

**Caveat:** rows at horizon N overlap by N−1 days, so long-horizon IC/AUC are computed on fewer effectively-independent observations than the row count suggests — read the 20d line with proportionally wider error bars.

## Regression head

| N (days) | OOS rows | folds | best RMSE/naive (model) | ridge IC | ridge hit | xgb IC | xgb RMSE/naive |
|---|---|---|---|---|---|---|---|
| 1 | 3391 | 54 | 0.997 (ridge) | 0.1076 | 0.537 | 0.0663 | 1.017 |
| 5 | 3383 | 54 | 0.999 (momentum) | 0.0922 | 0.544 | 0.0543 | 1.061 |
| 10 | 3373 | 54 | 1.000 (momentum) | 0.0642 | 0.536 | 0.0676 | 1.121 |
| 20 | 3353 | 54 | 1.000 (naive_zero) | 0.0623 | 0.560 | 0.0683 | 1.118 |

## Classification head

| N (days) | base rate | logistic acc | logistic AUC | xgb acc | xgb AUC |
|---|---|---|---|---|---|
| 1 | 0.521 | 0.543 | 0.545 | 0.531 | 0.533 |
| 5 | 0.543 | 0.538 | 0.526 | 0.527 | 0.520 |
| 10 | 0.539 | 0.526 | 0.519 | 0.534 | 0.522 |
| 20 | 0.544 | 0.545 | 0.517 | 0.546 | 0.523 |
