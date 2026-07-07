"""
models.py — the model zoo for stage 3, two heads that share one interface.

Every model implements the same tiny contract so the walk-forward harness in
`evaluate.py` can treat them uniformly:

    m.fit(X_train, y_train) -> None
    m.predict(X_test)       -> np.ndarray

  * REGRESSION head: `y` is the forward log return; `predict` returns a real
    return forecast (same units as y).
  * CLASSIFICATION head: `y` is the sign label {0,1} (1 = up over the horizon);
    `predict` returns P(up) in [0, 1]. Accuracy thresholds at 0.5; AUC uses the
    probability as a ranking score.

The point of the baselines is the README's bar: "Baselines must beat a naive
benchmark before the ML results are taken seriously." So the two momentum
baselines are deliberately just *one-feature* versions of the linear/logistic
models — if the full feature set and XGBoost can't beat pure trend, that is the
honest finding.

Leakage note: every fitted transform (imputer means, scaler moments, model
coefficients) is fit on the TRAIN slice only. The linear/logistic models wrap
imputation + scaling in an sklearn Pipeline so this holds automatically per fold;
XGBoost ingests NaNs natively (no imputation, no scaling needed).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


# --------------------------------------------------------------------------- #
# Regression head
# --------------------------------------------------------------------------- #
class NaiveZeroReg:
    """Random-walk benchmark: the best forecast of a forward return is 0.

    This is the bar every other model must clear. Its RMSE is the natural
    yardstick and its IC is 0 by construction (a constant has no rank
    information)."""

    name = "naive_zero"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveZeroReg":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class MomentumReg:
    """Trend-following baseline: forward return ~ trailing momentum.

    Calibrated, not hand-waved — we fit a 1-variable OLS of the forward return
    on a single trailing-momentum column on TRAIN, so it competes on the same
    RMSE footing as the richer models rather than emitting raw signal units."""

    name = "momentum"

    def __init__(self, signal_col: str):
        self.signal_col = signal_col
        self._ols = LinearRegression()

    def _sig(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.signal_col].to_numpy().reshape(-1, 1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MomentumReg":
        s = np.nan_to_num(self._sig(X), nan=0.0)
        self._ols.fit(s, y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        s = np.nan_to_num(self._sig(X), nan=0.0)
        return self._ols.predict(s)


def _linear_pipeline(alpha: float) -> Pipeline:
    # Impute (median) -> standardize -> Ridge. All fit on train only, per fold.
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


class RidgeReg:
    """Linear baseline on the full feature set (L2-regularized)."""

    name = "ridge"

    def __init__(self, alpha: float):
        self._pipe = _linear_pipeline(alpha)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeReg":
        self._pipe.fit(X, y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._pipe.predict(X)


class XGBReg:
    """Gradient-boosted trees. NaNs pass through natively (no imputation)."""

    name = "xgboost"

    def __init__(self, params: dict):
        self._m = XGBRegressor(objective="reg:squarederror", **params)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBReg":
        self._m.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._m.predict(X.to_numpy())


# --------------------------------------------------------------------------- #
# Classification head  (predict returns P(up) in [0, 1])
# --------------------------------------------------------------------------- #
class BaseRateClf:
    """Predict the training-set up-rate for everyone. The classification analogue
    of the naive benchmark: accuracy = majority class, AUC = 0.5 (no ranking)."""

    name = "base_rate"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseRateClf":
        self._p = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._p)


class MomentumClf:
    """Trend baseline for direction: 1-variable logistic of the up-label on a
    single trailing-momentum column. Symmetric with MomentumReg; gives a real
    probability (so AUC and log-loss are meaningful, not degenerate signs)."""

    name = "momentum"

    def __init__(self, signal_col: str):
        self.signal_col = signal_col
        self._lr = LogisticRegression()
        self._const: float | None = None  # fallback if train is single-class

    def _sig(self, X: pd.DataFrame) -> np.ndarray:
        return np.nan_to_num(X[self.signal_col].to_numpy().reshape(-1, 1), nan=0.0)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MomentumClf":
        if y.nunique() < 2:
            self._const = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        else:
            self._const = None
            self._lr.fit(self._sig(X), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._const is not None:
            return np.full(len(X), self._const)
        return self._lr.predict_proba(self._sig(X))[:, 1]


class LogisticClf:
    """Logistic regression on the full feature set (impute -> scale -> logit)."""

    name = "logistic"

    def __init__(self, C: float):
        self._pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=1000)),
        ])
        self._const: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticClf":
        if y.nunique() < 2:
            self._const = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        else:
            self._const = None
            self._pipe.fit(X, y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._const is not None:
            return np.full(len(X), self._const)
        return self._pipe.predict_proba(X)[:, 1]


class XGBClf:
    """Gradient-boosted classifier; returns P(up). NaNs pass through natively."""

    name = "xgboost"

    def __init__(self, params: dict):
        self._m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
        self._const: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBClf":
        if y.nunique() < 2:
            self._const = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        else:
            self._const = None
            self._m.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._const is not None:
            return np.full(len(X), self._const)
        return self._m.predict_proba(X.to_numpy())[:, 1]


# --------------------------------------------------------------------------- #
# Registries — built fresh per fold by the harness (fold-local fitted state)
# --------------------------------------------------------------------------- #
def build_regressors(cfg: dict) -> dict[str, Callable[[], object]]:
    """Return {display_name: factory} so each fold gets a fresh, unfitted model."""
    mc = cfg["model"]
    mom_col = mc["walk_forward"]["momentum_col"]
    return {
        "naive_zero": lambda: NaiveZeroReg(),
        "momentum": lambda: MomentumReg(mom_col),
        "ridge": lambda: RidgeReg(mc["ridge_alpha"]),
        "xgboost": lambda: XGBReg(mc["xgboost"]),
    }


def build_classifiers(cfg: dict) -> dict[str, Callable[[], object]]:
    mc = cfg["model"]
    mom_col = mc["walk_forward"]["momentum_col"]
    return {
        "base_rate": lambda: BaseRateClf(),
        "momentum": lambda: MomentumClf(mom_col),
        "logistic": lambda: LogisticClf(mc["logistic_C"]),
        "xgboost": lambda: XGBClf(mc["xgboost"]),
    }
