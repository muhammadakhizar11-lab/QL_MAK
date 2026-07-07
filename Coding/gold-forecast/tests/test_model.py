"""
Hermetic unit tests for stage-3 modelling logic (no network, no real data).
Run:  pytest -q

The load-bearing test here is the embargo: because the label is an N-day forward
return, a training row's label reaches N days into the future, so the walk-forward
splitter MUST purge those rows before each test block or the model peeks. Everything
else (baseline behaviour, prob range, harness plumbing) is checked on tiny data.
"""
import numpy as np
import pandas as pd

from src import evaluate, models


# --------------------------------------------------------------------------- #
# Walk-forward splitter — the leakage-critical invariant
# --------------------------------------------------------------------------- #
def test_embargo_purges_forward_label_overlap():
    """No training row's N-day-forward label may reach into (or past) the start of
    its test block. With embargo=N the last kept train row t satisfies t+N < ts."""
    n, min_train, test_days, N = 500, 100, 20, 5
    folds = evaluate.walk_forward_folds(n, min_train, test_days, embargo=N)
    assert folds, "expected at least one fold"
    for f in folds:
        train_hi = f.train_pos[-1]              # last training position
        test_lo = f.test_pos[0]                 # first test position
        # label of train_hi uses data through train_hi + N; it must land strictly
        # before the test block begins.
        assert train_hi + N <= test_lo, f"label overlap: {train_hi}+{N} vs {test_lo}"
        # exactly `embargo` (=N) rows are purged between the last train row and
        # the first test row
        assert test_lo - train_hi - 1 == N


def test_folds_are_time_ordered_and_disjoint_oos():
    folds = evaluate.walk_forward_folds(400, min_train=80, test_days=30, embargo=5)
    prev_end = -1
    for f in folds:
        assert f.train_pos[-1] < f.test_pos[0], "train must precede test"
        assert f.test_pos[0] > prev_end, "test blocks must not overlap / go backwards"
        prev_end = f.test_pos[-1]


def test_expanding_train_starts_at_zero_rolling_does_not():
    exp = evaluate.walk_forward_folds(400, 80, 30, 5, scheme="expanding")
    roll = evaluate.walk_forward_folds(400, 80, 30, 5, scheme="rolling")
    assert exp[-1].train_pos[0] == 0, "expanding window keeps all history"
    assert roll[-1].train_pos[0] > 0, "rolling window drops old history"
    assert len(roll[-1].train_pos) <= 80


def test_no_folds_when_insufficient_history():
    assert evaluate.walk_forward_folds(50, min_train=100, test_days=20, embargo=5) == []


# --------------------------------------------------------------------------- #
# Baselines behave as specified
# --------------------------------------------------------------------------- #
def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "ret_20d": rng.normal(0, 0.02, n),
        "other": rng.normal(0, 1, n),
    }, index=pd.RangeIndex(n))
    y = pd.Series(rng.normal(0, 0.01, n), index=X.index)
    return X, y


def test_naive_zero_predicts_zero():
    X, y = _toy()
    m = models.NaiveZeroReg().fit(X, y)
    assert np.all(m.predict(X) == 0.0)


def test_base_rate_predicts_train_up_rate():
    X, y = _toy()
    y_up = (y > 0).astype(int)
    m = models.BaseRateClf().fit(X, y_up)
    p = m.predict(X)
    assert np.allclose(p, y_up.mean())
    assert np.all((p >= 0) & (p <= 1))


def test_classifier_probs_in_unit_interval():
    X, y = _toy(seed=3)
    y_up = (y > 0).astype(int)
    for factory in models.build_classifiers(_cfg()).values():
        p = factory().fit(X, y_up).predict(X)
        assert np.all((p >= 0.0) & (p <= 1.0)), "P(up) must be a probability"


def test_momentum_reg_is_calibrated_to_signal():
    """MomentumReg regresses y on the momentum column, so a perfectly linear
    relationship is recovered (up to the intercept)."""
    n = 300
    idx = pd.RangeIndex(n)
    sig = np.linspace(-0.05, 0.05, n)
    X = pd.DataFrame({"ret_20d": sig, "noise": 0.0}, index=idx)
    y = pd.Series(2.0 * sig + 0.001, index=idx)   # exact linear target
    m = models.MomentumReg("ret_20d").fit(X, y)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.99


def _cfg():
    return {"model": {
        "walk_forward": {"momentum_col": "ret_20d"},
        "xgboost": {"n_estimators": 20, "max_depth": 2, "random_state": 0},
        "ridge_alpha": 1.0, "logistic_C": 1.0,
    }}


# --------------------------------------------------------------------------- #
# Harness plumbing runs end-to-end on synthetic data (no real files)
# --------------------------------------------------------------------------- #
def test_run_head_produces_disjoint_oos_predictions():
    X, y = _toy(n=300, seed=7)
    folds = evaluate.walk_forward_folds(len(X), min_train=100, test_days=40, embargo=5)
    out = evaluate._run_head(X, y, folds, models.build_regressors(_cfg()))
    oos_len = sum(len(f.test_pos) for f in folds)
    assert len(out) == oos_len, "one OOS prediction per test row, no gaps/overlaps"
    assert not out.index.duplicated().any(), "OOS predictions must be disjoint"
    assert set(out.columns) == {"naive_zero", "momentum", "ridge", "xgboost"}
    assert out.notna().all().all()
