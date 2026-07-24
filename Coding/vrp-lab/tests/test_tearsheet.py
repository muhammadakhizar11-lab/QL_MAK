"""Tearsheet + plots smoke tests: figures build, HTML lands on disk, and the
analysis helpers return sane tables. Headless (Agg) and file-I/O only to tmp."""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from vrp_lab.analysis import estimator_compare, regime_analysis
from vrp_lab.visualization import plots
from vrp_lab.visualization.tearsheet import generate_tearsheet


@pytest.fixture(scope="module")
def fake_equity():
    idx = pd.bdate_range("2019-01-01", periods=756)
    rng = np.random.default_rng(5)
    rets = rng.normal(4e-4, 0.008, len(idx))
    rets[300] = -0.06  # one ugly day for the drawdown/annotation paths
    return pd.Series(1e6 * np.exp(np.cumsum(rets)), index=idx)


def test_tearsheet_writes_html(fake_equity, tmp_path):
    out = generate_tearsheet(fake_equity, name="unit", out_path=tmp_path / "ts.html")
    html = out.read_text()
    assert out.exists()
    assert "base64" in html and "Risk metrics" in html
    assert html.count("data:image/png") >= 6


def test_core_figures_build(fake_equity):
    idx = fake_equity.index
    rets = fake_equity.pct_change()
    vix = pd.Series(18.0 + 10 * np.abs(np.sin(np.arange(len(idx)) / 40)), index=idx)
    vrp = pd.Series(0.03, index=idx) + rets.fillna(0)
    figs = [
        plots.plot_vrp_timeseries(vrp, vix),
        plots.plot_vrp_histogram(vrp),
        plots.plot_vrp_vs_vix(vrp, vix),
        plots.plot_equity_drawdown(fake_equity),
        plots.plot_monthly_heatmap(rets),
        plots.plot_rolling_sharpe(rets),
    ]
    assert all(f is not None for f in figs)
    import matplotlib.pyplot as plt
    plt.close("all")


def test_estimator_compare_tables(gbm_20):
    est = estimator_compare.run_all(gbm_20, window=21)
    stats = estimator_compare.divergence_stats(est)
    assert set(stats.index) == set(est.columns)
    assert stats.loc["close_to_close", "corr_vs_cc"] == pytest.approx(1.0)


def test_vrp_by_regime_table():
    idx = pd.bdate_range("2019-01-01", periods=500)
    rng = np.random.default_rng(2)
    vix = pd.Series(rng.uniform(10, 40, len(idx)), index=idx)
    vrp = pd.Series(rng.normal(0.03, 0.02, len(idx)), index=idx)
    tbl = regime_analysis.vrp_by_regime(vrp, vix)
    assert list(tbl.index) == ["low", "normal", "crisis"]
    assert (tbl["n_days"] > 0).all()
