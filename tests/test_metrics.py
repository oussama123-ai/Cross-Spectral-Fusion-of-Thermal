"""
Unit tests for evaluation metrics.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    mae, rmse, pearson_correlation, accuracy_3class,
    bootstrap_ci, compute_all_metrics, compute_stratified_metrics,
    format_results_table, paired_ttest_with_correction,
)


class TestBasicMetrics:

    def test_mae_perfect(self):
        pred = np.array([1.0, 2.0, 3.0])
        tgt  = np.array([1.0, 2.0, 3.0])
        assert mae(pred, tgt) == pytest.approx(0.0)

    def test_mae_known(self):
        pred = np.array([1.0, 3.0])
        tgt  = np.array([2.0, 2.0])
        assert mae(pred, tgt) == pytest.approx(1.0)

    def test_rmse_known(self):
        pred = np.array([0.0, 4.0])
        tgt  = np.array([0.0, 0.0])
        assert rmse(pred, tgt) == pytest.approx(2.0 * np.sqrt(2))

    def test_pcc_perfect(self):
        x = np.arange(10, dtype=float)
        assert pearson_correlation(x, x) == pytest.approx(1.0)

    def test_pcc_anticorrelated(self):
        x = np.arange(10, dtype=float)
        assert pearson_correlation(x, -x) == pytest.approx(-1.0)

    def test_pcc_constant_input(self):
        x = np.ones(10)
        y = np.arange(10, dtype=float)
        assert pearson_correlation(x, y) == pytest.approx(0.0)

    def test_accuracy_3class_all_low(self):
        pred = np.array([1.0, 2.0, 0.5])
        tgt  = np.array([1.0, 2.0, 0.5])
        assert accuracy_3class(pred, tgt) == pytest.approx(1.0)

    def test_accuracy_3class_mixed(self):
        pred = np.array([1.0, 5.0, 8.0])   # low, moderate, high
        tgt  = np.array([2.0, 4.0, 9.0])   # low, moderate, high
        assert accuracy_3class(pred, tgt) == pytest.approx(1.0)

    def test_accuracy_3class_all_wrong(self):
        pred = np.array([8.0, 8.0, 1.0])   # high, high, low
        tgt  = np.array([1.0, 2.0, 8.0])   # low, low, high
        assert accuracy_3class(pred, tgt) == pytest.approx(0.0)


class TestBootstrapCI:

    def test_ci_width_reasonable(self):
        rng = np.random.RandomState(42)
        pred = rng.uniform(0, 10, 200)
        tgt  = pred + rng.normal(0, 0.5, 200)
        point, ci_lo, ci_hi = bootstrap_ci(pred, tgt, mae, n_bootstrap=200)
        assert ci_lo < point < ci_hi
        assert (ci_hi - ci_lo) < 1.0   # 95% CI should be < 1 NRS unit here

    def test_ci_seed_reproducible(self):
        rng = np.random.RandomState(0)
        pred = rng.uniform(0, 10, 100)
        tgt  = pred + rng.normal(0, 1.0, 100)
        _, lo1, hi1 = bootstrap_ci(pred, tgt, mae, n_bootstrap=100, seed=42)
        _, lo2, hi2 = bootstrap_ci(pred, tgt, mae, n_bootstrap=100, seed=42)
        assert lo1 == lo2 and hi1 == hi2


class TestComputeAllMetrics:

    def test_returns_required_keys(self):
        rng = np.random.RandomState(1)
        pred = rng.uniform(0, 10, 50)
        tgt  = pred + rng.normal(0, 1.0, 50)
        results = compute_all_metrics(pred, tgt, n_bootstrap=50, compute_ci=True)
        for key in ("mae", "rmse", "pcc", "accuracy_3class",
                    "mae_ci_lower", "mae_ci_upper"):
            assert key in results, f"Missing key: {key}"

    def test_mae_matches_direct(self):
        rng = np.random.RandomState(2)
        pred = rng.uniform(0, 10, 100)
        tgt  = rng.uniform(0, 10, 100)
        results = compute_all_metrics(pred, tgt, n_bootstrap=0, compute_ci=False)
        assert results["mae"] == pytest.approx(mae(pred, tgt))


class TestStratifiedMetrics:

    def test_three_bins_present(self):
        rng = np.random.RandomState(3)
        pred = rng.uniform(0, 10, 300)
        tgt  = rng.uniform(0, 10, 300)
        result = compute_stratified_metrics(pred, tgt)
        assert len(result) == 3
        for bin_name in ("low_0_3", "moderate_4_6", "high_7_10"):
            assert bin_name in result

    def test_bin_n_samples_add_up(self):
        rng = np.random.RandomState(4)
        tgt = rng.uniform(0, 10, 300)
        pred = tgt + rng.normal(0, 0.5, 300)
        result = compute_stratified_metrics(pred, tgt)
        total = sum(v["n_samples"] for v in result.values())
        assert total == len(tgt)


class TestStatisticalTests:

    def test_significant_difference(self):
        rng = np.random.RandomState(5)
        n = 200
        errors_good = rng.uniform(0.5, 1.0, n)   # low errors
        errors_bad  = rng.uniform(1.5, 2.5, n)   # high errors
        t_stat, p_val = paired_ttest_with_correction(errors_bad, errors_good,
                                                     n_comparisons=1)
        assert p_val < 0.001

    def test_no_difference(self):
        rng = np.random.RandomState(6)
        n = 100
        errors = rng.uniform(0.5, 1.5, n)
        t_stat, p_val = paired_ttest_with_correction(errors, errors.copy(),
                                                     n_comparisons=1)
        assert p_val > 0.05


class TestFormatResultsTable:

    def test_output_is_string(self):
        results = {
            "RGB-Transformer": {"mae": 1.23, "pcc": 0.72, "icc": 0.69, "accuracy_3class": 0.728},
            "CSAF+Transformer": {"mae": 0.87, "pcc": 0.86, "icc": 0.83, "accuracy_3class": 0.824},
        }
        table = format_results_table(results, primary_method="CSAF+Transformer")
        assert isinstance(table, str)
        assert "CSAF+Transformer" in table
        assert "0.87" in table
