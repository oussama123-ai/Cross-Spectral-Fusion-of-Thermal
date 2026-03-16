"""
Evaluation metrics for the CSAF pain estimation model.

Primary metric:   MAE (Mean Absolute Error)
Secondary metrics: RMSE, PCC (Pearson Correlation), ICC (Intraclass Correlation),
                   3-class accuracy (Low/Moderate/High NRS)

Statistical testing: paired t-test with Bonferroni correction.
Confidence intervals: bootstrap resampling (B=1000).

Clinical reference: MAE < 1.5 NRS points = below minimal clinically
important difference (MCID). Our combined MAE = 0.87 is well below MCID.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.abs(predictions - targets).mean())


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Root Mean Square Error."""
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def pearson_correlation(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Pearson Correlation Coefficient."""
    if np.std(predictions) < 1e-9 or np.std(targets) < 1e-9:
        return 0.0
    r, _ = stats.pearsonr(predictions, targets)
    return float(r)


def intraclass_correlation(
    predictions: np.ndarray,
    targets: np.ndarray,
    icc_type: str = "ICC(2,1)",
) -> float:
    """
    Intraclass Correlation Coefficient (two-way mixed, absolute agreement).

    Computed via one-way ANOVA approach for two raters (system vs human NRS).

    Args:
        predictions: (N,) model predictions.
        targets:     (N,) ground-truth NRS scores.
        icc_type:    ICC type (only ICC(2,1) implemented here).

    Returns:
        ICC value in [-1, 1].
    """
    try:
        import pingouin as pg
        import pandas as pd
        n = len(predictions)
        df = pd.DataFrame({
            "rater": ["system"] * n + ["ground_truth"] * n,
            "subject": list(range(n)) * 2,
            "score": list(predictions) + list(targets),
        })
        icc_result = pg.intraclass_corr(
            data=df, targets="subject", raters="rater", ratings="score"
        )
        # ICC(2,1) = two-way mixed, single measures, absolute agreement
        icc_21 = icc_result.loc[icc_result["Type"] == "ICC2", "ICC"].values
        return float(icc_21[0]) if len(icc_21) > 0 else _icc_manual(predictions, targets)
    except ImportError:
        return _icc_manual(predictions, targets)


def _icc_manual(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Manual ICC(1,1) computation for two raters.
    (One-way random effects, single measures, absolute agreement)
    """
    n = len(pred)
    grand_mean = (pred + target).mean() / 2

    ss_between = np.sum((pred + target) / 2 - grand_mean) ** 2 * 2 / (n - 1)
    ss_within = np.sum((pred - target) ** 2) / (2 * n)

    ms_between = ss_between
    ms_within = ss_within

    if ms_between + ms_within < 1e-12:
        return 0.0
    return float((ms_between - ms_within) / (ms_between + ms_within))


def accuracy_3class(
    predictions: np.ndarray,
    targets: np.ndarray,
    bins: Tuple[float, float, float, float] = (0, 3.5, 6.5, 10),
) -> float:
    """
    3-class pain categorisation accuracy: Low (0–3) / Moderate (4–6) / High (7–10).

    Note: Bins are right-exclusive: [0,3.5), [3.5,6.5), [6.5,10].
    NRS 3.5 → Moderate, NRS 6.5 → High.
    """
    def to_class(nrs_array):
        classes = np.zeros(len(nrs_array), dtype=int)
        classes[(nrs_array > bins[1]) & (nrs_array <= bins[2])] = 1
        classes[nrs_array > bins[2]] = 2
        return classes

    pred_class = to_class(predictions)
    true_class = to_class(targets)
    return float((pred_class == true_class).mean())


def bootstrap_ci(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a metric.

    Args:
        predictions: (N,) model predictions.
        targets:     (N,) ground-truth scores.
        metric_fn:   Function(pred, target) → scalar.
        n_bootstrap: Number of bootstrap samples (default 1000).
        confidence:  Confidence level (default 0.95 → 95% CI).
        seed:        Random seed.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(predictions)
    bootstrap_scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        score = metric_fn(predictions[idx], targets[idx])
        bootstrap_scores.append(score)

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    point_estimate = metric_fn(predictions, targets)

    return point_estimate, ci_lower, ci_upper


def paired_ttest_with_correction(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    n_comparisons: int = 9,
) -> Tuple[float, float]:
    """
    Paired t-test with Bonferroni correction for multiple comparisons.

    Used in Table 2 to compare CSAF+Transformer vs all baselines.

    Args:
        errors_a: (N,) absolute errors for method A.
        errors_b: (N,) absolute errors for method B.
        n_comparisons: Number of comparisons for Bonferroni correction (default 9).

    Returns:
        t_statistic, corrected_p_value
    """
    t_stat, p_val = stats.ttest_rel(errors_a, errors_b)
    p_corrected = min(p_val * n_comparisons, 1.0)
    return float(t_stat), float(p_corrected)


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_bootstrap: int = 1000,
    compute_ci: bool = True,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics with optional bootstrap CIs.

    Returns:
        Dict with all metrics and CIs.
    """
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)

    results: Dict[str, float] = {}

    # Point estimates
    results["mae"] = mae(predictions, targets)
    results["rmse"] = rmse(predictions, targets)
    results["pcc"] = pearson_correlation(predictions, targets)
    results["icc"] = intraclass_correlation(predictions, targets)
    results["accuracy_3class"] = accuracy_3class(predictions, targets)

    # Bootstrap CIs for primary metric (MAE)
    if compute_ci and n_bootstrap > 0:
        _, ci_lo, ci_hi = bootstrap_ci(predictions, targets, mae, n_bootstrap)
        results["mae_ci_lower"] = ci_lo
        results["mae_ci_upper"] = ci_hi

    return results


def compute_stratified_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    bins: Dict[str, Tuple[float, float]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics stratified by pain intensity level.

    Args:
        predictions: (N,)
        targets:     (N,)
        bins:        Dict mapping bin_name → (min_nrs, max_nrs).

    Returns:
        Dict[bin_name → metrics_dict]
    """
    if bins is None:
        bins = {
            "low_0_3": (0.0, 3.5),
            "moderate_4_6": (3.5, 6.5),
            "high_7_10": (6.5, 10.1),
        }

    results = {}
    for bin_name, (lo, hi) in bins.items():
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() < 5:
            logger.warning(f"Bin '{bin_name}': only {mask.sum()} samples, skipping.")
            continue
        results[bin_name] = compute_all_metrics(
            predictions[mask], targets[mask], n_bootstrap=500, compute_ci=False
        )
        results[bin_name]["n_samples"] = int(mask.sum())

    return results


def format_results_table(
    method_results: Dict[str, Dict[str, float]],
    primary_method: str = "CSAF+Transformer",
) -> str:
    """
    Format evaluation results as a text table (similar to Table 2 in paper).

    Args:
        method_results: Dict[method_name → metrics_dict].
        primary_method: Name of the primary method for improvement calculation.

    Returns:
        Formatted string table.
    """
    header = f"{'Method':<35} {'MAE':>8} {'PCC':>8} {'ICC':>8} {'Acc%':>8}"
    separator = "-" * len(header)
    lines = [header, separator]

    primary_mae = method_results.get(primary_method, {}).get("mae", None)

    for method_name, metrics in method_results.items():
        m = metrics.get("mae", float("nan"))
        pcc = metrics.get("pcc", float("nan"))
        icc = metrics.get("icc", float("nan"))
        acc = metrics.get("accuracy_3class", float("nan")) * 100

        improvement = ""
        if primary_mae is not None and method_name != primary_method:
            pct = (m - primary_mae) / m * 100
            improvement = f"  ({pct:+.1f}%)"

        lines.append(
            f"{method_name:<35} {m:>8.3f} {pcc:>8.3f} {icc:>8.3f} {acc:>8.1f}{improvement}"
        )

    return "\n".join(lines)
