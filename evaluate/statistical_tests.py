"""
Statistical Tests for Publication.

Performs rigorous statistical comparisons between model configurations:
  - Bootstrap 95% CI: confidence intervals on all metrics
  - DeLong test (bootstrap approximation): compare AUC between runs
  - McNemar test: compare classification error patterns

Usage:
    python evaluate/statistical_tests.py [--config config.yaml]

Output:
    - outputs/metrics/statistical_tests.json
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, compute_metrics, save_json, setup_logger


# ============================================================================
# DeLong Test for comparing AUCs
# ============================================================================

def _compute_midrank(x):
    """Compute midranks for the DeLong test."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        for k in range(i, j):
            T[k] = 0.5 * (i + j - 1)
        i = j
    T2 = np.zeros(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong AUC variance computation."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.zeros(k)
    ty = np.zeros(k)
    tz = np.zeros(k)

    for r in range(k):
        tx[r] = _compute_midrank(positive_examples[r, :]).sum()
        ty[r] = _compute_midrank(negative_examples[r, :]).sum()
        tz[r] = _compute_midrank(predictions_sorted_transposed[r, :]).sum()

    aucs = (tz - tx) / (m * n)

    v01 = (tz[:, None] - tx[:, None]) / n - aucs[:, None] * np.ones((1, m))
    v10 = np.ones((k, 1)) * (1.0 / m) - (tz[:, None] - ty[:, None]) / m

    # Note: fixed indexing for positive/negative splitting
    sx = np.cov(
        np.hstack([
            _compute_midrank(predictions_sorted_transposed[i, :m]).reshape(1, -1)
            for i in range(k)
        ])
    ) if k == 1 else np.zeros((1, 1))

    return aucs, sx


def delong_test(y_true, y_prob_1, y_prob_2):
    """
    DeLong test for comparing two AUCs on the same set of samples.

    Args:
        y_true: Binary labels [N]
        y_prob_1: Predicted probabilities from model 1 [N]
        y_prob_2: Predicted probabilities from model 2 [N]

    Returns:
        dict with z_statistic, p_value, auc_1, auc_2, auc_diff
    """
    from sklearn.metrics import roc_auc_score

    auc_1 = roc_auc_score(y_true, y_prob_1)
    auc_2 = roc_auc_score(y_true, y_prob_2)

    # Simplified DeLong using bootstrap approximation
    n_bootstraps = 2000
    rng = np.random.RandomState(42)
    auc_diffs = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        auc_diff = (
            roc_auc_score(y_true[indices], y_prob_1[indices])
            - roc_auc_score(y_true[indices], y_prob_2[indices])
        )
        auc_diffs.append(auc_diff)

    auc_diffs = np.array(auc_diffs)
    mean_diff = np.mean(auc_diffs)
    std_diff = np.std(auc_diffs)

    if std_diff > 0:
        z = mean_diff / std_diff
        p_value = 2 * stats.norm.sf(abs(z))
    else:
        z = 0.0
        p_value = 1.0

    return {
        "auc_1": float(auc_1),
        "auc_2": float(auc_2),
        "auc_diff": float(auc_1 - auc_2),
        "z_statistic": float(z),
        "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
    }


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci(y_true, y_prob, n_bootstraps=1000, ci=0.95, seed=42):
    """
    Compute bootstrap confidence intervals for all metrics.

    Returns dict with metric: {mean, lower, upper, ci_level}
    """
    rng = np.random.RandomState(seed)
    metric_keys = ["auc", "sensitivity", "specificity", "f1", "accuracy", "ppv", "npv"]
    all_metrics = {k: [] for k in metric_keys}

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
        metrics = compute_metrics(y_true[indices], y_prob[indices])
        for k in metric_keys:
            all_metrics[k].append(metrics[k])

    alpha = (1 - ci) / 2
    results = {}
    for k in metric_keys:
        values = np.array(all_metrics[k])
        results[k] = {
            "mean": float(np.mean(values)),
            "lower": float(np.percentile(values, alpha * 100)),
            "upper": float(np.percentile(values, (1 - alpha) * 100)),
            "ci_level": ci,
        }

    return results


# ============================================================================
# McNemar Test
# ============================================================================

def mcnemar_test(y_true, y_pred_1, y_pred_2):
    """
    McNemar test for comparing two classifiers' errors.

    Tests if the two models make the same types of errors.
    """
    # Build contingency table
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)

    # b = model1 correct, model2 wrong
    b = np.sum(correct_1 & ~correct_2)
    # c = model1 wrong, model2 correct
    c = np.sum(~correct_1 & correct_2)

    # McNemar with continuity correction
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant_005": False, "b": int(b), "c": int(c)}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
        "b": int(b),
        "c": int(c),
    }


def main():
    parser = argparse.ArgumentParser(description="Statistical tests")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("stats", os.path.join(cfg["output"]["logs"], "statistical_tests.log"))

    metrics_dir = cfg["output"]["metrics"]
    results = {}

    # Load predictions for all runs dynamically
    predictions = {}
    import glob
    prediction_files = glob.glob(os.path.join(metrics_dir, "*_predictions.csv"))
    
    for path in prediction_files:
        run_name = os.path.basename(path).replace("_predictions.csv", "")
        df = pd.read_csv(path)
        # Skip ensemble/merged files without per-fold data
        if "fold" not in df.columns or "y_prob" not in df.columns:
            logger.info(f"  Skipping {run_name}: missing 'fold' or 'y_prob' column")
            continue
        predictions[run_name] = df

    if not predictions:
        logger.error("No prediction files found. Run training first.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 1. Bootstrap CIs for each condition
    # ----------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  Bootstrap 95% Confidence Intervals")
    logger.info("=" * 60)

    results["bootstrap_ci"] = {}
    for condition, df in predictions.items():
        ci = bootstrap_ci(df["y_true"].values, df["y_prob"].values, n_bootstraps=1000)
        results["bootstrap_ci"][condition] = ci

        logger.info(f"\n  {condition.upper()}-ACID:")
        for metric, vals in ci.items():
            logger.info(
                f"    {metric:15s}: {vals['mean']:.4f} "
                f"[{vals['lower']:.4f}, {vals['upper']:.4f}]"
            )

    # ----------------------------------------------------------------
    # 2. DeLong test: baseline_post vs baseline_pre
    # ----------------------------------------------------------------
    if "baseline_post" in predictions and "baseline_pre" in predictions:
        logger.info("\n" + "=" * 60)
        logger.info("  DeLong Test: Baseline Post vs Pre")
        logger.info("=" * 60)

        # Align predictions by patient_id
        post_df = predictions["baseline_post"].sort_values("patient_id")
        pre_df = predictions["baseline_pre"].sort_values("patient_id")

        common_patients = set(post_df["patient_id"]) & set(pre_df["patient_id"])
        post_aligned = post_df[post_df["patient_id"].isin(common_patients)].sort_values("patient_id")
        pre_aligned = pre_df[pre_df["patient_id"].isin(common_patients)].sort_values("patient_id")

        if len(post_aligned) > 0:
            delong = delong_test(
                post_aligned["y_true"].values,
                post_aligned["y_prob"].values,
                pre_aligned["y_prob"].values,
            )
            results["delong_post_vs_pre"] = delong

            logger.info(
                f"  Post AUC: {delong['auc_1']:.4f} vs Pre AUC: {delong['auc_2']:.4f}"
            )
            logger.info(f"  Δ AUC: {delong['auc_diff']:.4f}")
            logger.info(f"  Z: {delong['z_statistic']:.3f}, p: {delong['p_value']:.4f}")
            logger.info(
                f"  Significant (p<0.05): {delong['significant_005']}"
            )

            # McNemar test
            threshold_post = compute_metrics(
                post_aligned["y_true"].values, post_aligned["y_prob"].values
            )["threshold"]
            threshold_pre = compute_metrics(
                pre_aligned["y_true"].values, pre_aligned["y_prob"].values
            )["threshold"]

            y_pred_post = (post_aligned["y_prob"].values >= threshold_post).astype(int)
            y_pred_pre = (pre_aligned["y_prob"].values >= threshold_pre).astype(int)

            mcnemar = mcnemar_test(
                post_aligned["y_true"].values, y_pred_post, y_pred_pre
            )
            results["mcnemar_post_vs_pre"] = mcnemar

            logger.info(f"\n  McNemar test:")
            logger.info(f"    χ²: {mcnemar['statistic']:.3f}, p: {mcnemar['p_value']:.4f}")
            logger.info(f"    Post correct/Pre wrong: {mcnemar['b']}")
            logger.info(f"    Post wrong/Pre correct: {mcnemar['c']}")

    # ----------------------------------------------------------------
    # 3. Ablation statistical comparisons (Baseline Post vs Ablations)
    # ----------------------------------------------------------------
    if "baseline_post" in predictions:
        logger.info("\n" + "=" * 60)
        logger.info("  Ablation: DeLong tests vs Baseline Post Model")
        logger.info("=" * 60)

        results["ablation_comparisons"] = {}
        base_df = predictions["baseline_post"].sort_values("patient_id")
        
        for run_name, df in predictions.items():
            if run_name == "baseline_post" or "ablation" not in run_name:
                continue
                
            comp_df = df.sort_values("patient_id")
            common_patients = set(base_df["patient_id"]) & set(comp_df["patient_id"])
            base_aligned = base_df[base_df["patient_id"].isin(common_patients)].sort_values("patient_id")
            comp_aligned = comp_df[comp_df["patient_id"].isin(common_patients)].sort_values("patient_id")
            
            if len(base_aligned) > 0:
                delong = delong_test(
                    base_aligned["y_true"].values,
                    base_aligned["y_prob"].values,
                    comp_aligned["y_prob"].values,
                )
                results["ablation_comparisons"][run_name] = delong
                logger.info(
                    f"  {run_name:<25s}: AUC={delong['auc_2']:.4f} (Δ={-delong['auc_diff']:+.4f}) p={delong['p_value']:.4f}"
                )

    # Save all results
    save_json(results, os.path.join(metrics_dir, "statistical_tests.json"))
    logger.info(f"\n  Statistical tests complete! Saved to outputs/metrics/statistical_tests.json")


if __name__ == "__main__":
    main()
