"""
Piste 2 — Ensemble Pre+Post Acid Predictions.

Averages the predicted probabilities from independent pre-acid and post-acid
runs for each patient. Since errors from pre and post models are weakly
correlated, the ensemble typically improves AUC by 0.01–0.03.

Usage:
    python src/ensemble_pre_post.py [--run_name baseline] [--metrics_dir outputs/metrics]

Output:
    outputs/metrics/{run_name}_ensemble_predictions.csv
    outputs/metrics/{run_name}_ensemble_summary.json
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import compute_metrics, save_json, setup_logger


def ensemble_predictions(pre_df: pd.DataFrame, post_df: pd.DataFrame, weight_post: float = 0.5):
    """
    Merge and average pre/post predictions by patient_id.

    Args:
        pre_df: Predictions from pre-acid run (must have patient_id, y_true, y_prob)
        post_df: Predictions from post-acid run
        weight_post: Weight for post-acid predictions (0.5 = equal average)

    Returns:
        DataFrame with ensemble predictions
    """
    weight_pre = 1.0 - weight_post

    # Merge on patient_id (inner join: only patients present in both)
    merged = pre_df[["patient_id", "y_true", "y_prob"]].merge(
        post_df[["patient_id", "y_prob"]],
        on="patient_id",
        suffixes=("_pre", "_post"),
    )

    merged["y_prob_ensemble"] = (
        weight_pre * merged["y_prob_pre"] + weight_post * merged["y_prob_post"]
    )

    return merged


def main():
    parser = argparse.ArgumentParser(description="Ensemble Pre+Post Acid Predictions")
    parser.add_argument("--run_name", default="baseline", help="Run name prefix (e.g. 'baseline')")
    parser.add_argument("--metrics_dir", default="outputs_final/metrics", help="Metrics directory")
    parser.add_argument("--weight_post", type=float, default=0.5, help="Weight for post-acid (0.5 = equal)")
    args = parser.parse_args()

    logger = setup_logger("ensemble", "outputs_final/logs/ensemble.log")

    # Load predictions
    pre_path = os.path.join(args.metrics_dir, f"{args.run_name}_pre_predictions.csv")
    post_path = os.path.join(args.metrics_dir, f"{args.run_name}_post_predictions.csv")

    # Also try TTA versions
    pre_tta_path = os.path.join(args.metrics_dir, f"{args.run_name}_pre_predictions_tta.csv")
    post_tta_path = os.path.join(args.metrics_dir, f"{args.run_name}_post_predictions_tta.csv")

    for path, label in [(pre_path, "pre"), (post_path, "post")]:
        if not os.path.isfile(path):
            logger.error(f"Missing {label} predictions: {path}")
            sys.exit(1)

    pre_df = pd.read_csv(pre_path)
    post_df = pd.read_csv(post_path)

    logger.info(f"Pre predictions:  {len(pre_df)} patients")
    logger.info(f"Post predictions: {len(post_df)} patients")
    logger.info(f"Weight: pre={1 - args.weight_post:.2f}, post={args.weight_post:.2f}")

    # ── Standard ensemble ──
    merged = ensemble_predictions(pre_df, post_df, args.weight_post)
    logger.info(f"Merged (inner join): {len(merged)} patients")

    metrics = compute_metrics(
        merged["y_true"].values.astype(int),
        merged["y_prob_ensemble"].values,
    )

    # ── TTA ensemble (if available) ──
    tta_metrics = None
    if os.path.isfile(pre_tta_path) and os.path.isfile(post_tta_path):
        pre_tta_df = pd.read_csv(pre_tta_path)
        post_tta_df = pd.read_csv(post_tta_path)
        merged_tta = ensemble_predictions(pre_tta_df, post_tta_df, args.weight_post)
        tta_metrics = compute_metrics(
            merged_tta["y_true"].values.astype(int),
            merged_tta["y_prob_ensemble"].values,
        )

    # ── Per-fold ensemble ──
    fold_metrics = []
    if "fold" in pre_df.columns and "fold" in post_df.columns:
        for fold in sorted(pre_df["fold"].unique()):
            pre_fold = pre_df[pre_df["fold"] == fold]
            post_fold = post_df[post_df["fold"] == fold]
            merged_fold = ensemble_predictions(pre_fold, post_fold, args.weight_post)
            if len(merged_fold) > 0:
                fm = compute_metrics(
                    merged_fold["y_true"].values.astype(int),
                    merged_fold["y_prob_ensemble"].values,
                )
                fm["fold"] = int(fold)
                fold_metrics.append(fm)

    # ── Save results ──
    out_prefix = f"{args.run_name}_ensemble"

    merged.to_csv(
        os.path.join(args.metrics_dir, f"{out_prefix}_predictions.csv"),
        index=False,
    )

    summary = {
        "overall": metrics,
        "weight_pre": 1 - args.weight_post,
        "weight_post": args.weight_post,
    }
    if tta_metrics:
        summary["tta"] = tta_metrics
    if fold_metrics:
        summary["per_fold"] = fold_metrics
        auc_values = [fm["auc"] for fm in fold_metrics]
        summary["auc_mean"] = float(np.mean(auc_values))
        summary["auc_std"] = float(np.std(auc_values))

    save_json(summary, os.path.join(args.metrics_dir, f"{out_prefix}_summary.json"))

    # ── Print results ──
    logger.info("\n" + "=" * 60)
    logger.info("  ENSEMBLE RESULTS (Pre + Post Acid)")
    logger.info("=" * 60)
    logger.info(f"  AUC:         {metrics['auc']:.4f}")
    logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    logger.info(f"  F1:          {metrics['f1']:.4f}")
    logger.info(f"  Accuracy:    {metrics['accuracy']:.4f}")

    if tta_metrics:
        logger.info(f"\n  TTA Ensemble AUC: {tta_metrics['auc']:.4f}")

    if fold_metrics:
        logger.info(f"\n  Per-fold AUC:")
        for fm in fold_metrics:
            logger.info(f"    Fold {fm['fold']}: AUC={fm['auc']:.4f} Sens={fm['sensitivity']:.3f} Spec={fm['specificity']:.3f}")
        logger.info(f"  Mean AUC: {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")

    # Compare with individual runs
    pre_auc = compute_metrics(pre_df["y_true"].values.astype(int), pre_df["y_prob"].values)["auc"]
    post_auc = compute_metrics(post_df["y_true"].values.astype(int), post_df["y_prob"].values)["auc"]
    logger.info(f"\n  Comparison:")
    logger.info(f"    Pre-acid only:  AUC={pre_auc:.4f}")
    logger.info(f"    Post-acid only: AUC={post_auc:.4f}")
    logger.info(f"    Ensemble:       AUC={metrics['auc']:.4f} ({'+' if metrics['auc'] > max(pre_auc, post_auc) else ''}{metrics['auc'] - max(pre_auc, post_auc):.4f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
