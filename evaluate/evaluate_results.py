"""
Results Evaluation Script.

Aggregates predictions from all 5 folds, computes final per-fold and
ensemble metrics, performs comprehensive attention weight analysis
(global rankings, Friedman test, pairwise Wilcoxon with Bonferroni
correction), and produces a publication-ready results table.

Usage:
    python evaluate/evaluate_results.py [--config config.yaml]

Outputs:
    - outputs/metrics/final_results.json
    - outputs/metrics/final_results_table.csv
    - outputs/metrics/attention_analysis.json
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, compute_metrics, save_json, setup_logger


def analyze_attention_weights(predictions_df: pd.DataFrame, cfg: dict) -> dict:
    """
    Comprehensive analysis of attention weights across magnifications.

    Returns dict with:
    - global: mean/std per magnification
    - by_class: mean/std per magnification × class
    - by_fold: mean/std per magnification × fold
    - friedman_test: statistical test for differences between magnifications
    - ranking: magnifications ranked by mean attention weight
    """
    attn_cols = [c for c in predictions_df.columns if c.startswith("alpha_")]
    if not attn_cols:
        return {}

    mag_labels = [c.replace("alpha_", "") for c in attn_cols]
    analysis = {"magnification_labels": mag_labels}

    # Global attention weights
    analysis["global"] = {}
    for col, mag in zip(attn_cols, mag_labels):
        analysis["global"][mag] = {
            "mean": float(predictions_df[col].mean()),
            "std": float(predictions_df[col].std()),
            "median": float(predictions_df[col].median()),
            "q25": float(predictions_df[col].quantile(0.25)),
            "q75": float(predictions_df[col].quantile(0.75)),
        }

    # By class (Normal=0, Abnormal=1)
    analysis["by_class"] = {"normal": {}, "abnormal": {}}
    for col, mag in zip(attn_cols, mag_labels):
        for label, name in [(0, "normal"), (1, "abnormal")]:
            subset = predictions_df[predictions_df["y_true"] == label][col]
            analysis["by_class"][name][mag] = {
                "mean": float(subset.mean()),
                "std": float(subset.std()),
            }

    # By fold
    analysis["by_fold"] = {}
    for fold in predictions_df["fold"].unique():
        fold_data = predictions_df[predictions_df["fold"] == fold]
        analysis["by_fold"][f"fold_{fold}"] = {}
        for col, mag in zip(attn_cols, mag_labels):
            analysis["by_fold"][f"fold_{fold}"][mag] = {
                "mean": float(fold_data[col].mean()),
                "std": float(fold_data[col].std()),
            }

    # Friedman test (non-parametric repeated measures)
    # Tests if there's a statistically significant difference in attention
    # weights across magnifications
    attn_values = predictions_df[attn_cols].values  # [N, M]
    if attn_values.shape[0] > 10:
        try:
            stat, pval = stats.friedmanchisquare(*[attn_values[:, i] for i in range(len(attn_cols))])
            analysis["friedman_test"] = {
                "statistic": float(stat),
                "p_value": float(pval),
                "significant_005": bool(pval < 0.05),
                "significant_001": bool(pval < 0.01),
            }
        except Exception as e:
            analysis["friedman_test"] = {"error": str(e)}

    # Wilcoxon pairwise comparisons (post-hoc)
    analysis["pairwise_wilcoxon"] = {}
    n_comparisons = len(attn_cols) * (len(attn_cols) - 1) // 2
    bonferroni_alpha = 0.05 / max(n_comparisons, 1)

    for i in range(len(attn_cols)):
        for j in range(i + 1, len(attn_cols)):
            pair_name = f"{mag_labels[i]}_vs_{mag_labels[j]}"
            try:
                stat, pval = stats.wilcoxon(attn_values[:, i], attn_values[:, j])
                analysis["pairwise_wilcoxon"][pair_name] = {
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "significant_bonferroni": bool(pval < bonferroni_alpha),
                }
            except Exception as e:
                analysis["pairwise_wilcoxon"][pair_name] = {"error": str(e)}

    # Ranking
    means = {mag: analysis["global"][mag]["mean"] for mag in mag_labels}
    ranking = sorted(means.items(), key=lambda x: x[1], reverse=True)
    analysis["ranking"] = [{"magnification": m, "mean_attention": v} for m, v in ranking]

    return analysis


def build_results_table(conditions_results: dict) -> pd.DataFrame:
    """Build a publication-ready LaTeX/CSV results table."""
    rows = []
    for condition, results in conditions_results.items():
        row = {"Condition": condition.upper()}
        for metric in ["auc", "sensitivity", "specificity", "f1", "accuracy"]:
            if metric in results:
                mean = results[metric]["mean"]
                std = results[metric]["std"]
                row[metric.upper()] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)

    return pd.DataFrame(rows)


def compute_ensemble_metrics(metrics_dir: str, run_name: str) -> dict:
    """Load per-fold TTA predictions, ensemble by averaging probs per patient, compute metrics.

    Returns None if no per-fold prediction files found.
    """
    import glob as _glob
    patterns = [
        os.path.join(metrics_dir, f"{run_name}_fold_*_predictions_tta.csv"),
        os.path.join(metrics_dir, f"{run_name}_fold_*_predictions.csv"),
    ]
    for pattern in patterns:
        files = sorted(_glob.glob(
            os.path.abspath(os.path.join(metrics_dir, os.path.basename(pattern)))
        ))
        if files:
            break

    if not files:
        return None

    all_preds = []
    for f in files:
        df = pd.read_csv(f)
        all_preds.append(df)

    all_preds = pd.concat(all_preds, ignore_index=True)
    n_folds = all_preds["fold"].nunique() if "fold" in all_preds.columns else len(files)

    ensemble = all_preds.groupby("patient_id").agg({
        "y_true": "first",
        "y_prob": "mean",
    }).reset_index()

    metrics = compute_metrics(ensemble["y_true"].values, ensemble["y_prob"].values)
    metrics["n_patients"] = len(ensemble)
    metrics["n_folds"] = n_folds
    return {"metrics": metrics, "source": "tta" if "tta" in os.path.basename(files[0]) else "standard"}


def main():
    parser = argparse.ArgumentParser(description="Evaluate results")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("evaluate", os.path.join(cfg["output"]["logs"], "evaluate.log"))

    metrics_dir = cfg["output"]["metrics"]
    all_results = {}

    import glob
    prediction_files = glob.glob(os.path.join(metrics_dir, "*_predictions.csv"))
    
    if not prediction_files:
        logger.error(f"No prediction files found in {metrics_dir}")
        sys.exit(1)

    for pred_path in sorted(prediction_files):
        # Extract run_name from file name (e.g. 'baseline_post' from 'baseline_post_predictions.csv')
        run_name = os.path.basename(pred_path).replace("_predictions.csv", "")

        logger.info(f"\n{'='*60}")
        logger.info(f"  Evaluating: {run_name.upper()}")
        logger.info(f"{'='*60}")

        predictions_df = pd.read_csv(pred_path)

        # Skip files without 'fold' column (ensemble predictions, etc.)
        if "fold" not in predictions_df.columns:
            logger.info(f"  Skipping {run_name}: no 'fold' column (ensemble/merged file)")
            continue

        logger.info(f"  Loaded {len(predictions_df)} predictions from {predictions_df['fold'].nunique()} folds")

        # Per-fold metrics
        fold_metrics = []
        for fold in sorted(predictions_df["fold"].unique()):
            fold_data = predictions_df[predictions_df["fold"] == fold]
            metrics = compute_metrics(
                fold_data["y_true"].values, fold_data["y_prob"].values
            )
            metrics["fold"] = fold
            fold_metrics.append(metrics)
            logger.info(
                f"  Fold {fold}: AUC={metrics['auc']:.4f} "
                f"Sens={metrics['sensitivity']:.3f} Spec={metrics['specificity']:.3f} "
                f"F1={metrics['f1']:.3f}"
            )

        # Aggregated metrics
        metric_keys = ["auc", "sensitivity", "specificity", "f1", "accuracy", "ppv", "npv"]
        condition_results = {}
        for key in metric_keys:
            values = [m[key] for m in fold_metrics]
            condition_results[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "per_fold": [float(v) for v in values],
            }

        logger.info(f"\n  Aggregated ({run_name.upper()}):")
        for key in metric_keys:
            r = condition_results[key]
            logger.info(f"    {key:15s}: {r['mean']:.4f} ± {r['std']:.4f}")

        # Attention weight analysis
        attn_analysis = analyze_attention_weights(predictions_df, cfg)
        condition_results["attention_analysis"] = attn_analysis

        if attn_analysis.get("ranking"):
            logger.info(f"\n  Magnification Importance Ranking ({run_name.upper()}):")
            for i, item in enumerate(attn_analysis["ranking"], 1):
                logger.info(f"    #{i}: {item['magnification']} (α = {item['mean_attention']:.4f})")

        if attn_analysis.get("friedman_test"):
            ft = attn_analysis["friedman_test"]
            if "p_value" in ft:
                logger.info(
                    f"\n  Friedman test: χ²={ft['statistic']:.2f}, "
                    f"p={ft['p_value']:.2e} "
                    f"({'***' if ft['significant_001'] else '**' if ft['significant_005'] else 'n.s.'})"
                )

        all_results[run_name] = condition_results

        # Ensemble evaluation (average probabilities across folds)
        ensemble = compute_ensemble_metrics(metrics_dir, run_name)
        if ensemble is not None:
            em = ensemble["metrics"]
            n_folds_used = ensemble.get("n_folds", "?")
            src = ensemble.get("source", "?")
            logger.info(f"\n  Ensemble ({n_folds_used} folds, source={src}): AUC={em['auc']:.4f} "
                        f"Sens={em['sensitivity']:.3f} Spec={em['specificity']:.3f} "
                        f"F1={em['f1']:.3f}")

            condition_results["ensemble"] = {
                "n_folds": n_folds_used,
                "source": src,
                **em,
            }

    # Save all results
    save_json(all_results, os.path.join(metrics_dir, "final_results.json"))

    # Build and save results table
    if all_results:
        table = build_results_table(all_results)
        table.to_csv(os.path.join(metrics_dir, "final_results_table.csv"), index=False)
        logger.info(f"\n  Results table:\n{table.to_string(index=False)}")

    # Save attention analysis separately for visualization scripts
    for run_name, results in all_results.items():
        if "attention_analysis" in results:
            save_json(
                results["attention_analysis"],
                os.path.join(metrics_dir, f"{run_name}_attention_analysis.json"),
            )

    logger.info("\n  Evaluation complete! Results saved to outputs/metrics/")


if __name__ == "__main__":
    main()
