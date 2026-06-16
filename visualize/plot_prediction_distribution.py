"""
Piste 2 — Prediction Distribution & Threshold Analysis (Figure 7).

Generates:
  (a) Histogram of predicted probabilities (Normal vs Abnormal)
  (b) Per-fold decision threshold analysis
  (c) Overlap analysis: probability density curves

Usage:
    python visualize/plot_prediction_distribution.py [--config config_final.yaml]

Output:
    outputs/{figures}/fig7_prediction_distribution.pdf
    outputs/{figures}/fig7_prediction_distribution.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, compute_metrics

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

NORMAL_COLOR = "#66BB6A"
ABNORMAL_COLOR = "#EF5350"
FOLD_COLORS = ["#1E88E5", "#43A047", "#FB8C00", "#E53935", "#8E24AA"]


def plot_prediction_histogram(ax, df, title):
    """Histogram of predicted probabilities by class with density overlay."""
    y_prob_normal = df["y_prob"][df["y_true"] == 0].values
    y_prob_abnormal = df["y_prob"][df["y_true"] == 1].values

    bins = np.linspace(0, 1, 31)
    ax.hist(y_prob_normal, bins=bins, alpha=0.5, color=NORMAL_COLOR,
            edgecolor="white", density=True, label=f"Normal (n={len(y_prob_normal)})")
    ax.hist(y_prob_abnormal, bins=bins, alpha=0.5, color=ABNORMAL_COLOR,
            edgecolor="white", density=True, label=f"Abnormal (n={len(y_prob_abnormal)})")

    # KDE overlay
    x_grid = np.linspace(0, 1, 200)
    if len(y_prob_normal) > 3:
        kde_n = stats.gaussian_kde(y_prob_normal, bw_method="scott")
        ax.plot(x_grid, kde_n(x_grid), color=NORMAL_COLOR, linewidth=2,
                linestyle="--", alpha=0.8)
    if len(y_prob_abnormal) > 3:
        kde_a = stats.gaussian_kde(y_prob_abnormal, bw_method="scott")
        ax.plot(x_grid, kde_a(x_grid), color=ABNORMAL_COLOR, linewidth=2,
                linestyle="--", alpha=0.8)

    # Optimal threshold (Youden)
    metrics = compute_metrics(df["y_true"].values, df["y_prob"].values)
    threshold = metrics["threshold"]
    ax.axvline(x=threshold, color="black", linewidth=1.5, linestyle=":",
               label=f"Threshold = {threshold:.3f}")

    ax.set_xlabel("Predicted P(Abnormal)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0.8, fontsize=9)


def plot_threshold_by_fold(ax, df, title):
    """Threshold analysis per fold."""
    fold_thresholds = []
    fold_aucs = []

    for fold in sorted(df["fold"].unique()):
        fold_data = df[df["fold"] == fold]
        metrics = compute_metrics(fold_data["y_true"].values, fold_data["y_prob"].values)
        fold_thresholds.append(metrics["threshold"])
        fold_aucs.append(metrics["auc"])

    folds = sorted(df["fold"].unique())
    x = np.arange(len(folds))
    width = 0.35

    bars1 = ax.bar(x - width/2, fold_aucs, width, label="AUC",
                   color="#1565C0", edgecolor="white", alpha=0.85)
    ax_twin = ax.twinx()
    bars2 = ax_twin.bar(x + width/2, fold_thresholds, width, label="Threshold",
                        color="#FF5722", edgecolor="white", alpha=0.85)

    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC", color="#1565C0")
    ax_twin.set_ylabel("Decision Threshold", color="#FF5722")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f+1}" for f in folds])
    ax.set_ylim(0, 1)
    ax_twin.set_ylim(0, 1)
    ax.set_title(title, fontweight="bold")

    for bar, val in zip(bars1, fold_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=8, color="#1565C0")
    for bar, val in zip(bars2, fold_thresholds):
        ax_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.3f}", ha="center", fontsize=8, color="#FF5722")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", framealpha=0.8)


def plot_overlap_analysis(ax, df, title):
    """Probability density overlap between classes."""
    y_prob_normal = df["y_prob"][df["y_true"] == 0].values
    y_prob_abnormal = df["y_prob"][df["y_true"] == 1].values

    x_grid = np.linspace(0, 1, 200)

    if len(y_prob_normal) > 3:
        kde_n = stats.gaussian_kde(y_prob_normal, bw_method="scott")
        ax.fill_between(x_grid, kde_n(x_grid), alpha=0.4,
                        color=NORMAL_COLOR, label="Normal")
        ax.plot(x_grid, kde_n(x_grid), color=NORMAL_COLOR, linewidth=2)

    if len(y_prob_abnormal) > 3:
        kde_a = stats.gaussian_kde(y_prob_abnormal, bw_method="scott")
        ax.fill_between(x_grid, kde_a(x_grid), alpha=0.4,
                        color=ABNORMAL_COLOR, label="Abnormal")
        ax.plot(x_grid, kde_a(x_grid), color=ABNORMAL_COLOR, linewidth=2)

    # Overlap region
    if len(y_prob_normal) > 3 and len(y_prob_abnormal) > 3:
        min_density = np.minimum(kde_n(x_grid), kde_a(x_grid))
        ax.fill_between(x_grid, min_density, alpha=0.3, color="#FFC107",
                        label="Overlap region")

    ax.set_xlabel("Predicted P(Abnormal)")
    ax.set_ylabel("Probability Density")
    ax.set_title(title, fontweight="bold")
    ax.legend(framealpha=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_final.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["figures"], exist_ok=True)

    post_path = os.path.join(cfg["output"]["metrics"], "baseline_post_predictions.csv")
    if not os.path.isfile(post_path):
        print(f"ERROR: No post-acid predictions found: {post_path}")
        sys.exit(1)

    post_df = pd.read_csv(post_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_prediction_histogram(axes[0], post_df,
                              "(a) Prediction Distribution (Post-acid)")
    plot_threshold_by_fold(axes[1], post_df,
                           "(b) Per-Fold AUC & Threshold")
    plot_overlap_analysis(axes[2], post_df,
                          "(c) Class Overlap (Post-acid)")

    fig.suptitle(
        "Figure 7 — Prediction Distribution & Decision Boundary Analysis",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(cfg["output"]["figures"],
                    f"fig7_prediction_distribution.{fmt}"))
    plt.close()

    print(f"Figure 7 saved to {cfg['output']['figures']}/"
          f"fig7_prediction_distribution.[pdf|png]")


if __name__ == "__main__":
    main()
