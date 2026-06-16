"""
Piste 2 — Precision-Recall Curves Visualization (Figure 5).

Generates:
  (a) PR curves for 5 folds + mean (with std band)
  (b) Post-acid vs Pre-acid PR comparison

Usage:
    python visualize/plot_pr_curves.py [--config config_final.yaml]

Output:
    outputs/{figures}/fig5_pr_curves.pdf
    outputs/{figures}/fig5_pr_curves.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FOLD_COLORS = ["#1E88E5", "#43A047", "#FB8C00", "#E53935", "#8E24AA"]


def plot_fold_pr_curves(ax, df, title="Precision-Recall Curves"):
    """Plot PR curves for each fold + mean with std band."""
    precisions_list = []
    aps = []
    mean_recall = np.linspace(0, 1, 100)

    for fold in sorted(df["fold"].unique()):
        fold_data = df[df["fold"] == fold]
        precision, recall, _ = precision_recall_curve(fold_data["y_true"], fold_data["y_prob"])
        ap = average_precision_score(fold_data["y_true"], fold_data["y_prob"])
        aps.append(ap)

        # Interpolate for mean computation
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions_list.append(interp_precision)

        ax.step(recall, precision, color=FOLD_COLORS[fold % len(FOLD_COLORS)],
                alpha=0.4, linewidth=1, where="post",
                label=f"Fold {fold+1} (AP = {ap:.3f})")

    # Mean PR curve
    mean_precision = np.mean(precisions_list, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    ax.step(mean_recall, mean_precision, color="#1565C0", linewidth=2.5, where="post",
            label=f"Mean (AP = {mean_ap:.3f} ± {std_ap:.3f})")

    # Confidence band
    std_precision = np.std(precisions_list, axis=0)
    upper = np.minimum(mean_precision + std_precision, 1)
    lower = np.maximum(mean_precision - std_precision, 0)
    ax.fill_between(mean_recall, lower, upper, color="#1565C0", alpha=0.15)

    # Random baseline (positives / total)
    n_pos = (df["y_true"] == 1).sum()
    n_total = len(df)
    baseline = n_pos / n_total
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Random (baseline = {baseline:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)


def plot_comparison_pr(ax, post_df, pre_df):
    """Compare post-acid and pre-acid PR curves."""
    for df, label, color, lw in [
        (post_df, "Post-acid", "#E53935", 2.5),
        (pre_df, "Pre-acid", "#78909C", 2.0),
    ]:
        if df is None:
            continue
        precision, recall, _ = precision_recall_curve(df["y_true"], df["y_prob"])
        ap = average_precision_score(df["y_true"], df["y_prob"])
        ax.step(recall, precision, color=color, linewidth=lw, where="post",
                label=f"{label} (AP = {ap:.3f})")

    n_pos = (post_df["y_true"] == 1).sum()
    baseline = n_pos / len(post_df)
    ax.axhline(y=baseline, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("(b) Post-acid vs Pre-acid", fontweight="bold")
    ax.legend(loc="lower left", fontsize=11, framealpha=0.8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_final.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["figures"], exist_ok=True)

    post_df = None
    pre_df = None

    for run_name in ["baseline_post", "baseline_pre"]:
        path = os.path.join(cfg["output"]["metrics"], f"{run_name}_predictions.csv")
        if os.path.isfile(path):
            if "post" in run_name:
                post_df = pd.read_csv(path)
            else:
                pre_df = pd.read_csv(path)

    if post_df is None:
        print("ERROR: No post-acid predictions found.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_fold_pr_curves(axes[0], post_df, "(a) 5-Fold Precision-Recall (Post-acid)")
    plot_comparison_pr(axes[1], post_df, pre_df)

    fig.suptitle(
        "Figure 5 — Precision-Recall Analysis",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(cfg["output"]["figures"], f"fig5_pr_curves.{fmt}"))
    plt.close()

    print(f"Figure 5 saved to {cfg['output']['figures']}/fig5_pr_curves.[pdf|png]")


if __name__ == "__main__":
    main()
