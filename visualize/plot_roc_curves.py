"""
Piste 2 — ROC Curves Visualization (Figure 2).

Generates:
  (a) ROC curves for 5 folds + mean (with confidence band)
  (b) Post-acid vs Pre-acid ROC comparison

Usage:
    python visualize/plot_roc_curves.py [--config config.yaml]

Output:
    outputs/figures/fig2_roc_curves.pdf
    outputs/figures/fig2_roc_curves.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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


def plot_fold_rocs(ax, df, title="ROC Curves"):
    """Plot ROC curves for each fold + mean with confidence band."""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold in sorted(df["fold"].unique()):
        fold_data = df[df["fold"] == fold]
        fpr, tpr, _ = roc_curve(fold_data["y_true"], fold_data["y_prob"])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate for mean computation
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(fpr, tpr, color=FOLD_COLORS[fold % len(FOLD_COLORS)],
                alpha=0.4, linewidth=1,
                label=f"Fold {fold+1} (AUC = {roc_auc:.3f})")

    # Mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color="#1565C0", linewidth=2.5,
            label=f"Mean (AUC = {mean_auc:.3f} ± {std_auc:.3f})")

    # Confidence band
    std_tpr = np.std(tprs, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color="#1565C0", alpha=0.15)

    # Diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.8)
    ax.set_aspect("equal")


def plot_comparison_roc(ax, post_df, pre_df):
    """Compare post-acid and pre-acid ROC curves."""
    for df, label, color, lw in [
        (post_df, "Post-acid", "#E53935", 2.5),
        (pre_df, "Pre-acid", "#78909C", 2.0),
    ]:
        if df is None:
            continue
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=lw,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("(b) Post-acid vs Pre-acid", fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.8)
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["figures"], exist_ok=True)

    post_df = None
    pre_df = None

    for cond in ["baseline_post", "baseline_pre"]:
        path = os.path.join(cfg["output"]["metrics"], f"{cond}_predictions.csv")
        if os.path.isfile(path):
            if "post" in cond:
                post_df = pd.read_csv(path)
            else:
                pre_df = pd.read_csv(path)

    if post_df is None:
        print("ERROR: No post-acid predictions found.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Per-fold ROC
    plot_fold_rocs(axes[0], post_df, "(a) 5-Fold Cross-Validation ROC (Post-acid)")

    # (b) Post vs Pre comparison
    plot_comparison_roc(axes[1], post_df, pre_df)

    fig.suptitle(
        "Figure 2 — ROC Analysis",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(cfg["output"]["figures"], f"fig2_roc_curves.{fmt}"))
    plt.close()

    print(f"Figure 2 saved to {cfg['output']['figures']}/fig2_roc_curves.[pdf|png]")


if __name__ == "__main__":
    main()
