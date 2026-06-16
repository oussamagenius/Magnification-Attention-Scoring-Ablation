"""
Piste 2 — Confusion Matrices Visualization (Figure 4).

Generates aggregated confusion matrices (sum across 5 folds)
for post-acid and pre-acid conditions.

Usage:
    python visualize/plot_confusion_matrices.py [--config config.yaml]

Output:
    outputs/figures/fig4_confusion_matrices.pdf
    outputs/figures/fig4_confusion_matrices.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, compute_metrics

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_cm(ax, y_true, y_prob, title, threshold=None):
    """Plot a single confusion matrix with percentages and counts."""
    if threshold is None:
        metrics = compute_metrics(y_true, y_prob)
        threshold = metrics["threshold"]

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_pct = cm / cm.sum() * 100

    # Annotation with count + percentage
    annot = np.array([
        [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(2)]
        for i in range(2)
    ])

    sns.heatmap(
        cm, annot=annot, fmt="", cmap="Blues", ax=ax,
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
        cbar=True, cbar_kws={"label": "Count"},
        linewidths=1, linecolor="white",
        annot_kws={"size": 13, "fontweight": "bold"},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontweight="bold", fontsize=13)

    # Add metrics text
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    from sklearn.metrics import roc_auc_score
    auc_val = roc_auc_score(y_true, y_prob)

    text = (
        f"AUC: {auc_val:.3f}\n"
        f"Sens: {sens:.3f}\n"
        f"Spec: {spec:.3f}\n"
        f"Threshold: {threshold:.3f}"
    )
    ax.text(2.6, 0.5, text, transform=ax.transData,
            fontsize=10, verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["figures"], exist_ok=True)

    conditions = []
    for cond in ["baseline_post", "baseline_pre"]:
        path = os.path.join(cfg["output"]["metrics"], f"{cond}_predictions.csv")
        if os.path.isfile(path):
            display_name = "Post" if "post" in cond else "Pre"
            conditions.append((display_name, pd.read_csv(path)))

    if not conditions:
        print("ERROR: No prediction files found.")
        sys.exit(1)

    n_plots = len(conditions)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5.5))
    if n_plots == 1:
        axes = [axes]

    for i, (cond, df) in enumerate(conditions):
        plot_cm(
            axes[i],
            df["y_true"].values,
            df["y_prob"].values,
            f"({'abcde'[i]}) {cond.capitalize()}-acid (Aggregated 5-Fold)",
        )

    fig.suptitle(
        "Figure 4 — Confusion Matrices",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(cfg["output"]["figures"], f"fig4_confusion_matrices.{fmt}"))
    plt.close()

    print(f"Figure 4 saved to {cfg['output']['figures']}/fig4_confusion_matrices.[pdf|png]")


if __name__ == "__main__":
    main()
