"""
Piste 2 — Training Curves Visualization (Supplementary Figure).

Shows loss and AUC convergence across epochs for all 5 folds.

Usage:
    python visualize/plot_training_curves.py [--config config.yaml]

Output:
    outputs/figures/figS1_training_curves.pdf
    outputs/figures/figS1_training_curves.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--condition", default="baseline_post")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["figures"], exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for fold in range(cfg["training"]["n_folds"]):
        log_path = os.path.join(
            cfg["output"]["logs"], f"{args.condition}_fold_{fold}_train.csv"
        )
        if not os.path.isfile(log_path):
            continue

        log = pd.read_csv(log_path)
        color = FOLD_COLORS[fold % len(FOLD_COLORS)]

        # (a) Train Loss
        axes[0, 0].plot(log["epoch"], log["train_loss"], color=color,
                        alpha=0.7, linewidth=1.5, label=f"Fold {fold+1}")

        # (b) Val Loss
        axes[0, 1].plot(log["epoch"], log["val_loss"], color=color,
                        alpha=0.7, linewidth=1.5, label=f"Fold {fold+1}")

        # (c) Train AUC
        axes[1, 0].plot(log["epoch"], log["train_auc"], color=color,
                        alpha=0.7, linewidth=1.5, label=f"Fold {fold+1}")

        # (d) Val AUC
        axes[1, 1].plot(log["epoch"], log["val_auc"], color=color,
                        alpha=0.7, linewidth=1.5, label=f"Fold {fold+1}")

        # Mark best epoch
        best_epoch = log.loc[log["val_auc"].idxmax(), "epoch"]
        best_auc = log["val_auc"].max()
        axes[1, 1].scatter([best_epoch], [best_auc], color=color,
                          s=60, zorder=5, edgecolors="black", linewidth=0.5)

    # Formatting
    titles = [
        "(a) Training Loss", "(b) Validation Loss",
        "(c) Training AUC", "(d) Validation AUC"
    ]
    ylabels = ["Loss", "Loss", "AUC-ROC", "AUC-ROC"]

    for idx, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Supplementary Figure — Training Convergence ({args.condition.capitalize()}-acid)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(cfg["output"]["figures"], f"figS1_training_curves.{fmt}"))
    plt.close()

    print(f"Training curves saved to {cfg['output']['figures']}/figS1_training_curves.[pdf|png]")


if __name__ == "__main__":
    main()
