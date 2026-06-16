"""
Piste 2 — Calibration Analysis (Figure 6).

Generates:
  (a) Reliability diagram (calibration curve) with histogram
  (b) Post-acid vs Pre-acid calibration comparison

Usage:
    python visualize/plot_calibration.py [--config config_final.yaml]

Output:
    outputs/{figures}/fig6_calibration.pdf
    outputs/{figures}/fig6_calibration.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

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

NORMAL_COLOR = "#66BB6A"
ABNORMAL_COLOR = "#EF5350"


def plot_calibration_curve(ax, ax_hist, y_true, y_prob, n_bins=10, label="Post-acid"):
    """Plot reliability diagram with histogram of predicted probabilities."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    # Calibration curve
    ax.plot(prob_pred, prob_true, marker="o", color="#1565C0", linewidth=2.5,
            markersize=8, label=f"{label} (Brier = {brier:.4f})")

    # Perfect calibration
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Perfectly calibrated")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("(a) Reliability Diagram", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.8)

    # Histogram of predicted probabilities by class
    y_prob_normal = y_prob[y_true == 0]
    y_prob_abnormal = y_prob[y_true == 1]

    ax_hist.hist(y_prob_normal, bins=n_bins, range=(0, 1), alpha=0.6,
                 color=NORMAL_COLOR, edgecolor="white", label="Normal")
    ax_hist.hist(y_prob_abnormal, bins=n_bins, range=(0, 1), alpha=0.6,
                 color=ABNORMAL_COLOR, edgecolor="white", label="Abnormal")
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("(c) Prediction Distribution", fontweight="bold")
    ax_hist.legend(framealpha=0.8)
    ax_hist.set_xlim([0, 1])


def plot_comparison_calibration(ax, post_df, pre_df, n_bins=10):
    """Compare post-acid and pre-acid calibration."""
    if post_df is None or pre_df is None:
        ax.text(0.5, 0.5, "Both pre and post-acid data required",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(b) Calibration Comparison", fontweight="bold")
        return

    for df, label, color, lw in [
        (post_df, "Post-acid", "#E53935", 2.5),
        (pre_df, "Pre-acid", "#78909C", 2.0),
    ]:
        prob_true, prob_pred = calibration_curve(df["y_true"], df["y_prob"],
                                                  n_bins=n_bins, strategy="uniform")
        brier = brier_score_loss(df["y_true"], df["y_prob"])
        ax.plot(prob_pred, prob_true, marker="o", color=color, linewidth=lw,
                markersize=8, label=f"{label} (Brier = {brier:.4f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed fraction of positives")
    ax.set_title("(b) Pre-acid vs Post-acid", fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.8)


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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Reliability diagram for post-acid
    ax = axes[0]
    ax_hist = axes[2]
    plot_calibration_curve(ax, ax_hist, post_df["y_true"], post_df["y_prob"],
                            n_bins=10, label="Post-acid")

    # (b) Pre vs Post calibration comparison
    plot_comparison_calibration(axes[1], post_df, pre_df, n_bins=10)

    fig.suptitle(
        "Figure 6 — Model Calibration",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(cfg["output"]["figures"], f"fig6_calibration.{fmt}"))
    plt.close()

    print(f"Figure 6 saved to {cfg['output']['figures']}/fig6_calibration.[pdf|png]")


if __name__ == "__main__":
    main()
