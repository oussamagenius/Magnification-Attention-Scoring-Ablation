"""
Piste 2 — Attention Weights Visualization (Figure 1).

Generates publication-ready plots showing magnification importance:
  (a) Bar chart: mean α per magnification with error bars
  (b) Violin plot: full distribution of α
  (c) Grouped bars: α by magnification × class (Normal vs Abnormal)
  (d) Heatmap: α per patient sorted by prediction probability
  (e) Pre-acid vs Post-acid comparison

Usage:
    python visualize/plot_attention_weights.py [--config config.yaml]

Output:
    outputs/figures/fig1_attention_weights.pdf
    outputs/figures/fig1_attention_weights.png
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Color palette
COLORS = {
    "1x": "#2196F3",  # Blue
    "3x": "#4CAF50",  # Green
    "5x": "#FF9800",  # Orange
    "7x": "#E91E63",  # Pink
}
NORMAL_COLOR = "#66BB6A"
ABNORMAL_COLOR = "#EF5350"


def load_predictions(cfg, run_name="baseline_post"):
    """Load all predictions for a condition."""
    path = os.path.join(cfg["output"]["metrics"], f"{run_name}_predictions.csv")
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path)


def plot_panel_a(ax, df, attn_cols, mag_labels):
    """Bar chart: mean attention weight per magnification."""
    means = [df[col].mean() for col in attn_cols]
    stds = [df[col].std() for col in attn_cols]
    colors = [COLORS.get(m, "#999") for m in mag_labels]

    bars = ax.bar(mag_labels, means, yerr=stds, color=colors, edgecolor="white",
                  capsize=5, linewidth=0.8, alpha=0.85, error_kw={"linewidth": 1.5})

    ax.set_ylabel("Mean Attention Weight (α)")
    ax.set_xlabel("Magnification Level")
    ax.set_title("(a) Magnification Importance", fontweight="bold")
    ax.set_ylim(0, max(means) * 1.4)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5, label="Uniform (1/4)")
    ax.legend(loc="upper right", framealpha=0.7)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)


def plot_panel_b(ax, df, attn_cols, mag_labels):
    """Violin plot: distribution of attention weights."""
    data_long = []
    for col, mag in zip(attn_cols, mag_labels):
        for val in df[col].values:
            data_long.append({"Magnification": mag, "α": val})
    data_long = pd.DataFrame(data_long)

    palette = [COLORS.get(m, "#999") for m in mag_labels]
    parts = sns.violinplot(
        data=data_long, x="Magnification", y="α", hue="Magnification", ax=ax,
        palette=palette, inner="box", cut=0, alpha=0.8, legend=False,
    )

    ax.set_ylabel("Attention Weight (α)")
    ax.set_title("(b) Attention Weight Distribution", fontweight="bold")
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.5)


def plot_panel_c(ax, df, attn_cols, mag_labels):
    """Grouped bar chart: α by magnification × class."""
    normal = df[df["y_true"] == 0]
    abnormal = df[df["y_true"] == 1]

    x = np.arange(len(mag_labels))
    width = 0.35

    means_n = [normal[col].mean() for col in attn_cols]
    stds_n = [normal[col].std() for col in attn_cols]
    means_a = [abnormal[col].mean() for col in attn_cols]
    stds_a = [abnormal[col].std() for col in attn_cols]

    ax.bar(x - width / 2, means_n, width, yerr=stds_n, label="Normal",
           color=NORMAL_COLOR, edgecolor="white", capsize=3, alpha=0.85)
    ax.bar(x + width / 2, means_a, width, yerr=stds_a, label="Abnormal",
           color=ABNORMAL_COLOR, edgecolor="white", capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(mag_labels)
    ax.set_ylabel("Mean Attention Weight (α)")
    ax.set_xlabel("Magnification Level")
    ax.set_title("(c) Importance by Diagnosis", fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.3)


def plot_panel_d(ax, df, attn_cols, mag_labels):
    """Heatmap: attention weights per patient, sorted by prediction probability."""
    df_sorted = df.sort_values("y_prob", ascending=True).reset_index(drop=True)
    attn_matrix = df_sorted[attn_cols].values

    im = ax.imshow(attn_matrix.T, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=attn_matrix.max())

    ax.set_yticks(range(len(mag_labels)))
    ax.set_yticklabels(mag_labels)
    ax.set_xlabel("Patients (sorted by P(Abnormal) →)")
    ax.set_title("(d) Per-patient Attention Heatmap", fontweight="bold")

    # Add class boundary indicator
    n_normal = len(df_sorted[df_sorted["y_true"] == 0])
    ax.axvline(x=n_normal - 0.5, color="white", linewidth=2, linestyle="--")
    ax.text(n_normal / 2, -0.8, "Normal", ha="center", fontsize=9, color=NORMAL_COLOR)
    ax.text(n_normal + (len(df_sorted) - n_normal) / 2, -0.8, "Abnormal",
            ha="center", fontsize=9, color=ABNORMAL_COLOR)

    plt.colorbar(im, ax=ax, label="α", shrink=0.8)


def plot_panel_e(ax, post_df, pre_df, mag_labels):
    """Pre-acid vs Post-acid comparison."""
    if pre_df is None:
        ax.text(0.5, 0.5, "Pre-acid data\nnot available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("(e) Pre vs Post-acid", fontweight="bold")
        return

    x = np.arange(len(mag_labels))
    width = 0.35

    post_attn = [c for c in post_df.columns if c.startswith("alpha_")]
    pre_attn = [c for c in pre_df.columns if c.startswith("alpha_")]

    means_post = [post_df[col].mean() for col in post_attn]
    stds_post = [post_df[col].std() for col in post_attn]
    means_pre = [pre_df[col].mean() for col in pre_attn]
    stds_pre = [pre_df[col].std() for col in pre_attn]

    ax.bar(x - width / 2, means_pre, width, yerr=stds_pre, label="Pre-acid",
           color="#78909C", edgecolor="white", capsize=3, alpha=0.85)
    ax.bar(x + width / 2, means_post, width, yerr=stds_post, label="Post-acid",
           color="#FF7043", edgecolor="white", capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(mag_labels)
    ax.set_ylabel("Mean Attention Weight (α)")
    ax.set_xlabel("Magnification Level")
    ax.set_title("(e) Pre-acid vs Post-acid", fontweight="bold")
    ax.legend(framealpha=0.7)
    ax.axhline(y=0.25, color="gray", linestyle="--", alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["figures"], exist_ok=True)

    # Load data
    post_df = load_predictions(cfg, "baseline_post")
    pre_df = load_predictions(cfg, "baseline_pre")

    if post_df is None:
        print("ERROR: No post-acid predictions found. Run training first.")
        sys.exit(1)

    attn_cols = [c for c in post_df.columns if c.startswith("alpha_")]
    mag_labels = [c.replace("alpha_", "") for c in attn_cols]

    # Create figure with 5 panels
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # Panel (a) — Bar chart
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a(ax_a, post_df, attn_cols, mag_labels)

    # Panel (b) — Violin
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b(ax_b, post_df, attn_cols, mag_labels)

    # Panel (c) — By class
    ax_c = fig.add_subplot(gs[0, 2])
    plot_panel_c(ax_c, post_df, attn_cols, mag_labels)

    # Panel (d) — Heatmap
    ax_d = fig.add_subplot(gs[1, :2])
    plot_panel_d(ax_d, post_df, attn_cols, mag_labels)

    # Panel (e) — Pre vs Post
    ax_e = fig.add_subplot(gs[1, 2])
    plot_panel_e(ax_e, post_df, pre_df, mag_labels)

    fig.suptitle(
        "Figure 1 — Magnification Importance Scoring via Gated Attention",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Save
    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(cfg["output"]["figures"], f"fig1_attention_weights.{fmt}"))
    plt.close()

    print(f"Figure 1 saved to {cfg['output']['figures']}/fig1_attention_weights.[pdf|png]")


if __name__ == "__main__":
    main()
