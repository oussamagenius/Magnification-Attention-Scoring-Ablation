"""
Piste 2 — Ablation Results Visualization (Figure 3).

Generates:
  (a) Bar chart: AUC for each ablation configuration
  (b) Delta AUC: performance change when removing each magnification

Usage:
    python visualize/plot_ablation_results.py [--config config.yaml]

Output:
    outputs/figures/fig3_ablation_results.pdf
    outputs/figures/fig3_ablation_results.png
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

CONFIG_COLORS = {
    "full_4mag": "#1565C0",
    "no_attention": "#607D8B",
    "drop_1x": "#EF5350",
    "drop_3x": "#FF9800",
    "drop_5x": "#66BB6A",
    "drop_7x": "#AB47BC",
    "only_1x": "#E3F2FD",
    "only_3x": "#FFF3E0",
    "only_5x": "#E8F5E9",
    "only_7x": "#F3E5F5",
}

CONFIG_LABELS = {
    "full_4mag": "Full\n(1×,3×,5×,7×)",
    "no_attention": "No Attention\n(Mean Pooling)",
    "drop_1x": "Drop 1×\n(3×,5×,7×)",
    "drop_3x": "Drop 3×\n(1×,5×,7×)",
    "drop_5x": "Drop 5×\n(1×,3×,7×)",
    "drop_7x": "Drop 7×\n(1×,3×,5×)",
    "only_1x": "Only 1×",
    "only_3x": "Only 3×",
    "only_5x": "Only 5×",
    "only_7x": "Only 7×",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--condition", default="post")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output"]["figures"], exist_ok=True)

    # Load ablation results from final_results.json
    path = os.path.join(cfg["output"]["metrics"], "final_results.json")
    if not os.path.isfile(path):
        print(f"ERROR: Final results not found: {path}. Run evaluate_results.py first.")
        sys.exit(1)

    import json
    with open(path, "r") as f:
        data = json.load(f)
        
    records = []
    # Map our run names backward to old config names if needed, or directly use them.
    # The plotting script assumes names like: full_4mag, drop_1x, only_1x, etc.
    # Our run_names are: baseline_post, ablation_no_1x, ablation_1x_only, etc.
    name_mapping = {
        "baseline_post": "full_4mag",
        "ablation_no_attention": "no_attention",
        "ablation_no_1x": "drop_1x",
        "ablation_no_3x": "drop_3x",
        "ablation_no_5x": "drop_5x",
        "ablation_no_7x": "drop_7x",
        "ablation_1x_only": "only_1x",
        "ablation_3x_only": "only_3x",
        "ablation_5x_only": "only_5x",
        "ablation_7x_only": "only_7x",
        
    }
    
    for run_name, metrics in data.items():
        mapped_name = name_mapping.get(run_name)
        if mapped_name and "auc" in metrics:
            records.append({
                "config": mapped_name,
                "auc_mean": metrics["auc"]["mean"],
                "auc_std": metrics["auc"]["std"]
            })
            
    df = pd.DataFrame(records)
    if len(df) == 0:
        print("ERROR: No valid ablation records found in final_results.json")
        sys.exit(1)


    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ================================================================
    # (a) AUC Bar Chart for all configurations
    # ================================================================
    ax = axes[0]

    configs = df["config"].tolist()
    aucs = df["auc_mean"].tolist()
    stds = df["auc_std"].tolist()
    colors = [CONFIG_COLORS.get(c, "#999") for c in configs]
    labels = [CONFIG_LABELS.get(c, c) for c in configs]

    bars = ax.bar(range(len(configs)), aucs, yerr=stds, color=colors,
                  edgecolor="white", capsize=4, linewidth=0.8, alpha=0.9)

    # Baseline line
    full_auc = df[df["config"] == "full_4mag"]["auc_mean"].values[0]
    ax.axhline(y=full_auc, color="#1565C0", linestyle="--", alpha=0.6,
               label=f"Full model ({full_auc:.3f})")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("(a) Classification Performance by Configuration", fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.8)

    # Value labels
    for i, (bar, v) in enumerate(zip(bars, aucs)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[i] + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Separator between LOO and single configs
    ax.axvline(x=4.5, color="gray", linestyle=":", alpha=0.5)
    ax.text(2, ax.get_ylim()[0] + 0.02, "Leave-One-Out", ha="center", fontsize=9, color="gray")
    ax.text(7, ax.get_ylim()[0] + 0.02, "Single Magnification", ha="center", fontsize=9, color="gray")

    # ================================================================
    # (b) Delta AUC (drop impact)
    # ================================================================
    ax = axes[1]

    # Leave-one-out analysis
    loo_configs = ["drop_1x", "drop_3x", "drop_5x", "drop_7x"]
    loo_labels = ["Drop 1×", "Drop 3×", "Drop 5×", "Drop 7×"]
    loo_deltas = []
    loo_colors_bar = []

    for cfg_name in loo_configs:
        row = df[df["config"] == cfg_name]
        if len(row) > 0:
            delta = row["auc_mean"].values[0] - full_auc
            loo_deltas.append(delta)
            loo_colors_bar.append("#EF5350" if delta < 0 else "#66BB6A")
        else:
            loo_deltas.append(0)
            loo_colors_bar.append("#999")

    bars = ax.barh(range(len(loo_configs)), loo_deltas, color=loo_colors_bar,
                   edgecolor="white", alpha=0.85, height=0.6)

    ax.set_yticks(range(len(loo_configs)))
    ax.set_yticklabels(loo_labels)
    ax.set_xlabel("Δ AUC (vs Full Model)")
    ax.set_title("(b) Impact of Removing Each Magnification", fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)

    # Value labels
    for i, (bar, delta) in enumerate(zip(bars, loo_deltas)):
        offset = -0.003 if delta < 0 else 0.003
        ha = "right" if delta < 0 else "left"
        ax.text(delta + offset, bar.get_y() + bar.get_height() / 2,
                f"{delta:+.4f}", ha=ha, va="center", fontsize=10, fontweight="bold")

    # Annotation: most critical magnification
    most_critical_idx = np.argmin(loo_deltas)
    ax.annotate(
        f"Most critical:\n{loo_labels[most_critical_idx]}",
        xy=(loo_deltas[most_critical_idx], most_critical_idx),
        xytext=(loo_deltas[most_critical_idx] - 0.05, most_critical_idx + 0.8),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9, fontweight="bold", color="#E53935",
    )

    fig.suptitle(
        f"Figure 3 — Ablation Study",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for fmt in ["pdf", "png"]:
        fig.savefig(os.path.join(cfg["output"]["figures"], f"fig3_ablation_results.{fmt}"))
    plt.close()

    print(f"Figure 3 saved to {cfg['output']['figures']}/fig3_ablation_results.[pdf|png]")


if __name__ == "__main__":
    main()
