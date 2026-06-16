"""
Piste 2 — Wilcoxon Signed-Rank Test on Fold-AUCs.

Compares each ablation configuration against the full model (baseline_post)
using the 5 per-fold AUC values already stored in final_results_table.csv.
No retraining required.

Why Wilcoxon (not DeLong here)?
  DeLong is appropriate for pooled predictions across patients.
  Wilcoxon signed-rank on the 5 fold-AUCs treats each fold as a
  paired observation, which is the correct paired-CV comparison.
  NOTE: With n=5 pairs, the minimum achievable two-sided p is 0.0625.
  Results are reported for effect-size transparency, not for binary
  significance claims.

Usage:
    python evaluate/wilcoxon_fold_aucs.py [--config config.yaml]
    python evaluate/wilcoxon_fold_aucs.py --csv outputs_final/metrics/final_results_table.csv

Output:
    - outputs/metrics/wilcoxon_fold_aucs.csv
    - logged to outputs/logs/statistical_tests.log  (appended)
"""

import argparse
import csv
import logging
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── optional project utilities (graceful fallback if run standalone) ──────────
try:
    from src.utils import load_config, setup_logger, save_json
    HAS_PROJECT_UTILS = True
except ImportError:
    HAS_PROJECT_UTILS = False

    def load_config(path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    def setup_logger(name, log_path):
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        return logging.getLogger(name)

    def save_json(data, path):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# ── configs to evaluate ───────────────────────────────────────────────────────
# Maps human-readable label -> CSV prefix (lower-case, matching Condition column)
ABLATION_CONFIGS = {
    # Leave-one-out
    "No 1x (LOO)":      "ablation_no_1x",
    "No 3x (LOO)":      "ablation_no_3x",
    "No 5x (LOO)":      "ablation_no_5x",
    "No 7x (LOO)":      "ablation_no_7x",
    # Attention ablation
    "No Attention":      "ablation_no_attention",
    # Single-magnification
    "1x Only":           "ablation_1x_only",
    "3x Only":           "ablation_3x_only",
    "5x Only":           "ablation_5x_only",
    "7x Only":           "ablation_7x_only",
    # Two-magnification combos
    "1x+3x":             "ablation_1x_3x",
    "1x+5x":             "ablation_1x_5x",
    "1x+7x":             "ablation_1x_7x",
    "3x+5x":             "ablation_3x_5x",
    "3x+7x":             "ablation_3x_7x",
    "5x+7x":             "ablation_5x_7x",
}

BASELINE_PREFIX = "baseline_post"
N_FOLDS = 5
MIN_ACHIEVABLE_P = 0.0625  # theoretical min for n=5 two-sided Wilcoxon


# ── helpers ───────────────────────────────────────────────────────────────────

def load_fold_aucs(csv_path: Path) -> dict:
    """
    Read final_results_table.csv and return a dict mapping
    lower-case condition name -> float AUC.
    Handles the ± separator regardless of encoding.
    """
    auc_map = {}
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Condition"].strip().lower()
            m = re.match(r"^\s*([\d.]+)", row["AUC"])
            if m:
                auc_map[name] = float(m.group(1))
    return auc_map


def get_fold_aucs(auc_map: dict, prefix: str) -> list:
    """Return the N_FOLDS per-fold AUCs for a given config prefix."""
    keys = [f"{prefix}_fold_{i}" for i in range(N_FOLDS)]
    missing = [k for k in keys if k not in auc_map]
    if missing:
        raise KeyError(f"Missing fold entries in CSV: {missing}")
    return [auc_map[k] for k in keys]


def run_wilcoxon(baseline_folds: list, ablation_folds: list) -> dict:
    """
    Paired Wilcoxon signed-rank test between two lists of fold AUCs.
    Returns a dict with stat, p_value, delta, and per-fold info.
    """
    diffs = [a - b for a, b in zip(ablation_folds, baseline_folds)]
    bl_mean  = float(np.mean(baseline_folds))
    abl_mean = float(np.mean(ablation_folds))
    delta    = abl_mean - bl_mean

    if all(d == 0.0 for d in diffs):
        return {
            "baseline_mean_auc":  round(bl_mean, 4),
            "ablation_mean_auc":  round(abl_mean, 4),
            "delta_auc":          0.0,
            "W_statistic":        None,
            "p_value":            None,
            "note":               "all differences are zero",
            "baseline_fold_aucs": [round(v, 4) for v in baseline_folds],
            "ablation_fold_aucs": [round(v, 4) for v in ablation_folds],
        }

    stat, p = wilcoxon(ablation_folds, baseline_folds,
                       alternative="two-sided", zero_method="wilcox")
    return {
        "baseline_mean_auc":  round(bl_mean, 4),
        "ablation_mean_auc":  round(abl_mean, 4),
        "delta_auc":          round(delta, 4),
        "W_statistic":        round(float(stat), 2),
        "p_value":            round(float(p), 4),
        "baseline_fold_aucs": [round(v, 4) for v in baseline_folds],
        "ablation_fold_aucs": [round(v, 4) for v in ablation_folds],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wilcoxon signed-rank test on fold-AUCs (no retraining needed)"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Project config YAML (used to resolve output paths)")
    parser.add_argument("--csv", default=None,
                        help="Path to final_results_table.csv (overrides config)")
    args = parser.parse_args()

    # ── resolve paths ─────────────────────────────────────────────────────────
    if args.csv:
        csv_path = Path(args.csv)
        out_csv  = csv_path.parent / "wilcoxon_fold_aucs.csv"
        out_json = csv_path.parent / "wilcoxon_fold_aucs.json"
        log_path = csv_path.parent.parent / "logs" / "statistical_tests.log"
    elif HAS_PROJECT_UTILS and Path(args.config).exists():
        cfg      = load_config(args.config)
        metrics  = Path(cfg["output"]["metrics"])
        csv_path = metrics / "final_results_table.csv"
        out_csv  = metrics / "wilcoxon_fold_aucs.csv"
        out_json = metrics / "wilcoxon_fold_aucs.json"
        log_path = Path(cfg["output"]["logs"]) / "statistical_tests.log"
    else:
        # Fallback: look relative to script location
        root     = Path(__file__).resolve().parent.parent
        csv_path = root / "outputs_final" / "metrics" / "final_results_table.csv"
        out_csv  = csv_path.parent / "wilcoxon_fold_aucs.csv"
        out_json = csv_path.parent / "wilcoxon_fold_aucs.json"
        log_path = root / "outputs_final" / "logs" / "statistical_tests.log"

    logger = setup_logger("wilcoxon", str(log_path))

    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        sys.exit(1)

    # ── load data ─────────────────────────────────────────────────────────────
    auc_map  = load_fold_aucs(csv_path)
    baseline = get_fold_aucs(auc_map, BASELINE_PREFIX)
    bl_mean  = float(np.mean(baseline))

    logger.info("\n" + "=" * 72)
    logger.info("  Wilcoxon Signed-Rank Test: Ablation vs Full Model (n=5 fold-AUCs)")
    logger.info("=" * 72)
    logger.info(f"  Baseline ({BASELINE_PREFIX}) fold AUCs : {[round(v,4) for v in baseline]}")
    logger.info(f"  Baseline mean AUC               : {bl_mean:.4f}")
    logger.info(f"  Min achievable two-sided p (n=5): {MIN_ACHIEVABLE_P}")
    logger.info("=" * 72)

    hdr = (f"  {'Configuration':<20} {'BlAUC':>7} {'AblAUC':>7} "
           f"{'dAUC':>7} {'W':>6} {'p':>8}  Folds (abl)")
    logger.info(hdr)
    logger.info("  " + "-" * 70)

    all_results = {}
    csv_rows    = []

    for label, prefix in ABLATION_CONFIGS.items():
        try:
            abl_folds = get_fold_aucs(auc_map, prefix)
        except KeyError as e:
            logger.warning(f"  [{label}] Skipped — {e}")
            continue

        res = run_wilcoxon(baseline, abl_folds)
        all_results[label] = res

        w_str = f"{res['W_statistic']:.1f}" if res["W_statistic"] is not None else " n/a"
        p_str = f"{res['p_value']:.4f}"     if res["p_value"]     is not None else "  n/a"
        folds_str = str([round(v, 3) for v in abl_folds])

        logger.info(
            f"  {label:<20} {bl_mean:>7.4f} {res['ablation_mean_auc']:>7.4f} "
            f"{res['delta_auc']:>+7.4f} {w_str:>6} {p_str:>8}  {folds_str}"
        )

        csv_rows.append({
            "configuration":     label,
            "csv_prefix":        prefix,
            "baseline_mean_auc": res["baseline_mean_auc"],
            "ablation_mean_auc": res["ablation_mean_auc"],
            "delta_auc":         res["delta_auc"],
            "W_statistic":       res["W_statistic"] if res["W_statistic"] is not None else "",
            "p_value":           res["p_value"]     if res["p_value"]     is not None else "",
            "baseline_fold_aucs": str(res["baseline_fold_aucs"]),
            "ablation_fold_aucs": str(res["ablation_fold_aucs"]),
        })

    logger.info("  " + "-" * 70)
    logger.info(f"\n  NOTE: With n={N_FOLDS} pairs, p < 0.05 is mathematically unreachable")
    logger.info("  for a two-sided Wilcoxon test. Interpret W and p as effect-size")
    logger.info("  indicators, not binary significance decisions.")
    logger.info("  Lowest achievable p=0.0625 (W=0) indicates strongest evidence.\n")

    # ── save outputs ──────────────────────────────────────────────────────────
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    save_json({"wilcoxon_fold_aucs": all_results}, str(out_json))

    logger.info(f"  Results saved:")
    logger.info(f"    CSV  -> {out_csv}")
    logger.info(f"    JSON -> {out_json}\n")


if __name__ == "__main__":
    main()
