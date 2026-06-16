"""
Full Study Runner.

Orchestrates the complete experimental pipeline in three phases:
  1. Baseline training (post-acid and pre-acid conditions)
  2. Sensitivity analysis (corrected and strict label variants)
  3. Ablation study (16 configurations: single/pair/leave-one-out/no-attention)

Each phase calls src/train.py as a subprocess with appropriate CLI arguments.
After all phases complete, a Markdown summary report is generated.

Usage:
    python run_full_study.py [--phases all|1,2,3] [--config config.yaml] [--dry_run]
"""
import os
import subprocess
import argparse
import sys
import json
import pandas as pd

PHASES = {
    "1_baseline": [
        {"run_name": "baseline_post", "manifest": "manifeste_original.csv", "condition": "post"},
        {"run_name": "baseline_pre", "manifest": "manifeste_original.csv", "condition": "pre"},
        # {"run_name": "baseline_both", "manifest": "manifeste_original.csv", "condition": "both"}
    ],
    "2_sensitivity": [
        {"run_name": "sens_corrige_post", "manifest": "manifeste_corrige.csv", "condition": "post"},
        {"run_name": "sens_strict_post", "manifest": "manifeste_strict.csv", "condition": "post"}
    ],
    "3_ablation": [
        {"run_name": "ablation_1x_only", "manifest": "manifeste_original.csv", "condition": "post", "mags": "0", "no_save": True},
        {"run_name": "ablation_3x_only", "manifest": "manifeste_original.csv", "condition": "post", "mags": "1", "no_save": True},
        {"run_name": "ablation_5x_only", "manifest": "manifeste_original.csv", "condition": "post", "mags": "2", "no_save": True},
        {"run_name": "ablation_7x_only", "manifest": "manifeste_original.csv", "condition": "post", "mags": "3", "no_save": True},
        {"run_name": "ablation_1x_3x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "0,1", "no_save": True},
        {"run_name": "ablation_1x_5x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "0,2", "no_save": True},
        {"run_name": "ablation_1x_7x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "0,3", "no_save": True},
        {"run_name": "ablation_3x_5x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "1,2", "no_save": True},
        {"run_name": "ablation_3x_7x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "1,3", "no_save": True},
        {"run_name": "ablation_5x_7x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "2,3", "no_save": True},
        {"run_name": "ablation_no_1x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "1,2,3", "no_save": True},
        {"run_name": "ablation_no_7x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "0,1,2", "no_save": True},
        {"run_name": "ablation_no_3x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "0,2,3", "no_save": True},
        {"run_name": "ablation_no_5x", "manifest": "manifeste_original.csv", "condition": "post", "mags": "0,1,3", "no_save": True},
        {"run_name": "ablation_no_attention", "manifest": "manifeste_original.csv", "condition": "post", "pooling": "mean", "no_save": True},
    ]
}

def run_experiment(config, config_file="config.yaml", dry_run=False):
    """Run a single train.py configuration."""
    cmd = [
        sys.executable, "src/train.py",
        "--condition", config["condition"],
        "--manifest", config["manifest"],
        "--run_name", config["run_name"],
        "--config", config_file
    ]
    
    if "pooling" in config:
        cmd.extend(["--pooling", config["pooling"]])
    if "mags" in config:
        cmd.extend(["--mags", config["mags"]])
    if "data_root" in config:
        cmd.extend(["--data_root", config["data_root"]])
    if "no_save" in config and config["no_save"]:
        cmd.append("--no_save_checkpoint")
        
    print(f"\n{'='*80}\nRUNNING: {config['run_name']}\nCOMMAND: {' '.join(cmd)}\n{'='*80}")
    
    if not dry_run:
        subprocess.run(cmd, check=True)
    else:
        print("[DRY RUN] Would execute command above.")

def extract_metric(json_path, metric_key, use_tta=False):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if use_tta and "tta" in data:
            return f"{data['tta'][metric_key]['mean']:.3f} ± {data['tta'][metric_key]['std']:.3f}"
        return f"{data[metric_key]['mean']:.3f} ± {data[metric_key]['std']:.3f}"
    except (FileNotFoundError, KeyError):
        return "N/A"


def _compute_ensemble_auc(metrics_dir, run_name):
    import glob as _glob
    import pandas as _pd
    import numpy as _np
    from sklearn.metrics import roc_auc_score

    patterns = [
        os.path.join(metrics_dir, f"{run_name}_fold_*_predictions_tta.csv"),
        os.path.join(metrics_dir, f"{run_name}_fold_*_predictions.csv"),
    ]
    files = []
    for pattern in patterns:
        files = sorted(_glob.glob(pattern))
        if files:
            break
    if not files:
        return "N/A"

    dfs = []
    for f in files:
        dfs.append(_pd.read_csv(f))
    all_preds = _pd.concat(dfs, ignore_index=True)
    ensemble = all_preds.groupby("patient_id").agg({"y_true": "first", "y_prob": "mean"}).reset_index()
    auc = roc_auc_score(ensemble["y_true"].values, ensemble["y_prob"].values)
    return f"{auc:.3f}"

def generate_summary_report():
    print("\nGenerating Summary Report (summary_results.md)...")
    
    report = ["# Piste 2 — Ablation & Sensitivity Study Results (V3: TTA + Ensemble)", ""]
    
    # Baseline
    report.extend(["## 1. Baseline Performance (Original Labels)", ""])
    report.extend(["| Condition | AUC | TTA AUC | Ensemble AUC | Sensitivity | Specificity |", "|---|---|---|---|---|---|"])
    for run in PHASES["1_baseline"]:
        p = f"outputs_final/metrics/{run['run_name']}_summary.json"
        report.append(
            f"| {run['run_name']} | {extract_metric(p, 'auc')} | "
            f"{extract_metric(p, 'auc', use_tta=True)} | "
            f"{_compute_ensemble_auc('outputs_final/metrics', run['run_name'])} | "
            f"{extract_metric(p, 'sensitivity')} | {extract_metric(p, 'specificity')} |"
        )
    
    # Sensitivity
    report.extend(["", "## 2. Sensitivity Analysis (Label Correction)", ""])
    report.extend(["| Configuration | AUC | Sensitivity | Specificity |", "|---|---|---|---|"])
    for run in [{"run_name": "baseline_post", "manifest": "Original"}] + \
             [{"run_name": r["run_name"], "manifest": r["manifest"]} for r in PHASES["2_sensitivity"]]:
        p = f"outputs_final/metrics/{run['run_name']}_summary.json"
        report.append(f"| {run['manifest']} | {extract_metric(p, 'auc')} | {extract_metric(p, 'sensitivity')} | {extract_metric(p, 'specificity')} |")
        
    # Ablation
    report.extend(["", "## 3. Ablation Study (Configurations)", ""])
    report.extend(["| Configuration | AUC | Sensitivity | Specificity |", "|---|---|---|---|"])
    for run in [{"run_name": "baseline_post", "name": "Full Model (4 mags + Attention)"}] + \
             [{"run_name": r["run_name"], "name": r["run_name"]} for r in PHASES["3_ablation"]]:
        p = f"outputs_final/metrics/{run['run_name']}_summary.json"
        report.append(f"| {run['name']} | {extract_metric(p, 'auc')} | {extract_metric(p, 'sensitivity')} | {extract_metric(p, 'specificity')} |")
        
    with open("summary_results.md", "w") as f:
        f.write("\n".join(report))
        
    print("Report generated: summary_results.md")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="all", help="Comma separated phases (1,2,3) or 'all'")
    parser.add_argument("--config", default="config.yaml", help="Config file to use for all runs")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--report_only", action="store_true", help="Only generate the markdown report")
    parser.add_argument("--jobs", type=int, default=1, help="Number of configurations to run in parallel")
    args = parser.parse_args()

    if args.report_only:
        generate_summary_report()
        return

    phases_to_run = list(PHASES.keys())
    if args.phases != "all":
        selected_numbers = args.phases.split(",")
        phases_to_run = [p for p in PHASES.keys() if any(p.startswith(num) for num in selected_numbers)]

    print(f"Executing phases: {phases_to_run}")
    
    from concurrent.futures import ThreadPoolExecutor

    for phase in phases_to_run:
        print(f"\n\n{'#'*80}")
        print(f"STARTING PHASE: {phase.upper()} (Max parallel jobs: {args.jobs})")
        print(f"{'#'*80}")
        
        configs = PHASES[phase]
        if args.jobs > 1 and not args.dry_run:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                # We use a lambda to pass arguments
                futures = [executor.submit(run_experiment, config, args.config, args.dry_run) for config in configs]
                # Wait for all to finish
                for future in futures:
                    future.result() # This will raise any exceptions that occurred
        else:
            for config in configs:
                run_experiment(config, config_file=args.config, dry_run=args.dry_run)
            
    if not args.dry_run:
        generate_summary_report()

if __name__ == "__main__":
    main()
