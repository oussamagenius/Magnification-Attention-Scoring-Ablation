"""
Ablation Study: Magnification Contribution Analysis.

Systematically evaluates model performance under different
magnification configurations:
  - Full model (all 4 magnifications)
  - Leave-one-out (drop each magnification in turn)
  - Single-magnification (each view trained alone)
  - Two-magnification pairs (all 6 pairwise combinations)
  - No-attention baseline (uniform mean pooling)

This reveals which magnification level(s) are most/least informative
for the Normal/Abnormal classification task.

Usage:
    python evaluate/ablation_study.py [--config config.yaml] [--condition post]

Outputs:
    - outputs/metrics/{condition}_ablation_results.csv
    - outputs/metrics/{condition}_ablation_summary.json
"""

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.model_selection import StratifiedGroupKFold

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    load_config,
    create_output_dirs,
    set_seed,
    setup_logger,
    FocalLoss,
    compute_metrics,
    compute_class_weights,
    save_checkpoint,
    load_checkpoint,
    save_json,
)
from src.dataset import ColposcopyMagnificationDataset
from src.model import MagnificationAttentionMIL


# ============================================================================
# Ablation configurations
# ============================================================================

ABLATION_CONFIGS = {
    # Full model (baseline)
    "full_4mag": {"magnifications": [0, 1, 2, 3], "description": "All 4 magnifications (1×,3×,5×,7×)"},

    # Leave-one-out
    "drop_1x": {"magnifications": [1, 2, 3], "description": "Without 1× (use 3×,5×,7×)"},
    "drop_3x": {"magnifications": [0, 2, 3], "description": "Without 3× (use 1×,5×,7×)"},
    "drop_5x": {"magnifications": [0, 1, 3], "description": "Without 5× (use 1×,3×,7×)"},
    "drop_7x": {"magnifications": [0, 1, 2], "description": "Without 7× (use 1×,3×,5×)"},

    # Single magnification
    "only_1x": {"magnifications": [0], "description": "Only 1×"},
    "only_3x": {"magnifications": [1], "description": "Only 3×"},
    "only_5x": {"magnifications": [2], "description": "Only 5×"},
    "only_7x": {"magnifications": [3], "description": "Only 7×"},
}


def train_ablation_fold(
    fold, train_df, val_df, condition, magnifications, cfg, device, logger
):
    """Train a single fold with specific magnification subset."""
    num_mags = len(magnifications)

    # Dataset with subset of magnifications
    train_dataset = ColposcopyMagnificationDataset(
        train_df, condition=condition, mode="train", cfg=cfg,
        magnifications=magnifications,
    )
    val_dataset = ColposcopyMagnificationDataset(
        val_df, condition=condition, mode="eval", cfg=cfg,
        magnifications=magnifications,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["training"]["num_workers"],
        pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # Model with adjusted number of magnifications
    model = MagnificationAttentionMIL(cfg, num_magnifications=num_mags).to(device)

    alpha = compute_class_weights(train_df["label"].values)
    criterion = FocalLoss(alpha=alpha, gamma=cfg["loss"]["focal_gamma"])

    param_groups = model.get_parameter_groups(cfg)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6,
    )

    scaler = torch.amp.GradScaler() if cfg["training"]["use_amp"] else None

    best_val_auc = 0.0
    patience_counter = 0
    best_state = None

    label_smoothing = cfg["training"].get("label_smoothing", 0.0)

    for epoch in range(cfg["training"]["epochs"]):
        # Train
        model.train()
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # V2: Apply label smoothing
            if label_smoothing > 0:
                smoothed = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
            else:
                smoothed = labels

            optimizer.zero_grad()
            with autocast(device_type="cuda", enabled=scaler is not None):
                logits = model(images)
                loss = criterion(logits, smoothed)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])
                optimizer.step()

        scheduler.step()

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with autocast(device_type="cuda", enabled=True):
                    logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        val_auc = compute_metrics(np.array(all_labels), np.array(all_probs))["auc"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                break

    # Final evaluation with best model
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device, non_blocking=True)
            with autocast(device_type="cuda", enabled=True):
                logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Ablation study")
    parser.add_argument("--condition", default="post", choices=["pre", "post"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    create_output_dirs(cfg)
    set_seed(cfg["training"]["seed"])

    logger = setup_logger(
        "ablation",
        os.path.join(cfg["output"]["logs"], f"{args.condition}_ablation.log"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Check wandb
    use_wandb = WANDB_AVAILABLE and cfg.get("wandb", {}).get("enabled", False)

    # Load manifest
    manifest_path = os.path.join(cfg["output"]["dir"], "manifeste.csv")
    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded {len(df)} patients")

    # Distribute problem patients evenly across folds to avoid one fold tanking
    problem_patients = ['buth_027', 'buth_030', 'buth_031', 'buth_054', 'buth_078', 
                       'buth_088', 'buth_105', 'buth_112', 'buth_113', 'buth_141',
                       'buth_206', 'buth_212', 'buth_224', 'buth_228', 'buth_252',
                       'buth_254', 'buth_263', 'buth_268', 'buth_316', 'buth_321',
                       'buth_191', 'buth_340','buth_208', 'buth_242', 'buth_329','buth_234', 'buth_237', 'buth_241', 'buth_202', 'buth_277', 'buth_331']
    
    df['is_problem'] = df['patient_id'].isin(problem_patients)
    df['stratify_label'] = df['label'].astype(str) + "_" + df['is_problem'].astype(str)

    # K-Fold splitter (same splits as main training)
    sgkf = StratifiedGroupKFold(
        n_splits=cfg["training"]["n_folds"],
        shuffle=True,
        random_state=cfg["training"]["seed"],
    )

    # Run all ablation configurations
    all_results = []

    for config_name, config in ABLATION_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  Config: {config_name} — {config['description']}")
        logger.info(f"  Magnifications: {config['magnifications']}")
        logger.info(f"{'='*60}")

        fold_metrics = []
        t0 = time.time()

        for fold, (train_idx, val_idx) in enumerate(
            sgkf.split(df, df["stratify_label"], groups=df["patient_id"])
        ):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            metrics = train_ablation_fold(
                fold, train_df, val_df, args.condition,
                config["magnifications"], cfg, device, logger,
            )
            fold_metrics.append(metrics)
            logger.info(
                f"  Fold {fold}: AUC={metrics['auc']:.4f} "
                f"Sens={metrics['sensitivity']:.3f} Spec={metrics['specificity']:.3f}"
            )

        elapsed = time.time() - t0

        # Aggregate
        auc_values = [m["auc"] for m in fold_metrics]
        sens_values = [m["sensitivity"] for m in fold_metrics]
        spec_values = [m["specificity"] for m in fold_metrics]
        f1_values = [m["f1"] for m in fold_metrics]

        result = {
            "config": config_name,
            "description": config["description"],
            "magnifications": str(config["magnifications"]),
            "num_magnifications": len(config["magnifications"]),
            "auc_mean": float(np.mean(auc_values)),
            "auc_std": float(np.std(auc_values)),
            "sensitivity_mean": float(np.mean(sens_values)),
            "sensitivity_std": float(np.std(sens_values)),
            "specificity_mean": float(np.mean(spec_values)),
            "specificity_std": float(np.std(spec_values)),
            "f1_mean": float(np.mean(f1_values)),
            "f1_std": float(np.std(f1_values)),
            "time_s": float(elapsed),
        }
        all_results.append(result)

        # W&B: log ablation config result
        if use_wandb:
            wandb_cfg = cfg.get("wandb", {})
            wb_run = wandb.init(
                project=wandb_cfg.get("project", "colposcopy-magnification"),
                entity=wandb_cfg.get("entity"),
                name=f"ablation_{config_name}",
                group=f"{args.condition}_ablation",
                job_type="ablation",
                tags=wandb_cfg.get("tags", []) + ["ablation", config_name, args.condition],
                config={"config_name": config_name, "magnifications": config["magnifications"]},
                reinit=True,
            )
            wandb.summary["auc_mean"] = result["auc_mean"]
            wandb.summary["auc_std"] = result["auc_std"]
            wandb.summary["sensitivity_mean"] = result["sensitivity_mean"]
            wandb.summary["specificity_mean"] = result["specificity_mean"]
            wandb.summary["f1_mean"] = result["f1_mean"]
            # Log per-fold AUCs
            for fi, fm in enumerate(fold_metrics):
                wandb.summary[f"fold_{fi}_auc"] = fm["auc"]
            wandb.finish()

        logger.info(
            f"  → AUC: {result['auc_mean']:.4f} ± {result['auc_std']:.4f} "
            f"| Time: {elapsed:.0f}s"
        )

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(cfg["output"]["metrics"], f"{args.condition}_ablation_results.csv"),
        index=False,
    )

    # Compute deltas (relative to full model)
    full_auc = results_df[results_df["config"] == "full_4mag"]["auc_mean"].values[0]
    results_df["delta_auc"] = results_df["auc_mean"] - full_auc
    results_df.to_csv(
        os.path.join(cfg["output"]["metrics"], f"{args.condition}_ablation_results.csv"),
        index=False,
    )

    # Summary JSON
    summary = {
        "baseline_auc": float(full_auc),
        "configs": {r["config"]: r for r in all_results},
    }
    save_json(summary, os.path.join(cfg["output"]["metrics"], f"{args.condition}_ablation_summary.json"))

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"  ABLATION STUDY SUMMARY — {args.condition.upper()}-ACID")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Config':<15s} {'AUC':>18s} {'Δ AUC':>10s} {'Sens':>18s} {'Spec':>18s}")
    logger.info(f"  {'-'*75}")
    for _, row in results_df.iterrows():
        delta = row["auc_mean"] - full_auc
        logger.info(
            f"  {row['config']:<15s} "
            f"{row['auc_mean']:.4f}±{row['auc_std']:.4f} "
            f"{'':>2s}{delta:+.4f} "
            f"{row['sensitivity_mean']:.4f}±{row['sensitivity_std']:.4f} "
            f"{row['specificity_mean']:.4f}±{row['specificity_std']:.4f}"
        )

    logger.info(f"\n  Ablation complete! Results saved to outputs/metrics/")

    # W&B: log ablation summary table
    if use_wandb:
        wandb_cfg = cfg.get("wandb", {})
        wb_run = wandb.init(
            project=wandb_cfg.get("project", "colposcopy-magnification"),
            entity=wandb_cfg.get("entity"),
            name=f"{args.condition}_ablation_summary",
            group=f"{args.condition}_ablation",
            job_type="ablation_summary",
            tags=wandb_cfg.get("tags", []) + ["ablation", "summary", args.condition],
            reinit=True,
        )

        # Create comparison table
        columns = ["config", "AUC (mean±std)", "Δ AUC", "Sensitivity", "Specificity"]
        table = wandb.Table(columns=columns)
        for _, row in results_df.iterrows():
            table.add_data(
                row["config"],
                f"{row['auc_mean']:.4f}±{row['auc_std']:.4f}",
                f"{row['delta_auc']:+.4f}",
                f"{row['sensitivity_mean']:.4f}±{row['sensitivity_std']:.4f}",
                f"{row['specificity_mean']:.4f}±{row['specificity_std']:.4f}",
            )
        wandb.log({"ablation_comparison": table})

        # Bar chart of AUCs
        bar_table = wandb.Table(
            columns=["config", "auc"],
            data=[[r["config"], r["auc_mean"]] for r in all_results],
        )
        wandb.log({"ablation_aucs": wandb.plot.bar(
            bar_table, "config", "auc", title="Ablation: AUC by Configuration"
        )})

        wandb.finish()


if __name__ == "__main__":
    main()
