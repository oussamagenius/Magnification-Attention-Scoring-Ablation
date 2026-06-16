"""
Training Script with 5-Fold Stratified Cross-Validation.

Trains a MagnificationAttentionMIL model using a custom balanced
StratifiedGroupKFold split (stratified by label, grouped by patient_id
to prevent data leakage, with noisy/borderline patients evenly
distributed across folds).

Usage:
    python src/train.py --condition post [--config config.yaml]
    python src/train.py --condition pre  [--config config.yaml]

Outputs per fold:
    - outputs/checkpoints/{run_id}_fold_{k}_best.pt
    - outputs/logs/{run_id}_fold_{k}_train.csv
    - outputs/metrics/{run_id}_fold_{k}_predictions.csv
"""

import os
import sys
import argparse
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent to path for imports
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


def balanced_problem_split(df, problem_patients, n_folds, seed, logger):
    """
    Custom 2-phase fold splitter that guarantees even distribution of
    noisy/problem patients across all folds.

    Phase 1: Separate problem patients by label, shuffle, then assign
             round-robin to folds. This guarantees each fold gets
             ~equal counts of problem-Normal and problem-Abnormal.
    Phase 2: Split the remaining clean patients using StratifiedKFold
             on their label, and merge them into the fold assignments.

    Returns:
        List of (train_indices, val_indices) tuples, one per fold.
    """
    rng = np.random.RandomState(seed)

    is_problem = df['patient_id'].isin(problem_patients)
    problem_df = df[is_problem].copy()
    clean_df = df[~is_problem].copy()

    # ── Phase 1: Round-robin assignment of problem patients ──
    # Separate by label so each fold gets balanced noisy Normal AND noisy Abnormal
    prob_normal = problem_df[problem_df['label'] == 0].index.tolist()
    prob_abnormal = problem_df[problem_df['label'] == 1].index.tolist()
    rng.shuffle(prob_normal)
    rng.shuffle(prob_abnormal)

    # Round-robin into folds
    problem_fold_indices = [[] for _ in range(n_folds)]
    for i, idx in enumerate(prob_normal):
        problem_fold_indices[i % n_folds].append(idx)
    for i, idx in enumerate(prob_abnormal):
        problem_fold_indices[i % n_folds].append(idx)

    # ── Phase 2: StratifiedKFold for clean patients ──
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    clean_fold_indices = [[] for _ in range(n_folds)]

    for fold, (_, val_idx_in_clean) in enumerate(skf.split(clean_df, clean_df['label'])):
        # val_idx_in_clean is positional within clean_df; convert to df.index
        clean_fold_indices[fold] = clean_df.iloc[val_idx_in_clean].index.tolist()

    # ── Merge: each fold's val = its problem chunk + its clean chunk ──
    all_indices = set(df.index)
    fold_assignments = []

    for fold in range(n_folds):
        val_idx = np.array(problem_fold_indices[fold] + clean_fold_indices[fold])
        train_idx = np.array(sorted(all_indices - set(val_idx)))
        fold_assignments.append((train_idx, val_idx))

    # ── Logging: verify distribution ──
    logger.info("\n  Balanced Problem-Patient Split:")
    for fold in range(n_folds):
        val_idx = fold_assignments[fold][1]
        val_df_fold = df.loc[val_idx]
        n_prob = val_df_fold['patient_id'].isin(problem_patients).sum()
        n_clean = len(val_df_fold) - n_prob
        n_normal = (val_df_fold['label'] == 0).sum()
        n_abnormal = (val_df_fold['label'] == 1).sum()
        prob_normal_n = ((val_df_fold['label'] == 0) & val_df_fold['patient_id'].isin(problem_patients)).sum()
        prob_abnormal_n = ((val_df_fold['label'] == 1) & val_df_fold['patient_id'].isin(problem_patients)).sum()
        logger.info(
            f"    Fold {fold}: val={len(val_df_fold)} "
            f"(Normal={n_normal}, Abnormal={n_abnormal}) | "
            f"problem={n_prob} (N={prob_normal_n}, A={prob_abnormal_n}), clean={n_clean}"
        )

    return fold_assignments


def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device, max_grad_norm,
    label_smoothing=0.0, ema_state=None, ema_decay=0.999,
    mixup_alpha=0.0, cutmix_alpha=0.0, mixup_prob=0.0,
):
    """Train for one epoch with mixed precision.

    V2: supports label_smoothing to reduce overconfidence.
    V3: optional EMA (Exponential Moving Average) weight update per step.
    """
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch_idx, (images, labels, patient_ids) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Target Soft-labeling for problem patients (ASCUS etc)
        problem_patients = ['buth_027', 'buth_030', 'buth_031', 'buth_054', 'buth_078', 'buth_088', 'buth_105', 'buth_112', 'buth_113', 'buth_141', 'buth_206', 'buth_212', 'buth_224', 'buth_228', 'buth_252', 'buth_254', 'buth_263', 'buth_268', 'buth_316', 'buth_321', 'buth_191', 'buth_340','buth_208', 'buth_242', 'buth_329','buth_234', 'buth_237', 'buth_241', 'buth_202', 'buth_277', 'buth_331']

        # V2: Apply label smoothing (0 → ε/2, 1 → 1 - ε/2)
        if label_smoothing > 0:
            smoothed_labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            smoothed_labels = labels.clone()
            
        for i, pid in enumerate(patient_ids):
            if pid in problem_patients and labels[i] == 0:
                smoothed_labels[i] = 0.25  # Soft label for borderline normal cases

        # Mixup / Cutmix
        if mixup_prob > 0 and np.random.rand() < mixup_prob:
            use_cutmix = (np.random.rand() < 0.5) if cutmix_alpha > 0 else False
            if mixup_alpha > 0 and not use_cutmix:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
            elif cutmix_alpha > 0 and use_cutmix:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
            else:
                lam = 1.0

            if lam < 1.0:
                bsz = images.size(0)
                index = torch.randperm(bsz).to(device)

                if use_cutmix:
                    H, W = images.size(3), images.size(4)
                    cut_rat = np.sqrt(1. - lam)
                    cut_w = int(W * cut_rat)
                    cut_h = int(H * cut_rat)

                    cx = np.random.randint(W)
                    cy = np.random.randint(H)

                    bbx1 = np.clip(cx - cut_w // 2, 0, W)
                    bby1 = np.clip(cy - cut_h // 2, 0, H)
                    bbx2 = np.clip(cx + cut_w // 2, 0, W)
                    bby2 = np.clip(cy + cut_h // 2, 0, H)

                    images[:, :, :, bby1:bby2, bbx1:bbx2] = images[index, :, :, bby1:bby2, bbx1:bbx2]
                    lam = 1. - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
                else:
                    images = lam * images + (1 - lam) * images[index]

                smoothed_labels = lam * smoothed_labels + (1 - lam) * smoothed_labels[index]

        optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, smoothed_labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # EMA update: ema_w = decay * ema_w + (1 - decay) * current_w
        if ema_state is not None:
            for name, param in model.named_parameters():
                if name in ema_state:
                    ema_state[name].mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

        total_loss += loss.item() * len(labels)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    labels_np = all_labels.numpy().astype(int)

    metrics = compute_metrics(labels_np, probs)
    metrics["loss"] = total_loss / len(all_labels)

    return metrics


def _build_tta_ops():
    """Build TTA (Test-Time Augmentation) operations as (name, fn) pairs.

    NOTE: vflip removed — cervical anatomy is vertically asymmetric
    (os cervical at bottom, vaginal fornix at top). Vertical flip
    creates anatomically impossible images that degrade predictions.
    Evidence: TTA AUC (0.6753) was WORSE than standard AUC (0.7034).
    """
    return [
        ("identity", lambda x: x),
        ("hflip", lambda x: torch.flip(x, [-1])),
    ]


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, tta=False):
    """Evaluate on validation set, returning metrics and per-patient predictions.

    When tta=True, applies horizontal/vertical flips and averages predictions
    for more robust probability estimates.
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    all_patient_ids = []
    all_attention_weights = []

    tta_ops = _build_tta_ops() if tta else [("identity", lambda x: x)]

    for images, labels, patient_ids in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.size(0)
        num_mags = images.size(1)

        batch_logits = torch.zeros(batch_size, device=device)
        batch_attn = torch.zeros(batch_size, num_mags, device=device)

        for _op_name, op_fn in tta_ops:
            transformed = op_fn(images)
            with autocast(device_type="cuda", enabled=True):
                logits = model(transformed)
            batch_logits += logits
            attn = model.get_attention_weights()
            if attn is not None:
                batch_attn += attn

        logits = batch_logits / len(tta_ops)

        loss = criterion(logits, labels)
        total_loss += loss.item() * batch_size

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_patient_ids.extend(patient_ids)
        if attn is not None:
            all_attention_weights.append((batch_attn / len(tta_ops)).cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    labels_np = all_labels.numpy().astype(int)

    metrics = compute_metrics(labels_np, probs)
    metrics["loss"] = total_loss / len(all_labels)

    # Build predictions DataFrame
    predictions = pd.DataFrame({
        "patient_id": all_patient_ids,
        "y_true": labels_np,
        "y_prob": probs,
    })

    # Add attention weights columns
    if all_attention_weights:
        attn_matrix = torch.cat(all_attention_weights, dim=0).numpy()
        num_mags_attn = attn_matrix.shape[1]

        # Get magnification labels from dataset (supports dual-stream "both" mode)
        if hasattr(dataloader.dataset, 'effective_mag_labels'):
            mag_labels = dataloader.dataset.effective_mag_labels
        else:
            raw_labels = ["1x", "3x", "5x", "7x"]
            mag_indices = getattr(dataloader.dataset, "mag_indices", list(range(4)))
            mag_labels = [raw_labels[idx] if idx < len(raw_labels) else f"mag{idx}" for idx in mag_indices]

        # Fallback if something is weird
        if len(mag_labels) != num_mags_attn:
            mag_labels = [f"mag{i}" for i in range(num_mags_attn)]

        for i in range(num_mags_attn):
            col_name = f"alpha_{mag_labels[i]}"
            predictions[col_name] = attn_matrix[:, i]

    return metrics, predictions


def _use_wandb(cfg: dict) -> bool:
    """Check if wandb is available and enabled."""
    return WANDB_AVAILABLE and cfg.get("wandb", {}).get("enabled", False)


def _init_wandb_run(fold: int, condition: str, cfg: dict):
    """Initialize a wandb run for a single fold."""
    run_name = f"{condition}_fold_{fold}"
    wandb_cfg = cfg.get("wandb", {})
    run = wandb.init(
        project=wandb_cfg.get("project", "colposcopy-magnification"),
        entity=wandb_cfg.get("entity", None),
        name=run_name,
        group=f"{condition}_acid",
        job_type="train",
        tags=wandb_cfg.get("tags", []) + [f"fold_{fold}", condition],
        config={
            "fold": fold,
            "condition": condition,
            **{f"training/{k}": v for k, v in cfg["training"].items()},
            **{f"model/{k}": v for k, v in cfg["model"].items()},
            **{f"loss/{k}": v for k, v in cfg["loss"].items()},
            "data/image_size": cfg["data"]["image_size"],
            "data/num_magnifications": cfg["data"]["num_magnifications"],
        },
        reinit=True,
    )
    return run


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    condition: str,
    cfg: dict,
    device: torch.device,
    logger,
    run_id: str = None,
    save_to_disk: bool = True,
):
    """Train a single fold and return validation metrics + predictions."""
    if run_id is None:
        run_id = condition
    logger.info(f"\n{'='*60}")
    logger.info(f"  FOLD {fold + 1}/{cfg['training']['n_folds']} — Condition: {condition}")
    logger.info(f"  Train: {len(train_df)} patients | Val: {len(val_df)} patients")
    logger.info(f"{'='*60}")

    # Initialize wandb for this fold
    wb_run = None
    if _use_wandb(cfg):
        wb_run = _init_wandb_run(fold, condition, cfg)

    # Determine magnification indices (all 4 for main training, or custom)
    mag_indices = cfg.get("mags", [0, 1, 2, 3])
    num_mags = len(mag_indices)

    # For dual-stream ("both"), model sees 2x magnifications (pre + post)
    if condition == "both":
        model_num_mags = num_mags * 2
    else:
        model_num_mags = num_mags

    # Create datasets
    train_dataset = ColposcopyMagnificationDataset(
        train_df, condition=condition, mode="train", cfg=cfg, magnifications=mag_indices
    )
    val_dataset = ColposcopyMagnificationDataset(
        val_df, condition=condition, mode="eval", cfg=cfg, magnifications=mag_indices
    )

    # ── Weighted Random Sampler ──
    # Ensure 50/50 balance between normal (0) and abnormal (1) classes
    labels = train_df["label"].values.astype(int)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1)  # avoid div by zero
    sample_weights = np.array([class_weights[label] for label in labels])

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        sampler=sampler,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # Apply config overrides for pooling method (attention vs. mean)
    if cfg.get("override_pooling"):
        if cfg["override_pooling"] == "mean":
            cfg["model"]["gated_attention"] = False
            cfg["model"]["pooling"] = "mean"
        else:
            cfg["model"]["pooling"] = "attention"

    # Create model
    model = MagnificationAttentionMIL(cfg, num_magnifications=model_num_mags).to(device)
    
    # PyTorch 2.0+ graph compilation for potential speedup on modern GPUs
    if hasattr(torch, "compile"):
        logger.info("  Compiling model with torch.compile() for speedup...")
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.warning(f"  torch.compile() failed, falling back to eager mode: {e}")

    # EMA (Exponential Moving Average) of model weights — improves generalization on small datasets
    ema_decay = cfg["training"].get("ema_decay", 0.999)
    ema_state = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}

    # Compute class weights for focal loss
    alpha = compute_class_weights(train_df["label"].values)
    if cfg["loss"]["type"] == "focal":
        criterion = FocalLoss(alpha=alpha, gamma=cfg["loss"]["focal_gamma"])
    else:
        cfg_pos_weight = cfg.get("loss", {}).get("pos_weight")
        if cfg_pos_weight is not None and cfg_pos_weight > 0:
            pos_weight = torch.tensor([float(cfg_pos_weight)]).to(device)
        else:
            pos_weight = torch.tensor([0.8 * (alpha / (1 - alpha))]).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer with differential LR
    # foreach=False: avoids PyTorch 2.4+ _multi_tensor_adam TypeError
    # when lr values from YAML are not native Python floats.
    param_groups = model.get_parameter_groups(cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=float(cfg["training"]["weight_decay"]),
        foreach=False,
    )

    # Scheduler with optional linear warmup
    scheduler_name = cfg["training"].get("scheduler", "cosine")
    warmup_epochs = cfg["training"].get("warmup_epochs", 0)
    eta_min = float(cfg["training"].get("scheduler_eta_min", 1e-6))

    if scheduler_name == "cosine_with_restarts":
        T_0 = int(cfg["training"].get("scheduler_T0", 30))
        T_mult = int(cfg["training"].get("scheduler_T_mult", 1))
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=(cfg["training"]["epochs"] - warmup_epochs) if warmup_epochs > 0 else cfg["training"]["epochs"],
            eta_min=eta_min,
        )

    if warmup_epochs > 0 and scheduler_name != "cosine_with_restarts":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        if warmup_epochs > 0 and scheduler_name == "cosine_with_restarts":
            logger.warning("  warmup_epochs ignored: cosine_with_restarts starts at full LR")
        scheduler = main_scheduler

    # Mixed precision
    scaler = GradScaler() if cfg["training"]["use_amp"] else None

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0
    train_log = []
    best_state = None
    best_ema_state = None

    ckpt_path = os.path.join(
        cfg["output"]["checkpoints"], f"{run_id}_fold_{fold}_best.pt"
    )
    ema_ckpt_path = os.path.join(
        cfg["output"]["checkpoints"], f"{run_id}_fold_{fold}_best_ema.pt"
    )

    for epoch in range(cfg["training"]["epochs"]):
        t0 = time.time()

        # Train
        label_smoothing = cfg["training"].get("label_smoothing", 0.0)
        mixup_alpha = cfg.get("augmentation", {}).get("mixup_alpha", 0.0)
        cutmix_alpha = cfg.get("augmentation", {}).get("cutmix_alpha", 0.0)
        mixup_prob = cfg.get("augmentation", {}).get("mixup_prob", 0.0)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg["training"]["max_grad_norm"],
            label_smoothing=label_smoothing,
            ema_state=ema_state,
            ema_decay=ema_decay,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob,
        )

        # Validate — use EMA weights for stabler estimates
        if ema_state:
            backup = {}
            for name, param in model.named_parameters():
                if name in ema_state:
                    backup[name] = param.data.clone()
                    param.data.copy_(ema_state[name])
        val_metrics, _ = evaluate(model, val_loader, criterion, device)
        if ema_state:
            for name, param in model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t0

        # Log
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
            "val_f1": val_metrics["f1"],
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[1]["lr"],
            "time_s": elapsed,
        }
        train_log.append(log_entry)

        # W&B logging
        if wb_run is not None:
            wb_log = {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/auc": train_metrics["auc"],
                "train/sensitivity": train_metrics["sensitivity"],
                "train/specificity": train_metrics["specificity"],
                "val/loss": val_metrics["loss"],
                "val/auc": val_metrics["auc"],
                "val/sensitivity": val_metrics["sensitivity"],
                "val/specificity": val_metrics["specificity"],
                "val/f1": val_metrics["f1"],
                "lr/backbone": optimizer.param_groups[0]["lr"],
                "lr/head": optimizer.param_groups[1]["lr"],
            }

            # Log attention weight histograms periodically
            log_attn_every = cfg.get("wandb", {}).get("log_attention_every", 5)
            if log_attn_every > 0 and (epoch + 1) % log_attn_every == 0:
                val_metrics_tmp, val_preds_tmp = evaluate(model, val_loader, criterion, device)
                attn_cols = [c for c in val_preds_tmp.columns if c.startswith("alpha_")]
                for col in attn_cols:
                    mag = col.replace("alpha_", "")
                    wb_log[f"attention/{mag}"] = wandb.Histogram(val_preds_tmp[col].values)
                    wb_log[f"attention_mean/{mag}"] = val_preds_tmp[col].mean()

            wandb.log(wb_log)

        logger.info(
            f"  Epoch {epoch+1:3d}/{cfg['training']['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} AUC: {val_metrics['auc']:.4f} "
            f"Sens: {val_metrics['sensitivity']:.3f} Spec: {val_metrics['specificity']:.3f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping on val AUC
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            patience_counter = 0
            if save_to_disk:
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_auc, ckpt_path)
                if ema_state:
                    torch.save(ema_state, ema_ckpt_path)
                logger.info(f"  ★ New best val AUC: {best_val_auc:.4f} — checkpoint saved")
            else:
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if ema_state:
                    best_ema_state = {k: v.clone() for k, v in ema_state.items()}
                logger.info(f"  ★ New best val AUC: {best_val_auc:.4f} (in memory)")
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["early_stopping_patience"]:
                logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience_counter})")
                break

    # Save training log
    log_path = os.path.join(cfg["output"]["logs"], f"{run_id}_fold_{fold}_train.csv")
    pd.DataFrame(train_log).to_csv(log_path, index=False)

    # Load best model and get final val predictions
    if save_to_disk:
        load_checkpoint(model, ckpt_path, device)
        if os.path.exists(ema_ckpt_path):
            ema_state = torch.load(ema_ckpt_path, map_location=device, weights_only=False)
            for name, param in model.named_parameters():
                if name in ema_state:
                    param.data.copy_(ema_state[name])
    else:
        model.load_state_dict(best_state)
        if best_ema_state:
            for name, param in model.named_parameters():
                if name in best_ema_state:
                    param.data.copy_(best_ema_state[name])
    val_metrics, val_predictions = evaluate(model, val_loader, criterion, device)
    val_predictions["fold"] = fold

    # TTA evaluation
    use_tta = cfg.get("evaluation", {}).get("use_tta", True)
    if use_tta:
        val_metrics_tta, val_predictions_tta = evaluate(
            model, val_loader, criterion, device, tta=True
        )
        val_predictions_tta["fold"] = fold
        logger.info(
            f"  TTA:    AUC={val_metrics_tta['auc']:.4f} "
            f"Sens={val_metrics_tta['sensitivity']:.3f} Spec={val_metrics_tta['specificity']:.3f}"
        )
    else:
        val_metrics_tta = val_metrics
        val_predictions_tta = val_predictions

    # Save predictions
    pred_path = os.path.join(
        cfg["output"]["metrics"], f"{run_id}_fold_{fold}_predictions.csv"
    )
    val_predictions.to_csv(pred_path, index=False)

    # Save TTA predictions
    pred_path_tta = os.path.join(
        cfg["output"]["metrics"], f"{run_id}_fold_{fold}_predictions_tta.csv"
    )
    val_predictions_tta.to_csv(pred_path_tta, index=False)

    logger.info(f"  Fold {fold} best val AUC: {best_val_auc:.4f}")
    logger.info(f"  Final metrics: {val_metrics}")

    # W&B: log final fold results + model artifact
    if wb_run is not None:
        # Log final metrics as summary
        wandb.summary["best_val_auc"] = best_val_auc
        for k, v in val_metrics.items():
            wandb.summary[f"final/{k}"] = v
        if use_tta:
            for k, v in val_metrics_tta.items():
                wandb.summary[f"final_tta/{k}"] = v

        # Log attention weight bar chart
        attn_cols = [c for c in val_predictions.columns if c.startswith("alpha_")]
        if attn_cols:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mag_labels = [c.replace("alpha_", "") for c in attn_cols]
            means = [val_predictions[c].mean() for c in attn_cols]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(mag_labels, means, color=["#2196F3", "#4CAF50", "#FF9800", "#E91E63"][:len(mag_labels)])
            ax.set_ylabel("Mean Attention Weight (α)")
            ax.set_title(f"Fold {fold} — Magnification Importance")
            ax.axhline(y=1/len(mag_labels), color="gray", linestyle="--", alpha=0.5)
            wandb.log({"attention_chart": wandb.Image(fig)})
            plt.close(fig)

            # Log as wandb table
            table = wandb.Table(columns=["magnification", "mean_alpha", "std_alpha"])
            for col, mag in zip(attn_cols, mag_labels):
                table.add_data(mag, val_predictions[col].mean(), val_predictions[col].std())
            wandb.log({"attention_table": table})

        # Upload model checkpoint as artifact
        if cfg.get("wandb", {}).get("log_model", False) and os.path.exists(ckpt_path):
            artifact = wandb.Artifact(
                name=f"{condition}_fold_{fold}_model",
                type="model",
                description=f"Best model for {condition}-acid fold {fold} (AUC={best_val_auc:.4f})",
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

        wandb.finish()

    return val_metrics, val_predictions, val_metrics_tta, val_predictions_tta


def main():
    parser = argparse.ArgumentParser(description="Train Magnification Attention MIL")
    parser.add_argument("--condition", required=True, choices=["pre", "post", "both"],
                        help="Acid condition: 'pre', 'post', or 'both' (dual-stream 8 magnifications)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    
    # New arguments for the ablation study
    parser.add_argument("--manifest", default=None, help="Path to manifest file")
    parser.add_argument("--mags", default=None, help="Comma separated magnification indices (ex: 0,1,2,3 or 3 for 7x only)")
    parser.add_argument("--pooling", default=None, choices=["attention", "mean"], help="Pooling method")
    parser.add_argument("--run_name", default=None, help="Optional specific run name for metrics and logging")
    parser.add_argument("--data_root", default=None, help="Override data root directory (e.g. for raw dataset)")
    parser.add_argument("--save_checkpoint", action="store_true", default=True,
                        help="Save model checkpoints to disk (default: True)")
    parser.add_argument("--no_save_checkpoint", action="store_false", dest="save_checkpoint",
                        help="Skip disk checkpoint saving (RAM only, for ablation)")
    
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    
    # Override config with CLI args
    if args.mags is not None:
        cfg["mags"] = [int(m) for m in args.mags.split(",")]
        cfg["data"]["num_magnifications"] = len(cfg["mags"])
    else:
        cfg["mags"] = [0, 1, 2, 3]

    if args.pooling is not None:
        cfg["override_pooling"] = args.pooling
        
    if args.data_root is not None:
        cfg["data"]["root_dir"] = args.data_root
        
    create_output_dirs(cfg)
    set_seed(cfg["training"]["seed"])

    run_id = args.run_name if args.run_name else f"{args.condition}_custom_run"

    logger = setup_logger(
        "train",
        os.path.join(cfg["output"]["logs"], f"{run_id}_training.log"),
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load manifest
    if args.manifest:
        manifest_path = args.manifest
    else:
        manifest_path = os.path.join(cfg["output"]["dir"], "manifeste.csv")
        
    if not os.path.isfile(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest: {len(df)} patients")
    logger.info(f"  Normal: {len(df[df['label']==0])}, Abnormal: {len(df[df['label']==1])}")

    # Distribute problem patients evenly across folds to avoid one fold tanking
    problem_patients = ['buth_027', 'buth_030', 'buth_031', 'buth_054', 'buth_078', 
                       'buth_088', 'buth_105', 'buth_112', 'buth_113', 'buth_141',
                       'buth_206', 'buth_212', 'buth_224', 'buth_228', 'buth_252',
                       'buth_254', 'buth_263', 'buth_268', 'buth_316', 'buth_321',
                       'buth_191', 'buth_340','buth_208', 'buth_242', 'buth_329',
                       'buth_234', 'buth_237', 'buth_241', 'buth_202', 'buth_277', 'buth_331']

    df['is_problem'] = df['patient_id'].isin(problem_patients)

    n_folds = cfg["training"]["n_folds"]
    seed = cfg["training"]["seed"]
    fold_assignments = balanced_problem_split(df, problem_patients, n_folds, seed, logger)

    all_fold_metrics = []
    all_predictions = []
    all_fold_metrics_tta = []
    all_predictions_tta = []

    for fold, (train_idx, val_idx) in enumerate(fold_assignments):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # Verify no patient leakage
        train_patients = set(train_df["patient_id"])
        val_patients = set(val_df["patient_id"])
        assert len(train_patients & val_patients) == 0, "DATA LEAKAGE DETECTED!"

        fold_metrics, fold_predictions, fold_metrics_tta, fold_predictions_tta = train_fold(
            fold, train_df, val_df, args.condition, cfg, device, logger, run_id=run_id,
            save_to_disk=args.save_checkpoint,
        )

        all_fold_metrics.append(fold_metrics)
        all_predictions.append(fold_predictions)
        all_fold_metrics_tta.append(fold_metrics_tta)
        all_predictions_tta.append(fold_predictions_tta)

    # Aggregate results across folds
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_tta_df = pd.concat(all_predictions_tta, ignore_index=True)

    # Save with run_id
    all_predictions_df.to_csv(
        os.path.join(cfg["output"]["metrics"], f"{run_id}_predictions.csv"),
        index=False,
    )
    all_predictions_tta_df.to_csv(
        os.path.join(cfg["output"]["metrics"], f"{run_id}_predictions_tta.csv"),
        index=False,
    )

    # Summary statistics
    metric_keys = ["auc", "sensitivity", "specificity", "f1", "accuracy", "ppv", "npv"]
    summary = {}
    for key in metric_keys:
        values = [m[key] for m in all_fold_metrics]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": [float(v) for v in values],
        }

    # TTA summary statistics
    summary["tta"] = {}
    for key in metric_keys:
        values_tta = [m[key] for m in all_fold_metrics_tta]
        summary["tta"][key] = {
            "mean": float(np.mean(values_tta)),
            "std": float(np.std(values_tta)),
            "values": [float(v) for v in values_tta],
        }

    # Attention weight summary
    attn_cols = [c for c in all_predictions_df.columns if c.startswith("alpha_")]
    if attn_cols:
        summary["attention_weights"] = {}
        for col in attn_cols:
            mag = col.replace("alpha_", "")
            vals = all_predictions_df[col].values
            summary["attention_weights"][mag] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }

    save_json(summary, os.path.join(cfg["output"]["metrics"], f"{run_id}_summary.json"))

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info(f"  FINAL RESULTS — {run_id.upper()} (5-Fold CV)")
    logger.info("=" * 60)
    for key in metric_keys:
        logger.info(f"  {key:15s}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

    if "tta" in summary:
        logger.info("\n  ── TTA Results ──")
        for key in metric_keys:
            tta_vals = summary["tta"][key]
            logger.info(f"  {key:15s}: {tta_vals['mean']:.4f} ± {tta_vals['std']:.4f}")

    if "attention_weights" in summary:
        logger.info("\n  Attention Weights (Magnification Importance):")
        for mag, vals in summary["attention_weights"].items():
            bar = "█" * int(vals["mean"] * 40)
            logger.info(f"    {mag:4s}: {vals['mean']:.4f} ± {vals['std']:.4f}  {bar}")

    # W&B: log cross-fold summary as a final run
    if _use_wandb(cfg):
        wb_summary_run = wandb.init(
            project=cfg["wandb"]["project"],
            entity=cfg["wandb"].get("entity"),
            name=f"{args.condition}_summary",
            group=f"{args.condition}_acid",
            job_type="summary",
            tags=cfg["wandb"].get("tags", []) + ["summary", args.condition],
            reinit=True,
        )

        # Log aggregated metrics
        for key in metric_keys:
            wandb.summary[f"{key}/mean"] = summary[key]["mean"]
            wandb.summary[f"{key}/std"] = summary[key]["std"]
        if "tta" in summary:
            for key in metric_keys:
                wandb.summary[f"tta/{key}/mean"] = summary["tta"][key]["mean"]
                wandb.summary[f"tta/{key}/std"] = summary["tta"][key]["std"]

        # Log attention weights summary
        if "attention_weights" in summary:
            for mag, vals in summary["attention_weights"].items():
                wandb.summary[f"attention/{mag}/mean"] = vals["mean"]
                wandb.summary[f"attention/{mag}/std"] = vals["std"]

            # Log as table for easy comparison
            table = wandb.Table(columns=["magnification", "mean_alpha", "std_alpha"])
            for mag, vals in summary["attention_weights"].items():
                table.add_data(mag, vals["mean"], vals["std"])
            wandb.log({"summary/attention_table": table})

        # Log per-fold AUC as a bar chart
        fold_aucs = summary["auc"]["values"]
        table_folds = wandb.Table(
            columns=["fold", "auc"],
            data=[[f"Fold {i}", v] for i, v in enumerate(fold_aucs)],
        )
        wandb.log({"summary/fold_aucs": wandb.plot.bar(
            table_folds, "fold", "auc", title="AUC per Fold"
        )})

        wandb.finish()

    logger.info("=" * 60)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
