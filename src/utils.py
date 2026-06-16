"""
Shared utility functions.

Covers: reproducibility seeds, logging setup, binary classification
metrics (AUC, sensitivity, specificity, F1, PPV, NPV), Focal Loss,
checkpoint I/O, and JSON serialization helpers.
"""

import os
import random
import logging
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)


# ============================================================================
# Configuration
# ============================================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def create_output_dirs(cfg: dict):
    """Create all output directories from config."""
    for key in ["dir", "checkpoints", "logs", "metrics", "figures"]:
        path = cfg["output"][key]
        os.makedirs(path, exist_ok=True)


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behavior disabled for speed on modern GPUs
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================================
# Logging
# ============================================================================

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Setup a logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for binary classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the positive class. If None, computed from
               class distribution.
        gamma: Focusing parameter. Higher = more focus on hard examples.
    """

    def __init__(self, alpha: float = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model output (before sigmoid), shape [B]
            targets: Binary labels, shape [B]
        """
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Binary cross-entropy per sample
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t = prob of correct class
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal modulating factor
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce
        return loss.mean()


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = None) -> dict:
    """
    Compute comprehensive binary classification metrics.

    Args:
        y_true: Ground truth binary labels (0/1)
        y_prob: Predicted probabilities
        threshold: Decision threshold. If None, uses Youden's J.

    Returns:
        dict with AUC, sensitivity, specificity, F1, accuracy, PPV, NPV, threshold
    """
    # AUC
    auc = roc_auc_score(y_true, y_prob)

    # Optimal threshold via Youden's J statistic
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = thresholds[best_idx]

    # Binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ppv": float(ppv),
        "npv": float(npv),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_class_weights(labels: np.ndarray) -> float:
    """
    Compute focal loss alpha from class distribution.
    Alpha = proportion of negative class (so positives get more weight).
    """
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    alpha = n_neg / (n_pos + n_neg)
    return float(alpha)


# ============================================================================
# Checkpoint utilities
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, val_auc, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_auc": val_auc,
        },
        path,
    )


def load_checkpoint(model, path, device="cuda"):
    """Load model checkpoint (model weights only)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("val_auc", None)


# ============================================================================
# JSON serialization helper
# ============================================================================

def save_json(data: dict, path: str):
    """Save dict to JSON with numpy type handling."""
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)
