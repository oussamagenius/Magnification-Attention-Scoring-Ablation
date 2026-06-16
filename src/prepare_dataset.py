"""
Dataset Preparation Script.

Scans the raw BUTH dataset (01_Normal/ and 02_Abnormal/ directories),
resolves image paths with intelligent fallback for variant filenames
(suffix 'a'/'b', parenthesis variants, case-insensitive), excludes
patients with missing magnification views or unusable slides, and
generates a clean CSV manifest for training.

Usage:
    python src/prepare_dataset.py [--config config.yaml]

Output:
    outputs/manifeste.csv
"""

import os
import re
import sys
import argparse
from pathlib import Path

import pandas as pd

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger


def find_image_file(patient_dir: str, target_name: str) -> str:
    """
    Find the best matching image file for a target filename.

    Strategy:
    1. Exact match (e.g., 'p4.jpg')
    2. Case-insensitive exact match (e.g., 'P4.jpg')
    3. Variant with suffix 'a' (e.g., 'p4a.jpg') — first variant taken
    4. Variant with suffix 'b' or later
    5. Variant with parenthesis (e.g., 'p4(1).jpg')

    Args:
        patient_dir: Path to the patient directory
        target_name: Target filename (e.g., 'p4.jpg')

    Returns:
        Resolved filename or None if not found
    """
    files = os.listdir(patient_dir)
    files_lower = {f.lower(): f for f in files}

    # 1. Exact match
    if target_name in files:
        return target_name

    # 2. Case-insensitive
    if target_name.lower() in files_lower:
        return files_lower[target_name.lower()]

    # 3. Find variants (e.g., p4a.jpg, p4b.jpg for target p4.jpg)
    base = target_name.replace(".jpg", "")  # e.g., 'p4' or 'po3'
    pattern = re.compile(rf"^{re.escape(base)}[a-z]\.jpg$", re.IGNORECASE)

    variants = sorted([f for f in files if pattern.match(f.lower())])
    if variants:
        return variants[0]  # Take first variant (usually 'a')

    # 4. Parenthesis variant (e.g., p4(1).jpg)
    pattern_paren = re.compile(rf"^{re.escape(base)}\(\d+\)\.jpg$", re.IGNORECASE)
    paren_variants = [f for f in files if pattern_paren.match(f)]
    if paren_variants:
        return paren_variants[0]

    return None


def process_patient(patient_dir: str, label: int, cfg: dict) -> dict:
    """
    Process a single patient directory.

    Args:
        patient_dir: Full path to patient folder
        label: 0 for Normal, 1 for Abnormal
        cfg: Configuration dict

    Returns:
        dict with patient_id, label, and paths for all 8 images, or None if incomplete
    """
    patient_id = os.path.basename(patient_dir)

    # Check if patient is in exclusion list
    if patient_id in cfg["data"]["excluded_patients"]:
        return None

    # Check for problematic cytology labels
    txt_files = [f for f in os.listdir(patient_dir) if f.endswith(".txt")]
    if txt_files:
        txt_path = os.path.join(patient_dir, txt_files[0])
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                cytology = f.read().strip().upper()
            # Exclude patients with unusable slides
            if any(kw in cytology for kw in ["NO SLIDE", "POOR SLIDE", "INADEQUATE"]):
                return None
        except Exception:
            pass  # If can't read, proceed anyway

    # Resolve all 8 image paths
    record = {"patient_id": patient_id, "label": label}

    pre_files = cfg["data"]["pre_acid_files"]
    post_files = cfg["data"]["post_acid_files"]

    for i, fname in enumerate(pre_files):
        resolved = find_image_file(patient_dir, fname)
        if resolved is None:
            return None  # Missing magnification — exclude patient
        record[f"pre_{i+1}_path"] = os.path.join(patient_dir, resolved)

    for i, fname in enumerate(post_files):
        resolved = find_image_file(patient_dir, fname)
        if resolved is None:
            return None  # Missing magnification — exclude patient
        record[f"post_{i+1}_path"] = os.path.join(patient_dir, resolved)

    return record


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset manifest")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger("prepare", "outputs/logs/prepare_dataset.log")

    root_dir = cfg["data"]["root_dir"]
    normal_dir = os.path.join(root_dir, "01_Normal")
    abnormal_dir = os.path.join(root_dir, "02_Abnormal")

    # Validate dataset exists
    if not os.path.isdir(normal_dir):
        logger.error(f"Normal directory not found: {normal_dir}")
        sys.exit(1)
    if not os.path.isdir(abnormal_dir):
        logger.error(f"Abnormal directory not found: {abnormal_dir}")
        sys.exit(1)

    records = []
    excluded_count = {"unrecoverable": 0, "bad_label": 0, "explicit": 0}

    # Process Normal patients (label=0)
    for patient_name in sorted(os.listdir(normal_dir)):
        patient_path = os.path.join(normal_dir, patient_name)
        if not os.path.isdir(patient_path):
            continue

        record = process_patient(patient_path, label=0, cfg=cfg)
        if record is None:
            if patient_name in cfg["data"]["excluded_patients"]:
                excluded_count["explicit"] += 1
            else:
                excluded_count["unrecoverable"] += 1
        else:
            records.append(record)

    # Process Abnormal patients (label=1)
    for patient_name in sorted(os.listdir(abnormal_dir)):
        patient_path = os.path.join(abnormal_dir, patient_name)
        if not os.path.isdir(patient_path):
            continue

        record = process_patient(patient_path, label=1, cfg=cfg)
        if record is None:
            if patient_name in cfg["data"]["excluded_patients"]:
                excluded_count["explicit"] += 1
            else:
                excluded_count["unrecoverable"] += 1
        else:
            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Validate all image paths exist
    image_cols = [c for c in df.columns if c.endswith("_path")]
    missing = 0
    for col in image_cols:
        for idx, path in df[col].items():
            if not os.path.isfile(path):
                logger.warning(f"Missing file: {path} (patient {df.loc[idx, 'patient_id']})")
                missing += 1

    if missing > 0:
        logger.warning(f"Total missing files: {missing}")

    # Save manifest
    output_path = os.path.join(cfg["output"]["dir"], "manifeste.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Statistics
    n_normal = len(df[df["label"] == 0])
    n_abnormal = len(df[df["label"] == 1])

    logger.info("=" * 60)
    logger.info("  Dataset Manifest Generated")
    logger.info("=" * 60)
    logger.info(f"  Output        : {output_path}")
    logger.info(f"  Total patients: {len(df)}")
    logger.info(f"  Normal (0)    : {n_normal}")
    logger.info(f"  Abnormal (1)  : {n_abnormal}")
    logger.info(f"  Ratio N/A     : {n_normal/n_abnormal:.2f}")
    logger.info(f"  Excluded      : {sum(excluded_count.values())} total")
    logger.info(f"    - Explicit  : {excluded_count['explicit']}")
    logger.info(f"    - Unrecov.  : {excluded_count['unrecoverable']}")
    logger.info(f"    - Bad label : {excluded_count['bad_label']}")
    logger.info(f"  Images/patient: 8 (4 pre-acid + 4 post-acid)")
    logger.info(f"  Total images  : {len(df) * 8}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
