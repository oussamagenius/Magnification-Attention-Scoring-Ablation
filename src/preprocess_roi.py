"""
ROI Preprocessing for Colposcopy Images (V6 — Unified & Magnification-Aware).

Key design choices:
- Hard skin rejection: penalizes brown/dark-skin tones in Lab/HSV color space.
- Enhanced mucosa detection: focuses on a* (redness) and L* (luminance) balance.
- Multi-scale texture penalty: suppresses high-variance regions (pores, wrinkles).
- Robust centroid finding via contour-based peak detection.
- Magnification-aware cropping: crop size automatically scales with input
  magnification level (p1=1×, p2=3×, p3=5×, p4/p5=7×).
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger


# ── Text Overlay Masking ─────────────────────────────────────────────────────

def mask_text_overlay(image, top_pct=0.13, left_pct=0.55):
    """Replace the equipment text overlay region with a neutral fill color."""
    h, w = image.shape[:2]
    text_h = int(h * top_pct)
    text_w = int(w * left_pct)
    ref_y_start = min(text_h, h - 1)
    ref_y_end = min(text_h + 30, h)
    ref_strip = image[ref_y_start:ref_y_end, :text_w]

    if ref_strip.size > 0:
        fill_color = np.median(ref_strip.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        fill_color = np.array([40, 40, 40], dtype=np.uint8)
    image[:text_h, :text_w] = fill_color
    return image


# ── V6 Core Functions ────────────────────────────────────────────────────────

def _detect_colposcope_field(gray, h, w):
    """Detect the illuminated field of the colposcope (the bright circular area)."""
    _, field_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(field_mask, connectivity=8)
    center_y, center_x = h // 2, w // 2
    best_label = None
    best_area = 0

    for lbl in range(1, n_labels):
        if labels[center_y, center_x] == lbl:
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area > best_area:
                best_area = area
                best_label = lbl

    if best_label is None:
        cy_lo, cy_hi = h // 4, 3 * h // 4
        cx_lo, cx_hi = w // 4, 3 * w // 4
        for lbl in range(1, n_labels):
            region = labels[cy_lo:cy_hi, cx_lo:cx_hi] == lbl
            if region.any():
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area > best_area:
                    best_area = area
                    best_label = lbl

    if best_label is None or best_area < 0.05 * h * w:
        return None

    return (labels == best_label).astype(np.uint8) * 255

def _compute_skin_rejection_mask(image_bgr):
    """Identify skin-tone pixels to penalize them in the saliency map."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    
    skin_score = np.clip((b - a + 10) / 20.0, 0, 1)
    dark_penalty = np.clip((50 - L) / 30.0, 0, 1)
    return np.clip(skin_score + dark_penalty, 0, 1)

def _compute_mucosa_prob_v6(image_bgr, field_mask):
    """Detect cervical mucosa probability map with integrated skin rejection."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    
    a_score = np.clip((a - 142) / 15.0, 0, 1)
    L_score = np.clip((L - 70) / 50.0, 0, 1)
    b_penalty = np.clip((b - 145) / 20.0, 0, 1)
    
    prob = (a_score * 0.7 + L_score * 0.3) - b_penalty * 0.5
    prob = np.clip(prob, 0, 1)
    
    if field_mask is not None:
        prob *= (field_mask.astype(np.float32) / 255.0)
        
    return prob

def _compute_texture_penalty(gray):
    """Penalize high-variance (rough) regions and favor smooth tissue areas."""
    gray_f = gray.astype(np.float32)
    r = 5
    k = 2 * r + 1
    mean = cv2.boxFilter(gray_f, -1, (k, k))
    var = cv2.boxFilter(gray_f**2, -1, (k, k)) - mean**2
    std = np.sqrt(np.maximum(var, 0))
    penalty = np.clip(std / 40.0, 0, 1)
    return penalty


def _compute_speculum_penalty(image_bgr):
    """Penalize speculum-ring pixels (characteristic orange-red color, ~#fc8251)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    
    # Speculum ring has distinctly high a* and b* values (> 165) compared to mucosa (< 160)
    a_high = np.clip((a - 160) / 15.0, 0, 1)
    b_high = np.clip((b - 160) / 15.0, 0, 1)

    # Cross-check with HSV saturation and brightness
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    sv_high = np.clip((s - 150) / 25.0, 0, 1) * np.clip((v - 220) / 30.0, 0, 1)

    # Combined penalty (max when both Lab and HSV agree)
    return np.clip(a_high * b_high + sv_high, 0, 1)


# ── ROI Detection Unifiée ───────────────────────────────────────────────────

def _infer_magnification(filename):
    base = os.path.basename(filename).lower()
    if base.startswith(("p1", "po1")):   return "low"
    elif base.startswith(("p2", "po2")): return "mid"
    elif base.startswith(("p3", "po3")): return "high"
    else:                                return "very_high"

def detect_cervix_roi(image_bgr, filename="", padding_pct=0.15):
    """
    Unified V6 pipeline with aggressive skin rejection and magnification-adaptive cropping.
    """
    h, w = image_bgr.shape[:2]
    
    mag_level = _infer_magnification(filename)
    if mag_level == "low":
        target_area_pct = 0.15      # p1 : 10% adapté par utilisateur
    elif mag_level == "mid":
        target_area_pct = 0.50      # p2 : 50%
    elif mag_level == "high":
        target_area_pct = 0.90      # p3 : 75%
    else:
        target_area_pct = 0.95      # p4, p5 : 90%

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    field_mask = _detect_colposcope_field(gray, h, w)
    mucosa_prob = _compute_mucosa_prob_v6(image_bgr, field_mask)
    skin_color_penalty = _compute_skin_rejection_mask(image_bgr)
    texture_penalty = _compute_texture_penalty(gray)
    speculum_penalty = _compute_speculum_penalty(image_bgr)
    
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    cx_f, cy_f = w/2, h/2
    if field_mask is not None:
        M = cv2.moments(field_mask)
        if M["m00"] > 0:
            cx_f, cy_f = M["m10"]/M["m00"], M["m01"]/M["m00"]
    
    dist_sq = (X - cx_f)**2 + (Y - cy_f)**2
    sigma = min(h, w) * 0.20
    center_prior = np.exp(-dist_sq / (2 * sigma**2))
    
    saliency = mucosa_prob * (1.0 - 0.6 * skin_color_penalty) * (1.0 - 0.4 * texture_penalty) * (1.0 - 0.9 * speculum_penalty) * center_prior
    
    blur_k = max(31, min(h, w) // 15) | 1
    saliency_smooth = cv2.GaussianBlur(saliency, (blur_k, blur_k), 0)
    
    thresh_val = np.percentile(saliency_smooth, 95)
    _, binary = cv2.threshold((saliency_smooth * 255).astype(np.uint8), int(thresh_val * 255), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cx, best_cy = w // 2, h // 2
    best_score = -1
    
    if contours:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500: continue
            
            mask = np.zeros_like(binary)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Fill ratio: reject hollow shapes (closed speculum rings / arcs)
            actual_area = cv2.countNonZero(cv2.bitwise_and(binary, mask))
            fill_ratio = actual_area / area if area > 0 else 0
            
            if fill_ratio < 0.65:
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            mean_sal = cv2.mean(saliency_smooth, mask=mask)[0]
            
            # Weighted score: lower fill_ratio degrades the score
            score = mean_sal * (1.0 + 0.5 * circularity) * fill_ratio
            
            if score > best_score:
                best_score = score
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    best_cx = int(M["m10"] / M["m00"])
                    best_cy = int(M["m01"] / M["m00"])
    
    if best_score < 0.05:
        _, _, _, max_loc = cv2.minMaxLoc(saliency_smooth)
        cx, cy = max_loc
    else:
        cx, cy = best_cx, best_cy
    
    crop_w = int(w * np.sqrt(target_area_pct))
    crop_h = int(h * np.sqrt(target_area_pct))
    
    x1 = int(np.clip(cx - crop_w // 2, 0, w - crop_w))
    y1 = int(np.clip(cy - crop_h // 2, 0, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    
    strategy = f"v6_{mag_level}_{int(target_area_pct*100)}%"
    return (x1, y1, x2, y2), strategy, saliency_smooth


# ── Single Image Processing ─────────────────────────────────────────────────

def process_single_image(src_path, dst_path, cfg_preproc, no_mask=False):
    image = cv2.imread(src_path)
    if image is None:
        return None

    h_orig, w_orig = image.shape[:2]
    filename = os.path.basename(src_path)

    top_pct  = cfg_preproc.get("text_mask_top_pct", 0.13)
    left_pct = cfg_preproc.get("text_mask_left_pct", 0.55)
    image_clean = mask_text_overlay(image.copy(), top_pct, left_pct)

    padding = cfg_preproc.get("roi_padding_pct", 0.15)
    (x1, y1, x2, y2), strategy, saliency = detect_cervix_roi(image_clean, filename, padding)

    if y1 >= y2 or x1 >= x2:
        x1, y1, x2, y2 = 0, 0, w_orig, h_orig
        strategy += "_invalid_bbox"

    if no_mask:
        processed_image = image_clean
        area_ratio = 100.0
    else:
        mag_level = _infer_magnification(filename)
        _, max_sal, _, _ = cv2.minMaxLoc(saliency)
        
        if mag_level == "low":
            thresh = max_sal * 0.20
        else:
            thresh = max_sal * 0.01
            
        _, mask = cv2.threshold(saliency, thresh, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        if mag_level != "low":
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                biggest_cnt = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(biggest_cnt)
                cv2.drawContours(mask, [hull], -1, 255, -1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        processed_image = cv2.bitwise_and(image_clean, image_clean, mask=mask)
        area_ratio = np.sum(mask > 0) / max(1, w_orig * h_orig)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    quality = cfg_preproc.get("jpeg_quality", 95)
    cv2.imwrite(dst_path, processed_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    h_final, w_final = processed_image.shape[:2]

    return {
        "src":               str(src_path),
        "dst":               str(dst_path),
        "filename":          str(filename),
        "original_size":     [int(w_orig), int(h_orig)],
        "crop_size":         [int(w_final), int(h_final)],
        "roi_bbox":          [int(x1), int(y1), int(x2), int(y2)],
        "area_retained_pct": float(round(area_ratio * 100, 1)),
        "strategy":          str(strategy),
    }


# ── QC Visualization ────────────────────────────────────────────────────────

def generate_qc_figure(sample_stats, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_samples = min(8, len(sample_stats))
    if n_samples == 0:
        return

    fig, axes = plt.subplots(n_samples, 2,
                             figsize=(12, 3.5 * n_samples),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.35})
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("Preprocessing QC — Original vs Cropped (ROI in green)",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, stat in enumerate(sample_stats[:n_samples]):
        orig = cv2.imread(stat["src"])
        if orig is not None:
            x1, y1, x2, y2 = stat["roi_bbox"]
            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 3)
            axes[i, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"Original: {stat['filename']}", fontsize=9)
        axes[i, 0].axis("off")

        cropped = cv2.imread(stat["dst"])
        if cropped is not None:
            axes[i, 1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        axes[i, 1].set_title(
            f"Cropped [{stat['area_retained_pct']}%] {stat['strategy']}",
            fontsize=9
        )
        axes[i, 1].axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Main Processing Loop ────────────────────────────────────────────────────

def process_dataset(input_dir, output_dir, cfg_preproc, logger, no_mask=False):
    all_stats  = []
    qc_samples = []
    errors     = []
    skipped    = 0

    patient_dirs = []
    for class_folder in ["01_Normal", "02_Abnormal"]:
        class_path = os.path.join(input_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
        for patient in sorted(os.listdir(class_path)):
            patient_path = os.path.join(class_path, patient)
            if os.path.isdir(patient_path):
                patient_dirs.append((class_folder, patient, patient_path))

    logger.info(f"Found {len(patient_dirs)} patient directories")

    qc_sampled = {k: False for k in
                  ("p1", "p2", "p3", "p4", "p5", "po1", "po2", "po3", "po4", "po5")}

    STEM_TO_MAG = {
        "p1":  "pre_1x.jpg",
        "p2":  "pre_3x.jpg",
        "p3":  "pre_5x.jpg",
        "p4":  "pre_7x.jpg",
        "p5":  "pre_7x.jpg",
        "po1": "post_1x.jpg",
        "po2": "post_3x.jpg",
        "po3": "post_5x.jpg",
        "po4": "post_7x.jpg",
        "po5": "post_7x.jpg",
    }

    for class_folder, patient_name, patient_path in tqdm(patient_dirs, desc="Preprocessing"):
        dst_patient = os.path.join(output_dir, class_folder, patient_name)
        os.makedirs(dst_patient, exist_ok=True)

        for fname in sorted(os.listdir(patient_path)):
            src_file = os.path.join(patient_path, fname)
            if not os.path.isfile(src_file):
                continue
            # Rename to standard format (e.g., p1.jpg -> pre_1x.jpg)
            base_key = fname.lower().split(".")[0].rstrip("abcdefgh_")
            new_fname = STEM_TO_MAG.get(base_key, fname)
            dst_file = os.path.join(dst_patient, new_fname)

            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                stat = process_single_image(src_file, dst_file, cfg_preproc, no_mask)
                if stat is None:
                    errors.append(str(src_file))
                    logger.warning(f"Failed: {src_file}")
                else:
                    all_stats.append(stat)
                    base_key = fname.lower().split(".")[0].rstrip("abcdefgh")
                    if base_key in qc_sampled and not qc_sampled[base_key]:
                        qc_samples.append(stat)
                        qc_sampled[base_key] = True
            else:
                shutil.copy2(src_file, dst_file)
                skipped += 1

    return all_stats, qc_samples, errors, skipped


def compute_summary(all_stats, logger):
    if not all_stats:
        return {}

    mag_groups      = {}
    strategy_counts = {}

    for s in all_stats:
        fname = s["filename"].lower()
        if fname.startswith(("p1", "po1")):   mag = "1x"
        elif fname.startswith(("p2", "po2")): mag = "3x"
        elif fname.startswith(("p3", "po3")): mag = "5x"
        else:                                  mag = "7x+"

        mag_groups.setdefault(mag, []).append(s["area_retained_pct"])
        key = s.get("strategy", "unknown")
        strategy_counts[key] = strategy_counts.get(key, 0) + 1

    logger.info("=" * 60)
    logger.info("Preprocessing Summary")
    logger.info("=" * 60)
    logger.info(f"Total images: {len(all_stats)}")

    for mag in sorted(mag_groups.keys()):
        vals = mag_groups[mag]
        logger.info(f"{mag}: {len(vals)} imgs, "
                    f"retained {np.mean(vals):.1f}% ± {np.std(vals):.1f}%")

    logger.info("Strategies used:")
    for strat, count in sorted(strategy_counts.items()):
        logger.info(f"  {strat}: {count}")
    logger.info("=" * 60)

    return {
        "total_images": len(all_stats),
        "by_magnification": {
            k: {"count": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in mag_groups.items()
        },
        "strategies": strategy_counts,
    }


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess colposcopy images")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-mask", action="store_true",
                        help="Skip mucosa masking: apply text overlay removal, renaming and saving only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs("outputs/logs", exist_ok=True)
    logger = setup_logger("preprocess", "outputs/logs/preprocess_roi.log")

    cfg_preproc = cfg.get("preprocessing", {})
    input_dir   = cfg_preproc.get("input_dir", "/workspace/dataset")
    output_dir  = cfg_preproc.get("output_dir", "/workspace/dataset_preprocessed")

    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    if args.no_mask:
        logger.info("Mode --no-mask: skipping mucosa masking (clean + rename + save only)")

    if not os.path.isdir(input_dir):
        logger.error(f"Input not found: {input_dir}")
        sys.exit(1)

    all_stats, qc_samples, errors, skipped = process_dataset(
        input_dir, output_dir, cfg_preproc, logger, no_mask=args.no_mask
    )
    summary = compute_summary(all_stats, logger)

    if errors:
        logger.warning(f"Errors: {len(errors)}")

    os.makedirs("outputs/metrics", exist_ok=True)
    try:
        with open("outputs/metrics/preprocess_qc_report.json", "w") as f:
            json.dump({"summary": summary,
                       "errors": errors,
                       "sample_details": all_stats[:10]},
                      f, indent=2, default=str)
    except Exception as e:
        logger.error(f"JSON save error: {e}")
        np.save("outputs/metrics/preprocess_qc_report.npy", all_stats)

    if qc_samples:
        generate_qc_figure(qc_samples,
                           "outputs/figures/preprocessing_qc_samples.png")
        logger.info("QC figure saved")

    logger.info("Complete [DONE]")


if __name__ == "__main__":
    main()
