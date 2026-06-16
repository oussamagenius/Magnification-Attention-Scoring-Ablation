#!/bin/bash
# ============================================================================
# MagnificationAttentionMIL — Master Pipeline Runner
#
# Runs the complete pipeline end-to-end:
#   1. Environment setup
#   2. ROI preprocessing (crop + resize)
#   3. Dataset manifest generation
#   4. Manifest variants for sensitivity analysis (corrected/strict labels)
#   5. Full study: baseline training + sensitivity + ablation (16 configs)
#   6. Evaluation, ensemble metrics, and statistical tests
#   7. Publication figures
#   8. Architecture diagram
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Prerequisites:
#   - Raw BUTH dataset in the directory specified by preprocessing.input_dir
#     in config.yaml (default: /workspace/data/)
#   - Python environment with all dependencies installed (see requirements.txt)
#
# Estimated runtime on RTX 3090: ~5–6 hours for the full ablation study
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config.yaml"

START_TIME=$(date +%s)

echo ""
echo "╭──────────────────────────────────────────────────────────╮"
echo "│  MagnificationAttentionMIL — Colposcopy Pipeline         │"
echo "│  Does Attention Reflect Clinical Importance?            │"
echo "╰──────────────────────────────────────────────────────────╯"
echo ""
echo "Using config: ${CONFIG}"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 1: Setup
# ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/8] Environment Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash setup_vastai.sh
echo ""

# ──────────────────────────────────────────────────────────────
# Step 2: Preprocessing — crop cervix & resize to 224×224
# ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/8] Preprocessing — HSV cervix crop + smart resize"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 src/preprocess_roi.py --config "$CONFIG"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 3: Prepare Dataset (generate manifest from preprocessed)
# ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/8] Preparing Dataset Manifest"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 src/prepare_dataset.py --config "$CONFIG"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 4: Create manifest variants (original/corrected/strict)
# ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/8] Creating Manifest Variants for Sensitivity Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 create_manifests.py
echo ""

# ──────────────────────────────────────────────────────────────
# Step 5: Full Study (Baseline, Sensitivity, Ablation)
# ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/8] Executing Full Study (Training, Sensitivity, Ablation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 run_full_study.py --config "$CONFIG"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 6: Evaluation & Statistical Tests
# ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [6/8] Evaluation & Statistical Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 src/ensemble_pre_post.py --run_name baseline
python3 evaluate/evaluate_results.py --config "$CONFIG"
python3 evaluate/statistical_tests.py --config "$CONFIG"
python3 evaluate/wilcoxon_fold_aucs.py --config "$CONFIG"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 7: Generate Publication Figures
# ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [7/8] Generating Publication Figures"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 visualize/plot_attention_weights.py --config "$CONFIG"
python3 visualize/plot_roc_curves.py --config "$CONFIG"
python3 visualize/plot_ablation_results.py --config "$CONFIG"
python3 visualize/plot_confusion_matrices.py --config "$CONFIG"
python3 visualize/plot_training_curves.py --config "$CONFIG"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 8: Architecture Figure (standalone)
# ──────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [8/8] Generating Architecture Figure"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 generate_architecture_figure.py
echo ""

# ──────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

# Archive config + metadata for reproducibility
cp "$CONFIG" "outputs/config_snapshot.yaml"
cp manifeste_original.csv manifeste_corrige.csv manifeste_strict.csv outputs/ 2>/dev/null || true
echo "Run completed: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" > "outputs/run_metadata.txt"
echo "Config used: ${CONFIG}" >> "outputs/run_metadata.txt"
echo "Elapsed: ${HOURS}h ${MINUTES}m" >> "outputs/run_metadata.txt"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║         PIPELINE COMPLETE! ✓                            ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                        ║"
echo "║  Total runtime: ${HOURS}h ${MINUTES}m                            ║"
echo "║                                                        ║"
echo "║  Outputs:                                              ║"
echo "║    📊 outputs/config_snapshot.yaml (for reproducibility)║"
echo "║    📊 outputs/manifeste_*.csv                         ║"
echo "║    📊 outputs/preprocessing_report.csv                ║"
echo "║    📊 outputs/metrics/final_results.json              ║"
echo "║    📊 outputs/metrics/statistical_tests.json          ║"
echo "║    📊 outputs/metrics/*_attention_analysis.json       ║"
echo "║                                                        ║"
echo "║  Figures:                                              ║"
echo "║    🖼  outputs/figures/fig1_attention_weights.pdf      ║"
echo "║    🖼  outputs/figures/fig2_roc_curves.pdf             ║"
echo "║    🖼  outputs/figures/fig3_ablation_results.pdf       ║"
echo "║    🖼  outputs/figures/fig4_confusion_matrices.pdf     ║"
echo "║    🖼  outputs/figures/figS1_training_curves.pdf       ║"
echo "║                                                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Download the outputs/ folder to get all results and figures."
