# Does Attention Reflect Clinical Importance? Evidence from Multi-Magnification Colposcopy

> **Official code for the paper:**  
> *"Does Attention Reflect Clinical Importance? Evidence from Multi-Magnification Colposcopy"*  
> Oussama EL GARRAI, Othmane El Meslouhi, Karim Abouelmehdi  
> Submitted to *Computerized Medical Imaging and Graphics* (Elsevier), 2026.

---

## Overview

This repository contains the full experimental pipeline used to study whether **attention weights in a multiple instance learning (MIL) model truly reflect the clinical importance of each colposcopy magnification level**.

Standard digital colposcopy protocols capture images at four magnification levels — **1×, 3×, 5×, and 7×** — before and after acetic acid application. We train a **ResNet18-based MagnificationAttentionMIL** model and then run the most comprehensive ablation study to date on a public multi-magnification colposcopy dataset (**BUTH**, 321 patients), covering **16 configurations and >80 trained models**.

### Key Findings

| Finding | Result |
|---|---|
| Full model AUC (post-acid) | **0.701 ± 0.025** (5-fold CV) |
| AUC on clean subset (strict labels) | **0.783 ± 0.051** |
| Attention hierarchy (Friedman test) | 7× > 5× > 3× > 1× (χ² = 443.6, p < 10⁻⁹⁵) |
| Synergy paradox | Removing lowest-attended 1× causes 2nd-largest drop (ΔAUC = −0.024) |
| Minimal efficient protocol | **1× + 5×** achieves AUC = 0.698, within 0.003 of full model |

> **Conclusion:** Attention weights correctly rank magnification importance *globally*, but ablation is required to reveal complementary inter-feature synergies. Low-attention features can act as structural scaffolding — essential context that amplifies high-attention views.

---

## Repository Structure

```
.
├── src/
│   ├── model.py              # MagnificationAttentionMIL architecture
│   ├── dataset.py            # PyTorch Dataset (multi-magnification loader)
│   ├── train.py              # 5-fold CV training with EMA, TTA, W&B logging
│   ├── preprocess_roi.py     # V6 cervix ROI detection & crop pipeline
│   ├── prepare_dataset.py    # Manifest CSV generator from raw BUTH dataset
│   ├── ensemble_pre_post.py  # Pre/post-acid ensemble utility
│   └── utils.py              # Seeds, logging, metrics, FocalLoss, checkpoints
│
├── evaluate/
│   ├── ablation_study.py     # 16-configuration ablation study runner
│   ├── evaluate_results.py   # Fold aggregation, attention analysis, ensemble
│   ├── statistical_tests.py  # Bootstrap CIs, DeLong test, McNemar test
│   └── wilcoxon_fold_aucs.py # Wilcoxon signed-rank tests on paired fold AUCs
│
├── visualize/
│   ├── plot_attention_weights.py     # Fig. 1: attention distribution plots
│   ├── plot_roc_curves.py            # Fig. 2: ROC curves
│   ├── plot_ablation_results.py      # Fig. 3: ablation bar chart
│   ├── plot_confusion_matrices.py    # Fig. S: confusion matrices
│   ├── plot_pr_curves.py             # Precision-Recall curves
│   ├── plot_calibration.py           # Calibration / reliability diagram
│   ├── plot_prediction_distribution.py
│   └── plot_training_curves.py       # Training convergence plots
│
├── run_full_study.py         # Orchestrates all 3 experimental phases
├── run_all.sh                # End-to-end pipeline runner (bash)
├── create_manifests.py       # Generates corrected/strict label variants
├── config_final.yaml         # Configuration used for the paper results
├── config.yaml               # Default configuration
└── requirements.txt
```

---

## Dataset

This code is designed for the **BUTH multi-magnification colposcopy dataset**:

> Ogunlaja, A.O., et al. (2025). *A multi-magnification digital colposcopy image dataset for cervical cancer screening in Nigeria.* Data in Brief. To appear.

**Dataset structure expected:**
```
dataset/
├── 01_Normal/
│   ├── buth_001/
│   │   ├── p1.jpg   # pre-acid, 1×
│   │   ├── p2.jpg   # pre-acid, 3×
│   │   ├── p3.jpg   # pre-acid, 5×
│   │   ├── p4.jpg   # pre-acid, 7×
│   │   ├── po1.jpg  # post-acid, 1×
│   │   ├── po2.jpg  # post-acid, 3×
│   │   ├── po3.jpg  # post-acid, 5×
│   │   └── po4.jpg  # post-acid, 7×
│   └── ...
└── 02_Abnormal/
    └── ...
```

Update `preprocessing.input_dir` in `config_final.yaml` to point to your dataset location.

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/magnification-attention-mil.git
cd magnification-attention-mil
```

**2. Install PyTorch** (match your CUDA version)
```bash
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. Install remaining dependencies**
```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Option A — Full automated pipeline (bash)

```bash
chmod +x run_all.sh
./run_all.sh
```

This runs all 8 steps sequentially: preprocessing → manifest → training → ablation → evaluation → figures.

---

### Option B — Step by step

#### Step 1: Preprocessing (ROI crop + resize)

```bash
python src/preprocess_roi.py --config config_final.yaml
```

Applies the V6 cervix ROI detection pipeline: text overlay masking, skin rejection, mucosa saliency, and magnification-aware cropping. Outputs preprocessed images to `preprocessing.output_dir`.

#### Step 2: Generate dataset manifest

```bash
python src/prepare_dataset.py --config config_final.yaml
```

Scans the preprocessed dataset and produces `outputs/manifeste.csv` with resolved paths for all 8 images per patient.

#### Step 3: Create label variants (for sensitivity analysis)

```bash
python create_manifests.py
```

Produces `manifeste_corrige.csv` (ASC-US reclassified as Abnormal) and `manifeste_strict.csv` (ambiguous cases excluded).

#### Step 4: Train the full model

```bash
# Post-acid (main experiment)
python src/train.py --condition post --config config_final.yaml --run_name baseline_post

# Pre-acid (comparison)
python src/train.py --condition pre --config config_final.yaml --run_name baseline_pre
```

Runs 5-fold stratified group cross-validation. Saves checkpoints, per-fold predictions, and a JSON summary.

#### Step 5: Run the ablation study (all 16 configurations)

```bash
python run_full_study.py --phases 3 --config config_final.yaml
```

Or run the full three-phase study (baseline + sensitivity + ablation):

```bash
python run_full_study.py --config config_final.yaml
```

Individual ablation configurations can also be run directly via `train.py`:

```bash
# Single magnification (1× only)
python src/train.py --condition post --mags 0 --run_name ablation_1x_only

# Two-magnification pair (1× + 5×)
python src/train.py --condition post --mags 0,2 --run_name ablation_1x_5x

# Leave-one-out (no 1×)
python src/train.py --condition post --mags 1,2,3 --run_name ablation_no_1x

# No attention (mean pooling)
python src/train.py --condition post --pooling mean --run_name ablation_no_attention
```

#### Step 6: Evaluate and run statistical tests

```bash
python evaluate/evaluate_results.py --config config_final.yaml
python evaluate/statistical_tests.py --config config_final.yaml
python evaluate/wilcoxon_fold_aucs.py --config config_final.yaml
```

#### Step 7: Generate publication figures

```bash
python visualize/plot_attention_weights.py --config config_final.yaml
python visualize/plot_roc_curves.py --config config_final.yaml
python visualize/plot_ablation_results.py --config config_final.yaml
python visualize/plot_confusion_matrices.py --config config_final.yaml
python visualize/plot_training_curves.py --config config_final.yaml
```

---

## Configuration

The main configuration file used for all paper results is `config_final.yaml`. Key hyperparameters:

| Parameter | Value | Description |
|---|---|---|
| `training.backbone` | `resnet18` | Shared feature extractor (11.7M params) |
| `training.feature_dim` | `512` | Feature dimension per magnification |
| `model.gated_attention` | `true` | Use gated attention (Ilse et al., 2018) |
| `model.attention_dim` | `128` | Attention hidden dimension |
| `model.dropout` | `0.30` | Classifier head dropout |
| `training.lr_backbone` | `5e-5` | Backbone learning rate |
| `training.lr_head` | `3e-4` | Attention + classifier learning rate |
| `training.ema_decay` | `0.998` | EMA weight averaging decay |
| `loss.pos_weight` | `1.75` | BCE positive class weight (calibrated to 203:118 ratio) |
| `training.epochs` | `150` | Max epochs (early stopping patience = 30) |
| `data.image_size` | `448` | Input image size (px) |

---

## Model Architecture

```
Input: [B, 4, 3, 448, 448]  (batch × magnifications × RGB × H × W)
        ↓
Shared ResNet18 backbone (ImageNet pretrained, weight-sharing across magnifications)
        ↓  [B×4, 512]
Gated Attention Pooling (Ilse et al., ICML 2018)
  a_m = W_a · (tanh(V · f_m) ⊙ σ(U · f_m))
  α_m = softmax(a_m)            → interpretable magnification weights
        ↓  [B, 512]
Attention-weighted aggregation: f_patient = Σ α_m · f_m
        ↓
Classifier: Linear(512→256) → ReLU → Dropout(0.3) → Linear(256→1)
        ↓
Logit → sigmoid → P(Abnormal)
```

---

## Experiment Tracking (W&B)

Weights & Biases logging is supported. Enable it in `config.yaml`:

```yaml
wandb:
  enabled: true
  project: "magnification-attention-mil"
  entity: "your-entity"
  tags: ["colposcopy", "MIL", "ablation"]
```

---

## Results

### Baseline Performance (post-acid, 5-fold CV)

| Condition | AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|
| Post-acid | **0.701 ± 0.025** | 0.677 ± 0.069 | 0.735 ± 0.079 | 0.635 ± 0.017 |
| Pre-acid | 0.684 ± 0.055 | 0.780 ± 0.060 | 0.615 ± 0.106 | 0.640 ± 0.032 |
| Corrected labels | 0.718 ± 0.018 | 0.809 ± 0.079 | 0.622 ± 0.098 | 0.696 ± 0.037 |
| Strict subset (n=234) | **0.783 ± 0.051** | 0.726 ± 0.128 | 0.757 ± 0.125 | 0.656 ± 0.039 |

### Attention Hierarchy

| Magnification | Mean α | vs. 1× (Wilcoxon) |
|---|---|---|
| 1× (overview) | 0.106 ± 0.080 | — |
| 3× | 0.254 ± 0.093 | p < 10⁻⁴⁹ |
| 5× | 0.315 ± 0.095 | p < 10⁻⁵¹ |
| 7× (close-up) | 0.326 ± 0.109 | p < 10⁻⁵¹ |

### Key Ablation Results (ΔAUC from full model)

| Configuration | AUC | ΔAUC |
|---|---|---|
| Full model (4 mags + attention) | 0.701 | — |
| No attention (mean pooling) | 0.697 | −0.004 |
| 1× + 5× only | **0.698** | **−0.003** |
| Without 7× | 0.671 | −0.030 |
| Without 1× | 0.677 | −0.024 |
| 7× only | 0.680 | −0.021 |
| 1× only | 0.667 | −0.034 |

---

## Citation

If you use this code or the BUTH dataset analysis in your work, please cite:

```bibtex
@article{elgarrai2026attention,
  title     = {Does Attention Reflect Clinical Importance? Evidence from Multi-Magnification Colposcopy},
  author    = {EL GARRAI, Oussama and El Meslouhi, Othmane and Abouelmehdi, Karim},
  journal   = {Computerized Medical Imaging and Graphics},
  year      = {2026},
  publisher = {Elsevier},
  note      = {Under review}
}
```

Please also cite the BUTH dataset:

```bibtex
@article{ogunlaja2025multi,
  title   = {A multi-magnification digital colposcopy image dataset for cervical cancer screening in Nigeria},
  author  = {Ogunlaja, A.O. and Adeleke, O.T. and Ughonu, P.C. and others},
  journal = {Data in Brief},
  year    = {2025}
}
```

---

## License

This code is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Contact

**Oussama EL GARRAI** — ELITES Laboratory, Faculty of Sciences, Chouaib Doukkali University, El Jadida, Morocco  
✉ elgarrai.oussama@ucd.ac.ma
